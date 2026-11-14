"""
NutriGenus - live face screening with region marking
====================================================

Pipeline (per webcam frame):
  1. MediaPipe Face Landmarker  -> 478 face landmarks (face-only gate: no face = no prediction)
  2. Quality check              -> rejects dark / blurry / tiny faces
  3. EfficientNetV2-S           -> vitamin class probabilities on the face crop
  4. Grad-CAM                   -> where the model looked
  5. Landmark regions           -> forehead, cheeks, under-eyes, nose, lips, chin are scored
                                   against the Grad-CAM map and the strongest are highlighted
  6. Temporal smoothing         -> probabilities / region scores averaged over time (no flicker)

IMPORTANT: the highlighted regions show where the MODEL's attention was. They are not a
clinical finding. This is a screening aid, not a diagnosis.

Usage
-----
    pip install torch torchvision opencv-python mediapipe numpy
    python live_face_vitamin_screen.py \
        --checkpoint ./runs/effnetv2s/best_model.pt \
        --data_split ./runs/effnetv2s/data_split.json

Keys:  q = quit    h = toggle full-face heatmap
"""

import argparse
import json
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from torchvision.models import efficientnet_v2_s

from face_regions import (build_regions, region_scores, face_box_from_landmarks,
                          quality_check, REGION_NAMES)

LANDMARKER_URL = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/"
                  "face_landmarker/float16/1/face_landmarker.task")
LANDMARKER_PATH = Path("face_landmarker.task")
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)


# ---------------------------------------------------------------- model + Grad-CAM
def load_model(checkpoint, num_classes, device):
    model = efficientnet_v2_s(weights=None)
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    return model.to(device).eval()


class GradCAMHelper:
    """Captures last-conv activations and their gradients (no backward hooks -> safe with in-place SiLU)."""

    def __init__(self, layer):
        self.acts = None
        self.grads = None
        layer.register_forward_hook(self._on_forward)

    def _on_forward(self, module, inputs, output):
        self.acts = output
        if output.requires_grad:
            output.register_hook(lambda g: setattr(self, "grads", g))


def preprocess(face_bgr, device):
    rgb = cv2.cvtColor(cv2.resize(face_bgr, (224, 224)), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    rgb = (rgb - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(rgb.transpose(2, 0, 1)).unsqueeze(0).to(device)


def predict_with_cam(model, helper, tensor, use_half):
    model.zero_grad(set_to_none=True)
    with torch.autocast(device_type=tensor.device.type, dtype=torch.float16, enabled=use_half):
        logits = model(tensor)
    probs = torch.softmax(logits.float(), dim=1)[0]
    top = int(probs.argmax())
    logits[0, top].float().backward()
    weights = helper.grads.float().mean(dim=(2, 3), keepdim=True)
    cam = torch.relu((weights * helper.acts.detach().float()).sum(dim=1))[0]
    cam = cam / (cam.max() + 1e-8)
    return probs.detach().cpu().numpy(), cam.cpu().numpy()


# ---------------------------------------------------------------- drawing helpers
def ensure_landmarker():
    if not LANDMARKER_PATH.exists():
        print(f"Downloading face landmarker model to {LANDMARKER_PATH} ...")
        urllib.request.urlretrieve(LANDMARKER_URL, LANDMARKER_PATH)


def put_text(img, text, org, scale=0.6, color=(255, 255, 255), thick=1):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def draw_region(frame, mask, score, label):
    overlay = frame.copy()
    overlay[mask > 0] = (0, 0, 255)
    alpha = 0.20 + 0.35 * min(1.0, score)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(frame, contours, -1, (0, 0, 255), 2)
    ys, xs = np.nonzero(mask)
    if len(xs):
        put_text(frame, f"{label} {score:.2f}", (int(xs.mean()) - 40, int(ys.mean())), 0.45)


def heatmap_overlay(frame, cam_full):
    heat = cv2.applyColorMap((cam_full * 255).astype(np.uint8), cv2.COLORMAP_JET)
    sel = cam_full > 0.05
    out = frame.copy()
    out[sel] = cv2.addWeighted(frame, 0.55, heat, 0.45, 0)[sel]
    return out


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data_split", required=True, help="data_split.json from training (holds class names).")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--every", type=int, default=2, help="Run the classifier every N frames.")
    ap.add_argument("--conf_thr", type=float, default=0.60, help="Below this the result is 'Inconclusive'.")
    ap.add_argument("--region_thr", type=float, default=0.50, help="Highlight regions scoring above this.")
    ap.add_argument("--max_regions", type=int, default=3)
    ap.add_argument("--ema", type=float, default=0.30, help="Smoothing factor (higher = faster, noisier).")
    ap.add_argument("--half", action="store_true", help="fp16 autocast (faster; try only if you need FPS).")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_names = json.loads(Path(args.data_split).read_text())["class_names"]
    model = load_model(args.checkpoint, len(class_names), device)
    helper = GradCAMHelper(model.features[-1])
    print(f"Device: {device} | classes: {class_names}")

    ensure_landmarker()
    landmarker = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(LANDMARKER_PATH)),
        running_mode=vision.RunningMode.VIDEO, num_faces=1))

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit("Could not open the camera.")

    t_start, last_ts, frame_i = time.time(), -1, 0
    ema_probs, region_ema = None, {n: 0.0 for n in REGION_NAMES}
    small_cam = None
    show_heat, fps, t_prev = False, 0.0, time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_i += 1
        h, w = frame.shape[:2]

        ts = max(last_ts + 1, int((time.time() - t_start) * 1000))
        last_ts = ts
        rgb = np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        result = landmarker.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts)

        status = "No face detected"
        if result.face_landmarks:
            lms = result.face_landmarks[0]
            pts = np.array([[p.x * w, p.y * h] for p in lms], np.float32)
            x0, y0, x1, y1 = face_box_from_landmarks(pts, frame.shape)
            crop = frame[y0:y1, x0:x1]
            good, msg = quality_check(crop) if crop.size else (False, "Face out of frame")

            if not good:
                status = msg
                ema_probs, small_cam = None, None
            else:
                if frame_i % args.every == 0 or small_cam is None:
                    probs, small_cam = predict_with_cam(model, helper, preprocess(crop, device), args.half)
                    ema_probs = probs if ema_probs is None else args.ema * probs + (1 - args.ema) * ema_probs

                # paste the (7x7) Grad-CAM back into frame coordinates, aligned with the CURRENT face box
                cam_full = np.zeros((h, w), np.float32)
                cam_full[y0:y1, x0:x1] = cv2.resize(small_cam, (x1 - x0, y1 - y0), interpolation=cv2.INTER_CUBIC).clip(0, 1)
                masks = build_regions(pts, frame.shape)
                raw = region_scores(cam_full, masks)
                for n in REGION_NAMES:
                    region_ema[n] = args.ema * raw[n] + (1 - args.ema) * region_ema[n]

                if show_heat:
                    frame = heatmap_overlay(frame, cam_full)

                top_idx = int(np.argmax(ema_probs))
                conf = float(ema_probs[top_idx])
                if conf >= args.conf_thr:
                    status = f"{class_names[top_idx]}  ({conf * 100:.0f}%)"
                    ranked = sorted(region_ema.items(), key=lambda kv: kv[1], reverse=True)
                    shown = [(n, s) for n, s in ranked[:args.max_regions] if s >= args.region_thr]
                    for n, s in shown:
                        draw_region(frame, masks[n], s, n)
                    if shown:
                        put_text(frame, "Marked: " + ", ".join(n for n, _ in shown), (10, 58), 0.5, (0, 200, 255))
                else:
                    status = f"Inconclusive (best: {class_names[top_idx]} {conf * 100:.0f}%)"

            cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 255, 0), 1)

        now = time.time()
        fps = 0.9 * fps + 0.1 * (1.0 / max(now - t_prev, 1e-6))
        t_prev = now
        put_text(frame, status, (10, 28), 0.8, (0, 255, 255), 2)
        put_text(frame, f"{fps:.0f} FPS | q: quit  h: heatmap", (10, h - 32), 0.5)
        put_text(frame, "Screening aid only - not a medical diagnosis", (10, h - 10), 0.5, (200, 200, 255))
        cv2.imshow("NutriGenus live screening", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("h"):
            show_heat = not show_heat

    cap.release()
    landmarker.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
