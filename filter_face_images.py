"""
Face-only dataset filter for NutriGenus
=======================================

The Kaggle vitamin-deficiency dataset mixes body parts (nails, tongue, eyes, skin, ...).
This script keeps ONLY the images that contain a clearly visible face, so the model is
trained on the same kind of input it will see from the live camera.

What it does
------------
1. Walks  <data_dir>/<class_name>/*.jpg|png|jpeg  (retina / mask folders are skipped).
2. Runs the MediaPipe Face Detector on every image.
3. KEEPS an image only if a face is found with
      confidence   >= --min_conf   (default 0.60)
      face width   >= --min_face_frac of the image width (default 0.20)
   -> close-ups of a nail, tongue, hand or a lone eye/lip patch are rejected.
4. Saves the kept images (cropped to the face + margin by default) to <out_dir>/<class_name>/.
   Cropping makes the training images match what the live pipeline feeds the model.
5. Optionally drops near-duplicate images (--dedupe) so copies cannot land in both the
   train and test splits and inflate accuracy.
6. Writes report.csv (one row per image, with the reason for every rejection) and prints
   a per-class summary. Classes left with very few face images are flagged.

Usage
-----
    pip install mediapipe opencv-python pillow numpy
    python filter_face_images.py --data_dir /path/to/kaggle_dataset --out_dir ./faces_only --dedupe

Then train on the filtered folder (the training script is unchanged):
    python train_efficientnetv2s.py --data_dir ./faces_only --output_dir ./runs/effnetv2s

Always spot-check ~20 kept and ~20 rejected images per class by eye (rejected ones are
copied to <out_dir>_rejected/ with --save_rejected).
"""

import argparse
import csv
import shutil
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image, ImageFile

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

ImageFile.LOAD_TRUNCATED_IMAGES = True

VALID_EXTS = {".png", ".jpg", ".jpeg"}
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/face_detector/"
             "blaze_face_short_range/float16/1/blaze_face_short_range.tflite")
MODEL_PATH = Path("blaze_face_short_range.tflite")
MAX_SIDE = 1024  # detection runs on a downscaled copy for speed

POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def is_excluded_folder(name: str) -> bool:
    n = name.lower()
    return "retina" in n or "mask" in n


def ensure_model():
    if not MODEL_PATH.exists():
        print(f"Downloading face detector model to {MODEL_PATH} ...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)


def make_detector(min_conf: float):
    options = vision.FaceDetectorOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=vision.RunningMode.IMAGE,
        min_detection_confidence=min_conf,
    )
    return vision.FaceDetector.create_from_options(options)


def detect_best_face(detector, image: Image.Image):
    """Return (x, y, w, h, score) in ORIGINAL image pixels, or None."""
    w0, h0 = image.size
    scale = min(1.0, MAX_SIDE / max(w0, h0))
    small = image if scale == 1.0 else image.resize((int(w0 * scale), int(h0 * scale)), Image.LANCZOS)
    rgb = np.ascontiguousarray(np.asarray(small))
    result = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
    if not result.detections:
        return None
    det = max(result.detections, key=lambda d: d.categories[0].score)
    bb = det.bounding_box
    return (bb.origin_x / scale, bb.origin_y / scale, bb.width / scale, bb.height / scale,
            float(det.categories[0].score))


def crop_with_margin(image: Image.Image, box, margin: float):
    x, y, w, h, _ = box
    cx, cy = x + w / 2, y + h / 2
    side_w, side_h = w * (1 + 2 * margin), h * (1 + 2 * margin)
    left = max(0, int(cx - side_w / 2))
    top = max(0, int(cy - side_h / 2))
    right = min(image.width, int(cx + side_w / 2))
    bottom = min(image.height, int(cy + side_h / 2))
    return image.crop((left, top, right, bottom))


def average_hash(image: Image.Image) -> np.uint64:
    g = np.asarray(image.convert("L").resize((8, 8), Image.LANCZOS), dtype=np.float32)
    bits = (g > g.mean()).flatten()
    return np.packbits(bits).view(np.uint64)[0]


def hamming_to_all(hashes: np.ndarray, h: np.uint64) -> np.ndarray:
    xor = np.bitwise_xor(hashes, h).view(np.uint8).reshape(-1, 8)
    return POPCOUNT[xor].sum(axis=1)


def main():
    p = argparse.ArgumentParser(description="Keep only face images from the vitamin dataset.")
    p.add_argument("--data_dir", required=True)
    p.add_argument("--out_dir", default="./faces_only")
    p.add_argument("--min_conf", type=float, default=0.60, help="Minimum face-detector confidence.")
    p.add_argument("--min_face_frac", type=float, default=0.20,
                   help="Minimum face width as a fraction of image width (rejects tiny faces / body shots).")
    p.add_argument("--margin", type=float, default=0.25, help="Extra context around the face crop.")
    p.add_argument("--no_crop", action="store_true", help="Save the full image instead of the face crop.")
    p.add_argument("--dedupe", action="store_true", help="Drop near-duplicate images (average-hash).")
    p.add_argument("--dup_dist", type=int, default=3, help="Max Hamming distance counted as duplicate.")
    p.add_argument("--save_rejected", action="store_true", help="Copy rejected images to <out_dir>_rejected/.")
    p.add_argument("--min_class_images", type=int, default=50,
                   help="Warn if a class has fewer kept images than this.")
    args = p.parse_args()

    data_dir, out_dir = Path(args.data_dir), Path(args.out_dir)
    rejected_dir = Path(str(out_dir) + "_rejected")
    ensure_model()
    detector = make_detector(args.min_conf)

    class_dirs = sorted(d for d in data_dir.iterdir() if d.is_dir() and not is_excluded_folder(d.name))
    if not class_dirs:
        raise SystemExit(f"No class folders found in {data_dir}")

    rows = []
    stats = {}
    hash_store = np.zeros(200000, dtype=np.uint64)
    n_hashes = 0

    for class_dir in class_dirs:
        files = sorted(f for f in class_dir.rglob("*")
                       if f.is_file() and f.suffix.lower() in VALID_EXTS
                       and not is_excluded_folder(f.parent.name))
        kept = 0
        (out_dir / class_dir.name).mkdir(parents=True, exist_ok=True)
        print(f"\n[{class_dir.name}] {len(files)} images")

        for idx, f in enumerate(files):
            reason, score, frac = "", 0.0, 0.0
            try:
                img = Image.open(f).convert("RGB")
            except Exception as e:
                rows.append([str(f), class_dir.name, 0, f"unreadable ({e})", 0, 0])
                continue

            box = detect_best_face(detector, img)
            if box is None:
                reason = "no face"
            else:
                score, frac = box[4], box[2] / img.width
                if frac < args.min_face_frac:
                    reason = "face too small in frame"

            result_img = None
            if not reason:
                result_img = img if args.no_crop else crop_with_margin(img, box, args.margin)
                if args.dedupe:
                    h = average_hash(result_img)
                    if n_hashes and (hamming_to_all(hash_store[:n_hashes], h) <= args.dup_dist).any():
                        reason = "near-duplicate"
                    elif n_hashes < len(hash_store):
                        hash_store[n_hashes] = h
                        n_hashes += 1

            if reason:
                if args.save_rejected:
                    dst = rejected_dir / class_dir.name
                    dst.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(f, dst / f"{idx}_{f.name}")
                rows.append([str(f), class_dir.name, 0, reason, round(score, 3), round(frac, 3)])
            else:
                result_img.save(out_dir / class_dir.name / f"{f.stem}_{idx}.jpg", quality=95)
                rows.append([str(f), class_dir.name, 1, "kept", round(score, 3), round(frac, 3)])
                kept += 1

            if (idx + 1) % 200 == 0:
                print(f"  processed {idx + 1}/{len(files)}")

        stats[class_dir.name] = (kept, len(files))

    with open(out_dir / "report.csv", "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["path", "class", "kept", "reason", "face_conf", "face_width_frac"])
        writer.writerows(rows)

    print("\n================ SUMMARY ================")
    print(f"{'class':<28}{'kept':>8}{'total':>8}{'kept %':>9}")
    for name, (kept, total) in stats.items():
        pct = 100 * kept / total if total else 0
        flag = "   <-- too few face images" if kept < args.min_class_images else ""
        print(f"{name:<28}{kept:>8}{total:>8}{pct:>8.1f}%{flag}")
    print(f"\nFiltered dataset: {out_dir}\nReport: {out_dir / 'report.csv'}")
    print("Classes with almost no face images mean that vitamin has no visible FACE sign in this dataset:\n"
          "drop it from the face model (and say so in the paper) rather than training on a handful of images.")


if __name__ == "__main__":
    main()
