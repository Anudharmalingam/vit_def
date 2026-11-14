"""
Face-region helpers for the live NutriGenus pipeline (no PyTorch needed).

Turns the 478 MediaPipe Face Landmarker points into named face regions
(forehead, cheeks, under-eyes, nose, lips, chin) as binary masks, scores each
region against a Grad-CAM map, and checks that a frame is good enough to classify.

Landmark indices are MediaPipe FaceMesh indices. "right"/"left" are the SUBJECT's
own sides (the subject's right cheek appears on the left of an un-mirrored image).
"""

import cv2
import numpy as np

# Standard FaceMesh index sets
FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378,
             400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21,
             54, 103, 67, 109]
LIPS_OUTER = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37,
              39, 40, 185]
# Forehead polygon: upper face oval + both eyebrows (outer->inner, then across)
FOREHEAD = [54, 103, 67, 109, 10, 338, 297, 332, 284, 300, 293, 334, 296, 336,
            107, 66, 105, 63, 70]

REGION_NAMES = ["forehead", "right cheek", "left cheek", "right under-eye", "left under-eye",
                "nose", "lips / mouth area", "chin"]


def _poly_mask(shape, pts):
    m = np.zeros(shape[:2], np.uint8)
    cv2.fillPoly(m, [np.asarray(pts, np.int32)], 255)
    return m


def _ellipse_mask(shape, center, axes):
    m = np.zeros(shape[:2], np.uint8)
    cv2.ellipse(m, (int(center[0]), int(center[1])),
                (max(1, int(axes[0])), max(1, int(axes[1]))), 0, 0, 360, 255, -1)
    return m


def build_regions(pts: np.ndarray, shape):
    """pts: (478, 2) pixel landmarks. Returns {region_name: uint8 mask} limited to the face."""
    fw = float(np.linalg.norm(pts[454] - pts[234]))   # face width
    fh = float(np.linalg.norm(pts[152] - pts[10]))    # face height
    face_mask = _poly_mask(shape, pts[FACE_OVAL])

    masks = {}
    masks["forehead"] = _poly_mask(shape, pts[FOREHEAD])
    masks["right cheek"] = _ellipse_mask(shape, pts[205], (0.13 * fw, 0.10 * fh))
    masks["left cheek"] = _ellipse_mask(shape, pts[425], (0.13 * fw, 0.10 * fh))
    masks["right under-eye"] = _ellipse_mask(shape, pts[145] + [0, 0.045 * fh], (0.10 * fw, 0.035 * fh))
    masks["left under-eye"] = _ellipse_mask(shape, pts[374] + [0, 0.045 * fh], (0.10 * fw, 0.035 * fh))
    nose_center = (pts[6] + pts[4]) / 2
    masks["nose"] = _ellipse_mask(shape, nose_center, (0.07 * fw, 0.12 * fh))

    lips = _poly_mask(shape, cv2.convexHull(pts[LIPS_OUTER].astype(np.int32)).reshape(-1, 2))
    k = max(3, int(0.04 * fw)) | 1
    masks["lips / mouth area"] = cv2.dilate(lips, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    masks["chin"] = _ellipse_mask(shape, pts[152] - [0, 0.06 * fh], (0.13 * fw, 0.06 * fh))

    return {name: cv2.bitwise_and(m, face_mask) for name, m in masks.items()}


def region_scores(cam_full: np.ndarray, masks: dict) -> dict:
    """Mean Grad-CAM activation (0-1) inside each region."""
    scores = {}
    for name, m in masks.items():
        sel = m > 0
        scores[name] = float(cam_full[sel].mean()) if sel.any() else 0.0
    return scores


def face_box_from_landmarks(pts: np.ndarray, frame_shape, margin: float = 0.15):
    h, w = frame_shape[:2]
    x0, y0 = pts.min(axis=0)
    x1, y1 = pts.max(axis=0)
    bw, bh = x1 - x0, y1 - y0
    x0, x1 = max(0, int(x0 - margin * bw)), min(w, int(x1 + margin * bw))
    y0, y1 = max(0, int(y0 - margin * bh)), min(h, int(y1 + margin * bh))
    return x0, y0, x1, y1


def quality_check(crop_bgr: np.ndarray, min_side: int = 120):
    """Return (ok, message). Rejects tiny, dark, over-bright or blurry faces."""
    h, w = crop_bgr.shape[:2]
    if min(h, w) < min_side:
        return False, "Move closer"
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    mean = float(gray.mean())
    if mean < 50:
        return False, "Too dark - add light"
    if mean > 210:
        return False, "Too bright"
    if cv2.Laplacian(gray, cv2.CV_64F).var() < 40:
        return False, "Hold still / image blurry"
    return True, "OK"
