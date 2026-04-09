"""
train.py — Train the toothbrush defect detection model.

Strategy: per-pixel Gaussian model in LAB colour space.
  - Parallel image loading with ThreadPoolExecutor
  - Threshold tuned by maximising F1-score against ground-truth masks

Usage:
    python3 train.py
"""

import os
import numpy as np
import cv2
from concurrent.futures import ThreadPoolExecutor

# ── config ───────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.join(os.path.dirname(__file__), "toothbrush")
GOOD_DIR   = os.path.join(BASE_DIR, "train", "good")
DEFECT_DIR = os.path.join(BASE_DIR, "train", "defective")
GT_DIR     = os.path.join(BASE_DIR, "ground_truth", "defective")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "defect_model.npz")

IMG_SIZE       = (1024, 1024)
SMOOTH_K       = 7
EPSILON        = 1e-6
N_THRESH_STEPS = 400
N_WORKERS      = 8


# ── contrast enhancement ─────────────────────────────────────────────────────

def enhance_contrast(img_bgr):
    """
    Percentile contrast stretching on the L channel.

    Stretches the 2nd–98th percentile of L values to the full 0–255 range.
    This is a hard, direct contrast boost (no smoothing, no tiles):
      - Images that are globally dim get boosted
      - Images that are globally bright get normalised
      - Fine local differences (bristle defects) become proportionally larger
    Applied consistently at train and inference so the statistical model
    sees a normalised, high-contrast feature space.
    """
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2Lab)
    l, a, b = cv2.split(lab)

    lo, hi = np.percentile(l, 2), np.percentile(l, 98)
    if hi > lo:
        l_stretched = np.clip((l.astype(np.float32) - lo) * 255.0 / (hi - lo), 0, 255).astype(np.uint8)
    else:
        l_stretched = l

    return cv2.cvtColor(cv2.merge([l_stretched, a, b]), cv2.COLOR_Lab2BGR)


# ── parallel image loading ───────────────────────────────────────────────────

def _load_and_resize(path):
    img = cv2.imread(path)
    if img is None:
        return None
    img = cv2.resize(img, IMG_SIZE, interpolation=cv2.INTER_AREA)
    return enhance_contrast(img)


def load_images(directory, suffix=""):
    paths = sorted(
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if f.endswith(".png") and suffix in f
    )
    with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
        imgs = list(ex.map(_load_and_resize, paths))
    imgs = [img for img in imgs if img is not None]
    print(f"  Loaded {len(imgs)} images from {os.path.basename(directory)}/")
    return imgs


def to_lab(img_bgr):
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2Lab).astype(np.float32)


# ── 1. Build Gaussian model from good images ─────────────────────────────────

print("Loading good images …")
good_imgs = load_images(GOOD_DIR)
good_lab  = np.stack([to_lab(img) for img in good_imgs], axis=0)  # (N, H, W, 3)

mean_lab = good_lab.mean(axis=0)   # (H, W, 3)
std_lab  = good_lab.std(axis=0)    # (H, W, 3)

# Clamp std from below to avoid division by near-zero in stable regions
std_floor = np.percentile(std_lab, 5)
std_lab   = np.maximum(std_lab, max(std_floor, 0.5))

print(f"  Mean range : {mean_lab.min():.2f} – {mean_lab.max():.2f}")
print(f"  Std  range : {std_lab.min():.4f} – {std_lab.max():.2f}  (floor={std_floor:.4f})")


# ── anomaly score helper ─────────────────────────────────────────────────────

def anomaly_score(img_bgr):
    """Return (H_orig, W_orig) float32 anomaly map, background pixels zeroed."""
    h_orig, w_orig = img_bgr.shape[:2]
    small = cv2.resize(img_bgr, IMG_SIZE, interpolation=cv2.INTER_AREA)
    small = enhance_contrast(small)
    lab   = to_lab(small)
    z     = np.abs((lab - mean_lab) / (std_lab + EPSILON))
    score = z.max(axis=2)
    score = cv2.GaussianBlur(score, (SMOOTH_K, SMOOTH_K), 0)

    # Zero out dark background pixels so they don't influence threshold tuning
    gray       = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    foreground = gray > 30
    score[~foreground] = 0.0

    return cv2.resize(score, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR)


# ── 2. Tune threshold on defective images ────────────────────────────────────

print("\nLoading defective images and ground-truth masks …")
def_imgs = load_images(DEFECT_DIR)
gt_raw   = load_images(GT_DIR, suffix="_mask")

gt_masks = []
for m in gt_raw:
    gray = cv2.cvtColor(m, cv2.COLOR_BGR2GRAY) if m.ndim == 3 else m
    gt_masks.append((gray > 127).astype(np.uint8))

print("Computing anomaly scores …")
all_scores = []
all_labels = []

for img, gt in zip(def_imgs, gt_masks):
    all_scores.append(anomaly_score(img).ravel())
    gt_up = cv2.resize(gt, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
    all_labels.append(gt_up.ravel())

for img in good_imgs:
    all_scores.append(anomaly_score(img).ravel())
    all_labels.append(np.zeros(IMG_SIZE[0] * IMG_SIZE[1], dtype=np.uint8))

all_scores = np.concatenate(all_scores)
all_labels = np.concatenate(all_labels)

print(f"  Total pixels: {len(all_labels):,}   positives: {all_labels.sum():,}")

best_f1, best_thresh = 0.0, 1.0
lo, hi = np.percentile(all_scores, 1), np.percentile(all_scores, 99.5)
for t in np.linspace(lo, hi, N_THRESH_STEPS):
    pred = (all_scores >= t).astype(np.uint8)
    tp   = ((pred == 1) & (all_labels == 1)).sum()
    fp   = ((pred == 1) & (all_labels == 0)).sum()
    fn   = ((pred == 0) & (all_labels == 1)).sum()
    if tp == 0:
        continue
    prec = tp / (tp + fp + 1e-9)
    rec  = tp / (tp + fn + 1e-9)
    f1   = 2 * prec * rec / (prec + rec + 1e-9)
    if f1 > best_f1:
        best_f1, best_thresh = f1, t

print(f"\n  Best threshold : {best_thresh:.4f}")
print(f"  Best F1-score  : {best_f1:.4f}")


# ── 3. Save model ─────────────────────────────────────────────────────────────

np.savez_compressed(
    MODEL_PATH,
    mean_lab  = mean_lab,
    std_lab   = std_lab,
    threshold = np.float32(best_thresh),
    img_size  = np.array(IMG_SIZE),
)
print(f"\nModel saved → {MODEL_PATH}")
