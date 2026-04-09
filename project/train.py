"""
train.py — Train the toothbrush defect detection model.

Strategy: per-pixel Gaussian model in LAB colour space.
  - Preprocessing: CLAHE on the L channel equalises brightness across images
    taken under slightly different lighting, tightening the good-image std so
    defect z-scores are larger and easier to threshold.
    (Denoising and sharpening were tested but hurt F1: denoising blurs out
    bristle-tip defects; unsharp masking inflates variance among good images.)
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

# ── preprocessing parameters ─────────────────────────────────────────────────
DENOISE_H      = 3     # fastNlMeans strength — keep low to preserve defect texture
CLAHE_CLIP     = 1.0   # gentle clip limit; 2.0 over-equalises and washes out defects
CLAHE_GRID     = (2, 2) # 4×4 tiles on 1024px = 128px tiles — more global, less aggressive
UNSHARP_SIGMA  = 1.0   # small sigma — sharpen fine structure without amplifying noise
UNSHARP_AMOUNT = 0.5   # conservative blend — subtle edge boost only


# ── preprocessing pipeline ───────────────────────────────────────────────────

_clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=CLAHE_GRID)

def preprocess(img_bgr):
    """
    Three-stage preprocessing pipeline applied identically at train and inference.

    1. Denoising (gentle) — fastNlMeansDenoisingColored with h=3 removes sensor
       noise without blurring bristle-tip structure. Strength is kept low so
       actual defects (which are texture anomalies) are not smoothed away.

    2. Normalisation — CLAHE on the L channel of LAB space corrects for
       per-image brightness differences. Clip=1.0 and 4×4 tiles keep it gentle:
       aggressive settings (clip=2.0, 8×8) over-equalise locally and wash out
       the texture differences that reveal defects.

    3. Contrast enhancement (gentle) — unsharp mask on L with amount=0.5
       slightly boosts edges and fine structure without inflating inter-image
       variance the way a strong sharpening would.
    """
    # 1. Denoising
    denoised = cv2.fastNlMeansDenoisingColored(img_bgr, None,
                                               h=DENOISE_H, hColor=DENOISE_H,
                                               templateWindowSize=7,
                                               searchWindowSize=21)
    # 2. Normalisation — gentle CLAHE on L channel only
    lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2Lab)
    l, a, b = cv2.split(lab)
    l_eq = _clahe.apply(l)

    # 3. Contrast enhancement — unsharp mask on equalised L
    blur    = cv2.GaussianBlur(l_eq, (0, 0), UNSHARP_SIGMA)
    l_sharp = np.clip(
        cv2.addWeighted(l_eq, 1 + UNSHARP_AMOUNT, blur, -UNSHARP_AMOUNT, 0),
        0, 255
    ).astype(np.uint8)

    return cv2.cvtColor(cv2.merge([l_sharp, a, b]), cv2.COLOR_Lab2BGR)


# ── parallel image loading ───────────────────────────────────────────────────

def _load_and_resize(path):
    img = cv2.imread(path)
    if img is None:
        return None
    img = cv2.resize(img, IMG_SIZE, interpolation=cv2.INTER_AREA)
    return preprocess(img)


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
    """Return (H_orig, W_orig) float32 anomaly map for a BGR image."""
    h_orig, w_orig = img_bgr.shape[:2]
    small = cv2.resize(img_bgr, IMG_SIZE, interpolation=cv2.INTER_AREA)
    small = preprocess(small)   # same pipeline as training
    lab   = to_lab(small)
    z     = np.abs((lab - mean_lab) / (std_lab + EPSILON))
    score = z.max(axis=2)
    score = cv2.GaussianBlur(score, (SMOOTH_K, SMOOTH_K), 0)
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
    mean_lab       = mean_lab,
    std_lab        = std_lab,
    threshold      = np.float32(best_thresh),
    img_size       = np.array(IMG_SIZE),
    denoise_h      = np.float32(DENOISE_H),
    clahe_clip     = np.float32(CLAHE_CLIP),
    clahe_grid     = np.array(CLAHE_GRID),
    unsharp_sigma  = np.float32(UNSHARP_SIGMA),
    unsharp_amount = np.float32(UNSHARP_AMOUNT),
)
print(f"\nModel saved → {MODEL_PATH}")
