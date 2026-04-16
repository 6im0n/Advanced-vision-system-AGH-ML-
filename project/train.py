"""
train.py — Train the toothbrush defect detection model.

Strategy: per-pixel Gaussian model in LAB colour space.
  - Parallel image loading with ThreadPoolExecutor
  - Threshold tuned by maximising F1-score against ground-truth masks
  - Iterative FP removal: good images flagged by the model are dropped and
    the model is rebuilt until convergence
  - Void/hole detection: negative L-channel deviations are amplified so that
    missing-material defects (dark holes in bright bristle regions) are caught

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

IMG_SIZE        = (1024, 1024)
SMOOTH_K        = 13
EPSILON         = 1e-6
N_THRESH_STEPS  = 400
N_WORKERS       = 8

# Weight applied to the negative-L-deviation score so that dark holes
# (missing bristles) produce a stronger anomaly signal than the same
# magnitude of positive deviation.
VOID_WEIGHT = 1.5

# Fraction of the foreground area that must be flagged for a "good" image
# to be considered a false positive and removed from the training set.
FP_AREA_RATIO  = 0.001   # 1 % of foreground
MAX_REFINE_ITER = 5


# ── contrast enhancement ─────────────────────────────────────────────────────

def enhance_contrast(img_bgr):
    """
    Percentile contrast stretching on the L channel.

    Stretches the 2nd–98th percentile of L values to the full 0–255 range.
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


# ── model building ───────────────────────────────────────────────────────────

def build_model(good_imgs):
    """Return (mean_lab, std_lab) Gaussian model built from good images."""
    good_lab = np.stack([to_lab(img) for img in good_imgs], axis=0)  # (N, H, W, 3)
    m = good_lab.mean(axis=0)
    s = good_lab.std(axis=0)
    std_floor = np.percentile(s, 5)
    s = np.maximum(s, max(std_floor, 0.5))
    return m, s


# ── anomaly score ─────────────────────────────────────────────────────────────

def anomaly_score(img_bgr, m_lab, s_lab):
    """Return (H_orig, W_orig) float32 anomaly map, background pixels zeroed.

    Combines two signals:
      1. Max absolute z-score across LAB channels — catches colour/texture
         deviations in both directions.
      2. Void score — negative L deviation amplified by VOID_WEIGHT so that
         dark holes (missing bristles) produce a stronger signal than the
         symmetric z-score alone.
    """
    h_orig, w_orig = img_bgr.shape[:2]
    small = cv2.resize(img_bgr, IMG_SIZE, interpolation=cv2.INTER_AREA)
    small = enhance_contrast(small)
    lab   = to_lab(small)

    z     = np.abs((lab - m_lab) / (s_lab + EPSILON))
    score = z.max(axis=2)

    # Void/hole: amplify negative L deviation (missing material is darker)
    z_void = np.maximum(0, (m_lab[:, :, 0] - lab[:, :, 0]) / (s_lab[:, :, 0] + EPSILON))
    score  = np.maximum(score, z_void * VOID_WEIGHT)

    score = cv2.GaussianBlur(score, (SMOOTH_K, SMOOTH_K), 0)

    # Zero out dark background so it doesn't influence threshold tuning
    #gray       = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
   # score[gray <= 30] = 0.0

    return cv2.resize(score, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR)


# ── threshold tuning ─────────────────────────────────────────────────────────

def tune_threshold(good_imgs, def_imgs, gt_masks, m_lab, s_lab, tune_size=(256, 256)):
    """Tune the anomaly threshold by maximising pixel-level F1-score on a downsampled grid."""
    all_scores = []
    all_labels = []

    for img, gt in zip(def_imgs, gt_masks):
        score = anomaly_score(img, m_lab, s_lab)
        score_small = cv2.resize(score, tune_size, interpolation=cv2.INTER_AREA)
        gt_small = cv2.resize(gt, tune_size, interpolation=cv2.INTER_NEAREST)
        all_scores.append(score_small.ravel())
        all_labels.append(gt_small.ravel().astype(np.uint8))

    for img in good_imgs:
        score = anomaly_score(img, m_lab, s_lab)
        score_small = cv2.resize(score, tune_size, interpolation=cv2.INTER_AREA)
        all_scores.append(score_small.ravel())
        all_labels.append(np.zeros(score_small.size, dtype=np.uint8))

    all_scores = np.concatenate(all_scores).astype(np.float32, copy=False)
    all_labels = np.concatenate(all_labels).astype(np.uint8, copy=False)

    print(f"    Total sampled pixels: {len(all_labels):,}   positives: {all_labels.sum():,}")

    best_f1, best_thresh = 0.0, 1.0
    lo, hi = np.percentile(all_scores, 1), np.percentile(all_scores, 99.9)

    for t in np.linspace(lo, hi, N_THRESH_STEPS):
        pred = (all_scores >= t).astype(np.uint8)
        tp = ((pred == 1) & (all_labels == 1)).sum()
        fp = ((pred == 1) & (all_labels == 0)).sum()
        fn = ((pred == 0) & (all_labels == 1)).sum()

        if tp == 0:
            continue

        prec = tp / (tp + fp + 1e-9)
        rec = tp / (tp + fn + 1e-9)
        f1 = 2 * prec * rec / (prec + rec + 1e-9)

        if f1 > best_f1:
            best_f1, best_thresh = f1, t

    return best_thresh, best_f1


# ── main ──────────────────────────────────────────────────────────────────────

print("Loading good images …")
good_imgs = load_images(GOOD_DIR)

print("\nLoading defective images and ground-truth masks …")
def_imgs = load_images(DEFECT_DIR)
gt_raw   = load_images(GT_DIR, suffix="_mask")

gt_masks = []
for m in gt_raw:
    gray = cv2.cvtColor(m, cv2.COLOR_BGR2GRAY) if m.ndim == 3 else m
    gt_masks.append((gray > 127).astype(np.uint8))

# ── 1. Build initial Gaussian model ──────────────────────────────────────────
print("\n[1/3] Building initial Gaussian model …")
mean_lab, std_lab = build_model(good_imgs)
print(f"  Mean range : {mean_lab.min():.2f} – {mean_lab.max():.2f}")
print(f"  Std  range : {std_lab.min():.4f} – {std_lab.max():.2f}")

# ── 2. Tune initial threshold ─────────────────────────────────────────────────
print("\n[2/3] Tuning initial threshold …")
best_thresh, best_f1 = tune_threshold(good_imgs, def_imgs, gt_masks, mean_lab, std_lab)
print(f"  Threshold : {best_thresh:.4f}   F1 : {best_f1:.4f}")

# ── 3. Iterative FP removal ───────────────────────────────────────────────────
print(f"\n[3/3] Iterative FP removal (max {MAX_REFINE_ITER} rounds) …")

for refine_iter in range(MAX_REFINE_ITER):
    clean_imgs  = []
    n_removed   = 0

    for img in good_imgs:
        score = anomaly_score(img, mean_lab, std_lab)
        gray  = cv2.cvtColor(cv2.resize(img, IMG_SIZE), cv2.COLOR_BGR2GRAY)

        # Build a cleaned mask (same post-processing as inference)
        fg       = (gray > 30).astype(np.uint8) * 255
        mask     = (score >= best_thresh).astype(np.uint8) * 255
        mask     = cv2.bitwise_and(mask, fg)
        mask     = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
        mask     = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
        # Remove small noise blobs (≤ 350 pixels)
        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        for i in range(1, n_labels):
            if stats[i, cv2.CC_STAT_AREA] <= 350:
                mask[labels == i] = 0

        fg_area = int((gray > 30).sum())
        flagged = int((mask > 0).sum())

        if fg_area > 0 and flagged / fg_area > FP_AREA_RATIO:
            n_removed += 1  # this "good" image is a false positive — skip it
        else:
            clean_imgs.append(img)

    print(f"  Round {refine_iter + 1}: removed {n_removed} FP image(s)  "
          f"({len(clean_imgs)} good images remain)")

    if n_removed == 0:
        break  # converged

    good_imgs = clean_imgs
    mean_lab, std_lab = build_model(good_imgs)
    best_thresh, best_f1 = tune_threshold(good_imgs, def_imgs, gt_masks, mean_lab, std_lab)
    print(f"    Threshold : {best_thresh:.4f}   F1 : {best_f1:.4f}")

print(f"\n  Final threshold : {best_thresh:.4f}")
print(f"  Final F1-score  : {best_f1:.4f}")

# ── Save model ────────────────────────────────────────────────────────────────
np.savez_compressed(
    MODEL_PATH,
    mean_lab    = mean_lab,
    std_lab     = std_lab,
    threshold   = np.float32(best_thresh),
    img_size    = np.array(IMG_SIZE),
    void_weight = np.float32(VOID_WEIGHT),
)
print(f"\nModel saved → {MODEL_PATH}")
