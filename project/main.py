"""
main.py — Evaluate model.py against the toothbrush dataset.

Usage:
    python3 main.py           # terminal output only
    python3 main.py --show    # display each image, SPACE = next, Q = quit
"""

import os
import sys
import numpy as np
import cv2
from model import predict   # V2: PaDiM (ResNet18 features) — change to model for V1

BASE_DIR   = os.path.join(os.path.dirname(__file__), "toothbrush")
GOOD_DIR   = os.path.join(BASE_DIR, "train", "good")
DEFECT_DIR = os.path.join(BASE_DIR, "train", "defective")
GT_DIR     = os.path.join(BASE_DIR, "ground_truth", "defective")

SHOW         = "--show" in sys.argv
DISPLAY_SIZE = (512, 512)


def load_rgb(path):
    img = cv2.imread(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def f1(tp, fp, fn):
    if 2 * tp + fp + fn == 0:
        return 0.0
    return 2 * tp / (2 * tp + fp + fn)


def build_display(img_rgb, mask, gt_gray, label):
    """
    Three panels side by side (each DISPLAY_SIZE):
      1. Original image
      2. Detection result — red overlay on predicted defect pixels
      3. Ground truth mask (white = defect) — or black if good image
    """
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    # Panel 1 — original
    orig = cv2.resize(img_bgr, DISPLAY_SIZE)

    # Panel 2 — detection overlay
    overlay = img_bgr.copy()
    red = np.zeros_like(img_bgr)
    red[:, :, 2] = 255
    defect_px = mask > 0
    overlay[defect_px] = (0.6 * red[defect_px] + 0.4 * img_bgr[defect_px]).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0, 0, 255), 2)
    det = cv2.resize(overlay, DISPLAY_SIZE)

    # Panel 3 — ground truth
    gt_bgr = cv2.cvtColor(gt_gray, cv2.COLOR_GRAY2BGR)
    gt_panel = cv2.resize(gt_bgr, DISPLAY_SIZE)

    # Stack panels and add header labels
    canvas = np.hstack([orig, det, gt_panel])
    h = canvas.shape[0]

    for i, title in enumerate(["Original", "Detection", "Ground Truth"]):
        x = i * DISPLAY_SIZE[0] + 6
        cv2.putText(canvas, title, (x, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3)
        cv2.putText(canvas, title, (x, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

    # Bottom label with stats
    cv2.putText(canvas, label, (6, h - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
    cv2.putText(canvas, label, (6, h - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)

    return canvas


def show_and_wait(canvas):
    """Show canvas. SPACE = next, Q/Esc = quit. Returns False if quitting."""
    cv2.imshow("Evaluation  [SPACE = next | Q = quit]", canvas)
    while True:
        key = cv2.waitKey(0) & 0xFF
        if key == ord(' '):
            return True
        if key in (ord('q'), ord('Q'), 27):
            return False


def evaluate(img_path, gt_mask=None):
    """Run predict on one image. Returns (tp, fp, fn)."""
    img  = load_rgb(img_path)
    mask = predict(img)

    # Ground truth
    if gt_mask is None:
        gt_gray = np.zeros(mask.shape, dtype=np.uint8)
    else:
        gt_gray = cv2.resize(gt_mask, (mask.shape[1], mask.shape[0]),
                             interpolation=cv2.INTER_NEAREST)

    pred_fg = mask > 0
    gt_fg   = gt_gray > 0
    tp = int(np.sum( pred_fg &  gt_fg))
    fp = int(np.sum( pred_fg & ~gt_fg))
    fn = int(np.sum(~pred_fg &  gt_fg))

    if SHOW:
        prec  = tp / (tp + fp + 1e-9)
        rec   = tp / (tp + fn + 1e-9)
        label = (f"{os.path.basename(img_path)}"
                 f"   P={prec:.2f}  R={rec:.2f}  F1={f1(tp,fp,fn):.2f}")
        canvas = build_display(img, mask, gt_gray, label)
        if not show_and_wait(canvas):
            cv2.destroyAllWindows()
            sys.exit(0)

    return tp, fp, fn


# ── good images ───────────────────────────────────────────────────────────────

good_paths = sorted(
    os.path.join(GOOD_DIR, f)
    for f in os.listdir(GOOD_DIR) if f.endswith(".png")
)

print(f"{'─'*55}")
print(f"Good images ({len(good_paths)})  — expected: all black masks")
print(f"{'─'*55}")

g_tp = g_fp = g_fn = 0
for path in good_paths:
    tp, fp, fn = evaluate(path)
    flagged_pct = 100 * (tp + fp) / (1024 * 1024)
    status = "OK " if fp == 0 else "FP "
    print(f"  [{status}] {os.path.basename(path)}  flagged={flagged_pct:.2f}%")
    g_tp += tp; g_fp += fp; g_fn += fn

print(f"\n  False positive pixels : {g_fp:,}")


# ── defective images ──────────────────────────────────────────────────────────

def_paths = sorted(
    os.path.join(DEFECT_DIR, f)
    for f in os.listdir(DEFECT_DIR) if f.endswith(".png")
)
gt_paths = sorted(
    os.path.join(GT_DIR, f)
    for f in os.listdir(GT_DIR) if f.endswith(".png")
)

print(f"\n{'─'*55}")
print(f"Defective images ({len(def_paths)})")
print(f"{'─'*55}")

d_tp = d_fp = d_fn = 0
for img_path, gt_path in zip(def_paths, gt_paths):
    gt_mask = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
    tp, fp, fn = evaluate(img_path, gt_mask)
    prec = tp / (tp + fp + 1e-9)
    rec  = tp / (tp + fn + 1e-9)
    print(f"  {os.path.basename(img_path)}  P={prec:.2f}  R={rec:.2f}  F1={f1(tp,fp,fn):.2f}")
    d_tp += tp; d_fp += fp; d_fn += fn


# ── overall ───────────────────────────────────────────────────────────────────

all_tp = d_tp
all_fp = d_fp + g_fp
all_fn = d_fn

prec   = all_tp / (all_tp + all_fp + 1e-9)
rec    = all_tp / (all_tp + all_fn + 1e-9)
f1_all = f1(all_tp, all_fp, all_fn)

print(f"\n{'═'*55}")
print(f"  Precision : {prec:.4f}")
print(f"  Recall    : {rec:.4f}")
print(f"  F1        : {f1_all:.4f}")
print(f"{'═'*55}")

if SHOW:
    cv2.destroyAllWindows()
