import cv2
import os
import numpy as np

# Update policy: CONSERVATIVE
# Only pixels classified as background (mask == 0) are used to update the model.
# Foreground pixels are frozen — the model does not adapt to them.


def read_temporal_roi(roi_file):
    with open(roi_file, 'r') as f:
        line = f.readline()
    return map(int, line.split())


def read_frame_and_gt(folder_path, gt_path, frame_idx):
    img = cv2.imread(os.path.join(folder_path, f"in{frame_idx:06d}.jpg"))
    if img is None:
        return None, None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gt_img = cv2.imread(os.path.join(gt_path, f"gt{frame_idx:06d}.png"), cv2.IMREAD_GRAYSCALE)
    return gray, gt_img


def process_mask(diff):
    """Binarize + denoise a difference image into a foreground mask."""
    _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
    mask = cv2.medianBlur(thresh, 3)
    kernel = np.ones((3, 3), np.uint8)
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)


# ---------------------------------------------------------------------------
# Background model update — CONSERVATIVE variant
# Only background pixels (prev_mask == 0) drive the model update.
# ---------------------------------------------------------------------------

def mean_update(model, frame, prev_mask, alpha=0.01):
    """
    Exponential moving average:  BGN = alpha * I + (1-alpha) * BGN_prev
    Conservative: frozen where prev_mask indicates foreground.
    """
    candidate = alpha * frame.astype(np.float64) + (1.0 - alpha) * model
    new_model = np.where(prev_mask == 0, candidate, model)   # freeze foreground pixels
    diff = cv2.absdiff(frame, new_model.astype(np.uint8))
    return new_model, process_mask(diff)


def median_update(model, frame, prev_mask):
    """
    Median approximation: nudge model +1/-1 toward current frame.
    Conservative: pixels classified as foreground are not nudged.
    """
    delta = (model < frame).astype(np.int16) - (model > frame).astype(np.int16)
    delta[prev_mask != 0] = 0                                 # freeze foreground pixels
    new_model = (model.astype(np.int16) + delta).clip(0, 255).astype(np.uint8)
    diff = cv2.absdiff(frame, new_model)
    return new_model, process_mask(diff)


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def accumulate_stats(gt_img, mask, stats):
    """Add TP/FP/FN counts for one frame (CDnet scoring: 0=BG, 255=FG)."""
    if gt_img is None:
        return
    valid  = (gt_img == 0) | (gt_img == 255)
    gt_fg  = gt_img == 255
    pred_fg = mask == 255
    stats[0] += int(np.sum(pred_fg  &  gt_fg  & valid))   # TP
    stats[1] += int(np.sum(pred_fg  & ~gt_fg  & valid))   # FP
    stats[2] += int(np.sum(~pred_fg &  gt_fg  & valid))   # FN


def f1_score(stats):
    tp, fp, fn = stats
    denom = 2 * tp + fp + fn
    return (2 * tp) / denom if denom else 0.0


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run(folder_path, roi_file, alpha=0.01):
    start_frame, end_frame = read_temporal_roi(roi_file)
    gt_path = folder_path.replace("input", "groundtruth")
    print(f"Processing frames {start_frame} – {end_frame}  (alpha={alpha})")

    mean_model   = None
    median_model = None
    mean_mask    = None
    median_mask  = None
    stats_mean   = [0, 0, 0]
    stats_median = [0, 0, 0]

    for i in range(start_frame, end_frame + 1):
        frame, gt_img = read_frame_and_gt(folder_path, gt_path, i)
        if frame is None:
            print(f"Missing frame {i}, stopping.")
            break

        if mean_model is None:
            # First frame: initialise models; masks start as all-background
            mean_model   = frame.astype(np.float64)
            median_model = frame.copy()
            mean_mask    = np.zeros_like(frame)
            median_mask  = np.zeros_like(frame)
            continue

        # Conservative update — pass previous mask so foreground pixels are frozen
        mean_model,   mean_mask   = mean_update  (mean_model,   frame, mean_mask,   alpha)
        median_model, median_mask = median_update(median_model, frame, median_mask)

        accumulate_stats(gt_img, mean_mask,   stats_mean)
        accumulate_stats(gt_img, median_mask, stats_median)

        cv2.imshow("Frame",              frame)
        cv2.imshow("Mean mask",          mean_mask)
        cv2.imshow("Median mask",        median_mask)
        cv2.imshow("Mean model",         mean_model.astype(np.uint8))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    print(f"F1 – Approx Mean   (alpha={alpha}): {f1_score(stats_mean):.4f}")
    print(f"F1 – Approx Median            : {f1_score(stats_median):.4f}")


if __name__ == "__main__":
    sequence_folder = "./highway/input"
    roi_path        = "./highway/temporalROI.txt"

    if os.path.exists(sequence_folder):
        run(sequence_folder, roi_path)
        cv2.destroyAllWindows()
    else:
        print(f"Directory not found: {sequence_folder}")
