import cv2
import os
import numpy as np


def read_temporal_roi(roi_file):
    with open(roi_file, 'r') as f:
        line = f.readline()
    return map(int, line.split())

#Usefull for stat about the performance of agolrithn
def read_frame_and_gt(folder_path, gt_path, frame_idx):
    img = cv2.imread(os.path.join(folder_path, f"in{frame_idx:06d}.jpg"))
    if img is None:
        return None, None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gt_img = cv2.imread(os.path.join(gt_path, f"gt{frame_idx:06d}.png"), cv2.IMREAD_GRAYSCALE)
    return gray, gt_img


def accumulate_stats(gt_img, mask, stats):
    """Add TP/FP/FN counts for one frame (CDnet scoring: 0=BG, 255=FG)."""
    if gt_img is None:
        return
    valid   = (gt_img == 0) | (gt_img == 255)
    gt_fg   = gt_img == 255
    pred_fg = mask == 255
    stats[0] += int(np.sum( pred_fg &  gt_fg & valid))   # TP
    stats[1] += int(np.sum( pred_fg & ~gt_fg & valid))   # FP
    stats[2] += int(np.sum(~pred_fg &  gt_fg & valid))   # FN


def f1_score(stats):
    tp, fp, fn = stats
    denom = 2 * tp + fp + fn
    return (2 * tp) / denom if denom else 0.0


def run(folder_path, roi_file, history=500, var_threshold=16, learning_rate=-1):
    start_frame, end_frame = read_temporal_roi(roi_file)
    gt_path = folder_path.replace("input", "groundtruth")
    print(f"Processing frames {start_frame} – {end_frame}")
    print(f"  history={history}  varThreshold={var_threshold}  learningRate={learning_rate}")

    # MOG2 without shadow detection
    mog2 = cv2.createBackgroundSubtractorMOG2(
        history=history,
        varThreshold=var_threshold,
        detectShadows=False
    )

    stats = [0, 0, 0]

    for i in range(start_frame, end_frame + 1):
        frame, gt_img = read_frame_and_gt(folder_path, gt_path, i)
        if frame is None:
            print(f"Missing frame {i}, stopping.")
            break

        mask = mog2.apply(frame, learningRate=learning_rate)

        accumulate_stats(gt_img, mask, stats)

        cv2.imshow("Frame",    frame)
        cv2.imshow("MOG2 mask", mask)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    print(f"F1 – MOG2: {f1_score(stats):.4f}")


if __name__ == "__main__":
    sequence_folder = "./highway/input"
    roi_path        = "./highway/temporalROI.txt"

    if os.path.exists(sequence_folder):
        run(sequence_folder, roi_path)
        cv2.destroyAllWindows()
    else:
        print(f"Directory not found: {sequence_folder}")
