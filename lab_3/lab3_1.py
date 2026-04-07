import cv2
import os
import numpy as np


## mean and median based on the frame buffer

def read_temporal_roi(roi_file):
    with open(roi_file, 'r') as f:
        line = f.readline()
    return map(int, line.split())


def read_frame_and_gt(folder_path, gt_path, frame_idx):
    filename = f"in{frame_idx:06d}.jpg"
    file_path = os.path.join(folder_path, filename)
    img = cv2.imread(file_path)
    if img is None:
        return None, None, file_path

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    gt_filename = f"gt{frame_idx:06d}.png"
    gt_file_path = os.path.join(gt_path, gt_filename)
    gt_img = cv2.imread(gt_file_path, cv2.IMREAD_GRAYSCALE)

    return gray, gt_img, file_path


def init_buffer_if_needed(buf, gray, n):
    if buf is None:
        yy, xx = gray.shape
        return np.zeros((yy, xx, n), np.uint8)
    return buf


def push_fifo_frame(buf, write_idx, gray, frames_processed, n):
    buf[:, :, write_idx] = gray
    write_idx += 1
    if write_idx >= n:
        write_idx = 0

    valid_buf = buf[:, :, :frames_processed] if frames_processed < n else buf
    return write_idx, valid_buf


def update_stats_from_gt(gt_img, mean_mask, median_mask, stats_mean, stats_median):
    if gt_img is None:
        return

    # Evaluate only official CDnet scoring pixels.
    valid = (gt_img == 0) | (gt_img == 255)
    gt_fg = (gt_img == 255)

    for mask, stats in ((mean_mask, stats_mean), (median_mask, stats_median)):
        pred_fg = (mask == 255)

        tp = np.sum(pred_fg & gt_fg & valid)
        fp = np.sum(pred_fg & (~gt_fg) & valid)
        fn = np.sum((~pred_fg) & gt_fg & valid)

        stats[0] += int(tp)
        stats[1] += int(fp)
        stats[2] += int(fn)


def calculate_f1(stats):
    tp, fp, fn = stats
    if (2 * tp + fp + fn) == 0:
        return 0
    return (2 * tp) / (2 * tp + fp + fn)


def load_sequence(folder_path, roi_file):
    start_frame, end_frame = read_temporal_roi(roi_file)

    print(f"Loading sequence from frame {start_frame} to {end_frame}")

    # Initialize parameters
    N = 60
    iN = 0
    YY, XX = 0, 0
    BUF = None

    # Stats for F1 score: [TP, FP, FN]
    stats_mean = [0, 0, 0]
    stats_median = [0, 0, 0]

    gt_path = folder_path.replace("input", "groundtruth")

    frames_processed = 0
    for i in range(start_frame, end_frame + 1):
        frames_processed += 1
        gray, gt_img, file_path = read_frame_and_gt(folder_path, gt_path, i)
        if gray is None:
            print(f"Failed to load image: {file_path}")
            break

        BUF = init_buffer_if_needed(BUF, gray, N)
        iN, valid_buf = push_fifo_frame(BUF, iN, gray, frames_processed, N)

        # Process methods
        mean_mask = mean_method(valid_buf, gray)
        median_mask = median_method(valid_buf, gray)

        update_stats_from_gt(gt_img, mean_mask, median_mask, stats_mean, stats_median)

        cv2.imshow("Original", gray)
        cv2.imshow("Mean Method Mask", mean_mask)
        cv2.imshow("Median Method Mask", median_mask)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


    print(f"F1 Score (Mean): {calculate_f1(stats_mean):.4f}")
    print(f"F1 Score (Median): {calculate_f1(stats_median):.4f}")

def process_mask(diff):
    # Binarization
    _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)

    # Median filtering and morphological operations
    mask = cv2.medianBlur(thresh, 3)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    return mask

def mean_method(BUF, current_frame):
    model = np.mean(BUF, axis=2).astype(np.uint8)
    diff = cv2.absdiff(current_frame, model)
    return process_mask(diff)

def median_method(BUF, current_frame):
    model = np.median(BUF, axis=2).astype(np.uint8)
    diff = cv2.absdiff(current_frame, model)
    return process_mask(diff)


if __name__ == "__main__":
    sequence_folder = "./highway/input"
    roi_path = "./highway/temporalROI.txt"

    if os.path.exists(sequence_folder):
        load_sequence(sequence_folder, roi_path)
        cv2.destroyAllWindows()
    else:
        print(f"Directory not found: {sequence_folder}")
