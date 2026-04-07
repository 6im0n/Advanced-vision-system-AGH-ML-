import cv2
import os
import numpy as np

# "eez
# mean and median based on aproximation (mean and median)
# we can change the value of alpha to view the change betwen speed and quality.
#


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


def process_mask(diff):
    # Binarization
    _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)

    # Median filtering and morphological operations
    mask = cv2.medianBlur(thresh, 3)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    return mask


def mean_approx_method(model, current_frame, alpha=0.01):
    # BGN = alpha * IN + (1 - alpha) * BGN_prev
    # Model is stored as float64 to maintain precision
    new_model = alpha * current_frame.astype(np.float64) + (1 - alpha) * model
    diff = cv2.absdiff(current_frame, new_model.astype(np.uint8))
    return new_model, process_mask(diff)


def median_approx_method(model, current_frame):
    # Increment/decrement model to approach current frame using boolean arithmetic
    # new_model = model + (model < current_frame) - (model > current_frame)
    new_model = model.astype(np.int16)
    new_model += (model < current_frame)
    new_model -= (model > current_frame)
    new_model = new_model.astype(np.uint8)
    
    diff = cv2.absdiff(current_frame, new_model)
    return new_model, process_mask(diff)


def load_sequence(folder_path, roi_file):
    start_frame, end_frame = read_temporal_roi(roi_file)

    print(f"Loading sequence from frame {start_frame} to {end_frame}")

    # Stats for F1 score: [TP, FP, FN]
    stats_mean = [0, 0, 0]
    stats_median = [0, 0, 0]

    gt_path = folder_path.replace("input", "groundtruth")

    mean_model = None
    median_model = None
    alpha = 0.001

    for i in range(start_frame, end_frame + 1):
        gray, gt_img, file_path = read_frame_and_gt(folder_path, gt_path, i)
        if gray is None:
            print(f"Failed to load image: {file_path}")
            break

        if mean_model is None:
            # Initialize models with the first frame
            mean_model = gray.astype(np.float64)
            median_model = gray.copy()
            continue

        # Process methods
        mean_model, mean_mask = mean_approx_method(mean_model, gray, alpha)
        median_model, median_mask = median_approx_method(median_model, gray)

        update_stats_from_gt(gt_img, mean_mask, median_mask, stats_mean, stats_median)

        cv2.imshow("Original", gray)
        cv2.imshow("Mean Approx Mask", mean_mask)
        cv2.imshow("Median Approx Mask", median_mask)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    print(f"F1 Score (Approx Mean, alpha={alpha}): {calculate_f1(stats_mean):.4f}")
    print(f"F1 Score (Approx Median): {calculate_f1(stats_median):.4f}")


if __name__ == "__main__":
    sequence_folder = "./highway/input"
    roi_path = "./highway/temporalROI.txt"

    if os.path.exists(sequence_folder):
        load_sequence(sequence_folder, roi_path)
        cv2.destroyAllWindows()
    else:
        print(f"Directory not found: {sequence_folder}")
