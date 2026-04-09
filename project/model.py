import os
import numpy as np
import cv2

MODEL_PATH = os.path.join(os.path.dirname(__file__), "defect_model.npz")
EPSILON    = 1e-6
SMOOTH_K   = 7


def predict(image):
    """Detect toothbrush bristle defects and return a binary mask.

    Args:
        image: numpy array, uint8. Shape (H, W, 3) RGB or (H, W) grayscale.

    Returns:
        Binary mask as numpy array of shape (H, W), uint8 with values 0 or 255.
    """
    data      = np.load(MODEL_PATH)
    mean_lab  = data["mean_lab"]
    std_lab   = data["std_lab"]
    threshold = float(data["threshold"])
    img_size  = tuple(data["img_size"].tolist())

    # The test harness loads images with Pillow (RGB). Convert to BGR for cv2.
    if image.ndim == 2:
        img_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    h_orig, w_orig = img_bgr.shape[:2]

    # Contrast enhancement — percentile stretching on L channel
    small = cv2.resize(img_bgr, img_size, interpolation=cv2.INTER_AREA)
    lab   = cv2.cvtColor(small, cv2.COLOR_BGR2Lab)
    l, a, b = cv2.split(lab)
    lo, hi  = np.percentile(l, 2), np.percentile(l, 98)
    if hi > lo:
        l = np.clip((l.astype(np.float32) - lo) * 255.0 / (hi - lo), 0, 255).astype(np.uint8)
    small = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_Lab2BGR)

    # Anomaly score
    lab   = cv2.cvtColor(small, cv2.COLOR_BGR2Lab).astype(np.float32)
    z     = np.abs((lab - mean_lab) / (std_lab + EPSILON))
    score = cv2.GaussianBlur(z.max(axis=2), (SMOOTH_K, SMOOTH_K), 0)
    score = cv2.resize(score, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR)

    # Foreground mask — ignore dark background pixels (L < 30 in original image)
    gray_orig  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    foreground = (gray_orig > 30).astype(np.uint8) * 255

    # Mask: anomaly AND foreground only
    mask = (score >= threshold).astype(np.uint8) * 255
    mask = cv2.bitwise_and(mask, foreground)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)))
    return mask
