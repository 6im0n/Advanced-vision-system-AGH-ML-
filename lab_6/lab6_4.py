import sys
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# This task ask ready-made functions for feature point detection and descriptor of openCV (not hand made one)
#
# Find the feature points in the images. Use a descriptor such as:
#   cv2.SIFT_create() or cv2.xfeatures2d.SURF_create()  (SIFT more reliable) we use this one
# Display the feature points found using cv2.drawKeypoints.
#
# Match the feature points from both images via cv2.BFMatcher(NORM).
# For SIFT we use the norm is cv2.NORM_L2.
# Use Brute Force KNN (no tree).
# Weee use knnMatch with k=2, then ratio test:
#
#   best_matches = []
#   for m, n in matches:        # m, n - best and second best matches
#       if m.distance < 0.5 * n.distance:
#           best_matches.append([m])
#
# View matches with cv2.drawMatches.
#
# Determine homography between the images via cv2.findHomography.
# Convert keypoints first:
#
#   keypointsL = np.float32([kp.pt for kp in keypointsL])
#   keypointsR = np.float32([kp.pt for kp in keypointsR])
#   ptsA = np.float32([keypointsL[m.queryIdx] for m in matches])
#   ptsB = np.float32([keypointsR[m.trainIdx] for m in matches])
#
# Then we warp:
#   result = cv2.warpPerspective(image_left, H, (width, height))
# where (width, height) is the sum of dimensions of the two processed images.
# and use:
#   result[0:image_right.shape[0], 0:image_right.shape[1]] = image_right
#
# Finally crop the excess black background (display the image before)


RATIO_THRESHOLD = 0.5
N_BEST_DRAW    = 50


def load_image_cv(filename):
    path = os.path.join(os.path.dirname(__file__), 'Sources', filename)
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return img


def ask_user_images():
    sources_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'Sources'))
    files = [f for f in os.listdir(sources_dir)
             if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    print("Available images in ../Sources:")
    for i, f in enumerate(files):
        print(f"{i + 1}. {f}")
    choices = input("Select two image numbers (left right): ").split()
    return [files[int(c) - 1] for c in choices]


def detect_sift(img_bgr):
    grey = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(grey, None)
    return keypoints, descriptors


def match_knn(desc_left, desc_right, ratio=RATIO_THRESHOLD):
    bf = cv2.BFMatcher(cv2.NORM_L2)
    matches = bf.knnMatch(desc_left, desc_right, k=2)
    best = []
    for pair in matches:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < ratio * n.distance:
            best.append(m)
    return best


def crop_black_borders(img):
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    mask = grey > 0
    if not mask.any():
        return img
    ys, xs = np.where(mask)
    return img[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


def stitch(img_left, img_right):
    kp_left,  desc_left  = detect_sift(img_left)
    kp_right, desc_right = detect_sift(img_right)

    # Display detected keypoints
    vis_left  = cv2.drawKeypoints(img_left,  kp_left,  None,
                                  flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    vis_right = cv2.drawKeypoints(img_right, kp_right, None,
                                  flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    matches = match_knn(desc_left, desc_right)
    if len(matches) < 4:
        raise RuntimeError(f"Not enough matches: {len(matches)}")
    matches.sort(key=lambda m: m.distance)

    # Display matches
    match_vis = cv2.drawMatches(
        img_left, kp_left, img_right, kp_right,
        matches[:N_BEST_DRAW], None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )

    # Homography
    pts_left_all  = np.float32([kp.pt for kp in kp_left])
    pts_right_all = np.float32([kp.pt for kp in kp_right])
    pts_a = np.float32([pts_left_all[m.queryIdx]  for m in matches])
    pts_b = np.float32([pts_right_all[m.trainIdx] for m in matches])
    H, _ = cv2.findHomography(pts_a, pts_b, cv2.RANSAC, 5.0)

    # Warp left into right's frame, paste right on top
    h_l, w_l = img_left.shape[:2]
    h_r, w_r = img_right.shape[:2]
    out_w = w_l + w_r
    out_h = h_l + h_r
    result = cv2.warpPerspective(img_left, H, (out_w, out_h))
    result[0:h_r, 0:w_r] = img_right
    raw = result.copy()
    cropped = crop_black_borders(result)

    return vis_left, vis_right, match_vis, raw, cropped


def show_bgr(ax, img, title):
    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax.set_title(title)
    ax.axis('off')


def enable_scroll_zoom(fig, scale=1.3):
    """Mouse-wheel zoom around cursor on every axes of fig."""
    def on_scroll(event):
        ax = event.inaxes
        if ax is None:
            return
        x, y = event.xdata, event.ydata
        if x is None or y is None:
            return
        factor = 1 / scale if event.button == 'up' else scale
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        ax.set_xlim([x + (xi - x) * factor for xi in (x0, x1)])
        ax.set_ylim([y + (yi - y) * factor for yi in (y0, y1)])
        fig.canvas.draw_idle()
    fig.canvas.mpl_connect('scroll_event', on_scroll)


def main():
    filenames = ask_user_images()
    if len(filenames) < 2:
        return
    img_left  = load_image_cv(filenames[0])
    img_right = load_image_cv(filenames[1])

    vis_left, vis_right, match_vis, raw, cropped = stitch(img_left, img_right)

    fig1, axes = plt.subplots(1, 3, figsize=(18, 6))
    show_bgr(axes[0], vis_left,  f"Keypoints: {filenames[0]}")
    show_bgr(axes[1], vis_right, f"Keypoints: {filenames[1]}")
    show_bgr(axes[2], match_vis, f"KNN matches (ratio<{RATIO_THRESHOLD})")
    fig1.tight_layout()
    enable_scroll_zoom(fig1)

    fig2, axes2 = plt.subplots(1, 2, figsize=(18, 8))
    show_bgr(axes2[0], raw,     "Stitched (with black background)")
    show_bgr(axes2[1], cropped, "Stitched (cropped)")
    fig2.tight_layout()
    enable_scroll_zoom(fig2)

    plt.show()


if __name__ == "__main__":
    main()
