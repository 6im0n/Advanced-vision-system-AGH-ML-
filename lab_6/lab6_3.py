import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage
from PIL import Image

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Sources.pm import *

K_CONST = 0.04
PATCH_SIZE = 31
HALF_PATCH = PATCH_SIZE // 2  # 15
N_BEST = 50
FAST_THRESHOLD = 0.05

# Bresenham circle radius 3 — fits in 7×7, 16 pixels
FAST_CIRCLE = [
    (-3, 0), (-3, 1), (-2, 2), (-1, 3),
    (0, 3),  (1, 3),  (2, 2),  (3, 1),
    (3, 0),  (3, -1), (2, -2), (1, -3),
    (0, -3), (-1, -3),(-2, -2),(-3, -1),
]


def greyscaleimage(image):
    if len(image.shape) == 3:
        return np.dot(image[..., :3], [0.2989, 0.5870, 0.1140])
    return image


def fast_detector(image, threshold=FAST_THRESHOLD, n_consec=9):
    """FAST using Bresenham circle radius 3 (7×7 patch). Vectorized."""
    H, W = image.shape
    pad = 3
    img_pad = np.pad(image, pad, mode='reflect')

    # (16, H, W) — each slice is the circle pixel at offset (dy, dx)
    circle_stack = np.stack([
        img_pad[pad + dy:pad + dy + H, pad + dx:pad + dx + W]
        for dy, dx in FAST_CIRCLE
    ])

    brighter = circle_stack > image[np.newaxis] + threshold  # (16, H, W)
    darker   = circle_stack < image[np.newaxis] - threshold

    # Tile for circular consecutive-run check
    b2 = np.concatenate([brighter, brighter], axis=0)  # (32, H, W)
    d2 = np.concatenate([darker,   darker],   axis=0)

    mask = np.zeros((H, W), dtype=bool)
    n = len(FAST_CIRCLE)
    for start in range(n):
        mask |= np.all(b2[start:start + n_consec], axis=0)
        mask |= np.all(d2[start:start + n_consec], axis=0)

    # Border: need full 31×31 patch for descriptor
    mask[:HALF_PATCH]    = False
    mask[-HALF_PATCH:]   = False
    mask[:, :HALF_PATCH] = False
    mask[:, -HALF_PATCH:] = False

    return list(zip(*np.where(mask))) if mask.any() else []


def calculate_harris_measure(image, k=K_CONST):
    Ix = ndimage.sobel(image, axis=1)
    Iy = ndimage.sobel(image, axis=0)
    Ixx = ndimage.gaussian_filter(Ix ** 2, sigma=1)
    Iyy = ndimage.gaussian_filter(Iy ** 2, sigma=1)
    Ixy = ndimage.gaussian_filter(Ix * Iy,  sigma=1)
    det   = Ixx * Iyy - Ixy ** 2
    trace = Ixx + Iyy
    return det - k * trace ** 2


def non_max_suppression(keypoints, harris, nms_size=3):
    """Keep keypoint with max Harris score in each nms_size×nms_size window."""
    if not keypoints:
        return []
    H, W = harris.shape
    half = nms_size // 2
    used = np.zeros((H, W), dtype=bool)
    scored = sorted(keypoints, key=lambda p: harris[p[0], p[1]], reverse=True)
    result = []
    for y, x in scored:
        if not used[y, x]:
            result.append((y, x))
            y0, y1 = max(0, y - half), min(H, y + half + 1)
            x0, x1 = max(0, x - half), min(W, x + half + 1)
            used[y0:y1, x0:x1] = True
    return result


def get_orientation(image, y, x):
    """Intensity centroid over HALF_PATCH disc → orientation angle."""
    m10 = 0.0
    m01 = 0.0
    for dy in range(-HALF_PATCH, HALF_PATCH + 1):
        for dx in range(-HALF_PATCH, HALF_PATCH + 1):
            if dy * dy + dx * dx <= HALF_PATCH * HALF_PATCH:
                val = image[y + dy, x + dx]
                m10 += dx * val   # col moment
                m01 += dy * val   # row moment
    return np.arctan2(m01, m10)


def load_orb_pairs():
    path = os.path.join(os.path.dirname(__file__), 'Sources', 'orb_descriptor_positions.txt')
    try:
        return np.loadtxt(path)   # (256, 4): x1 y1 x2 y2
    except Exception:
        np.random.seed(42)
        return np.random.randint(-HALF_PATCH, HALF_PATCH, (256, 4)).astype(float)


def get_rotated_brief(image, y, x, angle, pairs):
    """Steered BRIEF: blur 31×31 patch with 5×5 Gaussian, rotate pairs by angle."""
    patch = image[y - HALF_PATCH:y + HALF_PATCH + 1,
                  x - HALF_PATCH:x + HALF_PATCH + 1].copy()
    # truncate=1.0 with sigma=2 → 5×5 kernel
    blurred = ndimage.gaussian_filter(patch, sigma=2, truncate=1.0)

    cos_a, sin_a = np.cos(angle), np.sin(angle)

    descriptor = []
    for px1, py1, px2, py2 in pairs:
        # eq 6.14/6.15: x' = cos*x − sin*y,  y' = sin*x + cos*y
        rx1 = int(round(cos_a * px1 - sin_a * py1))
        ry1 = int(round(sin_a * px1 + cos_a * py1))
        rx2 = int(round(cos_a * px2 - sin_a * py2))
        ry2 = int(round(sin_a * px2 + cos_a * py2))

        # Map offsets to patch indices, clamp to [0, PATCH_SIZE)
        r1 = int(np.clip(HALF_PATCH + ry1, 0, PATCH_SIZE - 1))
        c1 = int(np.clip(HALF_PATCH + rx1, 0, PATCH_SIZE - 1))
        r2 = int(np.clip(HALF_PATCH + ry2, 0, PATCH_SIZE - 1))
        c2 = int(np.clip(HALF_PATCH + rx2, 0, PATCH_SIZE - 1))

        descriptor.append(1 if blurred[r1, c1] < blurred[r2, c2] else 0)

    return np.packbits(descriptor)


def hamming_distance(a, b):
    return int(np.count_nonzero(np.unpackbits(a) ^ np.unpackbits(b)))


def load_images(filename):
    path = os.path.join(os.path.dirname(__file__), 'Sources', filename)
    return np.array(Image.open(path)).astype(np.float32) / 255.0


def ask_user_images():
    sources_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'Sources'))
    files = [f for f in os.listdir(sources_dir)
             if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    print("Available images in ../Sources:")
    for i, f in enumerate(files):
        print(f"{i + 1}. {f}")
    choices = input("Select two image numbers (separated by space): ").split()
    return [files[int(c) - 1] for c in choices]


def process_image(img_name, pairs):
    img  = load_images(img_name)
    grey = greyscaleimage(img)

    # Step 2: FAST detection + Harris measure
    candidates = fast_detector(grey)
    harris     = calculate_harris_measure(grey)

    # Step 3: NMS — keep max Harris in each 3×3 window
    nms_pts = non_max_suppression(candidates, harris, nms_size=3)

    # Step 4: border already excluded inside fast_detector (HALF_PATCH margin)

    # Step 5: sort by Harris, select N_BEST
    scored = sorted(nms_pts, key=lambda p: harris[p[0], p[1]], reverse=True)[:N_BEST]

    # Steps 6-7: orientation + steered BRIEF
    descriptors = []
    final_pts   = []
    for y, x in scored:
        angle = get_orientation(grey, y, x)
        desc  = get_rotated_brief(grey, y, x, angle, pairs)
        descriptors.append(desc)
        final_pts.append((y, x))

    return grey, descriptors, final_pts


def match_orb(desc1, pts1, desc2, pts2, n_matches=20):
    matches = []
    for i, d1 in enumerate(desc1):
        best_dist = float('inf')
        best_idx  = -1
        for j, d2 in enumerate(desc2):
            dist = hamming_distance(d1, d2)
            if dist < best_dist:
                best_dist = dist
                best_idx  = j
        if best_idx >= 0:
            matches.append((best_dist, pts1[i], pts2[best_idx]))
    matches.sort(key=lambda m: m[0])
    return matches[:n_matches]


def main():
    filenames = ask_user_images()
    if len(filenames) < 2:
        return
    pairs  = load_orb_pairs()
    grey1, desc1, pts1 = process_image(filenames[0], pairs)
    grey2, desc2, pts2 = process_image(filenames[1], pairs)
    best_matches = match_orb(desc1, pts1, desc2, pts2, n_matches=20)
    formatted = [([m[1][0], m[1][1]], [m[2][0], m[2][1]]) for m in best_matches]
    plot_matches(grey1, grey2, formatted)
    plt.show()


if __name__ == "__main__":
    main()
