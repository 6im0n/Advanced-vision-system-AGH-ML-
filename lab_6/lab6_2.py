import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage
import scipy.ndimage.filters as filters
from PIL import Image

# Add the parent directory to sys.path to allow importing from ../Sources
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Sources.pm import *

# Harris corner detection implementation
# code used from lab6_2 and now we add description of the points

# sigma for gaussian width
SIGMA = 5
K_CONST = 0.05  # (0.04-0.06)
MASK_SIZE = 7
THRESHOLD = 0.3
NEIGHBORHOOD_SIZE = 7


def greyscaleimage(image):
    if len(image.shape) == 3:
        return np.dot(image[..., :3], [0.2989, 0.5870, 0.1140])
    return image


def sobel_operator(image, size=7):
    # For size 3, use standard Sobel. For larger sizes, we use a generic
    Ix = ndimage.sobel(image, axis=1)
    Iy = ndimage.sobel(image, axis=0)

    if size > 3:
        # Smoothing the gradients to simulate a larger mask impact if size > 3
        # as standard Sobel is 3x3.
        Ix = ndimage.gaussian_filter(Ix, sigma=(size - 1) / 6.0)
        Iy = ndimage.gaussian_filter(Iy, sigma=(size - 1) / 6.0)
    return Ix, Iy


def autocorelation_matrix(greyscale_image, sigma, mask_size):
    Ix, Iy = sobel_operator(greyscale_image, size=mask_size)

    # Products of derivatives
    Ixx = Ix ** 2
    Iyy = Iy ** 2
    Ixy = Ix * Iy

    # Gaussian blur the products to get local neighborhood sums
    # Truncate is used to ensure the kernel fits approximately within the mask_size
    radius = mask_size // 2
    truncate = radius / sigma

    Sxx = ndimage.gaussian_filter(Ixx, sigma=sigma, truncate=truncate)
    Syy = ndimage.gaussian_filter(Iyy, sigma=sigma, truncate=truncate)
    Sxy = ndimage.gaussian_filter(Ixy, sigma=sigma, truncate=truncate)

    return Sxx, Syy, Sxy


def calculate_H_value(Sxx, Syy, Sxy, k):
    # H(x, y) = det(M(x, y)) − k * trace^2(M(x,y))
    det_M = (Sxx * Syy) - (Sxy ** 2)
    trace_M = Sxx + Syy
    H = det_M - k * (trace_M ** 2)

    # Normalise the resulting H image to the range 0–1
    h_min = np.min(H)
    h_max = np.max(H)
    if h_max > h_min:
        H = (H - h_min) / (h_max - h_min)
    else:
        H = np.zeros_like(H)

    return H


def find_max(image, size, threshold):
    data_max = filters.maximum_filter(image, size)
    maxima = (image == data_max)
    diff = image > threshold
    maxima[diff == 0] = 0
    return np.nonzero(maxima)


def load_images(filename):
    path = os.path.join(os.path.dirname(__file__), 'Sources', filename)
    img = np.array(Image.open(path)).astype(np.float32) / 255.0
    return img


def Drawing_corner_Result(ax, image, corners):
    ax.imshow(image, cmap='gray' if len(image.shape) == 2 else None)
    y, x = corners
    ax.plot(x, y, 'r*', markersize=5)
    ax.set_title("Corners Detected (*)")


def ask_user_images():
    sources_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'Sources'))
    files = [f for f in os.listdir(sources_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    print("Available images in ../Sources:")
    for i, f in enumerate(files):
        print(f"{i + 1}. {f}")

    choices = input("Select two image numbers (separated by space) to check repeatability: ").split()
    return [files[int(c) - 1] for c in choices]


def create_descriptors(image, corners, size):
    Y, X = image.shape
    pts = list(filter(lambda pt: pt[0] >= size and pt[0] < Y - size and pt[1] >= size and pt[1] < X - size, zip(corners[0], corners[1])))

    descriptors = []
    valid_points = []
    for y, x in pts:
        w = image[y - size:y + size + 1, x - size:x + size + 1].flatten()
        std = np.std(w)
        w_aff = (w - np.mean(w)) / std if std > 0 else w - np.mean(w)
        descriptors.append(w_aff)
        valid_points.append((y, x))
    return descriptors, valid_points


def match_descriptors(desc1, pts1, desc2, pts2, N):
    matches = []
    for i, d1 in enumerate(desc1):
        distances = []
        for j, d2 in enumerate(desc2):
            dist = np.sum(np.abs(d1 - d2))
            distances.append((dist, j))
        
        distances.sort(key=lambda x: x[0])
        best_dist, best_idx = distances[0]
        matches.append((best_dist, pts1[i], pts2[best_idx]))

    matches.sort(key=lambda x: x[0])
    return matches[:N]


def display_results(grey1, grey2, matches):
    formatted_matches = [([m[1][0], m[1][1]], [m[2][0], m[2][1]]) for m in matches]
    plot_matches(grey1, grey2, formatted_matches)
    plt.show()


def process_and_display(filenames):
    if len(filenames) < 2:
        print("Please select at least two images.")
        return

    greys = []
    all_kp_pts = []
    all_descs = []

    for fname in filenames[:2]:
        img = load_images(fname)
        grey = greyscaleimage(img)
        
        Sxx, Syy, Sxy = autocorelation_matrix(grey, SIGMA, MASK_SIZE)
        H = calculate_H_value(Sxx, Syy, Sxy, K_CONST)
        corners = find_max(H, MASK_SIZE, THRESHOLD)
        
        descs, pts = create_descriptors(grey, corners, NEIGHBORHOOD_SIZE)
        
        greys.append(grey)
        all_descs.append(descs)
        all_kp_pts.append(pts)

    best_matches = match_descriptors(all_descs[0], all_kp_pts[0], all_descs[1], all_kp_pts[1], N=20)
    display_results(greys[0], greys[1], best_matches)


if __name__ == "__main__":
    selected_images = ask_user_images()
    process_and_display(selected_images)
