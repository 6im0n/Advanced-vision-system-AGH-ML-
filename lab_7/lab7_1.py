import sys
import os
import numpy as np
import cv2

# the lab learn to become familliar with optical flow :
# - optical flow is a vector field describing the movement of pixels betwen two consicuted frame (in video sequence for exemple)
# this method is not efficient is to understand the concept of optical flow, but it is not used in real applications because of its slowness and inaccuracy (it is a block method, not a pixel method)
# this method use large value for dy and dy... and

SCALE_FACTOR = 2  # div: 1=same, 2=half, 4=quart
PATCH_SIZE = 7    ## must be odd; W2 = PATCH_SIZE // 2
SEARCH_RANGE = 3  # dX = dY = SEARCH_RANGE axis
THRESHOLD = 10

def block_method(I, J, W2=3, dX=3, dY=3):
    h, w = I.shape
    If = I.astype(np.float32)
    Jf = J.astype(np.float32)
    ksize = 2 * W2 + 1

    best_ssd = np.full((h, w), np.inf, dtype=np.float32)
    u = np.zeros((h, w), dtype=np.float32)
    v = np.zeros((h, w), dtype=np.float32)

    for dy in range(-dY, dY + 1):
        for dx in range(-dX, dX + 1):
            M = np.float32([[1, 0, dx], [0, 1, dy]])
            J_shift = cv2.warpAffine(Jf, M, (w, h),
                                     flags=cv2.INTER_NEAREST,
                                     borderMode=cv2.BORDER_REPLICATE)
            sq = (J_shift - If) ** 2
            ssd = cv2.boxFilter(sq, ddepth=-1, ksize=(ksize, ksize),
                                normalize=False, borderType=cv2.BORDER_ISOLATED)
            better = ssd < best_ssd
            best_ssd = np.where(better, ssd, best_ssd)
            u = np.where(better, np.float32(dx), u)
            v = np.where(better, np.float32(dy), v)

    u[:W2, :] = 0; u[-W2:, :] = 0; u[:, :W2] = 0; u[:, -W2:] = 0
    v[:W2, :] = 0; v[-W2:, :] = 0; v[:, :W2] = 0; v[:, -W2:] = 0
    return u, v


def diff_mask(I, J, thresh=10, kernel_size=5, iterations=2):
    diff = cv2.absdiff(I, J)
    _, bw = cv2.threshold(diff, thresh, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    return cv2.dilate(bw, kernel, iterations=iterations)


def filter_flow(u, v, mask):
    keep = mask.astype(bool)
    u_f = np.where(keep, u, 0).astype(np.float32)
    v_f = np.where(keep, v, 0).astype(np.float32)
    return u_f, v_f


def flow_to_hsv(u, v, swap_sv=True):
    mag, ang = cv2.cartToPolar(u.astype(np.float32), v.astype(np.float32))
    hsv = np.zeros((u.shape[0], u.shape[1], 3), dtype=np.uint8)
    hsv[..., 0] = (ang * 90 / np.pi).astype(np.uint8)
    s = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    if swap_sv:
        hsv[..., 1] = 255
        hsv[..., 2] = s
    else:
        hsv[..., 1] = s
        hsv[..., 2] = 255
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def display_flow_quiver(img, u, v, step=8, scale=1.0, color=(0, 255, 0)):
    out = img.copy()
    if out.ndim == 2:
        out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
    h, w = u.shape
    for j in range(0, h, step):
        for i in range(0, w, step):
            dx = float(u[j, i]) * scale
            dy = float(v[j, i]) * scale
            if dx == 0 and dy == 0:
                continue
            cv2.arrowedLine(out, (i, j), (int(i + dx), int(j + dy)),
                            color, 1, tipLength=0.3)
    return out


def display_image_diff(image):
    cv2.absdiff(image.astype(np.uint8), image.astype(np.bool))
    cv2.imshow("Image", image)
    cv2.waitKey(0)


def downscale_img(img):
    if SCALE_FACTOR != 1:
        width = max(1, img.shape[1] // SCALE_FACTOR)
        height = max(1, img.shape[0] // SCALE_FACTOR)
        img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
    return img

def grayscaleimage(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def load_image_cv(filename):
    path = os.path.join(os.path.dirname(__file__), 'sources', filename)
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return img


def ask_user_images():
    sources_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'sources'))
    files = [f for f in os.listdir(sources_dir)
             if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    print("Available images in ../Sources:")
    for i, f in enumerate(files):
        print(f"{i + 1}. {f}")
    choices = input("Select two image numbers (left right): ").split()
    return [files[int(c) - 1] for c in choices]


def cv_display_images(I, J, bgr, arrows):
    cv2.namedWindow("frame I", cv2.WINDOW_NORMAL)
    cv2.imshow("frame I", I)
    cv2.namedWindow("frame J", cv2.WINDOW_NORMAL)
    cv2.imshow("frame J", J)
    cv2.namedWindow("frame BGR", cv2.WINDOW_NORMAL)
    cv2.imshow("frame BGR", bgr)
    cv2.namedWindow("frame Arrows", cv2.WINDOW_NORMAL)
    cv2.imshow("frame Arrows", arrows)

def main():
    f1, f2 = ask_user_images()
    I = grayscaleimage(downscale_img(load_image_cv(f1)))
    J = grayscaleimage(downscale_img(load_image_cv(f2)))

    W2 = PATCH_SIZE // 2
    dX = dY = SEARCH_RANGE
    print(f"image: {I.shape}  W2={W2}  dX={dX}  dY={dY}")
    u, v = block_method(I, J, W2=W2, dX=dX, dY=dY)

    mask = diff_mask(I, J, thresh=THRESHOLD, kernel_size=5, iterations=2)
    u, v = filter_flow(u, v, mask)

    bgr = flow_to_hsv(u, v, swap_sv=True)
    arrows = display_flow_quiver(I, u, v, step=8, scale=2.0)

    cv2.namedWindow("diff mask", cv2.WINDOW_NORMAL)
    cv2.imshow("diff mask", mask)
    cv_display_images(I, J, bgr, arrows)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()


