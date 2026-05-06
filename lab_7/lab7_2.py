import sys
import os
import numpy as np
import cv2


#this part take the previous lab, to improve it, in the last lab we use block compute that is very not efficiant for realtime.
# in this methode we use the optical flow in mutiple space (we build a "piramide" of scalled images)


SCALE_FACTOR = 1  # div: 1=same, 2=half, 4=quart
PATCH_SIZE = 7    ## must be odd; W2 = PATCH_SIZE // 2
SEARCH_RANGE = 3  # dX = dY = SEARCH_RANGE axis
THRESHOLD = 10
MAX_SCALE = 3    # pyramid levels

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
            # shift J by (dy,dx); pad with large value so border SSD loses
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

    # zero out borders where window goes out
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

def of(J_org, I, J, W2=3, dY=3, dX=3):
    u, v = block_method(I, J, W2=W2, dX=dX, dY=dY)
    diff = cv2.absdiff(I, J)
    cv2.namedWindow("of: J_org", cv2.WINDOW_NORMAL)
    cv2.imshow("of: J_org", J_org)
    cv2.namedWindow("of: I", cv2.WINDOW_NORMAL)
    cv2.imshow("of: I", I)
    cv2.namedWindow("of: J (warped)", cv2.WINDOW_NORMAL)
    cv2.imshow("of: J (warped)", J)
    cv2.namedWindow("of: |I-J|", cv2.WINDOW_NORMAL)
    cv2.imshow("of: |I-J|", diff)
    return u, v


def vis_flow(u, v, YX, name):
    bgr = flow_to_hsv(u, v, swap_sv=True)
    bgr = cv2.resize(bgr, (YX[1], YX[0]), interpolation=cv2.INTER_NEAREST)
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.imshow(name, bgr)


def pyramid(im, max_scale):
    images = [im]
    for k in range(1, max_scale):
        images.append(cv2.resize(images[k - 1], (0, 0), fx=0.5, fy=0.5))
    return images


def warp_with_flow(J, u, v):
    h, w = J.shape
    J_new = J.copy()
    for j in range(h):
        for i in range(w):
            jj = int(round(j + v[j, i]))
            ii = int(round(i + u[j, i]))
            if 0 <= jj < h and 0 <= ii < w:
                J_new[j, i] = J[jj, ii]
    return J_new


def multiscale_of(I_in, J_in, W2=3, dX=3, dY=3, max_scale=3):
    IP = pyramid(I_in, max_scale)
    JP = pyramid(J_in, max_scale)
    H, W = I_in.shape
    u_total = np.zeros((H, W), dtype=np.float32)
    v_total = np.zeros((H, W), dtype=np.float32)

    J = JP[-1]
    for s in range(max_scale - 1, -1, -1):
        I = IP[s]
        u, v = of(J_in, I, J, W2=W2, dY=dY, dX=dX)

        u_up = cv2.resize((2 ** s) * u, (W, H), interpolation=cv2.INTER_LINEAR)
        v_up = cv2.resize((2 ** s) * v, (W, H), interpolation=cv2.INTER_LINEAR)
        u_total += u_up
        v_total += v_up

        if s > 0:
            J_next = JP[s - 1].copy()
            u_next = cv2.resize(2 * u, (0, 0), fx=2, fy=2,
                                interpolation=cv2.INTER_LINEAR)
            v_next = cv2.resize(2 * v, (0, 0), fx=2, fy=2,
                                interpolation=cv2.INTER_LINEAR)
            J_next = warp_with_flow(J_next, u_next, v_next)
            J = J_next

    return u_total, v_total


def main():
    f1, f2 = ask_user_images()
    I = grayscaleimage(downscale_img(load_image_cv(f1)))
    J = grayscaleimage(downscale_img(load_image_cv(f2)))

    W2 = PATCH_SIZE // 2
    dX = dY = SEARCH_RANGE
    print(f"image: {I.shape}  W2={W2}  dX={dX}  dY={dY}  scales={MAX_SCALE}")

    u_raw, v_raw = multiscale_of(I, J, W2=W2, dX=dX, dY=dY, max_scale=MAX_SCALE)
    print(f"raw flow: u[{u_raw.min():.2f},{u_raw.max():.2f}] "
          f"v[{v_raw.min():.2f},{v_raw.max():.2f}] "
          f"nonzero={int(np.count_nonzero(u_raw) + np.count_nonzero(v_raw))}")

    mask = diff_mask(I, J, thresh=THRESHOLD, kernel_size=5, iterations=2)
    print(f"mask nonzero pixels: {int(np.count_nonzero(mask))} / {mask.size}")
    u, v = filter_flow(u_raw, v_raw, mask)

    vis_flow(u_raw, v_raw, I.shape, "of raw (no filter, no swap)")
    bgr_white = flow_to_hsv(u_raw, v_raw, swap_sv=False)
    cv2.namedWindow("of raw white-bg", cv2.WINDOW_NORMAL)
    cv2.imshow("of raw white-bg", bgr_white)

    vis_flow(u, v, I.shape, f"of filtered ({MAX_SCALE} scales)")
    arrows = display_flow_quiver(I, u, v, step=8, scale=2.0)
    cv2.namedWindow("arrows", cv2.WINDOW_NORMAL)
    cv2.imshow("arrows", arrows)
    cv2.namedWindow("diff mask", cv2.WINDOW_NORMAL)
    cv2.imshow("diff mask", mask)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()


