import sys
import os
import numpy as np
import cv2


#this part take the previous lab, to improve it, in the last lab we use block compute that is very not efficiant for realtime.
# in this methode we use the optical flow in mutiple space (we build a "piramide" of scalled images)


SCALE_FACTOR = 1  # div: 1=same, 2=half, 4=quart
PATCH_SIZE = 7    ## must be odd; W2 = PATCH_SIZE // 2
SEARCH_RANGE = 3  # dX = dY = SEARCH_RANGE axis
THRESHOLD = 30
MAX_SCALE = 2    # pyramid levels

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
                                normalize=False, borderType=cv2.BORDER_REPLICATE)
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


def of(I, J, W2=3, dY=3, dX=3):
    return block_method(I, J, W2=W2, dX=dX, dY=dY)


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
    xs, ys = np.meshgrid(np.arange(w, dtype=np.float32),
                         np.arange(h, dtype=np.float32))
    map_x = xs + u.astype(np.float32)
    map_y = ys + v.astype(np.float32)
    return cv2.remap(J, map_x, map_y,
                     interpolation=cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REPLICATE)


def multiscale_of(I_in, J_in, W2=3, dX=3, dY=3, max_scale=3):
    IP = pyramid(I_in, max_scale)
    JP = pyramid(J_in, max_scale)

    h_top, w_top = IP[-1].shape
    u_acc = np.zeros((h_top, w_top), dtype=np.float32)
    v_acc = np.zeros((h_top, w_top), dtype=np.float32)

    for s in range(max_scale - 1, -1, -1):
        I = IP[s]
        J = JP[s]
        J_warped = warp_with_flow(J, u_acc, v_acc)
        u, v = of(I, J_warped, W2=W2, dY=dY, dX=dX)
        u_acc += u
        v_acc += v

        if s > 0:
            h_next, w_next = IP[s - 1].shape
            u_acc = cv2.resize(2.0 * u_acc, (w_next, h_next),
                               interpolation=cv2.INTER_LINEAR)
            v_acc = cv2.resize(2.0 * v_acc, (w_next, h_next),
                               interpolation=cv2.INTER_LINEAR)

    return u_acc, v_acc


def stack_grid(tiles, cols=3, pad=4, bg=0):
    h = max(t.shape[0] for t in tiles)
    w = max(t.shape[1] for t in tiles)
    norm = []
    for t in tiles:
        if t.ndim == 2:
            t = cv2.cvtColor(t, cv2.COLOR_GRAY2BGR)
        if t.shape[0] != h or t.shape[1] != w:
            t = cv2.resize(t, (w, h), interpolation=cv2.INTER_NEAREST)
        norm.append(t)
    rows = []
    for r in range(0, len(norm), cols):
        row_tiles = norm[r:r + cols]
        while len(row_tiles) < cols:
            row_tiles.append(np.full((h, w, 3), bg, dtype=np.uint8))
        row = np.hstack([np.pad(t, ((pad, pad), (pad, pad), (0, 0)),
                                constant_values=bg) for t in row_tiles])
        rows.append(row)
    return np.vstack(rows)


def label(tile, text):
    if tile.ndim == 2:
        tile = cv2.cvtColor(tile, cv2.COLOR_GRAY2BGR)
    out = tile.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 22), (0, 0, 0), -1)
    cv2.putText(out, text, (4, 16), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return out


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

    diff = cv2.absdiff(I, J)
    bgr_raw = flow_to_hsv(u_raw, v_raw, swap_sv=True)
    bgr_white = flow_to_hsv(u_raw, v_raw, swap_sv=False)
    bgr_filt = flow_to_hsv(u, v, swap_sv=True)
    arrows = display_flow_quiver(I, u, v, step=8, scale=2.0)

    tiles = [
        label(I, "I"),
        label(J, "J"),
        label(diff, "|I-J|"),
        label(bgr_raw, "flow raw (black bg)"),
        label(bgr_white, "flow raw (white bg)"),
        label(bgr_filt, f"flow filtered ({MAX_SCALE} scales)"),
        label(mask, "diff mask"),
        label(arrows, "arrows"),
    ]
    grid = stack_grid(tiles, cols=3)
    cv2.namedWindow("lab7_2 multiscale OF", cv2.WINDOW_NORMAL)
    cv2.imshow("lab7_2 multiscale OF", grid)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()


