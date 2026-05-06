import os
import time
import numpy as np
import cv2

from lab7_2 import (
    ask_user_images,
    load_image_cv,
    grayscaleimage,
    downscale_img,
    flow_to_hsv,
    display_flow_quiver,
    stack_grid,
    label,
)

# Task 1 - dense methods (Farneback, DIS, dense-grid LK).
# Task 2 - sparse Lucas-Kanade on uniform grid of points.

LK_MAG_CLIP = 10.0   # clip magnitude for dense-LK visualisation
GRID_STEP = 10      # grid used on the image.
LK_WIN = (21, 21)
LK_MAX_LEVEL = 3
LK_CRITERIA = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)


def flow_farneback(I, J):
    return cv2.calcOpticalFlowFarneback(
        I, J, None,
        pyr_scale=0.5, levels=3, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
    )


def flow_dis(I, J, preset=cv2.DISOPTICAL_FLOW_PRESET_MEDIUM):
    dis = cv2.DISOpticalFlow_create(preset)
    return dis.calc(I, J, None)


def flow_dense_lk(I, J, step=1):
    h, w = I.shape
    ys, xs = np.mgrid[0:h:step, 0:w:step]
    p0 = np.stack([xs.ravel(), ys.ravel()], axis=1).astype(np.float32).reshape(-1, 1, 2)
    p1, st, _ = cv2.calcOpticalFlowPyrLK(
        I, J, p0, None,
        winSize=LK_WIN, maxLevel=LK_MAX_LEVEL, criteria=LK_CRITERIA,
    )
    flow = np.zeros((h, w, 2), dtype=np.float32)
    p0r = p0.reshape(-1, 2)
    p1r = p1.reshape(-1, 2)
    stm = st.ravel().astype(bool)
    dx = p1r[:, 0] - p0r[:, 0]
    dy = p1r[:, 1] - p0r[:, 1]
    xs_i = p0r[:, 0].astype(np.int32)
    ys_i = p0r[:, 1].astype(np.int32)
    flow[ys_i[stm], xs_i[stm], 0] = dx[stm]
    flow[ys_i[stm], xs_i[stm], 1] = dy[stm]
    if step > 1:
        flow[..., 0] = cv2.resize(flow[..., 0], (w, h), interpolation=cv2.INTER_LINEAR)
        flow[..., 1] = cv2.resize(flow[..., 1], (w, h), interpolation=cv2.INTER_LINEAR)
    return flow


def clip_magnitude(flow, max_mag):
    u = flow[..., 0]
    v = flow[..., 1]
    mag = np.sqrt(u * u + v * v)
    scale = np.where(mag > max_mag, max_mag / np.maximum(mag, 1e-6), 1.0)
    out = flow.copy()
    out[..., 0] = u * scale
    out[..., 1] = v * scale
    return out


def flow_vis(flow, swap_sv=True):
    return flow_to_hsv(flow[..., 0], flow[..., 1], swap_sv=swap_sv)


def sparse_lk_grid(I, J, step=GRID_STEP):
    h, w = I.shape
    ys, xs = np.mgrid[step:h:step, step:w:step]
    p0 = np.stack([xs.ravel(), ys.ravel()], axis=1).astype(np.float32).reshape(-1, 1, 2)
    p1, st, _ = cv2.calcOpticalFlowPyrLK(
        I, J, p0, None,
        winSize=LK_WIN, maxLevel=LK_MAX_LEVEL, criteria=LK_CRITERIA,
    )
    return p0, p1, st


def draw_sparse(I, p0, p1, st):
    img = cv2.cvtColor(I, cv2.COLOR_GRAY2BGR) if I.ndim == 2 else I.copy()
    p0r = p0.reshape(-1, 2)
    p1r = p1.reshape(-1, 2)
    stm = st.ravel().astype(bool)
    for (x0, y0), (x1, y1), ok in zip(p0r, p1r, stm):
        if not ok:
            continue
        cv2.line(img, (int(x0), int(y0)), (int(x1), int(y1)), (0, 0, 255), 1)
        cv2.circle(img, (int(x0), int(y0)), 1, (0, 255, 0), -1)
    return img


def time_call(fn, *args, **kwargs):
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    return out, (time.perf_counter() - t0) * 1000.0


def main():
    f1, f2 = ask_user_images()
    I = grayscaleimage(downscale_img(load_image_cv(f1)))
    J = grayscaleimage(downscale_img(load_image_cv(f2)))
    print(f"image: {I.shape}")

    flow_fb,  ms_fb  = time_call(flow_farneback, I, J)
    flow_dis_, ms_dis = time_call(flow_dis, I, J)
    flow_lk,  ms_lk  = time_call(flow_dense_lk, I, J, 1)

    flow_lk_clipped = clip_magnitude(flow_lk, LK_MAG_CLIP)

    print(f"Farneback: {ms_fb:.1f} ms")
    print(f"DIS      : {ms_dis:.1f} ms")
    print(f"dense LK : {ms_lk:.1f} ms (mag clipped to {LK_MAG_CLIP})")

    (p0, p1, st), ms_sparse = time_call(sparse_lk_grid, I, J, GRID_STEP)
    print(f"sparse LK ({GRID_STEP}px grid): {ms_sparse:.1f} ms  "
          f"tracked={int(st.sum())}/{len(st)}")

    sparse_img = draw_sparse(I, p0, p1, st)

    tiles = [
        label(I, "I"),
        label(J, "J"),
        label(cv2.absdiff(I, J), "|I-J|"),
        label(flow_vis(flow_fb),         f"Farneback ({ms_fb:.0f} ms)"),
        label(flow_vis(flow_dis_),       f"DIS ({ms_dis:.0f} ms)"),
        label(flow_vis(flow_lk_clipped), f"dense LK clip<={int(LK_MAG_CLIP)} ({ms_lk:.0f} ms)"),
        label(flow_to_hsv(flow_fb[..., 0], flow_fb[..., 1], swap_sv=False),
              "Farneback (white bg)"),
        label(display_flow_quiver(I, flow_fb[..., 0], flow_fb[..., 1], step=12, scale=2.0),
              "Farneback arrows"),
        label(sparse_img, f"sparse LK grid={GRID_STEP}px"),
    ]
    grid = stack_grid(tiles, cols=3)
    cv2.namedWindow("lab7_3 OpenCV optical flow", cv2.WINDOW_NORMAL)
    cv2.imshow("lab7_3 OpenCV optical flow", grid)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
