"""
AVS Lab 12.1 - Simple thermal image analysis.

Pipeline:
  1. Binarisation, filtering, object indexing (labelling).
  2. Keep only objects with the required height-to-width proportions.
  3. Merge detected parts into a single bounding box.

Works on a single frame (frame_003090.png) or the IR video (vid1_IR.avi).
"""

import cv2
import numpy as np

# Parameters
THRESH = 50            # binarisation threshold (people are mid-bright, bg dark)
MIN_AREA = 150         # reject small noise blobs (px)
MIN_ASPECT = 1.02       # keep tall objects: height / width >= MIN_ASPECT
MERGE_GAP = 10         # px gap below which two boxes are merged into one


# Pipeline
def binarise(gray):
    """Blur + fixed threshold -> binary mask of hot objects."""
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, mask = cv2.threshold(blur, THRESH, 255, cv2.THRESH_BINARY)
    return mask


def filter_mask(mask):
    """Morphology: remove speckle (open), then close gaps inside silhouettes."""
    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 15))  # tall: join head/body
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k_open)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close)
    return mask


def label_objects(mask):
    """Connected-component labelling -> list of (x, y, w, h) boxes."""
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    boxes = []
    for i in range(1, n):  # skip background (label 0)
        x, y, w, h, area = stats[i]
        if area < MIN_AREA:
            continue
        if h / max(w, 1) < MIN_ASPECT:  # keep tall (person-shaped) objects only
            continue
        boxes.append([x, y, w, h])
    return boxes


def boxes_overlap(a, b, gap):
    """True if boxes a, b overlap or are within `gap` px (inflate then test)."""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    return not (ax > bx + bw + gap or bx > ax + aw + gap or
                ay > by + bh + gap or by > ay + ah + gap)


def merge_boxes(boxes, gap=MERGE_GAP):
    """Merge nearby/overlapping parts into a single bounding rectangle."""
    boxes = [list(b) for b in boxes]
    changed = True
    while changed:
        changed = False
        out = []
        while boxes:
            cur = boxes.pop(0)
            i = 0
            while i < len(boxes):
                if boxes_overlap(cur, boxes[i], gap):
                    bx, by, bw, bh = boxes.pop(i)
                    x1 = min(cur[0], bx)
                    y1 = min(cur[1], by)
                    x2 = max(cur[0] + cur[2], bx + bw)
                    y2 = max(cur[1] + cur[3], by + bh)
                    cur = [x1, y1, x2 - x1, y2 - y1]
                    changed = True
                else:
                    i += 1
            out.append(cur)
        boxes = out
    return boxes


def detect(gray):
    """Full pipeline on a greyscale frame -> list of bounding boxes."""
    mask = binarise(gray)
    mask = filter_mask(mask)
    boxes = label_objects(mask)
    boxes = merge_boxes(boxes)
    # second aspect check after merging (a merged blob can become person-shaped)
    boxes = [b for b in boxes if b[3] / max(b[2], 1) >= MIN_ASPECT]
    return boxes, mask


def draw(frame, boxes):
    for x, y, w, h in boxes:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
    return frame


def label(img, text):
    """Write a title in the top-left corner of a panel."""
    cv2.putText(img, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return img


def make_grid(frame, mask, boxes):
    """Build the 2x2 view: original | mask | mask+detection | original+detection."""
    original = frame.copy()

    mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    # mask painted red over the frame, plus detection boxes
    red = np.zeros_like(frame)
    red[:, :, 2] = mask
    merge = cv2.addWeighted(frame, 1.0, red, 0.5, 0)
    draw(merge, boxes)

    detection = draw(frame.copy(), boxes)

    top = np.hstack((label(original, "Original"), label(mask_bgr, "Mask (filtered)")))
    bottom = np.hstack((label(merge, "Mask + detection"), label(detection, "Detection")))
    return np.vstack((top, bottom))


# Run
def run_image(path):
    frame = cv2.imread(path)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    boxes, mask = detect(gray)
    grid = make_grid(frame, mask, boxes)
    cv2.imwrite("out_detection.png", draw(frame.copy(), boxes))
    cv2.imwrite("out_mask.png", mask)
    print(f"{path}: {len(boxes)} objects -> out_detection.png, out_mask.png")
    cv2.imshow("thermal", grid)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def run_video(path):
    """Frame-by-frame viewer. Right/d = next, Left/a = previous, ESC = quit."""
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    LEFT = (81, 65361, 2424832, ord("a"))    # left arrow on different backends, or 'a'
    RIGHT = (83, 65363, 2555904, ord("d"), 32)  # right arrow, or 'd', or space

    idx = 0
    while True:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        boxes, mask = detect(gray)
        grid = make_grid(frame, mask, boxes)

        cv2.imshow("thermal (left/right = step, ESC = quit)", grid)
        key = cv2.waitKeyEx(0)
        if key in (27, ord("q")):
            break
        elif key in RIGHT:
            idx = min(total - 1, idx + 1)
        elif key in LEFT:
            idx = max(0, idx - 1)
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    import sys
    src = sys.argv[1] if len(sys.argv) > 1 else "frame_003090.png"
    if src.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
        run_image(src)
    else:
        run_video(src)