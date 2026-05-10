"""Visualization helpers: draw boxes/IDs/trails, write video."""
from __future__ import annotations
from pathlib import Path
import cv2
import numpy as np
from .config import MOT_CLASS_NAMES

_PALETTE = [
    (31, 119, 180), (255, 127, 14), (44, 160, 44), (214, 39, 40),
    (148, 103, 189), (140, 86, 75), (227, 119, 194), (127, 127, 127),
    (188, 189, 34), (23, 190, 207),
]


def color_for_id(tid: int):
    return _PALETTE[int(tid) % len(_PALETTE)]


def draw_tracks(frame, tracks, draw_trail: bool = True):
    img = frame.copy()
    for t in tracks:
        x, y, w, h = t.bbox
        col = color_for_id(t.track_id)
        cls_name = MOT_CLASS_NAMES.get(getattr(t, "cls_id", 1), "?")
        label = f"ID{t.track_id} {cls_name}"
        cv2.rectangle(img, (int(x), int(y)), (int(x + w), int(y + h)), col, 2)
        cv2.putText(img, label, (int(x), max(0, int(y) - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
        if draw_trail and len(t.history) > 1:
            pts = np.array(t.history, np.int32)
            cv2.polylines(img, [pts], False, col, 2)
    return img


def draw_boxes(frame, boxes_xywh, color=(0, 255, 0), label: str | None = None):
    img = frame.copy()
    for b in boxes_xywh:
        x, y, w, h = b[:4]
        cv2.rectangle(img, (int(x), int(y)), (int(x + w), int(y + h)), color, 2)
        if label:
            cv2.putText(img, label, (int(x), max(0, int(y) - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return img


class VideoWriter:
    def __init__(self, path: Path, fps: int, size_wh):
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.w = cv2.VideoWriter(str(path), fourcc, fps, size_wh)

    def write(self, frame):
        self.w.write(frame)

    def close(self):
        self.w.release()
