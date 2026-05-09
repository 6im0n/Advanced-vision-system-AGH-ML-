"""Ensemble detector: union of det.txt (challenge-supplied) + YOLOv8 own
detections, deduplicated by class-agnostic NMS.

Rationale: det.txt is the official tracker input but has limited recall on
small / occluded / off-domain pedestrians. Own YOLO (finetuned on EVS-MOT)
catches a complementary set. Union → tracker gets best of both.

Same .detect(frame) → ndarray[N, 6] interface as the other detectors.
Stateful: increments an internal frame counter to index det.txt; one
detector instance per sequence.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np

from .detector_yolo import YoloDetector
from .detector_dettxt import DetTxtDetector


def _iou_xywh(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU. a: [N,4], b: [M,4] in xywh. Returns [N,M]."""
    if a.size == 0 or b.size == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    ax2 = a[:, 0] + a[:, 2]
    ay2 = a[:, 1] + a[:, 3]
    bx2 = b[:, 0] + b[:, 2]
    by2 = b[:, 1] + b[:, 3]
    inter_x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    inter_y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    inter_x2 = np.minimum(ax2[:, None], bx2[None, :])
    inter_y2 = np.minimum(ay2[:, None], by2[None, :])
    iw = np.clip(inter_x2 - inter_x1, 0, None)
    ih = np.clip(inter_y2 - inter_y1, 0, None)
    inter = iw * ih
    area_a = (a[:, 2] * a[:, 3])[:, None]
    area_b = (b[:, 2] * b[:, 3])[None, :]
    union = area_a + area_b - inter + 1e-6
    return (inter / union).astype(np.float32)


def _nms_xywh(dets: np.ndarray, iou_th: float) -> np.ndarray:
    """Greedy NMS, class-agnostic. dets: [N,6] (x,y,w,h,score,cls)."""
    if len(dets) == 0:
        return dets
    order = np.argsort(-dets[:, 4])
    kept = []
    boxes = dets[order, :4]
    while len(order) > 0:
        i = order[0]
        kept.append(i)
        if len(order) == 1:
            break
        rest = order[1:]
        ious = _iou_xywh(boxes[:1], dets[rest, :4])[0]
        keep_mask = ious < iou_th
        order = rest[keep_mask]
        boxes = dets[order, :4]
    return dets[np.array(kept, dtype=int)]


class EnsembleDetector:
    """det.txt ∪ YOLO → NMS."""

    def __init__(self, seq_dir: Path, nms_iou: float = 0.5,
                 dettxt_score_th: float | None = None,
                 yolo_kwargs: dict | None = None):
        self.dettxt = DetTxtDetector(seq_dir, score_th=dettxt_score_th)
        self.yolo = YoloDetector(**(yolo_kwargs or {}))
        self.nms_iou = nms_iou

    def detect(self, frame_bgr: np.ndarray) -> np.ndarray:
        a = self.dettxt.detect(frame_bgr)
        b = self.yolo.detect(frame_bgr)
        if a.size == 0:
            merged = b
        elif b.size == 0:
            merged = a
        else:
            merged = np.concatenate([a, b], axis=0)
        return _nms_xywh(merged, self.nms_iou)
