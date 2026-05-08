"""Faster R-CNN wrapper (torchvision) for multi-class detection.

Detected COCO classes are remapped to MOT-Challenge class ids via
config.COCO_TO_MOT_CLASS. Detections from unmapped classes are dropped.
"""
from __future__ import annotations
import numpy as np
import torch
import torchvision.transforms.functional as TF
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn,
    FasterRCNN_ResNet50_FPN_Weights,
)
from .config import CFG, DEVICE, use_amp, COCO_TO_MOT_CLASS


class FasterRCNNDetector:
    def __init__(self):
        weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        self.model = fasterrcnn_resnet50_fpn(weights=weights).to(DEVICE).eval()
        self._amp = use_amp(DEVICE)
        self._coco_keys = np.array(sorted(COCO_TO_MOT_CLASS.keys()))

    @torch.no_grad()
    def detect(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Frame BGR → ndarray[N, 6]: x, y, w, h, score, mot_cls."""
        rgb = frame_bgr[:, :, ::-1].copy()
        tensor = TF.to_tensor(rgb).to(DEVICE, non_blocking=True)
        if self._amp:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                out = self.model([tensor])[0]
        else:
            out = self.model([tensor])[0]
        boxes = out["boxes"].detach().cpu().numpy()
        scores = out["scores"].detach().cpu().numpy()
        labels = out["labels"].detach().cpu().numpy()

        keep = (scores >= CFG.det_score_th) & np.isin(labels, self._coco_keys)
        boxes = boxes[keep]
        scores = scores[keep]
        labels = labels[keep]
        if len(boxes) == 0:
            return np.zeros((0, 6), dtype=np.float32)

        mot_cls = np.array([COCO_TO_MOT_CLASS[int(l)] for l in labels],
                           dtype=np.float32)
        xywh = np.stack(
            [
                boxes[:, 0],
                boxes[:, 1],
                boxes[:, 2] - boxes[:, 0],
                boxes[:, 3] - boxes[:, 1],
            ],
            axis=1,
        )
        return np.concatenate([xywh, scores[:, None], mot_cls[:, None]], axis=1).astype(np.float32)
