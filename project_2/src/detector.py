"""Faster R-CNN wrapper (torchvision) for person detection."""
from __future__ import annotations
import numpy as np
import torch
import torchvision.transforms.functional as TF
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn,
    FasterRCNN_ResNet50_FPN_Weights,
)
from .config import CFG, DEVICE


class FasterRCNNDetector:
    def __init__(self):
        weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        self.model = fasterrcnn_resnet50_fpn(weights=weights).to(DEVICE).eval()

    @torch.no_grad()
    def detect(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Frame BGR → ndarray[N, 5]: x, y, w, h, score (filtered to person class)."""
        rgb = frame_bgr[:, :, ::-1].copy()
        tensor = TF.to_tensor(rgb).to(DEVICE)
        out = self.model([tensor])[0]
        boxes = out["boxes"].detach().cpu().numpy()
        scores = out["scores"].detach().cpu().numpy()
        labels = out["labels"].detach().cpu().numpy()

        keep = (labels == CFG.det_class_id) & (scores >= CFG.det_score_th)
        boxes = boxes[keep]
        scores = scores[keep]
        if len(boxes) == 0:
            return np.zeros((0, 5), dtype=np.float32)

        xywh = np.stack(
            [
                boxes[:, 0],
                boxes[:, 1],
                boxes[:, 2] - boxes[:, 0],
                boxes[:, 3] - boxes[:, 1],
            ],
            axis=1,
        )
        return np.concatenate([xywh, scores[:, None]], axis=1).astype(np.float32)
