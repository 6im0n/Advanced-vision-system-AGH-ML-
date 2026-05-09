"""YOLOv8 detector wrapper. Same .detect(frame) → ndarray[N, 6] interface
as FasterRCNNDetector so the tracker pipeline is detector-agnostic.

Loads finetuned weights from weights/yolov8_mot.pt by default. If absent,
falls back to ultralytics yolov8m.pt (COCO-pretrained, person class only).
"""
from __future__ import annotations
from pathlib import Path
import numpy as np

from .config import CFG, DEVICE, WEIGHTS_DIR


YOLO_FT_WEIGHTS = WEIGHTS_DIR / "yolov8_mot.pt"
YOLO_FALLBACK = "yolov8m.pt"
COCO_PERSON_CLS = 0   # COCO class id for "person" in pretrained YOLOv8


class YoloDetector:
    """Pedestrian-only detector. Output rows: [x, y, w, h, score, mot_cls=1]."""

    def __init__(self, weights_path: str | None = None, imgsz: int = 1280,
                 score_th: float | None = None, iou: float | None = None):
        from ultralytics import YOLO

        wp = Path(weights_path) if weights_path else YOLO_FT_WEIGHTS
        if wp.exists():
            self.model = YOLO(str(wp))
            self._finetuned = True
            print(f"[yolo] loaded finetuned weights {wp}")
        else:
            self.model = YOLO(YOLO_FALLBACK)
            self._finetuned = False
            print(f"[yolo] no finetuned weights at {wp} — using {YOLO_FALLBACK} (COCO)")

        self.imgsz = imgsz
        self.score_th = CFG.det_score_th if score_th is None else score_th
        self.iou = CFG.det_nms_iou if iou is None else iou
        self._device = "cuda" if DEVICE.type == "cuda" else "cpu"

    def detect(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Frame BGR → ndarray[N, 6]: x, y, w, h, score, mot_cls."""
        results = self.model.predict(
            source=frame_bgr,
            imgsz=self.imgsz,
            conf=self.score_th,
            iou=self.iou,
            device=self._device,
            verbose=False,
            half=True,
        )
        if not results:
            return np.zeros((0, 6), dtype=np.float32)
        r = results[0]
        if r.boxes is None or r.boxes.shape[0] == 0:
            return np.zeros((0, 6), dtype=np.float32)

        boxes = r.boxes.xyxy.detach().cpu().numpy()
        scores = r.boxes.conf.detach().cpu().numpy()
        labels = r.boxes.cls.detach().cpu().numpy().astype(int)

        if self._finetuned:
            # Single-class finetune: every output is pedestrian.
            keep = np.ones(len(boxes), dtype=bool)
        else:
            keep = labels == COCO_PERSON_CLS

        boxes = boxes[keep]
        scores = scores[keep]
        if len(boxes) == 0:
            return np.zeros((0, 6), dtype=np.float32)

        xywh = np.stack(
            [
                boxes[:, 0],
                boxes[:, 1],
                boxes[:, 2] - boxes[:, 0],
                boxes[:, 3] - boxes[:, 1],
            ],
            axis=1,
        )
        mot_cls = np.full((len(boxes), 1), 1.0, dtype=np.float32)  # MOT pedestrian
        return np.concatenate(
            [xywh.astype(np.float32), scores[:, None].astype(np.float32), mot_cls],
            axis=1,
        )
