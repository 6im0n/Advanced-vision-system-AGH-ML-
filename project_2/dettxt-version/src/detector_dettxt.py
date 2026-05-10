"""det.txt detector — serves the pre-supplied detections shipped with each
sequence. Per challenge rules ("detekcji, których należy użyć do inicjalizacji
algorytmu śledzenia"), these are the canonical input to the tracker.

.detect(frame) → ndarray[N, 6]: x, y, w, h, score, mot_cls=1.
Stateful: each call advances an internal frame counter (1-indexed). One
detector instance per sequence.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np

from .config import CFG
from .io_mot import parse_mot_csv


class DetTxtDetector:
    """Loads seq_dir/det/det.txt at init and emits per-frame detections."""

    def __init__(self, seq_dir: Path, score_th: float | None = None):
        det_path = Path(seq_dir) / "det" / "det.txt"
        if not det_path.exists():
            raise FileNotFoundError(f"no det.txt at {det_path}")
        raw = parse_mot_csv(det_path)
        self.score_th = CFG.det_score_th if score_th is None else score_th
        # det.txt cols: frame, -1, x, y, w, h, score
        self._by_frame: dict[int, np.ndarray] = {}
        if raw.size:
            keep = raw[:, 6] >= self.score_th
            raw = raw[keep]
            for f in np.unique(raw[:, 0].astype(int)):
                rows = raw[raw[:, 0].astype(int) == f]
                xywh = rows[:, 2:6].astype(np.float32)
                scores = rows[:, 6:7].astype(np.float32)
                cls = np.ones((len(rows), 1), dtype=np.float32)  # MOT pedestrian
                self._by_frame[int(f)] = np.concatenate([xywh, scores, cls], axis=1)
        self._cursor = 0
        print(f"[dettxt] {det_path}: {sum(len(v) for v in self._by_frame.values())} dets "
              f"across {len(self._by_frame)} frames (score_th={self.score_th})")

    def detect(self, frame_bgr: np.ndarray) -> np.ndarray:  # noqa: ARG002
        self._cursor += 1
        return self._by_frame.get(self._cursor, np.zeros((0, 6), dtype=np.float32))
