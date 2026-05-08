"""MOT tracker: Track lifecycle + Hungarian association on IoU + appearance."""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from scipy.optimize import linear_sum_assignment
from .config import CFG
from .siamfc import SiamEmbedder, crop_person, cosine_sim_matrix


@dataclass
class Track:
    track_id: int
    bbox: np.ndarray              # [x, y, w, h]
    feat: np.ndarray              # appearance embedding (EMA, L2-normed)
    hits: int = 1
    age: int = 0                  # frames since last successful match
    state: str = "tentative"      # tentative | confirmed | deleted
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(4))
    history: list = field(default_factory=list)

    def predict(self):
        # Only propagate position (x, y); freeze size (w, h) during prediction.
        # Decay velocity to avoid drift during long misses.
        step = self.velocity.copy()
        step[2:] = 0.0
        self.bbox = self.bbox + step
        self.velocity *= 0.9
        self.age += 1

    def update(self, det_bbox: np.ndarray, det_feat: np.ndarray):
        v = det_bbox - self.bbox
        # Only smooth position velocity; size taken straight from detection.
        self.velocity[:2] = 0.5 * self.velocity[:2] + 0.5 * v[:2]
        self.velocity[2:] = 0.0
        self.bbox = det_bbox.copy()

        feat = CFG.ema_alpha * self.feat + (1.0 - CFG.ema_alpha) * det_feat
        self.feat = feat / (np.linalg.norm(feat) + 1e-9)

        self.hits += 1
        self.age = 0
        if self.state == "tentative" and self.hits >= CFG.n_init:
            self.state = "confirmed"

        cx = self.bbox[0] + self.bbox[2] / 2.0
        cy = self.bbox[1] + self.bbox[3] / 2.0
        self.history.append((cx, cy))
        if len(self.history) > CFG.trail_len:
            self.history.pop(0)

    def mark_missed(self):
        if self.state == "tentative":
            self.state = "deleted"
        elif self.age > CFG.max_age:
            self.state = "deleted"


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """xywh inputs → IoU matrix [N, M]."""
    if a.shape[0] == 0 or b.shape[0] == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=np.float32)
    a1, a2 = a[:, :2], a[:, :2] + a[:, 2:4]
    b1, b2 = b[:, :2], b[:, :2] + b[:, 2:4]
    tl = np.maximum(a1[:, None], b1[None])
    br = np.minimum(a2[:, None], b2[None])
    wh = np.clip(br - tl, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    area_a = a[:, 2] * a[:, 3]
    area_b = b[:, 2] * b[:, 3]
    union = area_a[:, None] + area_b[None] - inter + 1e-9
    return (inter / union).astype(np.float32)


class MOTTracker:
    def __init__(self, embedder: SiamEmbedder):
        self.embedder = embedder
        self.tracks: list[Track] = []
        self.next_id = 1

    def update(self, frame_bgr: np.ndarray, dets_xywhs: np.ndarray) -> list[Track]:
        # 1. Predict track motion
        for t in self.tracks:
            t.predict()

        # 2. Embed current detections
        det_xywh = dets_xywhs[:, :4] if dets_xywhs.size else np.zeros((0, 4), dtype=np.float32)
        crops = [crop_person(frame_bgr, b) for b in det_xywh]
        det_feats = self.embedder.embed(crops)

        # 3. Cost matrix and Hungarian
        T, D = len(self.tracks), len(det_xywh)
        if T > 0 and D > 0:
            track_box = np.stack([t.bbox for t in self.tracks])
            track_feat = np.stack([t.feat for t in self.tracks])
            ious = iou_matrix(track_box, det_xywh)
            sims = cosine_sim_matrix(track_feat, det_feats)
            cost = CFG.w_iou * (1.0 - ious) + CFG.w_app * (1.0 - sims)
            cost[ious < CFG.iou_gate] = 1e6
            row, col = linear_sum_assignment(cost)
        else:
            cost = np.zeros((T, D))
            row, col = np.array([], dtype=int), np.array([], dtype=int)

        matched_t, matched_d = set(), set()
        for r, c in zip(row, col):
            if cost[r, c] < CFG.cost_max:
                self.tracks[r].update(det_xywh[c], det_feats[c])
                matched_t.add(r)
                matched_d.add(c)

        # 4. Mark unmatched tracks
        for i, t in enumerate(self.tracks):
            if i not in matched_t:
                t.mark_missed()

        # 5. Spawn tentative tracks for unmatched detections
        for j in range(D):
            if j not in matched_d:
                self.tracks.append(Track(
                    track_id=self.next_id,
                    bbox=det_xywh[j].astype(np.float32).copy(),
                    feat=det_feats[j].astype(np.float32).copy(),
                ))
                self.next_id += 1

        # 6. Cull deleted
        self.tracks = [t for t in self.tracks if t.state != "deleted"]

        return [t for t in self.tracks if t.state == "confirmed"]
