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
    cls_id: int = 1               # MOT-Challenge class id
    hits: int = 1
    age: int = 0                  # frames since last successful match
    state: str = "tentative"      # tentative | confirmed | lost | deleted
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(4))
    history: list = field(default_factory=list)
    # (frame_idx, bbox) buffered while tentative; flushed by runner on first confirm.
    bbox_history: list = field(default_factory=list)
    emitted: bool = False         # runner sets True after flushing pending bboxes

    def predict(self):
        # Only propagate position (x, y); freeze size (w, h) during prediction.
        # Decay velocity to avoid drift during long misses.
        step = self.velocity.copy()
        step[2:] = 0.0
        self.bbox = self.bbox + step
        self.velocity *= 0.9
        self.age += 1

    def update(self, det_bbox: np.ndarray, det_feat: np.ndarray, frame_idx: int):
        v = det_bbox - self.bbox
        # Only smooth position velocity; size taken straight from detection.
        self.velocity[:2] = 0.5 * self.velocity[:2] + 0.5 * v[:2]
        self.velocity[2:] = 0.0
        self.bbox = det_bbox.copy()
        if self.state == "tentative":
            self.bbox_history.append((frame_idx, det_bbox.astype(np.float32).copy()))

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
        # Tentative tracks die instantly when unmatched.
        # Confirmed tracks first move to "lost" (kept in shadow pool for
        # resurrection by appearance), then deleted after lost_max_age.
        if self.state == "tentative":
            self.state = "deleted"
        elif self.state == "confirmed" and self.age > CFG.max_age:
            self.state = "lost"
        elif self.state == "lost" and self.age > (CFG.max_age + CFG.lost_max_age):
            self.state = "deleted"

    def resurrect(self, det_bbox: np.ndarray, det_feat: np.ndarray):
        """Reactivate a lost track when matched again by appearance."""
        self.bbox = det_bbox.copy()
        self.velocity[:] = 0.0
        feat = CFG.ema_alpha * self.feat + (1.0 - CFG.ema_alpha) * det_feat
        self.feat = feat / (np.linalg.norm(feat) + 1e-9)
        self.age = 0
        self.hits += 1
        self.state = "confirmed"
        cx = self.bbox[0] + self.bbox[2] / 2.0
        cy = self.bbox[1] + self.bbox[3] / 2.0
        self.history.append((cx, cy))
        if len(self.history) > CFG.trail_len:
            self.history.pop(0)


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

    def update(self, frame_bgr: np.ndarray, dets: np.ndarray, frame_idx: int = 0) -> list[Track]:
        """`dets` is ndarray[D, 6]: x, y, w, h, score, mot_cls."""
        active_tracks = [t for t in self.tracks if t.state in ("tentative", "confirmed")]
        lost_tracks = [t for t in self.tracks if t.state == "lost"]

        for t in active_tracks:
            t.predict()
        for t in lost_tracks:
            t.age += 1

        if dets.size:
            det_xywh = dets[:, :4]
            det_cls = dets[:, 5].astype(int)
        else:
            det_xywh = np.zeros((0, 4), dtype=np.float32)
            det_cls = np.zeros((0,), dtype=int)
        crops = [crop_person(frame_bgr, b) for b in det_xywh]
        det_feats = self.embedder.embed(crops)

        T, D = len(active_tracks), len(det_xywh)
        if T > 0 and D > 0:
            track_box = np.stack([t.bbox for t in active_tracks])
            track_feat = np.stack([t.feat for t in active_tracks])
            track_cls = np.array([t.cls_id for t in active_tracks])
            ious = iou_matrix(track_box, det_xywh)
            sims = cosine_sim_matrix(track_feat, det_feats)
            cost = CFG.w_iou * (1.0 - ious) + CFG.w_app * (1.0 - sims)
            cost[ious < CFG.iou_gate] = 1e6
            cost[track_cls[:, None] != det_cls[None, :]] = 1e6   # forbid cross-class match
            row, col = linear_sum_assignment(cost)
        else:
            cost = np.zeros((T, D))
            row, col = np.array([], dtype=int), np.array([], dtype=int)

        matched_t, matched_d = set(), set()
        for r, c in zip(row, col):
            if cost[r, c] < CFG.cost_max:
                active_tracks[r].update(det_xywh[c], det_feats[c], frame_idx)
                matched_t.add(r)
                matched_d.add(c)

        for i, t in enumerate(active_tracks):
            if i not in matched_t:
                t.mark_missed()

        # Resurrection: appearance-only, but still class-aware.
        unmatched_d = [j for j in range(D) if j not in matched_d]
        if unmatched_d and lost_tracks:
            lost_feat = np.stack([t.feat for t in lost_tracks])
            lost_cls = np.array([t.cls_id for t in lost_tracks])
            ud_feat = det_feats[unmatched_d]
            ud_cls = det_cls[unmatched_d]
            sims_l = cosine_sim_matrix(lost_feat, ud_feat)
            cost_l = 1.0 - sims_l
            cost_l[lost_cls[:, None] != ud_cls[None, :]] = 1e6
            row_l, col_l = linear_sum_assignment(cost_l)
            resurrected_d = set()
            for r, c in zip(row_l, col_l):
                if cost_l[r, c] < CFG.resurrect_cost_max:
                    j = unmatched_d[c]
                    lost_tracks[r].resurrect(det_xywh[j], det_feats[j])
                    resurrected_d.add(j)
            unmatched_d = [j for j in unmatched_d if j not in resurrected_d]

        for j in unmatched_d:
            t = Track(
                track_id=self.next_id,
                bbox=det_xywh[j].astype(np.float32).copy(),
                feat=det_feats[j].astype(np.float32).copy(),
                cls_id=int(det_cls[j]),
            )
            t.bbox_history.append((frame_idx, det_xywh[j].astype(np.float32).copy()))
            self.tracks.append(t)
            self.next_id += 1

        for t in lost_tracks:
            t.mark_missed()
        self.tracks = [t for t in self.tracks if t.state != "deleted"]

        return [t for t in self.tracks if t.state == "confirmed"]
