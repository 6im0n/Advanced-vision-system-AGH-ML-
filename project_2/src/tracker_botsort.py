"""BoT-SORT adapter exposing the same MOTTracker interface as tracker_legacy.

Vendored upstream lives in third_party/BoT-SORT. We bypass its FastReID
dependency by injecting our own ConvNeXt-Small SiamEmbedder as the encoder.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import sys
import types
import numpy as np

from .config import CFG, BOTSORT_CFG
from .siamfc import SiamEmbedder, crop_person


_BOTSORT_DIR = Path(__file__).resolve().parent.parent / "third_party" / "BoT-SORT"


def _patch_numpy_aliases() -> None:
    """Restore np.float / np.float_ / np.int_ aliases that BoT-SORT vendored
    code still references but were removed in numpy>=1.24."""
    if not hasattr(np, "float"):
        np.float = float           # type: ignore[attr-defined]
    if not hasattr(np, "float_"):
        np.float_ = np.float64     # type: ignore[attr-defined]
    if not hasattr(np, "int_"):
        np.int_ = np.int64         # type: ignore[attr-defined]
    if not hasattr(np, "bool_"):
        np.bool_ = np.bool         # type: ignore[attr-defined]


def _stub_fast_reid() -> None:
    """Pre-register an empty fast_reid package so bot_sort.py's
    `from fast_reid.fast_reid_interfece import FastReIDInterface` succeeds
    without pulling in fast_reid dependencies."""
    if "fast_reid.fast_reid_interfece" in sys.modules:
        return
    pkg = types.ModuleType("fast_reid")
    sub = types.ModuleType("fast_reid.fast_reid_interfece")

    class _StubFastReID:
        def __init__(self, *a, **k):
            raise RuntimeError("FastReID stub: should not be instantiated")

        def inference(self, *a, **k):
            raise RuntimeError("FastReID stub: should not be called")

    sub.FastReIDInterface = _StubFastReID
    sys.modules["fast_reid"] = pkg
    sys.modules["fast_reid.fast_reid_interfece"] = sub


def _stub_cython_bbox() -> None:
    """Provide a numpy-only `cython_bbox.bbox_overlaps` so BoT-SORT's
    matching.py works without compiling the C extension (which fails on
    Windows toolchains missing the SDK headers)."""
    if "cython_bbox" in sys.modules:
        return

    def bbox_overlaps(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """tlbr inputs (N×4 and M×4) → IoU matrix [N, M], float64."""
        a = np.ascontiguousarray(a, dtype=np.float64)
        b = np.ascontiguousarray(b, dtype=np.float64)
        if a.size == 0 or b.size == 0:
            return np.zeros((a.shape[0], b.shape[0]), dtype=np.float64)
        tl = np.maximum(a[:, None, :2], b[None, :, :2])
        br = np.minimum(a[:, None, 2:4], b[None, :, 2:4])
        wh = np.clip(br - tl, 0.0, None)
        inter = wh[..., 0] * wh[..., 1]
        area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
        area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
        union = area_a[:, None] + area_b[None, :] - inter + 1e-12
        return inter / union

    mod = types.ModuleType("cython_bbox")
    mod.bbox_overlaps = bbox_overlaps
    sys.modules["cython_bbox"] = mod


def _bootstrap_botsort() -> None:
    if not (_BOTSORT_DIR / "tracker" / "bot_sort.py").exists():
        raise ModuleNotFoundError(
            f"BoT-SORT not vendored at {_BOTSORT_DIR}. Clone it:\n"
            f"  git clone --depth 1 https://github.com/NirAharon/BoT-SORT.git "
            f"{_BOTSORT_DIR}"
        )
    if str(_BOTSORT_DIR) not in sys.path:
        sys.path.insert(0, str(_BOTSORT_DIR))
    _patch_numpy_aliases()
    _stub_fast_reid()
    _stub_cython_bbox()


_bootstrap_botsort()

# Imports below depend on the bootstrap above — keep order.
from tracker.bot_sort import BoTSORT  # noqa: E402
from tracker.basetrack import BaseTrack  # noqa: E402
from tracker.kalman_filter import KalmanFilter  # noqa: E402
from tracker.gmc import GMC  # noqa: E402


class _SiamEncoder:
    """Adapter making SiamEmbedder match the FastReIDInterface.inference signature."""

    def __init__(self, embedder: SiamEmbedder):
        self.embedder = embedder
        self.t_crop = 0.0
        self.t_embed = 0.0

    def inference(self, img: np.ndarray, dets: np.ndarray) -> np.ndarray:
        import time
        if len(dets) == 0:
            return np.zeros((0, SiamEmbedder.EMBED_DIM), dtype=np.float32)
        t0 = time.perf_counter()
        crops = []
        for box in dets[:, :4]:
            x1, y1, x2, y2 = box
            xywh = (float(x1), float(y1), float(x2 - x1), float(y2 - y1))
            crops.append(crop_person(img, xywh))
        t1 = time.perf_counter()
        out = self.embedder.embed(crops)
        t2 = time.perf_counter()
        self.t_crop += t1 - t0
        self.t_embed += t2 - t1
        return out


class _BoTSORTNoFastReID(BoTSORT):
    """BoTSORT subclass that accepts an externally-supplied encoder rather
    than instantiating FastReIDInterface. All other logic inherited."""

    def __init__(self, args, frame_rate: int, encoder):
        BaseTrack.clear_count()
        self.tracked_stracks = []
        self.lost_stracks = []
        self.removed_stracks = []
        self.frame_id = 0
        self.args = args
        self.track_high_thresh = args.track_high_thresh
        self.track_low_thresh = args.track_low_thresh
        self.new_track_thresh = args.new_track_thresh
        self.buffer_size = int(frame_rate / 30.0 * args.track_buffer)
        self.max_time_lost = self.buffer_size
        self.kalman_filter = KalmanFilter()
        self.proximity_thresh = args.proximity_thresh
        self.appearance_thresh = args.appearance_thresh
        self.encoder = encoder
        self.gmc = GMC(
            method=args.cmc_method,
            downscale=getattr(args, "gmc_downscale", 2),
            verbose=[args.name, args.ablation],
        )


@dataclass
class _ArgsBundle:
    """Mirror of BoT-SORT's argparse Namespace, fed to the tracker."""
    track_high_thresh: float
    track_low_thresh: float
    new_track_thresh: float
    track_buffer: int
    match_thresh: float
    proximity_thresh: float
    appearance_thresh: float
    with_reid: bool
    cmc_method: str
    gmc_downscale: int
    mot20: bool
    fuse_score: bool
    name: str
    ablation: bool
    device: str
    fast_reid_config: str = ""
    fast_reid_weights: str = ""


@dataclass
class Track:
    """Track shape compatible with project.py's run_track loop and visualize.draw_tracks."""
    track_id: int
    bbox: np.ndarray              # [x, y, w, h]
    cls_id: int = 1
    age: int = 0
    state: str = "confirmed"
    history: list = field(default_factory=list)
    bbox_history: list = field(default_factory=list)
    emitted: bool = False


class MOTTracker:
    """Drop-in replacement for tracker_legacy.MOTTracker backed by BoT-SORT."""

    PEDESTRIAN_CLS = 1

    def __init__(self, embedder: SiamEmbedder, frame_rate: int = 30,
                 overrides: dict | None = None):
        self.embedder = embedder
        cfg = BOTSORT_CFG
        def _g(k):
            if overrides and k in overrides:
                return overrides[k]
            return getattr(cfg, k)
        args = _ArgsBundle(
            track_high_thresh=_g("track_high_thresh"),
            track_low_thresh=_g("track_low_thresh"),
            new_track_thresh=_g("new_track_thresh"),
            track_buffer=_g("track_buffer"),
            match_thresh=_g("match_thresh"),
            proximity_thresh=_g("proximity_thresh"),
            appearance_thresh=_g("appearance_thresh"),
            with_reid=_g("with_reid"),
            cmc_method=_g("cmc_method"),
            gmc_downscale=_g("gmc_downscale"),
            mot20=_g("mot20"),
            fuse_score=_g("fuse_score"),
            name=_g("name"),
            ablation=_g("ablation"),
            device=_g("device"),
        )
        encoder = _SiamEncoder(embedder) if cfg.with_reid else None
        self._inner = _BoTSORTNoFastReID(args, frame_rate=frame_rate, encoder=encoder)
        self._tracks_by_id: dict[int, Track] = {}

    @staticmethod
    def _to_xyxy_with_cls(dets: np.ndarray) -> np.ndarray:
        """Detector returns [x, y, w, h, score, cls]. Filter pedestrian only,
        convert to BoT-SORT format [x1, y1, x2, y2, score, cls]."""
        if dets.size == 0:
            return np.zeros((0, 6), dtype=np.float32)
        mask = dets[:, 5].astype(int) == MOTTracker.PEDESTRIAN_CLS
        d = dets[mask]
        if d.size == 0:
            return np.zeros((0, 6), dtype=np.float32)
        out = np.zeros((d.shape[0], 6), dtype=np.float32)
        out[:, 0] = d[:, 0]
        out[:, 1] = d[:, 1]
        out[:, 2] = d[:, 0] + d[:, 2]
        out[:, 3] = d[:, 1] + d[:, 3]
        out[:, 4] = d[:, 4]
        out[:, 5] = d[:, 5]
        return out

    def update(self, frame_bgr: np.ndarray, dets: np.ndarray, frame_idx: int = 0) -> list[Track]:
        det_xyxy = self._to_xyxy_with_cls(dets)
        stracks = self._inner.update(det_xyxy, frame_bgr)
        self.last_n_dets = len(det_xyxy)

        out: list[Track] = []
        live_ids = set()
        for st in stracks:
            tid = int(st.track_id)
            live_ids.add(tid)
            tlwh = st.tlwh.astype(np.float32).copy()
            cx = float(tlwh[0] + tlwh[2] / 2.0)
            cy = float(tlwh[1] + tlwh[3] / 2.0)

            tr = self._tracks_by_id.get(tid)
            if tr is None:
                tr = Track(track_id=tid, bbox=tlwh, cls_id=self.PEDESTRIAN_CLS)
                tr.bbox_history.append((frame_idx, tlwh.copy()))
                self._tracks_by_id[tid] = tr
            else:
                tr.bbox = tlwh
                if not tr.emitted:
                    tr.bbox_history.append((frame_idx, tlwh.copy()))
            tr.age = 0
            tr.history.append((cx, cy))
            if len(tr.history) > CFG.trail_len:
                tr.history.pop(0)
            out.append(tr)

        # Drop dead tracks from cache to bound memory.
        for tid in list(self._tracks_by_id.keys()):
            if tid not in live_ids:
                del self._tracks_by_id[tid]
        return out
