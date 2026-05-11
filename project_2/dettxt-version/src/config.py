"""Central config: paths, hyperparams, device autodetect."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import torch


def pick_device() -> torch.device:
    """CUDA if available, otherwise CPU."""
    if torch.cuda.is_available():
        # Speedup: pick fastest conv algos for fixed input shapes
        torch.backends.cudnn.benchmark = True
        return torch.device("cuda")
    return torch.device("cpu")


def use_amp(device: torch.device) -> bool:
    return device.type == "cuda"


ROOT = Path(__file__).resolve().parent.parent


def _resolve_dataset_dir() -> Path:
    """Locate evs_mot_public_dataset. Check parent (shared across versions,
    Windows-friendly — no symlink needed) then local fallback."""
    for cand in (ROOT.parent / "evs_mot_public_dataset",
                 ROOT / "evs_mot_public_dataset"):
        if cand.is_dir():
            return cand
    return ROOT.parent / "evs_mot_public_dataset"


DATA_DIR = _resolve_dataset_dir()
TRAIN_DIR = DATA_DIR / "evs_mot-train"
TEST_DIR = DATA_DIR / "evs_mot-test"
WEIGHTS_DIR = ROOT / "weights"
RESULTS_DIR = ROOT / "results"
TRACKEVAL_DIR = ROOT / "trackeval_workdir"
REID_WEIGHTS = WEIGHTS_DIR / "reid_convnext_small.pth"

WEIGHTS_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)


# MOT-Challenge foreground class ids and human-readable names (used by
# reid_dataset to filter GT and by visualize to label boxes).
FG_MOT_CLASSES = (1, 2, 3, 4, 5, 6)
MOT_CLASS_NAMES = {1: "ped", 2: "ped-veh", 3: "car", 4: "bike", 5: "moto", 6: "veh"}


@dataclass
class TrackerCfg:
    # Detector (dettxt) — score floor applied when reading det.txt
    det_score_th: float = 0.05

    # Appearance encoder (ReID inference)
    crop_h: int = 256                # person ReID standard
    crop_w: int = 128
    context_amount: float = 0.1      # less padding for tall person crops

    # Exemplar EMA (used by tracker for embedding refresh)
    ema_alpha: float = 0.9

    # Visualization
    draw_trajectory: bool = True
    trail_len: int = 60


@dataclass
class BotSortCfg:
    # Detection score gates (BoT-SORT two-stage)
    track_high_thresh: float = 0.3
    track_low_thresh: float = 0.05
    new_track_thresh: float = 0.35

    # Lifecycle
    track_buffer: int = 90           # frames a lost track is kept

    # Association
    match_thresh: float = 0.8        # 1st-stage IoU/embed cost gate
    proximity_thresh: float = 0.5    # IoU gate above which appearance is ignored
    appearance_thresh: float = 0.25  # cosine-distance accept gate for ReID match

    # ReID + CMC
    with_reid: bool = True
    cmc_method: str = "sparseOptFlow"
    gmc_downscale: int = 2

    # Misc BoT-SORT flags
    mot20: bool = False              # disables fuse_score, used for crowded scenes
    fuse_score: bool = True          # fuse detection score into IoU cost
    name: str = "evs_mot"
    ablation: bool = False
    device: str = "cuda"


# Per-sequence overrides for detector + tracker hyperparams.
# Applied in project.run_track based on seqinfo name. Keys:
#   "detector": kwargs forwarded to DetTxtDetector (e.g. score_th)
#   "botsort":  attribute overrides on BotSortCfg
PER_SEQ_OVERRIDES: dict[str, dict] = {
    # MOT_07 det.txt: 10% of dets below score 0.32, 5% below 0.14.
    # Default new_track_thresh=0.35 / track_high_thresh=0.3 reject these
    # → can't seed new tracks for small/distant pedestrians → FN.
    "MOT_07": {
        "botsort": {
            "track_high_thresh": 0.15,
            "new_track_thresh": 0.20,
        },
    },
}


CFG = TrackerCfg()
BOTSORT_CFG = BotSortCfg()
DEVICE = pick_device()
