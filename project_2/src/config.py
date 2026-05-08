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
DATA_DIR = ROOT / "evs_mot_public_dataset"
TRAIN_DIR = DATA_DIR / "evs_mot-train"
TEST_DIR = DATA_DIR / "evs_mot-test"
WEIGHTS_DIR = ROOT / "weights"
RESULTS_DIR = ROOT / "results"
TRACKEVAL_DIR = ROOT / "trackeval_workdir"
REID_WEIGHTS = WEIGHTS_DIR / "reid_convnext_small.pth"

WEIGHTS_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)


# COCO class id (torchvision FRCNN) → MOT-Challenge class id mapping.
# Only entries listed here are kept by the detector.
COCO_TO_MOT_CLASS = {
    1: 1,   # person       → pedestrian
    2: 4,   # bicycle      → bicycle
    3: 3,   # car          → car
    4: 5,   # motorcycle   → motorbike
    6: 3,   # bus          → car (treated as wheeled vehicle)
    8: 3,   # truck        → car
}
FG_MOT_CLASSES = (1, 2, 3, 4, 5, 6)
MOT_CLASS_NAMES = {1: "ped", 2: "ped-veh", 3: "car", 4: "bike", 5: "moto", 6: "veh"}


@dataclass
class TrackerCfg:
    # Detector
    det_score_th: float = 0.5
    det_nms_iou: float = 0.5

    # Appearance encoder
    crop_h: int = 256                # person ReID standard
    crop_w: int = 128
    context_amount: float = 0.1      # less padding for tall person crops
    siam_weights: str = str(WEIGHTS_DIR / "siamfc_alexnet_e50.pth")

    # Track lifecycle
    n_init: int = 2
    max_age: int = 60                # frames a confirmed track stays "active" while unmatched
    lost_max_age: int = 120          # frames a "lost" track waits for resurrection

    # Association weights (single-pass Hungarian)
    w_iou: float = 0.3
    w_app: float = 0.7
    iou_gate: float = 0.05
    cost_max: float = 0.7            # main-association accept gate
    resurrect_cost_max: float = 0.4  # appearance-only cosine-distance gate for resurrection

    # Exemplar EMA
    ema_alpha: float = 0.9

    # Visualization
    draw_trajectory: bool = True
    trail_len: int = 30


CFG = TrackerCfg()
DEVICE = pick_device()
