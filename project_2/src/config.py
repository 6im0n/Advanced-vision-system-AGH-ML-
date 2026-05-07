"""Central config: paths, hyperparams, device autodetect."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import torch


def pick_device() -> torch.device:
    """CUDA if available, otherwise CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "evs_mot_public_dataset"
TRAIN_DIR = DATA_DIR / "evs_mot-train"
TEST_DIR = DATA_DIR / "evs_mot-test"
WEIGHTS_DIR = ROOT / "weights"
RESULTS_DIR = ROOT / "results"
TRACKEVAL_DIR = ROOT / "trackeval_workdir"

WEIGHTS_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)


@dataclass
class TrackerCfg:
    # Detector
    det_score_th: float = 0.5
    det_class_id: int = 1            # COCO person
    det_nms_iou: float = 0.5

    # SiamFC
    exemplar_size: int = 127
    instance_size: int = 255
    context_amount: float = 0.5
    siam_weights: str = str(WEIGHTS_DIR / "siamfc_alexnet_e50.pth")

    # Track lifecycle
    n_init: int = 3
    max_age: int = 30

    # Association weights (single-pass Hungarian)
    w_iou: float = 0.4
    w_app: float = 0.6
    iou_gate: float = 0.1
    cost_max: float = 0.9            # cells above this = forbid match

    # Exemplar EMA
    ema_alpha: float = 0.9

    # Visualization
    draw_trajectory: bool = True
    trail_len: int = 30


CFG = TrackerCfg()
DEVICE = pick_device()
