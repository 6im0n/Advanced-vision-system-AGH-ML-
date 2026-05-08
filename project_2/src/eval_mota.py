"""TrackEval CLEAR/Identity wrapper.

Builds the directory layout TrackEval expects, then runs evaluation via the
trackeval Python API (no CLI dependence).
"""
from __future__ import annotations
from pathlib import Path
import shutil
from .config import TRACKEVAL_DIR, RESULTS_DIR

BENCHMARK = "EVS_MOT"
SPLIT = "train"


def prepare_dirs(seq_dirs: list[Path], tracker_name: str = "SiamMOT"):
    """Mirror GT into TrackEval's expected layout. Return (gt_root, tr_root)."""
    gt_root = TRACKEVAL_DIR / "gt" / "mot_challenge" / f"{BENCHMARK}-{SPLIT}"
    tr_root = (TRACKEVAL_DIR / "trackers" / "mot_challenge"
               / f"{BENCHMARK}-{SPLIT}" / tracker_name / "data")
    seqmap_dir = TRACKEVAL_DIR / "gt" / "mot_challenge" / "seqmaps"
    gt_root.mkdir(parents=True, exist_ok=True)
    tr_root.mkdir(parents=True, exist_ok=True)
    seqmap_dir.mkdir(parents=True, exist_ok=True)

    seqmap = seqmap_dir / f"{BENCHMARK}-{SPLIT}.txt"
    with open(seqmap, "w") as f:
        f.write("name\n")
        for sd in seq_dirs:
            f.write(f"{sd.name}\n")
            dst = gt_root / sd.name
            (dst / "gt").mkdir(parents=True, exist_ok=True)
            shutil.copy(sd / "gt" / "gt.txt", dst / "gt" / "gt.txt")
            shutil.copy(sd / "seqinfo.ini", dst / "seqinfo.ini")
    return gt_root, tr_root


def copy_results(tr_root: Path):
    for f in RESULTS_DIR.glob("*.txt"):
        shutil.copy(f, tr_root / f.name)


def run_eval(tracker_name: str = "SiamMOT"):
    """Run TrackEval CLEAR + Identity metrics. Prints summary."""
    import trackeval

    eval_config = trackeval.Evaluator.get_default_eval_config()
    eval_config["DISPLAY_LESS_PROGRESS"] = True
    eval_config["PRINT_RESULTS"] = True
    eval_config["PRINT_CONFIG"] = False

    dataset_config = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
    dataset_config["BENCHMARK"] = BENCHMARK
    dataset_config["SPLIT_TO_EVAL"] = SPLIT
    dataset_config["GT_FOLDER"] = str(TRACKEVAL_DIR / "gt" / "mot_challenge")
    dataset_config["TRACKERS_FOLDER"] = str(TRACKEVAL_DIR / "trackers" / "mot_challenge")
    dataset_config["TRACKERS_TO_EVAL"] = [tracker_name]
    # MOT-Challenge style preprocessing: filters distractor classes (7-12) and
    # restricts evaluation to CLASSES_TO_EVAL (default 'pedestrian'). Requires
    # the class column in our predictions to be the MOT-Challenge class id.
    dataset_config["DO_PREPROC"] = True

    metrics_config = {"METRICS": ["CLEAR", "Identity"], "THRESHOLD": 0.5}

    evaluator = trackeval.Evaluator(eval_config)
    dataset_list = [trackeval.datasets.MotChallenge2DBox(dataset_config)]
    metrics_list = [
        trackeval.metrics.CLEAR(metrics_config),
        trackeval.metrics.Identity(metrics_config),
    ]
    return evaluator.evaluate(dataset_list, metrics_list)
