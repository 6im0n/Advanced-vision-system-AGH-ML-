"""MOT dataset I/O: load sequence, parse gt/det, write predictions."""
from __future__ import annotations
from pathlib import Path
import configparser
import numpy as np
import cv2


def load_seqinfo(seq_dir: Path) -> dict:
    cp = configparser.ConfigParser()
    cp.read(seq_dir / "seqinfo.ini")
    s = cp["Sequence"]
    return {
        "name": s["name"],
        "frame_rate": int(s["frameRate"]),
        "seq_length": int(s["seqLength"]),
        "im_width": int(s["imWidth"]),
        "im_height": int(s["imHeight"]),
        "im_dir": s["imDir"],
        "im_ext": s["imExt"],
    }


def frame_iter(seq_dir: Path):
    """Yield (frame_idx, BGR ndarray) for each image in the sequence."""
    info = load_seqinfo(seq_dir)
    img_dir = seq_dir / info["im_dir"]
    for i in range(1, info["seq_length"] + 1):
        path = img_dir / f"{i:06d}{info['im_ext']}"
        img = cv2.imread(str(path))
        if img is None:
            continue
        yield i, img


def parse_mot_csv(path: Path) -> np.ndarray:
    """Parse gt.txt or det.txt → ndarray[N, >=7]: frame,id,x,y,w,h,conf,cls,vis."""
    if not path.exists():
        return np.zeros((0, 9))
    return np.loadtxt(path, delimiter=",", ndmin=2)


def filter_frame(arr: np.ndarray, frame_idx: int) -> np.ndarray:
    """Subset rows for a given frame index."""
    if arr.size == 0:
        return arr
    return arr[arr[:, 0].astype(int) == frame_idx]


def write_mot_results(path: Path, rows):
    """rows: iterable of (frame, id, x, y, w, h, conf). Writes MOT-Challenge format."""
    with open(path, "w") as f:
        for frame, tid, x, y, w, h, conf in rows:
            f.write(
                f"{int(frame)},{int(tid)},"
                f"{x:.2f},{y:.2f},{w:.2f},{h:.2f},"
                f"{conf:.2f},-1,-1,-1\n"
            )
