"""Convert MOT GT into YOLO training format.

Layout produced under WEIGHTS_DIR.parent / 'yolo_data':
    images/train/<seq>_<frame>.jpg   (symlink or copy of source frame)
    images/val/<seq>_<frame>.jpg
    labels/train/<seq>_<frame>.txt   YOLO rows: cls cx cy w h (normalized)
    labels/val/<seq>_<frame>.txt
    data.yaml                        ultralytics dataset descriptor

A single class — pedestrian (id 0 in YOLO; original MOT class 1).
Val split: configurable per-sequence stride (default 5 = 20%).
"""
from __future__ import annotations
from pathlib import Path
import shutil
import argparse
import yaml

from .config import TRAIN_DIR, ROOT
from .io_mot import load_seqinfo, parse_mot_csv


YOLO_DATA_DIR = ROOT / "yolo_data"
PEDESTRIAN_CLS_MOT = 1   # MOT-Challenge class id for pedestrian
YOLO_PEDESTRIAN = 0      # YOLO class id (single-class)


def _link_or_copy(src: Path, dst: Path):
    if dst.exists():
        return
    try:
        dst.symlink_to(src)
    except (OSError, NotImplementedError):
        shutil.copy2(src, dst)


def _write_yolo_label(label_path: Path, gt_rows, im_w: int, im_h: int):
    """gt_rows: iterable of (frame, tid, x, y, w, h, conf, cls, vis)."""
    lines = []
    for r in gt_rows:
        cls = int(r[7])
        if cls != PEDESTRIAN_CLS_MOT:
            continue
        x, y, w, h = float(r[2]), float(r[3]), float(r[4]), float(r[5])
        cx = (x + w / 2.0) / im_w
        cy = (y + h / 2.0) / im_h
        nw = w / im_w
        nh = h / im_h
        if nw <= 0 or nh <= 0:
            continue
        cx = max(0.0, min(1.0, cx))
        cy = max(0.0, min(1.0, cy))
        nw = max(0.0, min(1.0, nw))
        nh = max(0.0, min(1.0, nh))
        lines.append(f"{YOLO_PEDESTRIAN} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
    label_path.write_text("\n".join(lines) + ("\n" if lines else ""))


def build(seqs: list[str], val_stride: int = 5, force: bool = False) -> Path:
    """Generate YOLO dataset under YOLO_DATA_DIR."""
    if force and YOLO_DATA_DIR.exists():
        shutil.rmtree(YOLO_DATA_DIR)
    for sub in ("images/train", "images/val", "labels/train", "labels/val"):
        (YOLO_DATA_DIR / sub).mkdir(parents=True, exist_ok=True)

    n_train = n_val = 0
    for seq_name in seqs:
        seq_dir = TRAIN_DIR / seq_name
        info = load_seqinfo(seq_dir)
        gt = parse_mot_csv(seq_dir / "gt" / "gt.txt")
        # Group GT rows by frame for fast lookup.
        rows_by_frame: dict[int, list] = {}
        for r in gt:
            f = int(r[0])
            rows_by_frame.setdefault(f, []).append(r)

        im_w, im_h = info["im_width"], info["im_height"]
        ext = info.get("im_ext", ".jpg")

        for frame_idx in range(1, info["seq_length"] + 1):
            split = "val" if (frame_idx % val_stride == 0) else "train"
            stem = f"{seq_name}_{frame_idx:06d}"
            src_img = seq_dir / "img1" / f"{frame_idx:06d}{ext}"
            if not src_img.exists():
                continue
            dst_img = YOLO_DATA_DIR / "images" / split / f"{stem}{ext}"
            dst_lbl = YOLO_DATA_DIR / "labels" / split / f"{stem}.txt"
            _link_or_copy(src_img.resolve(), dst_img)
            _write_yolo_label(dst_lbl, rows_by_frame.get(frame_idx, []), im_w, im_h)
            if split == "train":
                n_train += 1
            else:
                n_val += 1

    data_yaml = YOLO_DATA_DIR / "data.yaml"
    data_yaml.write_text(yaml.safe_dump({
        "path": str(YOLO_DATA_DIR.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": {YOLO_PEDESTRIAN: "pedestrian"},
        "nc": 1,
    }, sort_keys=False))
    print(f"[yolo-dataset] train={n_train}  val={n_val}  → {YOLO_DATA_DIR}")
    return data_yaml


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seqs", nargs="+",
                    default=["MOT_02", "MOT_03", "MOT_04", "MOT_05"])
    ap.add_argument("--val_stride", type=int, default=5,
                    help="every Nth frame goes to val (default 5 → 20%% val)")
    ap.add_argument("--force", action="store_true",
                    help="wipe existing yolo_data/ before rebuilding")
    args = ap.parse_args()
    build(args.seqs, val_stride=args.val_stride, force=args.force)


if __name__ == "__main__":
    main()
