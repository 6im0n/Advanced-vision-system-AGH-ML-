"""V1 MOT pipeline — Faster R-CNN detector + SiamFC appearance + Hungarian assoc.

Examples
--------
# Run on one training sequence and save video:
    python project.py --seq MOT_02

# Run all training sequences then evaluate with TrackEval:
    python project.py --seq all --eval

# Visualize ground truth only (sanity check, no tracker):
    python project.py --seq MOT_02 --mode gt

# Visualize raw detections (no tracker, no IDs):
    python project.py --seq MOT_02 --mode det
"""
from __future__ import annotations
import argparse
from pathlib import Path
from tqdm import tqdm

from src.config import CFG, TRAIN_DIR, TEST_DIR, RESULTS_DIR
from src.io_mot import (
    load_seqinfo, frame_iter, write_mot_results, parse_mot_csv, filter_frame,
)
from src import visualize as viz


def _open_writer(info, suffix: str):
    out = RESULTS_DIR / f"{info['name']}_{suffix}.mp4"
    return viz.VideoWriter(out, info["frame_rate"], (info["im_width"], info["im_height"]))


def run_track(seq_dir: Path, save_video: bool):
    """Full tracker run: detector → tracker → write MOT txt + video."""
    from src.detector import FasterRCNNDetector
    from src.siamfc import SiamEmbedder
    from src.tracker import MOTTracker

    info = load_seqinfo(seq_dir)
    print(f"[track] {info['name']}: {info['seq_length']} frames @ {info['frame_rate']} fps")

    detector = FasterRCNNDetector()
    embedder = SiamEmbedder()
    tracker = MOTTracker(embedder)

    rows = []
    writer = _open_writer(info, "track") if save_video else None
    try:
        for fi, frame in tqdm(frame_iter(seq_dir), total=info["seq_length"]):
            dets = detector.detect(frame)
            active = tracker.update(frame, dets)
            for t in active:
                x, y, w, h = t.bbox
                rows.append((fi, t.track_id, x, y, w, h, 1.0))
            if writer is not None:
                writer.write(viz.draw_tracks(frame, active, CFG.draw_trajectory))
    finally:
        if writer is not None:
            writer.close()

    out_txt = RESULTS_DIR / f"{info['name']}.txt"
    write_mot_results(out_txt, rows)
    print(f"[track] wrote {len(rows)} rows -> {out_txt}")
    return out_txt


def run_visualize_gt(seq_dir: Path):
    """Render GT-only video for dataset interpretation."""
    info = load_seqinfo(seq_dir)
    gt = parse_mot_csv(seq_dir / "gt" / "gt.txt")
    writer = _open_writer(info, "gt")
    try:
        for fi, frame in tqdm(frame_iter(seq_dir), total=info["seq_length"]):
            rows = filter_frame(gt, fi)
            tracks_like = []
            for r in rows:
                tid = int(r[1])
                bbox = r[2:6]
                tracks_like.append(_FakeTrack(tid, bbox))
            writer.write(viz.draw_tracks(frame, tracks_like, draw_trail=False))
    finally:
        writer.close()
    print(f"[gt] wrote {RESULTS_DIR / f'{info['name']}_gt.mp4'}")


def run_visualize_det(seq_dir: Path):
    """Render raw Faster R-CNN detections (no tracker)."""
    from src.detector import FasterRCNNDetector

    info = load_seqinfo(seq_dir)
    detector = FasterRCNNDetector()
    writer = _open_writer(info, "det")
    try:
        for fi, frame in tqdm(frame_iter(seq_dir), total=info["seq_length"]):
            dets = detector.detect(frame)
            writer.write(viz.draw_boxes(frame, dets[:, :4], (0, 255, 0), "det"))
    finally:
        writer.close()
    print(f"[det] wrote {RESULTS_DIR / f'{info['name']}_det.mp4'}")


class _FakeTrack:
    """Tiny shim so visualize.draw_tracks can render GT rows."""
    def __init__(self, tid, bbox):
        self.track_id = tid
        self.bbox = bbox
        self.history = []


def _resolve_seqs(spec: str, split: str) -> list[Path]:
    """Resolve --seq + --split into a list of sequence directories."""
    if split == "train":
        root = TRAIN_DIR
    elif split == "test":
        root = TEST_DIR
    elif split == "all":
        return sorted(TRAIN_DIR.glob("MOT_*")) + sorted(TEST_DIR.glob("MOT_*"))
    else:
        raise ValueError(f"unknown split={split}")

    if spec == "all":
        return sorted(root.glob("MOT_*"))
    # explicit name(s) — auto-detect which split contains it
    names = spec.split(",")
    out = []
    for n in names:
        for r in (TRAIN_DIR, TEST_DIR):
            cand = r / n
            if cand.exists():
                out.append(cand)
                break
        else:
            raise FileNotFoundError(n)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", default="MOT_02",
                    help="sequence name(s), comma-separated, or 'all'")
    ap.add_argument("--split", default="train", choices=["train", "test", "all"],
                    help="resolves --seq=all to this split (ignored for explicit names)")
    ap.add_argument("--mode", default="track", choices=["track", "gt", "det"])
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--eval", action="store_true",
                    help="run TrackEval after tracking (only valid for train split)")
    args = ap.parse_args()

    seq_dirs = _resolve_seqs(args.seq, args.split)

    for sd in seq_dirs:
        if args.mode == "track":
            run_track(sd, save_video=not args.no_video)
        elif args.mode == "gt":
            if not (sd / "gt" / "gt.txt").exists():
                print(f"[skip] {sd.name}: no GT (test split)")
                continue
            run_visualize_gt(sd)
        elif args.mode == "det":
            run_visualize_det(sd)

    if args.eval and args.mode == "track":
        from src import eval_mota
        seqs_with_gt = [sd for sd in seq_dirs if (sd / "gt" / "gt.txt").exists()]
        if not seqs_with_gt:
            print("[eval] no sequences with GT — skipping (test set has no GT)")
            return
        gt_root, tr_root = eval_mota.prepare_dirs(seqs_with_gt)
        eval_mota.copy_results(tr_root)
        eval_mota.run_eval()


if __name__ == "__main__":
    main()
