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


def _build_tracker(name: str, embedder, frame_rate: int):
    if name == "botsort":
        from src.tracker_botsort import MOTTracker
        return MOTTracker(embedder, frame_rate=frame_rate)
    if name == "legacy":
        from src.tracker_legacy import MOTTracker
        return MOTTracker(embedder)
    raise ValueError(f"unknown tracker: {name}")


def _build_detector(name: str):
    if name == "yolo":
        from src.detector_yolo import YoloDetector
        return YoloDetector()
    if name == "frcnn":
        from src.detector import FasterRCNNDetector
        return FasterRCNNDetector()
    raise ValueError(f"unknown detector: {name}")


def run_track(seq_dir: Path, save_video: bool, tracker_name: str = "botsort",
              detector_name: str = "frcnn", profile: bool = False):
    """Full tracker run: detector → tracker → write MOT txt + video."""
    import time
    from src.siamfc import SiamEmbedder

    info = load_seqinfo(seq_dir)
    print(f"[track] {info['name']}: {info['seq_length']} frames @ {info['frame_rate']} fps "
          f"(detector={detector_name}, tracker={tracker_name})")

    detector = _build_detector(detector_name)
    embedder = SiamEmbedder()
    tracker = _build_tracker(tracker_name, embedder, frame_rate=int(info["frame_rate"]))

    rows = []
    writer = _open_writer(info, "track") if save_video else None
    timings = {"io": 0.0, "det": 0.0, "track": 0.0, "draw": 0.0, "total": 0.0}
    n_frames = 0
    try:
        t_prev = time.perf_counter()
        for fi, frame in tqdm(frame_iter(seq_dir), total=info["seq_length"]):
            t_io = time.perf_counter()
            dets = detector.detect(frame)
            t_det = time.perf_counter()
            active = tracker.update(frame, dets, fi)
            t_trk = time.perf_counter()
            for t in active:
                if not t.emitted:
                    for f_buf, bb in t.bbox_history:
                        rows.append((f_buf, t.track_id,
                                     float(bb[0]), float(bb[1]),
                                     float(bb[2]), float(bb[3]), 1.0, t.cls_id))
                    t.emitted = True
                    t.bbox_history.clear()
                else:
                    x, y, w, h = t.bbox
                    rows.append((fi, t.track_id, x, y, w, h, 1.0, t.cls_id))
            if writer is not None:
                drawable = [t for t in active if t.age == 0]
                writer.write(viz.draw_tracks(frame, drawable, CFG.draw_trajectory))
            t_end = time.perf_counter()
            timings["io"] += t_io - t_prev
            timings["det"] += t_det - t_io
            timings["track"] += t_trk - t_det
            timings["draw"] += t_end - t_trk
            timings["total"] += t_end - t_prev
            n_frames += 1
            t_prev = time.perf_counter()
    finally:
        if writer is not None:
            writer.close()

    if profile and n_frames > 0:
        print(f"[profile] avg ms/frame over {n_frames} frames "
              f"(total {timings['total']*1000/n_frames:.1f}ms = "
              f"{n_frames/timings['total']:.1f} fps):")
        for k in ("io", "det", "track", "draw"):
            ms = timings[k] * 1000 / n_frames
            pct = 100 * timings[k] / timings["total"]
            print(f"  {k:>6}: {ms:6.1f} ms  ({pct:4.1f}%)")
        enc = getattr(getattr(tracker, "_inner", None), "encoder", None)
        if enc is not None and hasattr(enc, "t_embed"):
            print(f"  [reid] crop:  {enc.t_crop*1000/n_frames:6.1f} ms  "
                  f"embed: {enc.t_embed*1000/n_frames:6.1f} ms")

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
    ap.add_argument("--tracker", default="botsort", choices=["botsort", "legacy"],
                    help="association backend (default: botsort)")
    ap.add_argument("--detector", default="frcnn", choices=["frcnn", "yolo"],
                    help="detector backend (default: frcnn)")
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--profile", action="store_true",
                    help="print per-stage timing breakdown")
    ap.add_argument("--eval", action="store_true",
                    help="run TrackEval after tracking (only valid for train split)")
    args = ap.parse_args()

    seq_dirs = _resolve_seqs(args.seq, args.split)

    for sd in seq_dirs:
        if args.mode == "track":
            run_track(sd, save_video=not args.no_video,
                      tracker_name=args.tracker, detector_name=args.detector,
                      profile=args.profile)
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
