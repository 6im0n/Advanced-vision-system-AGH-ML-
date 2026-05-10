"""Finetune YOLOv8 on MOT pedestrian GT.

Usage
-----
    python -m src.yolo_dataset --force            # one-time dataset build
    python -m src.yolo_train --model yolov8m.pt --epochs 80

Output: weights at weights/yolov8_mot.pt (best.pt of the run).
"""
from __future__ import annotations
import argparse
import shutil
from pathlib import Path

from .config import WEIGHTS_DIR
from .yolo_dataset import YOLO_DATA_DIR, build as build_dataset


YOLO_WEIGHTS = WEIGHTS_DIR / "yolov8_mot.pt"


def train(model: str, epochs: int, imgsz: int, batch: int, device: str,
          patience: int, project: str, name: str) -> Path:
    from ultralytics import YOLO

    data_yaml = YOLO_DATA_DIR / "data.yaml"
    if not data_yaml.exists():
        print("[yolo-train] no dataset, building defaults…")
        build_dataset(["MOT_02", "MOT_03", "MOT_04", "MOT_05"], val_stride=5, force=False)

    yolo = YOLO(model)
    results = yolo.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        patience=patience,
        project=project,
        name=name,
        cos_lr=True,
        amp=True,
        single_cls=True,
        verbose=True,

        # Color / light variation — test sequences include low-light, different
        # camera response, different scene. Push HSV hard.
        hsv_h=0.1,                  # hue jitter (small — keep skin/clothing tone)
        hsv_s=0.9,                  # saturation: full range, simulates desaturated/dim
        hsv_v=0.9,                  # value (brightness): max → covers dark + bright

        # Geometric transforms — different camera angles in test sequences.
        degrees=15.0,               # rotation ±15°
        translate=0.15,             # shift ±15%
        scale=0.65,                  # zoom 0.4×–1.6×
        shear=4.0,                  # shear ±4°
        perspective=0.0008,         # mild perspective warp
        fliplr=0.5,                 # horizontal flip
        flipud=0.0,               # never vertical (people don't fly upside-down)

        # Mix-style / occlusion aug — boost generalization on small dataset.
        mosaic=1.0,                 # always-on mosaic = 4× context per training sample
        mixup=0.2,                  # blend two images → robustness
        copy_paste=0.4,             # paste GT instances elsewhere → crowd density
        close_mosaic=15,            # disable mosaic last 15 epochs for clean finetune
        erasing=0.4,                # random erasing → simulate occlusion

        # Inference-time aug during val (multi-scale recall).
        augment=False,              # leave train-time off; turn on at predict
        bgr=0.0,
    )
    save_dir = Path(results.save_dir) if hasattr(results, "save_dir") else Path(yolo.trainer.save_dir)
    best = save_dir / "weights" / "best.pt"
    if best.exists():
        shutil.copy2(best, YOLO_WEIGHTS)
        print(f"[yolo-train] saved best -> {YOLO_WEIGHTS}")
    else:
        print(f"[yolo-train] WARNING best.pt not found in {save_dir}")
    return YOLO_WEIGHTS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolov8m.pt",
                    help="ultralytics base weights (yolov8n/s/m/l/x .pt)")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--imgsz", type=int, default=1280,
                    help="train image size (1280 retains small pedestrians)")
    ap.add_argument("--batch", type=int, default=8,
                    help="lower if 10GB GPU OOM at imgsz=1280")
    ap.add_argument("--device", default="0")
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--project", default="runs/yolo_mot")
    ap.add_argument("--name", default="finetune")
    args = ap.parse_args()
    train(args.model, args.epochs, args.imgsz, args.batch,
          args.device, args.patience, args.project, args.name)


if __name__ == "__main__":
    main()
