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
        # Slightly more conservative aug for MOT-style data (heavy crowds, small boxes).
        mosaic=0.5,
        mixup=0.0,
        close_mosaic=10,
        hsv_h=0.015, hsv_s=0.4, hsv_v=0.3,
        translate=0.05, scale=0.3, fliplr=0.5,
        single_cls=True,
        verbose=True,
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
