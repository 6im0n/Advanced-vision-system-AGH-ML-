# EVS-MOT pedestrian tracker

Multi-object tracking on the EVS-MOT dataset. Detector + ReID encoder + BoT-SORT
association. This README describes the algorithms used for detection and for
training the two learned components.

## 1. Detection

### Model

YOLOv8 (Ultralytics), single-class ("pedestrian"). Backbone is the standard
YOLOv8m: CSPDarknet trunk, PAN-FPN neck, anchor-free decoupled head.

Weights live at `weights/yolov8_mot.pt` (finetuned on the EVS-MOT train split).
If the file is missing, the wrapper falls back to the COCO-pretrained
`yolov8m.pt` and keeps only the COCO `person` class (id 0).

Code: `src/detector_yolo.py`.

### Inference pipeline

For each frame:

1. **Optional CLAHE preprocessing.** Contrast-limited adaptive histogram
   equalization on the L channel of LAB. Three modes:
   - `off`: never apply.
   - `on`: always apply (default).
   - `auto`: apply only when frame mean intensity < 70 (low-light heuristic).

   CLAHE was added because part of the test split is dim and the model trained
   on the brighter train split was missing low-contrast pedestrians.

2. **YOLO forward pass.** `model.predict` with:
   - `imgsz=1920` (default) — resize so the long side is 1920 px, letterbox
     pad. MOT_07 uses `imgsz=2560` (per-sequence override) because median
     pedestrian height in that sequence is ~78 px and the smaller scale was
     dropping them.
   - `conf=0.15` global score threshold (0.05 for MOT_07).
   - `iou=0.4` NMS IoU.
   - `half=True` (fp16 on CUDA).
   - `augment=True` (Ultralytics test-time augmentation, multi-scale + flip
     ensemble — boosts recall at the cost of latency).

3. **Class filter.** Finetuned weights are single-class: keep all outputs.
   Fallback (COCO) weights: keep only class 0 (`person`).

4. **Format.** Output is `[N, 6]`: `x, y, w, h, score, mot_cls=1`. Same shape
   as the older Faster R-CNN detector, so the tracker is detector-agnostic.

### Per-sequence overrides

`src/config.py` exposes `PER_SEQ_OVERRIDES`. The tracking pipeline reads it in
`run_track` and applies overrides for both the detector kwargs and the
BoT-SORT thresholds. Currently only MOT_07 is overridden:

```python
"MOT_07": {
    "detector": {"imgsz": 2560, "score_th": 0.05},
    "botsort":  {"track_low_thresh": 0.05, "new_track_thresh": 0.5},
}
```

Reason: small-pedestrian sequence, default thresholds were dropping ~75% of
ground-truth detections.

---

## 2. Training — YOLO detector

Script: `src/yolo_train.py`. Wraps `ultralytics.YOLO.train`.

### Data

Built by `src/yolo_dataset.py` from MOT GT files (`MOT_02..MOT_05`).
GT bboxes are converted to YOLO format (cx, cy, w, h normalized) and split
into train/val by frame stride (every 5th frame to val). All classes collapsed
to a single `person` class (`single_cls=True`).

### Algorithm

Standard supervised finetune. Loss is the YOLOv8 default:

- **Box regression**: CIoU loss + DFL (Distribution Focal Loss) on the
  anchor-free regression head.
- **Classification**: BCE on the (single) class logit.
- **Objectness**: handled implicitly by anchor-free design — no separate
  objectness branch in v8.

Optimizer: SGD (Ultralytics default), cosine LR schedule (`cos_lr=True`),
AMP enabled (`amp=True`).

### Augmentation strategy

Hard augmentation because the train split is small (~140k bboxes across 4
sequences) and the test split has lighting / scale shifts.

| Aug | Value | Why |
|---|---|---|
| `hsv_h` | 0.1 | Small hue jitter — keep skin/clothing tone realistic. |
| `hsv_s` | 0.9 | Heavy saturation jitter — covers desaturated/IR-ish frames. |
| `hsv_v` | 0.9 | Heavy value jitter — covers dark + bright sequences. |
| `degrees` | 15° | Camera roll variation. |
| `translate` | 0.15 | Camera shift. |
| `scale` | 0.65 | Zoom 0.4×–1.6× — train sees multi-scale pedestrians. |
| `shear` | 4° | Mild shear. |
| `perspective` | 0.0008 | Mild perspective warp. |
| `fliplr` | 0.5 | Horizontal flip. |
| `flipud` | 0.0 | No vertical flip — pedestrians don't appear upside-down. |
| `mosaic` | 1.0 | Always-on 4-image mosaic — 4× context per sample. |
| `mixup` | 0.2 | Image blend — robustness. |
| `copy_paste` | 0.4 | Paste GT instances into other images — fake crowd density. |
| `close_mosaic` | 15 | Disable mosaic for the last 15 epochs (clean finetune). |
| `erasing` | 0.4 | Random rectangle erase — simulate occlusion. |

Default schedule: 80 epochs, `imgsz=1280`, `batch=8`, patience 20.

run on 220 epochs with `imgsz=1024` and `batch=8` for best results.

Output: `weights/yolov8_mot.pt` (best.pt of the run).

---

## 3. Training — ReID encoder

Script: `src/reid_train.py`. Used at inference time inside `src/siamfc.py`
(`SiamEmbedder`) to produce a 768-D appearance vector per detection. The
BoT-SORT tracker uses cosine distance between embeddings as the appearance
term in its association cost.

### Architecture

`src/reid_model.py:ReIDNet`:

- **Trunk**: ConvNeXt-Small, ImageNet-1k pretrained (torchvision).
  Output features → adaptive avg pool → LayerNorm2d → flatten → 768-D.
- **BNNeck** (Luo et al., "Bag of Tricks", CVPR 2019): a `BatchNorm1d` between
  the embedding and the classifier head. Training uses the BN-normalized
  feature to compute classification logits, but the L2-normalized
  pre-BN feature is what the triplet loss and inference both use. This
  decouples the two losses' geometries (they want different things — CE wants
  features pushed apart in Euclidean space, triplet wants them close on the
  unit sphere).
  - `bias=False` on the classifier and BN bias frozen at 0 — also from
    Bag-of-Tricks, removes a degree of freedom that hurts ReID metrics.
- **Classifier head**: `Linear(768 → num_classes)`, no bias, Kaiming init.
  Discarded after training. `num_classes` = number of unique pedestrian IDs
  across all training sequences.

After training, only the trunk weights (`features.*`, `norm.*`) are exported
via `export_backbone_state_dict()` and saved to
`weights/reid_convnext_small.pth`. `SiamEmbedder` loads them at runtime.

### Loss

Combined CE + batch-hard triplet:

```
L = λ_ce · CE(logits, gid) + λ_tri · BatchHardTriplet(emb, gid)
```

with `λ_ce = λ_tri = 1.0` by default.

- **Cross-entropy** with label smoothing 0.1 — identity classification on
  BN-necked features.
- **Batch-hard triplet** (Hermans et al., "In Defense of the Triplet Loss for
  Person Re-Identification", 2017) on cosine distance:
  - `dist = 1 - emb @ emb.T` (since emb is L2-normalized, this is cosine
    distance).
  - For each anchor, hardest positive = max distance to a same-ID sample.
  - Hardest negative = min distance to a different-ID sample.
  - Loss = `relu(hard_pos - hard_neg + margin)`, default margin 0.3.

Hardest-mining is what makes this version of triplet work without the usual
mining headaches — every batch is constructed so each ID has K samples, so
hard pos/neg always exist.

### Sampling — PK batches

`src/reid_dataset.py:PKSampler` builds batches of `P×K` crops (default P=16
identities, K=4 samples each → batch size 64). Each iteration:

1. Sample P identities uniformly.
2. For each, sample K crops uniformly without replacement.

This is the canonical sampling for batch-hard triplet — it guarantees enough
positives and negatives per anchor.

### Optimizer / schedule

- AdamW, lr=3e-4, weight_decay=5e-4.
- Cosine annealing over `epochs` (default 30).
- AMP (fp16) on CUDA.
- 200 iterations per epoch (configurable).

### Augmentation

`src/reid_dataset.py` applies (when `augment=True`):
- Resize to 256×128 (standard ReID crop size).
- Random horizontal flip.
- Color jitter.
- Random erasing.
- ImageNet mean/std normalization.

Inference uses the same crop size, no augmentation.

---

## 4. End-to-end pipeline (per frame)

```
frame
  │
  ├─► YOLOv8 (+ optional CLAHE) ──► detections [x,y,w,h,score,cls]
  │                                       │
  │   crop each det ──► ConvNeXt-Small ──► 768-D embedding
  │                                       │
  └─► BoT-SORT tracker ◄──────────────────┘
        ├─ Kalman filter predict (per active track)
        ├─ Camera-motion compensation (sparse optical flow)
        ├─ 1st-pass association: high-conf dets ↔ tracks
        │     cost = (1 - IoU) fused with detection score, with
        │     ReID cosine distance gating outside `proximity_thresh`
        ├─ 2nd-pass: leftover low-conf dets ↔ remaining tracks (IoU only)
        ├─ Confirm new tracks (`new_track_thresh`)
        └─ Bury lost tracks after `track_buffer` frames
```

Tracker code: `src/tracker_botsort.py` (wraps `third_party/BoT-SORT`).

---

## 5. Layout

```
src/
  config.py            # paths + hyperparameters + per-seq overrides
  detector_yolo.py     # YOLOv8 wrapper (CLAHE, per-seq imgsz/conf)
  detector.py          # legacy Faster R-CNN wrapper (kept as fallback)
  yolo_train.py        # YOLO finetune entry point
  yolo_dataset.py      # builds Ultralytics-format dataset from MOT GT
  reid_train.py        # ConvNeXt-S ReID training (CE + batch-hard triplet)
  reid_model.py        # ReIDNet (trunk + BNNeck + classifier)
  reid_dataset.py      # PK sampler + crop loader
  siamfc.py            # SiamEmbedder — ConvNeXt-S inference encoder
  tracker_botsort.py   # MOTTracker — BoT-SORT wrapper, accepts overrides
  tracker_legacy.py    # original Hungarian + Kalman tracker (kept for compare)
  io_mot.py            # MOT-Challenge txt I/O, seqinfo parser
  eval_mota.py         # TrackEval wrapper (CLEAR metrics)
  visualize.py         # video writers for det/gt/track modes

project.py             # CLI entry point: track | gt | det modes
weights/               # yolov8_mot.pt, reid_convnext_small.pth, siamfc_alexnet
evs_mot_public_dataset/
  evs_mot-train/       # MOT_02..MOT_05 (with gt/)
  evs_mot-test/        # MOT_01, MOT_06, MOT_07 (no gt/, server-side eval)
results/               # per-seq output txt + mp4
```

## 6. Quick commands

```
# Train detector (one time, ~hours on a single GPU)
python -m src.yolo_dataset --force
python -m src.yolo_train --epochs 80

# Train ReID encoder
python -m src.reid_train --epochs 30

# Run on test set
python project.py --seq all --split test --detector yolo --tracker botsort --no-video

# Run on a train sequence + local TrackEval
python project.py --seq MOT_02 --eval

# Visualize raw detections only
python project.py --seq MOT_02 --mode det

# Profile per-stage timing
python project.py --seq MOT_02 --profile --no-video
```

reid train ** python -m src.reid_train --epochs 350 --P 16 --K 8 --iters_per_epoch 250  **