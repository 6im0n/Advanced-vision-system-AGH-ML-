# EVS-MOT pedestrian tracker — dettxt version

Multi-object tracking on the EVS-MOT dataset using the **provided per-sequence
`det.txt`** as the canonical detector input (per challenge rules:
*"detekcji, których należy użyć do inicjalizacji algorytmu śledzenia"*),
combined with a ConvNeXt-Small ReID encoder and the BoT-SORT association
backend.

This version contains no learned detector. The YOLO / Faster R-CNN /
ensemble detectors and the legacy Hungarian tracker have been stripped out.
The only remaining trainable component is the ReID encoder.

## 1. Detection

### Source

`evs_mot_public_dataset/<split>/<seq>/det/det.txt`. One line per detection,
MOT-Challenge format:

```
frame, -1, bb_left, bb_top, bb_width, bb_height, confidence
```

These are precomputed by the dataset organizers. They are the official
input to the tracker; this project does not run its own detector at
inference time.

### Loader

Code: `src/detector_dettxt.py:DetTxtDetector`.

Behavior:

1. Construct one `DetTxtDetector(seq_dir)` per sequence.
2. At init, parse the full `det.txt`, drop rows with `score < CFG.det_score_th`
   (default 0.05), bucket the survivors by frame index.
3. Each call to `.detect(frame)` advances a 1-indexed internal cursor and
   returns the rows for that frame as `[N, 6]`: `x, y, w, h, score, cls=1`
   (single class, pedestrian).

The detector is stateful — sequence iteration must start at frame 1 and
proceed in order. `project.run_track` already does this via `frame_iter`.

### Per-sequence overrides

`src/config.py:PER_SEQ_OVERRIDES`. Read in `run_track` based on `seqinfo`
name. Two override blocks:

- `detector` — kwargs for `DetTxtDetector` (currently only `score_th`).
- `botsort` — attribute overrides on `BotSortCfg`.

Active overrides:

```python
"MOT_07": {
    "botsort": {
        "track_high_thresh": 0.15,
        "new_track_thresh": 0.20,
    },
},
```

Reason: MOT_07's `det.txt` has a long low-confidence tail. 10% of detections
score below 0.32, 5% below 0.14. With the default
`new_track_thresh=0.35` / `track_high_thresh=0.3`, those small/distant
pedestrians never seed new tracks → high FN. Lowering the gates lets them
in.

---

## 2. Tracker — BoT-SORT

Code: `src/tracker_botsort.py:MOTTracker` (wraps `third_party/BoT-SORT`).

Per-frame algorithm:

1. **Kalman predict** for each active track.
2. **Camera-motion compensation (CMC)** via sparse optical flow on the
   frame, applied to the predicted boxes (`gmc_downscale=2` for speed).
3. **First-stage association**: high-confidence detections
   (`score ≥ track_high_thresh`) ↔ active tracks. Cost is `1 - IoU` (with
   `fuse_score=True`, IoU is fused with the detection score), gated by
   `match_thresh`. Inside the proximity zone (`IoU ≥ proximity_thresh`),
   appearance distance is ignored; outside it, ReID cosine distance must
   pass `appearance_thresh`.
4. **Second-stage association**: leftover low-confidence detections
   (`track_low_thresh ≤ score < track_high_thresh`) ↔ remaining tracks,
   IoU only.
5. **New tracks** spawned from unmatched detections with
   `score ≥ new_track_thresh`.
6. **Lost tracks** kept alive for `track_buffer` frames before deletion;
   re-acquired on later match.

ReID is on (`with_reid=True`). The tracker calls into `SiamEmbedder`
(see §3) to get a 768-D embedding per detection crop and updates each
track's exemplar with an EMA on every confirmed match.

### `BotSortCfg` defaults

| Field | Default | Role |
|---|---|---|
| `track_high_thresh` | 0.30 | gate for primary association |
| `track_low_thresh` | 0.05 | floor — below this dets dropped |
| `new_track_thresh` | 0.35 | min score to start a new track |
| `track_buffer` | 90 | frames a lost track stays revivable |
| `match_thresh` | 0.80 | first-stage cost accept gate |
| `proximity_thresh` | 0.50 | IoU above which appearance is ignored |
| `appearance_thresh` | 0.25 | cosine-distance gate for ReID match |
| `cmc_method` | sparseOptFlow | camera-motion model |
| `gmc_downscale` | 2 | downsample factor for CMC |
| `fuse_score` | True | multiply IoU cost by detection score |

---

## 3. Training — ReID encoder

The only trainable component in this version. The detector is fixed (det.txt),
the tracker is hand-tuned. ReID quality drives **IDSW** (identity switches)
and lost-track resurrection.

Script: `src/reid_train.py`. Inference wrapper: `src/siamfc.py:SiamEmbedder`,
which loads `weights/reid_convnext_small.pth`.

### Architecture

`src/reid_model.py:ReIDNet`:

- **Trunk**: ConvNeXt-Small, ImageNet-1k pretrained (torchvision).
  Output features → adaptive avg pool → LayerNorm2d → flatten → 768-D.
- **BNNeck** (Luo et al., "Bag of Tricks", CVPR 2019): a `BatchNorm1d`
  between the embedding and the classifier head. The classifier loss is
  computed on the BN-normalized feature; the L2-normalized **pre-BN**
  feature is what the triplet loss and inference both use. Decouples the
  two losses' geometries — CE wants features pushed apart in Euclidean
  space, triplet wants them close on the unit sphere.
  - `bias=False` on the classifier and BN bias frozen at 0 — also from
    Bag-of-Tricks, removes a degree of freedom that hurts ReID metrics.
- **Classifier head**: `Linear(768 → num_classes)`, no bias, Kaiming init.
  Discarded after training. `num_classes` = unique pedestrian IDs across
  the training sequences.

After training, only the trunk weights (`features.*`, `norm.*`) are
exported via `export_backbone_state_dict()` and saved to
`weights/reid_convnext_small.pth`. `SiamEmbedder` loads them at runtime.

### Loss

```
L = λ_ce · CE(logits, gid) + λ_tri · BatchHardTriplet(emb, gid)
```

Defaults: `λ_ce = λ_tri = 1.0`.

- **Cross-entropy** with label smoothing 0.1 on BN-necked features.
- **Batch-hard triplet** (Hermans et al., "In Defense of the Triplet Loss
  for Person Re-Identification", 2017) on cosine distance:
  - `dist = 1 - emb @ emb.T` (since `emb` is L2-normalized, this is
    cosine distance).
  - For each anchor: hardest positive = max distance to a same-ID sample;
    hardest negative = min distance to a different-ID sample.
  - Loss = `relu(hard_pos − hard_neg + margin)`, default margin 0.3.

Hard mining works because each batch is built so every ID present has K
samples, guaranteeing positives exist.

### Sampling — PK batches

`src/reid_dataset.py:PKSampler` builds batches of `P × K` crops:

1. Sample P identities uniformly.
2. For each, sample K crops uniformly without replacement.

Default `P=16, K=4 → batch=64`. Canonical sampling for batch-hard triplet.

### Optimizer / schedule

- AdamW, `lr=3e-4`, `weight_decay=5e-4`.
- Cosine annealing over `--epochs` (default 30).
- AMP (fp16) on CUDA.
- 200 iterations per epoch (configurable via `--iters_per_epoch`).

### Augmentation (train only)

`src/reid_dataset.py` applies, when `augment=True`:

- Resize to 256×128 (standard ReID crop size).
- Random horizontal flip.
- Color jitter.
- Random erasing.
- ImageNet mean/std normalization.

Inference uses the same crop size, no augmentation.

### Data

GT crops from MOT_02..MOT_05. Each pedestrian identity appears across
many frames; `MOTReIDDataset` indexes them and exposes a global identity id.

---

## 4. End-to-end pipeline (per frame)

```
frame
  │
  ├─► DetTxtDetector ──► detections [x,y,w,h,score,cls=1]   (lookup, no model)
  │                              │
  │   crop each det ──► ConvNeXt-Small (SiamEmbedder) ──► 768-D embedding
  │                              │
  └─► BoT-SORT tracker ◄─────────┘
        ├─ Kalman filter predict (per active track)
        ├─ Camera-motion compensation (sparse optical flow)
        ├─ 1st-pass association: high-conf dets ↔ tracks
        │     cost = (1 − IoU) ⊗ score, ReID cosine distance gate outside `proximity_thresh`
        ├─ 2nd-pass: low-conf dets ↔ remaining tracks (IoU only)
        ├─ Confirm new tracks (score ≥ `new_track_thresh`)
        └─ Bury lost tracks after `track_buffer` frames
  │
  └─► write [frame, id, x, y, w, h, conf, cls] to results/<seq>.txt
```

---

## 5. Layout

```
src/
  config.py            # paths + TrackerCfg + BotSortCfg + PER_SEQ_OVERRIDES
  detector_dettxt.py   # det.txt loader / per-frame replay
  io_mot.py            # MOT-Challenge txt I/O, seqinfo parser
  siamfc.py            # SiamEmbedder — ConvNeXt-S inference encoder
  reid_model.py        # ReIDNet (trunk + BNNeck + classifier)
  reid_dataset.py      # PK sampler + crop loader for MOT GT
  reid_train.py        # CE + batch-hard triplet training entry
  tracker_botsort.py   # MOTTracker — BoT-SORT wrapper, accepts overrides
  eval_mota.py         # TrackEval wrapper (CLEAR metrics)
  visualize.py         # video writers for det/gt/track modes

project.py             # CLI: track | gt | det modes (BoT-SORT + dettxt only)
weights/               # reid_convnext_small.pth
evs_mot_public_dataset → ../evs_mot_public_dataset (symlink to shared dataset)
results/               # per-seq output txt + mp4
third_party/           # vendored BoT-SORT
```

## 6. Quick commands

```bash
# Train ReID encoder (only training step in this version)
python -m src.reid_train --epochs 30

# Track a single training sequence (with TrackEval)
python project.py --seq MOT_02 --eval

# Track all training sequences with TrackEval
python project.py --seq all --split train --eval

# Track all test sequences (no GT, server-side eval) — submission run
python project.py --seq all --split test --no-video

# Visualize det.txt (no tracker)
python project.py --seq MOT_02 --mode det

# Visualize ground truth
python project.py --seq MOT_02 --mode gt

# Profile per-stage timing
python project.py --seq MOT_02 --profile --no-video
```

CLI flags:

| Flag | Default | Notes |
|---|---|---|
| `--seq` | MOT_02 | comma-separated names or `all` |
| `--split` | train | `train` / `test` / `all` (used when `--seq all`) |
| `--mode` | track | `track` / `gt` / `det` |
| `--no-video` | off | skip mp4 output (faster) |
| `--profile` | off | per-stage ms breakdown |
| `--eval` | off | run TrackEval after tracking (train split only) |

There is no `--detector` or `--tracker` flag in this version — the detector
is hardwired to dettxt and the tracker to BoT-SORT.

## 7. Notes

- `weights/reid_convnext_small.pth` is required for tracking. Train it
  first (see §3) or copy it from the yolo-version sibling.
- The dataset directory is a symlink to `../evs_mot_public_dataset` shared
  with the yolo-version; do not duplicate it.
- `track_buffer=90` is generous (~3 s at 30 fps). Lower it if test sequences
  show ID resurrection issues; raise it if FN bursts during long occlusions.
