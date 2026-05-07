"""ReID dataset: pedestrian crops from MOT GT + PK sampler for triplet training."""
from __future__ import annotations
from collections import defaultdict
from pathlib import Path
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler
from .config import CFG, TRAIN_DIR
from .io_mot import load_seqinfo, parse_mot_csv
from .siamfc import crop_person

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _global_id(seq: str, tid: int) -> str:
    return f"{seq}_{int(tid)}"


class MOTReIDDataset(Dataset):
    """Index over (seq, frame, bbox, gid) entries from MOT GT.

    Crops are produced on-the-fly via crop_exemplar to match the inference
    distribution exactly. Augmentations: hflip, color jitter, random erasing,
    small random crop+resize.
    """

    def __init__(self, seq_names: list[str], target_hw=None, augment: bool = True):
        self.target_hw = target_hw if target_hw is not None else (CFG.crop_h, CFG.crop_w)
        self.augment = augment
        self.entries: list[tuple[str, int, np.ndarray, int]] = []
        self.gid_to_int: dict[str, int] = {}
        self.seq_dirs = {s: TRAIN_DIR / s for s in seq_names}
        self.seq_info: dict[str, dict] = {}
        self._frame_cache: dict[tuple[str, int], np.ndarray] = {}
        self._cache_max = 64

        for seq in seq_names:
            seq_dir = self.seq_dirs[seq]
            self.seq_info[seq] = load_seqinfo(seq_dir)
            gt = parse_mot_csv(seq_dir / "gt" / "gt.txt")
            if gt.size == 0:
                continue
            # MOT format: frame,id,x,y,w,h,conf,class,vis
            keep = (gt[:, 6] > 0) & (gt[:, 7] == 1)
            gt = gt[keep]
            for row in gt:
                frame = int(row[0])
                tid = int(row[1])
                bbox = row[2:6].astype(np.float32)
                if bbox[2] < 8 or bbox[3] < 8:
                    continue
                gid_str = _global_id(seq, tid)
                if gid_str not in self.gid_to_int:
                    self.gid_to_int[gid_str] = len(self.gid_to_int)
                self.entries.append((seq, frame, bbox, self.gid_to_int[gid_str]))

        self.num_classes = len(self.gid_to_int)
        self.gid_to_entries: dict[int, list[int]] = defaultdict(list)
        for i, (_, _, _, gid) in enumerate(self.entries):
            self.gid_to_entries[gid].append(i)

        # Drop classes with <2 samples (triplet cannot mine positives otherwise)
        bad = {g for g, lst in self.gid_to_entries.items() if len(lst) < 2}
        if bad:
            self.entries = [e for e in self.entries if e[3] not in bad]
            kept = sorted(set(e[3] for e in self.entries))
            remap = {old: new for new, old in enumerate(kept)}
            self.entries = [(s, f, b, remap[g]) for s, f, b, g in self.entries]
            self.gid_to_int = {k: remap[v] for k, v in self.gid_to_int.items() if v in remap}
            self.num_classes = len(remap)
            self.gid_to_entries = defaultdict(list)
            for i, (_, _, _, gid) in enumerate(self.entries):
                self.gid_to_entries[gid].append(i)

        print(f"[reid] {len(self.entries)} crops, {self.num_classes} identities, "
              f"sequences={seq_names}")

    def __len__(self):
        return len(self.entries)

    def _read_frame(self, seq: str, frame: int) -> np.ndarray:
        key = (seq, frame)
        cached = self._frame_cache.get(key)
        if cached is not None:
            return cached
        info = self.seq_info[seq]
        path = self.seq_dirs[seq] / info["im_dir"] / f"{frame:06d}{info['im_ext']}"
        img = cv2.imread(str(path))
        if len(self._frame_cache) > self._cache_max:
            self._frame_cache.pop(next(iter(self._frame_cache)))
        self._frame_cache[key] = img
        return img

    def _augment(self, crop: np.ndarray) -> np.ndarray:
        if np.random.rand() < 0.5:
            crop = crop[:, ::-1].copy()
        if np.random.rand() < 0.5:
            alpha = 1.0 + (np.random.rand() - 0.5) * 0.4
            beta = (np.random.rand() - 0.5) * 30.0
            crop = np.clip(alpha * crop.astype(np.float32) + beta, 0, 255).astype(np.uint8)
        if np.random.rand() < 0.3:
            h, w = crop.shape[:2]
            eh = np.random.randint(max(2, h // 8), max(3, h // 3))
            ew = np.random.randint(max(2, w // 8), max(3, w // 3))
            ey = np.random.randint(0, h - eh)
            ex = np.random.randint(0, w - ew)
            crop[ey:ey + eh, ex:ex + ew] = np.random.randint(
                0, 255, size=(eh, ew, 3), dtype=np.uint8
            )
        if np.random.rand() < 0.3:
            h, w = crop.shape[:2]
            scale = np.random.uniform(0.85, 1.0)
            ch, cw = int(h * scale), int(w * scale)
            y0 = np.random.randint(0, h - ch + 1)
            x0 = np.random.randint(0, w - cw + 1)
            crop = crop[y0:y0 + ch, x0:x0 + cw]
            crop = cv2.resize(crop, (w, h))
        return crop

    def __getitem__(self, idx):
        seq, frame, bbox, gid = self.entries[idx]
        img = self._read_frame(seq, frame)
        crop = crop_person(img, bbox, target_hw=self.target_hw, context=CFG.context_amount)
        if self.augment:
            crop = self._augment(crop)
        rgb = crop[:, :, ::-1].astype(np.float32) / 255.0
        rgb = (rgb - _IMAGENET_MEAN) / _IMAGENET_STD
        x = torch.from_numpy(rgb).permute(2, 0, 1).contiguous().float()
        return x, gid


class PKSampler(Sampler):
    """Each batch = P identities × K samples per identity (for batch-hard triplet)."""

    def __init__(self, dataset: MOTReIDDataset, p: int = 16, k: int = 4, num_iters: int = 200):
        self.dataset = dataset
        self.p = p
        self.k = k
        self.num_iters = num_iters
        self.gids = list(dataset.gid_to_entries.keys())

    def __len__(self):
        return self.num_iters

    def __iter__(self):
        for _ in range(self.num_iters):
            chosen = np.random.choice(self.gids,
                                      size=self.p,
                                      replace=len(self.gids) < self.p)
            batch: list[int] = []
            for gid in chosen:
                pool = self.dataset.gid_to_entries[int(gid)]
                replace = len(pool) < self.k
                idxs = np.random.choice(pool, size=self.k, replace=replace)
                batch.extend(int(i) for i in idxs)
            yield batch
