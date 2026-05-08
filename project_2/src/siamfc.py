"""Siamese-style appearance embedder for MOT association.

Architecture: ResNet50 (ImageNet pretrained) trunk → global avg pool →
2048-D L2-normalized embedding. Auto-loads finetuned ReID weights from
weights/reid_resnet50.pth if present.

Public API
----------
    crop_person(frame_bgr, bbox_xywh) -> 256×128 BGR crop (H×W)
    SiamEmbedder().embed(list_of_crops)  -> ndarray[N, D] L2-normalized
    cosine_sim_matrix(a, b)              -> ndarray[N, M]
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import cv2
from .config import CFG, DEVICE, use_amp, REID_WEIGHTS


_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def crop_person(frame_bgr: np.ndarray, bbox_xywh, target_hw=None, context: float = None) -> np.ndarray:
    """Crop person bbox with small context margin, resize to (H, W) = (256, 128).

    Padding with image mean handles out-of-bounds boxes near image edges.
    """
    if target_hw is None:
        target_hw = (CFG.crop_h, CFG.crop_w)
    if context is None:
        context = CFG.context_amount
    H, W = target_hw
    x, y, w, h = bbox_xywh[:4]
    pad_w = context * w
    pad_h = context * h
    x0 = int(round(x - pad_w))
    y0 = int(round(y - pad_h))
    x1 = int(round(x + w + pad_w))
    y1 = int(round(y + h + pad_h))

    fh, fw = frame_bgr.shape[:2]
    pl, pt = max(0, -x0), max(0, -y0)
    pr, pb = max(0, x1 - fw), max(0, y1 - fh)
    if pl or pt or pr or pb:
        mean = frame_bgr.reshape(-1, 3).mean(axis=0)
        img = cv2.copyMakeBorder(frame_bgr, pt, pb, pl, pr,
                                 cv2.BORDER_CONSTANT, value=mean.tolist())
        x0 += pl
        x1 += pl
        y0 += pt
        y1 += pt
    else:
        img = frame_bgr
    crop = img[y0:y1, x0:x1]
    if crop.size == 0:
        crop = np.zeros((H, W, 3), dtype=np.uint8)
    return cv2.resize(crop, (W, H))


# Back-compat alias for any older import sites.
crop_exemplar = crop_person


class _ConvNeXtSmallTrunk(nn.Module):
    """ConvNeXt-Small features + avgpool + LayerNorm → 768-D embedding."""

    def __init__(self):
        super().__init__()
        bb = torchvision.models.convnext_small(
            weights=torchvision.models.ConvNeXt_Small_Weights.IMAGENET1K_V1
        )
        self.features = bb.features
        self.avgpool = bb.avgpool
        self.norm = bb.classifier[0]              # LayerNorm2d on B×C×1×1
        self.flatten = bb.classifier[1]           # Flatten

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = self.norm(x)
        return self.flatten(x)                    # [N, 768]


class SiamEmbedder:
    """ConvNeXt-Small encoder → L2-normalized 768-D appearance embedding."""

    EMBED_DIM = 768

    def __init__(self, weights_path: str | None = None):
        self.net = _ConvNeXtSmallTrunk().to(DEVICE).eval()
        self._mean = _IMAGENET_MEAN.to(DEVICE)
        self._std = _IMAGENET_STD.to(DEVICE)
        self._amp = use_amp(DEVICE)

        # Auto-load finetuned ReID weights if user has trained them.
        path = Path(weights_path) if weights_path else REID_WEIGHTS
        if path.exists():
            sd = torch.load(path, map_location=DEVICE, weights_only=True)
            missing, unexpected = self.net.load_state_dict(sd, strict=False)
            print(f"[siamfc] loaded finetuned ReID weights {path}  "
                  f"(missing={len(missing)}, unexpected={len(unexpected)})")
        else:
            print(f"[siamfc] no finetuned weights at {path} — using ImageNet pretrained")

        print(f"[siamfc] ConvNeXt-Small encoder ready on {DEVICE}, dim={self.EMBED_DIM}, amp={self._amp}")

    @torch.no_grad()
    def embed(self, crops_bgr: list[np.ndarray]) -> np.ndarray:
        """List of BGR crops (any uniform HxW) → ndarray[N, EMBED_DIM] L2-normalized."""
        if len(crops_bgr) == 0:
            return np.zeros((0, self.EMBED_DIM), dtype=np.float32)
        rgb = [c[:, :, ::-1].astype(np.float32) / 255.0 for c in crops_bgr]
        x = torch.from_numpy(np.stack(rgb)).permute(0, 3, 1, 2).contiguous().to(
            DEVICE, non_blocking=True
        )
        x = (x - self._mean) / self._std
        if self._amp:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                feat = self.net(x)
            feat = feat.float()
        else:
            feat = self.net(x)
        feat = F.normalize(feat, dim=1)
        return feat.detach().cpu().numpy()


def cosine_sim_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """L2-normalized a [N,D], b [M,D] → [N,M] cosine similarity."""
    if a.shape[0] == 0 or b.shape[0] == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=np.float32)
    return (a @ b.T).astype(np.float32)
