"""Siamese-style appearance embedder for MOT association.

Original plan was SiamFC's AlexNetV1 + pretrained weights.
We pivoted to torchvision ResNet18 (ImageNet pretrained) as the shared encoder
because the SiamFC weight mirrors went offline. The Siamese *principle* is
unchanged: a single CNN encodes both track exemplars and detection crops, and
cosine similarity in the embedding space drives the association cost.

Public API
----------
    crop_exemplar(frame_bgr, bbox_xywh) -> 127x127 BGR crop with context margin
    SiamEmbedder().embed(list_of_crops)  -> ndarray[N, D] L2-normalized
    cosine_sim_matrix(a, b)              -> ndarray[N, M]
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import cv2
from pathlib import Path
from .config import CFG, DEVICE, use_amp, REID_WEIGHTS


_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def _crop_pad(img: np.ndarray, cx, cy, sz) -> np.ndarray:
    """Square crop sz×sz around (cx, cy); pad with image mean if OOB."""
    h, w = img.shape[:2]
    sz = int(round(sz))
    half = sz // 2
    x0, y0 = int(round(cx - half)), int(round(cy - half))
    x1, y1 = x0 + sz, y0 + sz
    pl, pt = max(0, -x0), max(0, -y0)
    pr, pb = max(0, x1 - w), max(0, y1 - h)
    if pl or pt or pr or pb:
        mean = img.reshape(-1, 3).mean(axis=0)
        img = cv2.copyMakeBorder(img, pt, pb, pl, pr, cv2.BORDER_CONSTANT, value=mean.tolist())
        x0 += pl
        y0 += pt
    return img[y0:y0 + sz, x0:x0 + sz]


def crop_exemplar(frame_bgr: np.ndarray, bbox_xywh, target=127, context=0.5) -> np.ndarray:
    """SiamFC-style exemplar crop: context-padded square around bbox, resized to target."""
    x, y, w, h = bbox_xywh[:4]
    cx, cy = x + w / 2.0, y + h / 2.0
    wc = w + context * (w + h)
    hc = h + context * (w + h)
    sz = float(np.sqrt(max(wc * hc, 1.0)))
    crop = _crop_pad(frame_bgr, cx, cy, sz)
    return cv2.resize(crop, (target, target))


class _ResNet18Trunk(nn.Module):
    """ResNet18 cut after layer3 + global avg pool → 256-D embedding."""

    def __init__(self):
        super().__init__()
        backbone = torchvision.models.resnet18(
            weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1
        )
        self.stem = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool,
        )
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool(x).flatten(1)
        return x  # [N, 256]


class SiamEmbedder:
    """Shared CNN encoder → L2-normalized 256-D appearance embedding."""

    EMBED_DIM = 256

    def __init__(self, weights_path: str | None = None):
        self.net = _ResNet18Trunk().to(DEVICE).eval()
        self._mean = _IMAGENET_MEAN.to(DEVICE)
        self._std = _IMAGENET_STD.to(DEVICE)
        self._amp = use_amp(DEVICE)

        # Auto-load finetuned ReID weights if user has trained them.
        path = Path(weights_path) if weights_path else REID_WEIGHTS
        if path.exists():
            sd = torch.load(path, map_location=DEVICE)
            missing, unexpected = self.net.load_state_dict(sd, strict=False)
            print(f"[siamfc] loaded finetuned ReID weights {path}  "
                  f"(missing={len(missing)}, unexpected={len(unexpected)})")
        else:
            print(f"[siamfc] no finetuned weights at {path} — using ImageNet pretrained")

        print(f"[siamfc] ResNet18 encoder ready on {DEVICE}, dim={self.EMBED_DIM}, amp={self._amp}")

    @torch.no_grad()
    def embed(self, crops_bgr: list[np.ndarray]) -> np.ndarray:
        """List of BGR crops → ndarray[N, EMBED_DIM] L2-normalized."""
        if len(crops_bgr) == 0:
            return np.zeros((0, self.EMBED_DIM), dtype=np.float32)
        rgb = [c[:, :, ::-1].astype(np.float32) / 255.0 for c in crops_bgr]
        x = torch.from_numpy(np.stack(rgb)).permute(0, 3, 1, 2).to(DEVICE, non_blocking=True)
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
