"""
model_v2.py — Inference for V2 PaDiM-style defect detector.

Loads the model saved by train_v2.py and produces a binary defect mask.
The ResNet18 extractor is loaded once and cached for the lifetime of the process.

Drop-in replacement for model.py — same predict(image) API.
"""

import os
import numpy as np
import cv2
import torch
import torch.nn.functional as F
import torchvision.models as models

MODEL_PATH = os.path.join(os.path.dirname(__file__), "defect_model_v2.npz")
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

_MEAN  = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_STD   = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

# CLAHE — must match train_v2.py exactly (same clipLimit and tileGridSize)
_CLAHE = cv2.createCLAHE(clipLimit=1000, tileGridSize=(8, 8))

# Minimum connected-component area to keep as a real defect (pixels).
# Set low enough to catch small bristle holes (~350px GT area seen in dataset).
_MIN_BLOB_PX = 120


# ── feature extractor (must match train_v2.py exactly) ───────────────────────

class _ResNetExtractor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        local_weights = os.path.join(os.path.dirname(__file__), "wide_resnet50_2-9ba9bcbe.pth")
        if os.path.exists(local_weights):
            backbone = models.wide_resnet50_2(weights=None)
            backbone.load_state_dict(torch.load(local_weights, map_location="cpu"))
        else:
            backbone = models.wide_resnet50_2(weights=models.Wide_ResNet50_2_Weights.IMAGENET1K_V2)
        self.stem   = torch.nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool
        )
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3

    def forward(self, x):
        x  = self.stem(x)
        f1 = self.layer1(x)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        sz = f1.shape[2:]
        f2 = F.interpolate(f2, size=sz, mode="bilinear", align_corners=False)
        f3 = F.interpolate(f3, size=sz, mode="bilinear", align_corners=False)
        return torch.cat([f1, f2, f3], dim=1)


# ── toothbrush ROI ───────────────────────────────────────────────────────────

def _toothbrush_roi(img_bgr: np.ndarray) -> np.ndarray:
    """
    Return a binary mask (uint8, 0/255) covering the toothbrush body.

    Identical algorithm to train_v2.toothbrush_roi — must stay in sync.

    Steps:
      1. Otsu threshold (reliable at ~82 for this dataset).
      2. Morphological close 20×20 — bridges inter-bristle gaps.
      3. Convex hull of bright pixels — fills the toothbrush outline without
         being confused by inter-bristle gaps that connect to the background.
      4. Dilation 20×20 — includes boundary-adjacent defects (e.g. outer
         bristle row missing or broken).
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    close_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
    closed  = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, close_k)

    coords = np.column_stack(np.where(closed > 0))    # (row, col)
    if len(coords) == 0:
        return np.ones(gray.shape, dtype=np.uint8) * 255   # fallback

    hull = cv2.convexHull(coords[:, ::-1].astype(np.float32))    # (x, y)
    roi  = np.zeros(gray.shape, dtype=np.uint8)
    cv2.fillPoly(roi, [hull.astype(np.int32)], 255)

    dil_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
    return cv2.dilate(roi, dil_k)


# ── lazy model cache ──────────────────────────────────────────────────────────

_cache: dict = {}


def _load():
    """Load model weights and ResNet extractor once, cache in module global."""
    if _cache:
        return _cache

    data = np.load(MODEL_PATH)

    # ResNet50 layer1(256) + layer2(512) + layer3(1024) = 1792 channels.
    # This is the only valid feature dimension for this code.
    expected_feat_dim = 1792   # ResNet50 layer1(256)+layer2(512)+layer3(1024)

    pca_mean       = data["pca_mean"].astype(np.float32)
    pca_components = data["pca_components"].astype(np.float32)
    mean_feat      = data["mean_feat"].astype(np.float32)
    precision      = data["precision"].astype(np.float32)

    if pca_mean.shape != (expected_feat_dim,) or pca_components.shape[1] != expected_feat_dim:
        raise RuntimeError(
            f"Stale model file: '{MODEL_PATH}'\n"
            f"  pca_mean      : {pca_mean.shape}   (expected ({expected_feat_dim},))\n"
            f"  pca_components: {pca_components.shape}   (expected (n_components, {expected_feat_dim}))\n"
            f"\nThe model was built by a different version of train_v2.py.\n"
            f"Re-run:  python3 train_v2.py"
        )

    _cache["mean_feat"]      = mean_feat
    _cache["precision"]      = precision
    _cache["pca_components"] = pca_components
    _cache["pca_mean"]       = pca_mean
    _cache["threshold"]      = float(data["threshold"])
    _cache["img_size"]       = tuple(int(v) for v in data["img_size"])
    _cache["feat_size"]      = tuple(int(v) for v in data["feat_size"])

    extractor = _ResNetExtractor().to(DEVICE)
    extractor.eval()
    _cache["extractor"] = extractor
    return _cache


# ── public API ────────────────────────────────────────────────────────────────

def predict(image: np.ndarray) -> np.ndarray:
    """
    Detect toothbrush bristle defects and return a binary mask.

    Args:
        image: uint8 numpy array, shape (H, W, 3) RGB or (H, W) grayscale.

    Returns:
        Binary mask, shape (H, W), uint8, values 0 or 255.
    """
    m = _load()
    extractor      = m["extractor"]
    mean_feat      = m["mean_feat"]
    precision      = m["precision"]
    pca_components = m["pca_components"]
    pca_mean       = m["pca_mean"]
    threshold      = m["threshold"]
    img_h, img_w   = m["img_size"]
    feat_h, feat_w = m["feat_size"]

    # Convert input to BGR
    if image.ndim == 2:
        img_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    h_orig, w_orig = img_bgr.shape[:2]

    # ── CLAHE preprocessing (must match train_v2.py) ──────────────────────────
    # Locally amplifies contrast so dark holes and bright blemishes both deviate
    # clearly from the "normal" feature distribution learned on good images.
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2Lab)
    l, a, b = cv2.split(lab)
    l = _CLAHE.apply(l)
    img_enhanced = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_Lab2BGR)

    # ── feature extraction ────────────────────────────────────────────────────
    img_rgb = cv2.cvtColor(img_enhanced, cv2.COLOR_BGR2RGB)
    img_res = cv2.resize(img_rgb, (img_w, img_h), interpolation=cv2.INTER_AREA)
    tensor  = torch.from_numpy(img_res).permute(2, 0, 1).float() / 255.0
    tensor  = ((tensor - _MEAN) / _STD).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        feat = extractor(tensor)                                       # (1, 1792, fh, fw)
    feat = feat.squeeze(0).permute(1, 2, 0).cpu().numpy()             # (fh, fw, 1792)
    feat = feat.reshape(-1, feat.shape[2])                             # (P, 1792)

    # ── PCA + Mahalanobis distance ────────────────────────────────────────────
    feat_pca = (feat - pca_mean) @ pca_components.T                   # (P, d)
    diff     = feat_pca - mean_feat                                    # (P, d)
    tmp      = np.einsum("pi,pij->pj", diff, precision)               # (P, d)
    dist2    = np.einsum("pi,pi->p",   tmp,  diff)                    # (P,)
    dist     = np.sqrt(np.maximum(dist2, 0.0))                        # (P,)

    score_map = dist.reshape(feat_h, feat_w)
    score_up  = cv2.resize(score_map, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR)
    score_up  = cv2.GaussianBlur(score_up, (0, 0), sigmaX=2.0)

    # ── binary mask ───────────────────────────────────────────────────────────
    mask = (score_up >= threshold).astype(np.uint8) * 255

    # Toothbrush ROI — zero out everything outside the toothbrush body.
    # Computed from the original (non-CLAHE) image so that the Otsu threshold
    # sees the real brightness distribution and correctly excludes background.
    roi  = _toothbrush_roi(img_bgr)
    mask = cv2.bitwise_and(mask, roi)

    # Morphological cleanup — keep kernels small to avoid inflating detections:
    #   OPEN  3×3 : removes isolated salt noise
    #   CLOSE 5×5 : fills small intra-defect gaps without over-expanding borders
    #               (was 7×7 — reduced to limit balloon effect on small defects)
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    )
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    )

    # Remove blobs too small to be real defects.
    # _MIN_BLOB_PX=120 is below the smallest GT defect blob in the dataset (272px)
    # while still rejecting single-patch noise (~64px at 64×64→1024 resolution).
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    for i in range(1, n_labels):
        if stats[i, cv2.CC_STAT_AREA] <= _MIN_BLOB_PX:
            mask[labels == i] = 0

    return mask