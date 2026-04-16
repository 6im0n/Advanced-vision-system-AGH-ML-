"""
train_v2.py — V2: PaDiM-style defect detection using pretrained ResNe50 patch features.

Why better than V1 (LAB Gaussian)?
  V1 computes z-scores in raw 3-channel LAB space — very low-dimensional, misses
  texture/shape information entirely.  V2 extracts 448-dimensional patch descriptors
  from three ResNet50 layers (layer1/2/3), reduces via PCA to 50 dims, and fits a
  full multivariate Gaussian per spatial position.  The Mahalanobis distance in this
  rich feature space is far more discriminative.

Training strategy (unsupervised):
  - Only good images are used to build the Gaussian model.
  - Defective images + GT masks are used only for threshold calibration.
  - No labels or bounding boxes are needed for the core model.

Usage:
    pip install torch torchvision scikit-learn   # one-time
    python3 train_v2.py
"""

import os
import numpy as np
import cv2
import torch
import torch.nn.functional as F
import torchvision.models as models
from sklearn.decomposition import PCA
from concurrent.futures import ThreadPoolExecutor

# ── config ────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.join(os.path.dirname(__file__), "toothbrush")
GOOD_DIR   = os.path.join(BASE_DIR, "train", "good")
DEFECT_DIR = os.path.join(BASE_DIR, "train", "defective")
GT_DIR     = os.path.join(BASE_DIR, "ground_truth", "defective")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "defect_model_v2.npz")

IMG_SIZE     = 256    # resize input for ResNet (must be ≥224, keep power of 2)
N_COMPONENTS = 100     # PCA dims — keep below number of good training images
N_THRESH     = 30   # threshold search steps
N_WORKERS    = 4
REG_COEF     = 0.01   # diagonal regularisation added to covariance matrices

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ImageNet normalisation constants
_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

# CLAHE for local contrast enhancement (created once, reused per image).
# clipLimit=2 prevents over-amplifying noise; tileGridSize=(8,8) operates at
# a scale fine enough to catch both large bright defects and small dark holes.
_CLAHE = cv2.createCLAHE(clipLimit=1000, tileGridSize=(8, 8))

print(f"Device : {DEVICE}")
print(f"IMG_SIZE={IMG_SIZE}  N_COMPONENTS={N_COMPONENTS}  REG={REG_COEF}")


# ── feature extractor ─────────────────────────────────────────────────────────

class ResNetExtractor(torch.nn.Module):
    """
    Multi-scale patch features from a frozen pretrained ResNet50.

    Extracts layer1 (64-ch), layer2 (128-ch), layer3 (256-ch) feature maps,
    upsamples layer2 and layer3 to match layer1's spatial resolution, and
    concatenates them → 448 channels per patch.

    For IMG_SIZE=256 the output spatial grid is 64×64 (stride 4).
    """

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
        self.layer1 = backbone.layer1   # → (B, 64,  H/4,  W/4)
        self.layer2 = backbone.layer2   # → (B, 128, H/8,  W/8)
        self.layer3 = backbone.layer3   # → (B, 256, H/16, W/16)

    def forward(self, x):
        x  = self.stem(x)
        f1 = self.layer1(x)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        sz = f1.shape[2:]   # target spatial size
        f2 = F.interpolate(f2, size=sz, mode="bilinear", align_corners=False)
        f3 = F.interpolate(f3, size=sz, mode="bilinear", align_corners=False)
        return torch.cat([f1, f2, f3], dim=1)   # (B, 448, sz_h, sz_w)


def preprocess(img_bgr: np.ndarray) -> torch.Tensor:
    """
    BGR uint8 → normalised ImageNet float tensor (1, 3, IMG_SIZE, IMG_SIZE).

    CLAHE is applied to the L channel before ResNet normalisation so that:
      - Dark holes (missing bristles) are locally amplified relative to their
        neighbours → features become distinct from the normal pattern.
      - Bright blemishes are also amplified, but remain different from dark holes.
    Must be applied identically at training and inference.
    """
    # Local contrast enhancement on L channel
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2Lab)
    l, a, b = cv2.split(lab)
    l = _CLAHE.apply(l)
    img_enhanced = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_Lab2BGR)

    img_rgb = cv2.cvtColor(img_enhanced, cv2.COLOR_BGR2RGB)
    img     = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
    return ((t - _MEAN) / _STD).unsqueeze(0)


@torch.no_grad()
def extract_features(img_bgr: np.ndarray, extractor: ResNetExtractor) -> np.ndarray:
    """
    Extract patch features for a single image.

    Returns: float32 array (P, 448)  where P = (IMG_SIZE/4)².
    """
    t    = preprocess(img_bgr).to(DEVICE)
    feat = extractor(t)                                     # (1, 448, fh, fw)
    feat = feat.squeeze(0).permute(1, 2, 0).cpu().numpy()  # (fh, fw, 448)
    return feat.reshape(-1, feat.shape[2])                  # (P, 448)


# ── toothbrush ROI ───────────────────────────────────────────────────────────

def toothbrush_roi(img_bgr: np.ndarray) -> np.ndarray:
    """
    Return a binary mask (uint8, 0/255) covering the toothbrush body.

    Strategy:

      1. Otsu threshold → isolates bright bristle material from background/gaps.
      2. Morphological close (20×20) → merges individual bristle tips into a
         solid mass (bridges ~10px inter-bristle gaps).
      3. Convex hull of non-zero pixels → fills the enclosed toothbrush shape.
         Fill-holes is not used because inter-bristle gaps are connected to the
         real background through the bristle array edges, so they would not be
         filled by a border-flood-fill approach.
      4. Dilation (20×20) → extends the ROI to include defects sitting exactly
         on the boundary (e.g. missing bristles at the outer edge).

    This mask is applied during threshold tuning so that background pixels
    (always low-anomaly true-negatives) do not dominate the F1 calculation.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    close_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
    closed  = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, close_k)

    coords = np.column_stack(np.where(closed > 0))   # (row, col)
    if len(coords) == 0:
        return np.ones(gray.shape, dtype=np.uint8) * 255   # fallback

    hull = cv2.convexHull(coords[:, ::-1].astype(np.float32))   # (x, y)
    roi  = np.zeros(gray.shape, dtype=np.uint8)
    cv2.fillPoly(roi, [hull.astype(np.int32)], 255)

    dil_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
    return cv2.dilate(roi, dil_k)


# ── image loading ─────────────────────────────────────────────────────────────

def _load_raw(path: str):
    img = cv2.imread(path)
    return img   # keep original size; resize is done inside preprocess

def load_dir(directory: str, suffix: str = "") -> list:
    paths = sorted(
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if f.endswith(".png") and suffix in f
    )
    with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
        imgs = list(ex.map(_load_raw, paths))
    imgs = [img for img in imgs if img is not None]
    print(f"  Loaded {len(imgs)} images from {os.path.basename(directory)}/")
    return imgs


# ── main ──────────────────────────────────────────────────────────────────────

print("\nBuilding feature extractor (ResNet50, ImageNet pretrained) …")
extractor = ResNetExtractor().to(DEVICE)
extractor.eval()

print("\nLoading images …")
good_imgs = load_dir(GOOD_DIR)
def_imgs  = load_dir(DEFECT_DIR)
gt_raw    = load_dir(GT_DIR, suffix="_mask")

gt_masks = []
for m in gt_raw:
    gray = cv2.cvtColor(m, cv2.COLOR_BGR2GRAY) if m.ndim == 3 else m
    gt_masks.append((gray > 127).astype(np.uint8))


# ── Step 1: extract patch features from good images ──────────────────────────
print(f"\n[1/3] Extracting patch features from {len(good_imgs)} good images …")

good_feats = []
for i, img in enumerate(good_imgs):
    feat = extract_features(img, extractor)   # (P, 448)
    good_feats.append(feat)
    if (i + 1) % 10 == 0 or i == len(good_imgs) - 1:
        print(f"  {i + 1}/{len(good_imgs)}")

good_feats_arr = np.stack(good_feats, axis=0)   # (N, P, 448)
N, P, D = good_feats_arr.shape
FEAT_H = FEAT_W = IMG_SIZE // 4
print(f"  Feature array : {good_feats_arr.shape}  "
      f"({good_feats_arr.nbytes / 1e6:.0f} MB)  "
      f"spatial grid {FEAT_H}×{FEAT_W}")


# ── Step 2: PCA + per-position Gaussian ──────────────────────────────────────
print(f"\n[2/3] PCA {D}→{N_COMPONENTS} dims + Gaussian fitting …")

feats_flat = good_feats_arr.reshape(N * P, D)
pca = PCA(n_components=N_COMPONENTS, svd_solver="randomized", random_state=42)
pca.fit(feats_flat)
print(f"  Explained variance : {pca.explained_variance_ratio_.sum():.1%}")

good_feats_pca = pca.transform(feats_flat).reshape(N, P, N_COMPONENTS)  # (N, P, d)

# Per-position mean and full covariance
mean_feat = good_feats_pca.mean(axis=0)      # (P, d)
centered  = good_feats_pca - mean_feat        # (N, P, d)

print("  Computing covariance matrices …")
cov = np.einsum("npi,npj->pij", centered, centered) / max(N - 1, 1)  # (P, d, d)
cov += REG_COEF * np.eye(N_COMPONENTS)[None]   # regularise diagonal

print("  Inverting covariance matrices (batched) …")
precision = np.linalg.inv(cov)   # (P, d, d)  — numpy batches this automatically
print(f"  Precision array : {precision.shape}  ({precision.nbytes / 1e6:.0f} MB)")


# ── Step 3: threshold tuning ─────────────────────────────────────────────────
print(f"\n[3/3] Tuning threshold on {len(def_imgs)} defective + {len(good_imgs)} good images …")


def _score_tile(tile_bgr: np.ndarray) -> np.ndarray:
    """Score a single IMG_SIZE×IMG_SIZE tile → (FEAT_H, FEAT_W) distance map."""
    feat     = extract_features(tile_bgr, extractor)
    feat_pca = (feat - pca.mean_) @ pca.components_.T
    diff     = feat_pca - mean_feat
    tmp      = np.einsum("pi,pij->pj", diff, precision)
    dist2    = np.einsum("pi,pi->p",   tmp,  diff)
    return np.sqrt(np.maximum(dist2, 0.0)).reshape(FEAT_H, FEAT_W)


def score_map_for(img_bgr: np.ndarray) -> np.ndarray:
    """Return (H, W) score map for threshold calibration.

    Uses a single forward pass on the resized image — much faster than tiling
    (1 pass vs 16+) while giving a score distribution close enough to calibrate
    a good threshold. Tiling is used only at inference time (model_v2.py).
    """
    feat     = extract_features(img_bgr, extractor)
    feat_pca = (feat - pca.mean_) @ pca.components_.T
    diff     = feat_pca - mean_feat
    tmp      = np.einsum("pi,pij->pj", diff, precision)
    dist2    = np.einsum("pi,pi->p",   tmp,  diff)
    dist     = np.sqrt(np.maximum(dist2, 0.0)).reshape(FEAT_H, FEAT_W)
    h, w     = img_bgr.shape[:2]
    return cv2.resize(dist, (w, h), interpolation=cv2.INTER_LINEAR)


all_scores = []
all_labels = []

for img, gt in zip(def_imgs, gt_masks):
    score_up = score_map_for(img)
    roi      = toothbrush_roi(img).ravel().astype(bool)
    all_scores.append(score_up.ravel()[roi])
    all_labels.append(gt.ravel()[roi])

for img in good_imgs:
    score_up = score_map_for(img)
    roi      = toothbrush_roi(img).ravel().astype(bool)
    all_scores.append(score_up.ravel()[roi])
    all_labels.append(np.zeros(roi.sum(), dtype=np.uint8))

all_scores = np.concatenate(all_scores).astype(np.float32)
all_labels = np.concatenate(all_labels).astype(np.uint8)
print(f"  Pixels : {len(all_labels):,}   positives : {all_labels.sum():,}")

best_f1, best_thresh = 0.0, 1.0
lo = float(np.percentile(all_scores, 1))
hi = float(np.percentile(all_scores, 99.9))

for t in np.linspace(lo, hi, N_THRESH):
    pred = (all_scores >= t).astype(np.uint8)
    tp = int(((pred == 1) & (all_labels == 1)).sum())
    fp = int(((pred == 1) & (all_labels == 0)).sum())
    fn = int(((pred == 0) & (all_labels == 1)).sum())
    if tp == 0:
        continue
    prec = tp / (tp + fp + 1e-9)
    rec  = tp / (tp + fn + 1e-9)
    f1   = 2 * prec * rec / (prec + rec + 1e-9)
    if f1 > best_f1:
        best_f1, best_thresh = f1, float(t)

print(f"\n  Threshold : {best_thresh:.4f}")
print(f"  F1-score  : {best_f1:.4f}")


# ── save ──────────────────────────────────────────────────────────────────────
np.savez_compressed(
    MODEL_PATH,
    mean_feat      = mean_feat.astype(np.float32),
    precision      = precision.astype(np.float32),
    pca_components = pca.components_.astype(np.float32),   # (d, 448)
    pca_mean       = pca.mean_.astype(np.float32),         # (448,)
    threshold      = np.float32(best_thresh),
    img_size       = np.array([IMG_SIZE, IMG_SIZE]),
    feat_size      = np.array([FEAT_H, FEAT_W]),
)
print(f"\nModel saved → {MODEL_PATH}")
