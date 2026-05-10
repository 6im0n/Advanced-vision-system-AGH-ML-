"""ConvNeXt-Small trunk + BNNeck + classifier head for ReID training.

After training, only the trunk weights (features + classifier[0] LayerNorm)
are exported and loaded by SiamEmbedder for inference. The BNNeck and
classifier head are discarded.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class ReIDNet(nn.Module):
    EMBED_DIM = 768

    def __init__(self, num_classes: int):
        super().__init__()
        bb = torchvision.models.convnext_small(
            weights=torchvision.models.ConvNeXt_Small_Weights.IMAGENET1K_V1
        )
        self.features = bb.features
        self.avgpool = bb.avgpool
        self.norm = bb.classifier[0]              # LayerNorm2d
        self.flatten = bb.classifier[1]
        self.bnneck = nn.BatchNorm1d(self.EMBED_DIM)
        nn.init.constant_(self.bnneck.bias, 0)
        self.bnneck.bias.requires_grad_(False)
        self.classifier = nn.Linear(self.EMBED_DIM, num_classes, bias=False)
        nn.init.kaiming_normal_(self.classifier.weight)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = self.norm(x)
        x = self.flatten(x)                       # [N, 768]
        emb = F.normalize(x, dim=1)
        feat_bn = self.bnneck(x)
        logits = self.classifier(feat_bn)
        return emb, logits

    def export_backbone_state_dict(self) -> dict:
        """Keys compatible with siamfc._ConvNeXtSmallTrunk."""
        keep = ("features.", "norm.")
        return {k: v for k, v in self.state_dict().items() if k.startswith(keep)}
