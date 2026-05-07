"""ResNet50 trunk + BNNeck + classifier head for ReID training.

After training, only the trunk weights (stem + layer1..4) are exported and
loaded by SiamEmbedder for inference. Classifier head is discarded.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class ReIDNet(nn.Module):
    EMBED_DIM = 2048

    def __init__(self, num_classes: int):
        super().__init__()
        bb = torchvision.models.resnet50(
            weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2
        )
        self.stem = nn.Sequential(bb.conv1, bb.bn1, bb.relu, bb.maxpool)
        self.layer1 = bb.layer1
        self.layer2 = bb.layer2
        self.layer3 = bb.layer3
        self.layer4 = bb.layer4
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.bnneck = nn.BatchNorm1d(self.EMBED_DIM)
        nn.init.constant_(self.bnneck.bias, 0)
        self.bnneck.bias.requires_grad_(False)
        self.classifier = nn.Linear(self.EMBED_DIM, num_classes, bias=False)
        nn.init.kaiming_normal_(self.classifier.weight)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x).flatten(1)              # [N, 2048] embedding (pre-BN)
        emb = F.normalize(x, dim=1)              # used for triplet
        feat_bn = self.bnneck(x)
        logits = self.classifier(feat_bn)
        return emb, logits

    def export_backbone_state_dict(self) -> dict:
        """Keys compatible with siamfc._ResNet50Trunk."""
        keep = ("stem.", "layer1.", "layer2.", "layer3.", "layer4.")
        return {k: v for k, v in self.state_dict().items() if k.startswith(keep)}
