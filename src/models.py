"""
CNN models with transfer learning for DeepFake detection.
Supports optional auxiliary inputs (frequency, texture) for artifact-aware classification.
"""

import torch
import torch.nn as nn
from torchvision import models
from typing import Optional, List


def get_backbone(name: str, pretrained: bool = True):
    """Get pretrained backbone and feature dimension."""
    if name == "efficientnet_b0":
        m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None)
        return m.features, m.classifier[1].in_features  # 1280
    if name == "resnet50":
        m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
        return nn.Sequential(*list(m.children())[:-1]), 2048
    if name == "mobilenet_v3_small":
        m = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None)
        return m.features, 576
    raise ValueError(f"Unknown backbone: {name}")


class DeepFakeClassifier(nn.Module):
    """
    Transfer-learning classifier for real vs deepfake.
    Optionally fuses auxiliary artifact features (frequency, texture) via a small head.
    """

    def __init__(
        self,
        backbone_name: str = "efficientnet_b0",
        pretrained: bool = True,
        num_classes: int = 2,
        dropout: float = 0.3,
        use_aux: bool = True,
        aux_channels: int = 2,
        aux_size: int = 56,
    ):
        super().__init__()
        self.use_aux = use_aux
        self.backbone, feat_dim = get_backbone(backbone_name, pretrained)
        self.feat_dim = feat_dim

        # Auxiliary branch: process frequency + texture maps
        if use_aux:
            self.aux_conv = nn.Sequential(
                nn.Conv2d(aux_channels, 32, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
            )
            aux_dim = 32
            classifier_in = feat_dim + aux_dim
        else:
            self.aux_conv = None
            classifier_in = feat_dim

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(classifier_in, num_classes),
        )

    def forward(self, rgb: torch.Tensor, aux: Optional[torch.Tensor] = None):
        features = self.backbone(rgb)  # (B, C, H, W)
        pooled = self.pool(features).flatten(1)

        if self.use_aux and self.aux_conv is not None and aux is not None:
            aux_feat = self.aux_conv(aux)
            pooled = torch.cat([pooled, aux_feat], dim=1)

        return self.classifier(pooled)

    def freeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = True
