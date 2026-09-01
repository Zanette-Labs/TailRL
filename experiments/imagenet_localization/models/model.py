from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models


class LocalizationPolicy(nn.Module):
    """ResNet-50 backbone (spatial map preserved) + 1x1 spatial bottleneck +
    4 independent linear heads for bin classification.

    The spec patch (spec_patch_architecture.md) strips the AdaptiveAvgPool2d so
    the (B, 2048, 7, 7) spatial feature map flows into a 1x1 conv bottleneck
    (2048 -> 64 channels), which is then flattened to (B, 64*7*7) = (B, 3136).
    Each of 4 independent nn.Linear(3136, K) heads reads the flattened spatial
    features. This preserves spatial information, which is essential for
    localization (the previous arch globally pooled it away).
    """

    HEAD_NAMES: tuple[str, ...] = ("x_c", "y_c", "w", "h")
    BACKBONE_CHANNELS: int = 2048    # ResNet-50 output channels at layer4
    SPATIAL_DIM: int = 7             # ResNet-50 spatial size for 224x224 input
    REDUCED_CHANNELS: int = 64       # 1x1 conv bottleneck output channels
    FEATURE_DIM: int = REDUCED_CHANNELS * SPATIAL_DIM * SPATIAL_DIM  # 3136

    def __init__(self, K: int, pretrained: bool = True, seed: int = 42):
        """
        Args:
            K:          number of bins per coordinate (e.g., 50)
            pretrained: whether to load ImageNet-pretrained ResNet-50 weights
                        (IMAGENET1K_V2). For offline tests, set False.
            seed:       seed used to initialize spatial_reduce + heads
                        reproducibly across seed sweeps. (Backbone init is
                        determined by the torchvision weights / default init.)
        """
        super().__init__()
        self.K = K
        self.pretrained = pretrained

        # --- Backbone: ResNet-50 up to (but not including) avgpool + fc ---
        # children()[:-2] drops the final AdaptiveAvgPool2d AND the Linear
        # classifier, keeping the (B, 2048, 7, 7) spatial map.
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        backbone = models.resnet50(weights=weights)
        self.features = nn.Sequential(*list(backbone.children())[:-2])

        # --- Spatial bottleneck: 1x1 conv (2048 -> 64) + ReLU + Flatten ---
        # NOT part of self.features so it remains trainable during warmup
        # (heads-only phase) — this is what gives the heads-only phase actual
        # localization signal that the old arch lacked.
        self.spatial_reduce = nn.Sequential(
            nn.Conv2d(self.BACKBONE_CHANNELS, self.REDUCED_CHANNELS, kernel_size=1),
            nn.ReLU(),
            nn.Flatten(),  # (B, 64, 7, 7) -> (B, 3136)
        )

        # --- 4 heads: each nn.Linear(3136, K), seeded for reproducibility ---
        gen = torch.Generator().manual_seed(seed)
        self.heads = nn.ModuleDict()
        for h in self.HEAD_NAMES:
            lin = nn.Linear(self.FEATURE_DIM, K)
            with torch.no_grad():
                nn.init.kaiming_uniform_(lin.weight, a=5 ** 0.5, generator=gen)
                bound = 1.0 / (self.FEATURE_DIM ** 0.5)
                nn.init.uniform_(lin.bias, -bound, bound, generator=gen)
            self.heads[h] = lin

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Args:
            x: (B, 3, 224, 224) image batch
        Returns:
            dict with keys 'x_c', 'y_c', 'w', 'h', each mapping to (B, K) logits.
        """
        feat_map = self.features(x)              # (B, 2048, 7, 7)
        f = self.spatial_reduce(feat_map)        # (B, 3136)
        return {h: self.heads[h](f) for h in self.HEAD_NAMES}

    def freeze_backbone(self) -> None:
        """Freeze all parameters in self.features (head-only warmup phase).
        Note: spatial_reduce and heads remain trainable."""
        for p in self.features.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self) -> None:
        """Unfreeze all parameters in self.features."""
        for p in self.features.parameters():
            p.requires_grad = True


class LocalizationRegressor(nn.Module):
    """MSE regression baseline: ResNet-50 (spatial map preserved) + 1x1 conv
    bottleneck + 4-way linear with sigmoid output.

    Same architecture changes as LocalizationPolicy (spatial info preserved via
    1x1 conv bottleneck rather than destroyed by AdaptiveAvgPool2d).
    Predicts (x_c, y_c, w, h) in [0, 1]^4 directly (no bins, no rollouts).
    """

    BACKBONE_CHANNELS: int = 2048
    SPATIAL_DIM: int = 7
    REDUCED_CHANNELS: int = 64
    FEATURE_DIM: int = REDUCED_CHANNELS * SPATIAL_DIM * SPATIAL_DIM  # 3136

    def __init__(self, pretrained: bool = True, seed: int = 42):
        super().__init__()
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        backbone = models.resnet50(weights=weights)
        self.features = nn.Sequential(*list(backbone.children())[:-2])
        self.spatial_reduce = nn.Sequential(
            nn.Conv2d(self.BACKBONE_CHANNELS, self.REDUCED_CHANNELS, kernel_size=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        gen = torch.Generator().manual_seed(seed)
        self.head = nn.Linear(self.FEATURE_DIM, 4)
        with torch.no_grad():
            nn.init.kaiming_uniform_(self.head.weight, a=5 ** 0.5, generator=gen)
            bound = 1.0 / (self.FEATURE_DIM ** 0.5)
            nn.init.uniform_(self.head.bias, -bound, bound, generator=gen)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, 3, 224, 224) -> (B, 4) predicted xywh in [0, 1]^4 (sigmoid)."""
        feat_map = self.features(x)
        f = self.spatial_reduce(feat_map)
        return torch.sigmoid(self.head(f))

    def freeze_backbone(self) -> None:
        for p in self.features.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self) -> None:
        for p in self.features.parameters():
            p.requires_grad = True
