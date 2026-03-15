"""
model.py
────────
CNN architecture for radar RF signal classification.

Architecture overview
─────────────────────
RadarCNN is a 4-block convolutional backbone followed by a 2-layer
classification head:

  Input (1, F, T)
  │
  ├─ ConvBlock 1 : Conv2d 1→32,  3×3 | BN | ReLU | MaxPool 2×2
  ├─ ConvBlock 2 : Conv2d 32→64, 3×3 | BN | ReLU | MaxPool 2×2
  ├─ ConvBlock 3 : Conv2d 64→128,3×3 | BN | ReLU | MaxPool 2×2  + ResSkip
  ├─ ConvBlock 4 : Conv2d 128→256,3×3| BN | ReLU | AdaptiveAvgPool
  │
  ├─ Flatten
  ├─ FC 256→128  | BN | ReLU | Dropout(0.4)
  └─ FC 128→num_classes

  Output: logits (N, num_classes)

Two additional lightweight variants are provided:
  • RadarCNN_Lite   – 2-block version for rapid prototyping / CPU runs
  • RadarCNN_Deep   – 5-block version for production / GPU runs
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import logging

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Building blocks
# ──────────────────────────────────────────────────────────────────────────────
class ConvBNReLU(nn.Module):
    """Conv2d → BatchNorm2d → ReLU (standard vision block)."""

    def __init__(
        self,
        in_channels:  int,
        out_channels: int,
        kernel_size:  int = 3,
        stride:       int = 1,
        padding:      int = 1,
    ) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size,
                      stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResidualBlock(nn.Module):
    """
    Lightweight residual block with a 1×1 projection shortcut when
    channel dimensions differ.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = ConvBNReLU(channels, channels)
        self.conv2 = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.act(out + residual)
        return out


# ──────────────────────────────────────────────────────────────────────────────
# Main model
# ──────────────────────────────────────────────────────────────────────────────
class RadarCNN(nn.Module):
    """
    4-block CNN for spectrogram-based radar signal classification.

    Parameters
    ----------
    num_classes : int   – number of signal classes
    input_channels : int – 1 for grayscale spectrograms
    dropout_rate   : float – dropout probability in the FC head
    """

    def __init__(
        self,
        num_classes:    int   = 6,
        input_channels: int   = 1,
        dropout_rate:   float = 0.4,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes

        # ── Feature extractor ────────────────────────────────────────────
        self.features = nn.Sequential(
            # Block 1
            ConvBNReLU(input_channels, 32),
            nn.MaxPool2d(2, 2),                 # ÷2

            # Block 2
            ConvBNReLU(32, 64),
            nn.MaxPool2d(2, 2),                 # ÷4

            # Block 3 + residual skip
            ConvBNReLU(64, 128),
            ResidualBlock(128),
            nn.MaxPool2d(2, 2),                 # ÷8

            # Block 4
            ConvBNReLU(128, 256),
            nn.AdaptiveAvgPool2d((4, 4)),       # fixed 4×4 spatial output
        )

        # ── Classification head ──────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(128, num_classes),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)

    def get_feature_maps(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw feature-map tensor before the classifier head."""
        return self.features(x)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ──────────────────────────────────────────────────────────────────────────────
# Lightweight variant (CPU / quick test)
# ──────────────────────────────────────────────────────────────────────────────
class RadarCNN_Lite(nn.Module):
    """2-block CNN for rapid prototyping or CPU-only runs."""

    def __init__(self, num_classes: int = 6, input_channels: int = 1) -> None:
        super().__init__()
        self.features = nn.Sequential(
            ConvBNReLU(input_channels, 32),
            nn.MaxPool2d(2, 2),
            ConvBNReLU(32, 64),
            nn.AdaptiveAvgPool2d((8, 8)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity="relu")
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ──────────────────────────────────────────────────────────────────────────────
# Deep variant (GPU / production)
# ──────────────────────────────────────────────────────────────────────────────
class RadarCNN_Deep(nn.Module):
    """5-block CNN with two residual stages for maximum accuracy."""

    def __init__(
        self,
        num_classes:  int   = 6,
        input_channels: int = 1,
        dropout_rate: float = 0.5,
    ) -> None:
        super().__init__()
        self.features = nn.Sequential(
            ConvBNReLU(input_channels, 32),
            nn.MaxPool2d(2, 2),

            ConvBNReLU(32, 64),
            nn.MaxPool2d(2, 2),

            ConvBNReLU(64, 128),
            ResidualBlock(128),
            nn.MaxPool2d(2, 2),

            ConvBNReLU(128, 256),
            ResidualBlock(256),
            nn.MaxPool2d(2, 2),

            ConvBNReLU(256, 512),
            nn.AdaptiveAvgPool2d((2, 2)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 2 * 2, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(1024, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(256, num_classes),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ──────────────────────────────────────────────────────────────────────────────
# Factory
# ──────────────────────────────────────────────────────────────────────────────
def get_model(
    variant:     str = "standard",
    num_classes: int = 6,
) -> nn.Module:
    """
    Return a model instance by name.

    Parameters
    ----------
    variant     : 'standard' | 'lite' | 'deep'
    num_classes : int

    Returns
    -------
    nn.Module
    """
    registry = {
        "standard": RadarCNN,
        "lite":     RadarCNN_Lite,
        "deep":     RadarCNN_Deep,
    }
    if variant not in registry:
        raise ValueError(f"Unknown variant '{variant}'. Choose from {list(registry)}")
    model = registry[variant](num_classes=num_classes)
    logger.info(
        "Built RadarCNN-%s | classes=%d | params=%s",
        variant, num_classes, f"{model.count_parameters():,}",
    )
    return model