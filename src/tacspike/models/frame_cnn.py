from __future__ import annotations

from typing import Dict, Tuple

import torch
from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TacSpikeFrameCNN(nn.Module):
    """Small non-spiking CNN upper-bound for TacSpike windows.

    Input is the same [B, T, C, H, W] tensor used by the SNN. The default
    temporal mode flattens time and polarity into image channels, preserving
    1 ms bin identity while avoiding recurrent dynamics.
    """

    def __init__(
        self,
        input_channels: int = 2,
        time_steps: int = 20,
        num_classes: int = 2,
        width: int = 32,
        temporal_mode: str = "time_channels",
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if temporal_mode not in {"time_channels", "sum"}:
            raise ValueError(f"Unsupported temporal_mode={temporal_mode!r}")
        self.temporal_mode = temporal_mode
        in_channels = input_channels * time_steps if temporal_mode == "time_channels" else input_channels
        self.features = nn.Sequential(
            ConvBlock(in_channels, width, stride=1),
            ConvBlock(width, width * 2, stride=2),
            ConvBlock(width * 2, width * 4, stride=2),
            ConvBlock(width * 4, width * 4, stride=1),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(width * 4 * 4 * 4, width * 4),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(width * 4, num_classes),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if x.ndim != 5:
            raise ValueError(f"Expected x shape [B, T, C, H, W], got {tuple(x.shape)}")
        if self.temporal_mode == "time_channels":
            x = x.flatten(1, 2)
        else:
            x = x.sum(dim=1)
        features = self.features(x)
        logits = self.classifier(features)
        stats = {
            "feature_abs_mean": features.detach().abs().mean(),
            "feature_nonzero_frac": (features.detach() != 0).float().mean(),
        }
        return logits, stats
