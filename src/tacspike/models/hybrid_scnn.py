from __future__ import annotations

from typing import Dict, Tuple

import torch
from torch import nn

from .lite_scnn import LIFCell


class TacSpikeTimeChannelSCNN(nn.Module):
    """Time-channel convolutional SNN with LIF hidden activations.

    The input window [B, T, C, H, W] is flattened into [B, T*C, H, W],
    matching the strongest FrameCNN input representation while keeping
    spiking LIF hidden layers.
    """

    def __init__(
        self,
        input_channels: int = 2,
        time_steps: int = 100,
        num_classes: int = 2,
        beta: float = 0.85,
        threshold: float = 0.1,
        surrogate_alpha: float = 2.0,
        width: int = 32,
        hidden: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        in_channels = input_channels * time_steps
        self.conv1 = nn.Conv2d(in_channels, width, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width)
        self.lif1 = LIFCell(beta=beta, threshold=threshold, surrogate_alpha=surrogate_alpha)
        self.conv2 = nn.Conv2d(width, width * 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width * 2)
        self.lif2 = LIFCell(beta=beta, threshold=threshold, surrogate_alpha=surrogate_alpha)
        self.conv3 = nn.Conv2d(width * 2, width * 4, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width * 4)
        self.lif3 = LIFCell(beta=beta, threshold=threshold, surrogate_alpha=surrogate_alpha)
        self.conv4 = nn.Conv2d(width * 4, width * 4, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(width * 4)
        self.lif4 = LIFCell(beta=beta, threshold=threshold, surrogate_alpha=surrogate_alpha)
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.fc1 = nn.Linear(width * 4 * 4 * 4, hidden, bias=False)
        self.bn_fc = nn.BatchNorm1d(hidden)
        self.lif5 = LIFCell(beta=beta, threshold=threshold, surrogate_alpha=surrogate_alpha)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden, num_classes)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if x.ndim != 5:
            raise ValueError(f"Expected x shape [B, T, C, H, W], got {tuple(x.shape)}")
        z = x.flatten(1, 2)
        z = self.bn1(self.conv1(z))
        s1, _ = self.lif1(z)
        z = self.bn2(self.conv2(s1))
        s2, _ = self.lif2(z)
        z = self.bn3(self.conv3(s2))
        s3, _ = self.lif3(z)
        z = self.bn4(self.conv4(s3))
        s4, _ = self.lif4(z)
        z = self.pool(s4).flatten(1)
        z = self.bn_fc(self.fc1(z))
        s5, _ = self.lif5(z)
        logits = self.fc2(self.dropout(s5))
        stats = {
            "lif1_firing_rate": s1.detach().mean(),
            "lif2_firing_rate": s2.detach().mean(),
            "lif3_firing_rate": s3.detach().mean(),
            "lif4_firing_rate": s4.detach().mean(),
            "lif5_firing_rate": s5.detach().mean(),
            "output_spike_mean": x.new_tensor(0.0),
        }
        return logits, stats


class TacSpikeTemporalConvSCNN(nn.Module):
    """3D temporal-convolution SNN with LIF hidden activations."""

    def __init__(
        self,
        input_channels: int = 2,
        num_classes: int = 2,
        beta: float = 0.85,
        threshold: float = 0.1,
        surrogate_alpha: float = 2.0,
        width: int = 16,
        hidden: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv3d(
            input_channels,
            width,
            kernel_size=(5, 3, 3),
            stride=(1, 1, 1),
            padding=(2, 1, 1),
            bias=False,
        )
        self.bn1 = nn.BatchNorm3d(width)
        self.lif1 = LIFCell(beta=beta, threshold=threshold, surrogate_alpha=surrogate_alpha)
        self.conv2 = nn.Conv3d(
            width,
            width * 2,
            kernel_size=(5, 3, 3),
            stride=(2, 2, 2),
            padding=(2, 1, 1),
            bias=False,
        )
        self.bn2 = nn.BatchNorm3d(width * 2)
        self.lif2 = LIFCell(beta=beta, threshold=threshold, surrogate_alpha=surrogate_alpha)
        self.conv3 = nn.Conv3d(
            width * 2,
            width * 4,
            kernel_size=(3, 3, 3),
            stride=(2, 2, 2),
            padding=(1, 1, 1),
            bias=False,
        )
        self.bn3 = nn.BatchNorm3d(width * 4)
        self.lif3 = LIFCell(beta=beta, threshold=threshold, surrogate_alpha=surrogate_alpha)
        self.pool = nn.AdaptiveAvgPool3d((4, 4, 4))
        self.fc1 = nn.Linear(width * 4 * 4 * 4 * 4, hidden, bias=False)
        self.bn_fc = nn.BatchNorm1d(hidden)
        self.lif4 = LIFCell(beta=beta, threshold=threshold, surrogate_alpha=surrogate_alpha)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden, num_classes)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if x.ndim != 5:
            raise ValueError(f"Expected x shape [B, T, C, H, W], got {tuple(x.shape)}")
        z = x.permute(0, 2, 1, 3, 4).contiguous()
        z = self.bn1(self.conv1(z))
        s1, _ = self.lif1(z)
        z = self.bn2(self.conv2(s1))
        s2, _ = self.lif2(z)
        z = self.bn3(self.conv3(s2))
        s3, _ = self.lif3(z)
        z = self.pool(s3).flatten(1)
        z = self.bn_fc(self.fc1(z))
        s4, _ = self.lif4(z)
        logits = self.fc2(self.dropout(s4))
        stats = {
            "lif1_firing_rate": s1.detach().mean(),
            "lif2_firing_rate": s2.detach().mean(),
            "lif3_firing_rate": s3.detach().mean(),
            "lif4_firing_rate": s4.detach().mean(),
            "output_spike_mean": x.new_tensor(0.0),
        }
        return logits, stats
