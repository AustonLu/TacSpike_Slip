from __future__ import annotations

from typing import Dict, Tuple

import torch
from torch import nn

from .lite_scnn import LIFCell


class TacSpikeDeepSCNN(nn.Module):
    """Stronger LIF-SCNN for accuracy exploration.

    This keeps LIF hidden dynamics but adds BatchNorm and a wider three-block
    convolutional stem. The readout uses raw logits over time because prior
    output spike/membrane readouts underfit this dataset.
    """

    def __init__(
        self,
        input_channels: int = 2,
        num_classes: int = 2,
        beta: float = 0.85,
        threshold: float = 0.1,
        surrogate_alpha: float = 2.0,
        width: int = 32,
        hidden: int = 128,
        readout: str = "logit_mean",
    ) -> None:
        super().__init__()
        if readout not in {"logit_mean", "logit_sum"}:
            raise ValueError(f"Unsupported readout={readout!r}")
        self.readout = readout
        self.conv1 = nn.Conv2d(input_channels, width, kernel_size=5, stride=1, padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(width)
        self.lif1 = LIFCell(beta=beta, threshold=threshold, surrogate_alpha=surrogate_alpha)
        self.conv2 = nn.Conv2d(width, width * 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width * 2)
        self.lif2 = LIFCell(beta=beta, threshold=threshold, surrogate_alpha=surrogate_alpha)
        self.conv3 = nn.Conv2d(width * 2, width * 4, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width * 4)
        self.lif3 = LIFCell(beta=beta, threshold=threshold, surrogate_alpha=surrogate_alpha)
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.fc1 = nn.Linear(width * 4 * 4 * 4, hidden, bias=False)
        self.bn_fc = nn.BatchNorm1d(hidden)
        self.lif4 = LIFCell(beta=beta, threshold=threshold, surrogate_alpha=surrogate_alpha)
        self.fc2 = nn.Linear(hidden, num_classes)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if x.ndim != 5:
            raise ValueError(f"Expected x shape [B, T, C, H, W], got {tuple(x.shape)}")
        state1 = state2 = state3 = state4 = None
        raw_logits = []
        firing_sums = {
            "lif1": x.new_tensor(0.0),
            "lif2": x.new_tensor(0.0),
            "lif3": x.new_tensor(0.0),
            "lif4": x.new_tensor(0.0),
        }
        firing_counts = {name: 0 for name in firing_sums}

        for t in range(x.shape[1]):
            z = self.bn1(self.conv1(x[:, t]))
            s1, state1 = self.lif1(z, state1)
            z = self.bn2(self.conv2(s1))
            s2, state2 = self.lif2(z, state2)
            z = self.bn3(self.conv3(s2))
            s3, state3 = self.lif3(z, state3)
            z = self.pool(s3).flatten(1)
            z = self.bn_fc(self.fc1(z))
            s4, state4 = self.lif4(z, state4)
            raw_logits.append(self.fc2(s4))

            for name, spikes in (("lif1", s1), ("lif2", s2), ("lif3", s3), ("lif4", s4)):
                firing_sums[name] = firing_sums[name] + spikes.detach().sum()
                firing_counts[name] += spikes.numel()

        logit_seq = torch.stack(raw_logits, dim=1)
        logits = logit_seq.mean(dim=1) if self.readout == "logit_mean" else logit_seq.sum(dim=1)
        stats = {
            f"{name}_firing_rate": firing_sums[name] / max(firing_counts[name], 1)
            for name in firing_sums
        }
        stats["output_spike_mean"] = x.new_tensor(0.0)
        return logits, stats
