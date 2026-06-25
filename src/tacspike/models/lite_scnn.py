from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
from torch import nn
from torch.nn import functional as F


class SurrogateSpike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, alpha: float) -> torch.Tensor:
        ctx.save_for_backward(x)
        ctx.alpha = alpha
        return (x >= 0).to(x.dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        (x,) = ctx.saved_tensors
        alpha = ctx.alpha
        grad = grad_output / (alpha * x.abs() + 1.0).pow(2)
        return grad, None


def spike_fn(x: torch.Tensor, alpha: float = 2.0) -> torch.Tensor:
    return SurrogateSpike.apply(x, alpha)


@dataclass
class LIFState:
    mem: torch.Tensor


class LIFCell(nn.Module):
    def __init__(self, beta: float = 0.85, threshold: float = 1.0, surrogate_alpha: float = 2.0) -> None:
        super().__init__()
        self.beta = float(beta)
        self.threshold = float(threshold)
        self.surrogate_alpha = float(surrogate_alpha)

    def forward(self, input_current: torch.Tensor, state: LIFState | None = None) -> Tuple[torch.Tensor, LIFState]:
        if state is None:
            mem = torch.zeros_like(input_current)
        else:
            mem = state.mem
        mem = self.beta * mem + input_current
        spike = spike_fn(mem - self.threshold, self.surrogate_alpha)
        mem = mem - spike.detach() * self.threshold
        return spike, LIFState(mem=mem)


class TacSpikeLiteSCNN(nn.Module):
    """Lightweight LIF SCNN for [B, T, 2, 32, 32] TacSpike windows."""

    def __init__(
        self,
        input_channels: int = 2,
        num_classes: int = 2,
        beta: float = 0.85,
        threshold: float = 1.0,
        surrogate_alpha: float = 2.0,
        readout: str = "spike_count",
    ) -> None:
        super().__init__()
        if readout not in {"spike_count", "membrane", "logit_mean", "logit_sum"}:
            raise ValueError(f"Unsupported readout={readout!r}")
        self.readout = readout
        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=5, stride=1, padding=2, bias=False)
        self.lif1 = LIFCell(beta=beta, threshold=threshold, surrogate_alpha=surrogate_alpha)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.lif2 = LIFCell(beta=beta, threshold=threshold, surrogate_alpha=surrogate_alpha)
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.fc1 = nn.Linear(32 * 4 * 4, 64, bias=False)
        self.lif3 = LIFCell(beta=beta, threshold=threshold, surrogate_alpha=surrogate_alpha)
        self.fc2 = nn.Linear(64, num_classes, bias=False)
        self.out_lif = LIFCell(beta=beta, threshold=threshold, surrogate_alpha=surrogate_alpha)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if x.ndim != 5:
            raise ValueError(f"Expected x shape [B, T, C, H, W], got {tuple(x.shape)}")
        batch, steps = x.shape[0], x.shape[1]
        state1 = state2 = state3 = state_out = None
        out_spikes = []
        out_mems = []
        raw_logits = []
        firing_sums = {
            "lif1": x.new_tensor(0.0),
            "lif2": x.new_tensor(0.0),
            "lif3": x.new_tensor(0.0),
            "out": x.new_tensor(0.0),
        }
        firing_counts = {
            "lif1": 0,
            "lif2": 0,
            "lif3": 0,
            "out": 0,
        }

        for t in range(steps):
            z = self.conv1(x[:, t])
            s1, state1 = self.lif1(z, state1)
            z = self.conv2(s1)
            s2, state2 = self.lif2(z, state2)
            z = self.pool(s2).flatten(1)
            z = self.fc1(z)
            s3, state3 = self.lif3(z, state3)
            z = self.fc2(s3)
            raw_logits.append(z)
            sout, state_out = self.out_lif(z, state_out)
            out_spikes.append(sout)
            out_mems.append(state_out.mem)

            firing_sums["lif1"] = firing_sums["lif1"] + s1.detach().sum()
            firing_sums["lif2"] = firing_sums["lif2"] + s2.detach().sum()
            firing_sums["lif3"] = firing_sums["lif3"] + s3.detach().sum()
            firing_sums["out"] = firing_sums["out"] + sout.detach().sum()
            firing_counts["lif1"] += s1.numel()
            firing_counts["lif2"] += s2.numel()
            firing_counts["lif3"] += s3.numel()
            firing_counts["out"] += sout.numel()

        spike_seq = torch.stack(out_spikes, dim=1)
        spike_count = spike_seq.sum(dim=1)
        raw_logit_seq = torch.stack(raw_logits, dim=1)
        if self.readout == "membrane":
            logits = torch.stack(out_mems, dim=1).mean(dim=1)
        elif self.readout == "logit_mean":
            logits = raw_logit_seq.mean(dim=1)
        elif self.readout == "logit_sum":
            logits = raw_logit_seq.sum(dim=1)
        else:
            logits = spike_count
        stats = {
            f"{name}_firing_rate": firing_sums[name] / max(firing_counts[name], 1)
            for name in firing_sums
        }
        stats["output_spike_mean"] = spike_count.detach().mean()
        return logits, stats


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
