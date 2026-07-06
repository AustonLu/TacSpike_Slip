from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

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
        conv1_channels: int = 16,
        conv2_channels: int = 32,
        hidden: int = 64,
        readout_start_frac: float = 0.0,
    ) -> None:
        super().__init__()
        if readout not in {"spike_count", "membrane", "logit_mean", "logit_sum"}:
            raise ValueError(f"Unsupported readout={readout!r}")
        self.readout = readout
        self.readout_start_frac = float(readout_start_frac)
        if not 0.0 <= self.readout_start_frac < 1.0:
            raise ValueError(f"readout_start_frac must be in [0, 1), got {readout_start_frac}")
        self.conv1 = nn.Conv2d(input_channels, conv1_channels, kernel_size=5, stride=1, padding=2, bias=False)
        self.lif1 = LIFCell(beta=beta, threshold=threshold, surrogate_alpha=surrogate_alpha)
        self.conv2 = nn.Conv2d(conv1_channels, conv2_channels, kernel_size=3, stride=2, padding=1, bias=False)
        self.lif2 = LIFCell(beta=beta, threshold=threshold, surrogate_alpha=surrogate_alpha)
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.fc1 = nn.Linear(conv2_channels * 4 * 4, hidden, bias=False)
        self.lif3 = LIFCell(beta=beta, threshold=threshold, surrogate_alpha=surrogate_alpha)
        self.fc2 = nn.Linear(hidden, num_classes, bias=False)
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
        raw_logit_seq = torch.stack(raw_logits, dim=1)
        out_mem_seq = torch.stack(out_mems, dim=1)
        readout_start = int(steps * self.readout_start_frac)
        spike_readout = spike_seq[:, readout_start:]
        raw_logit_readout = raw_logit_seq[:, readout_start:]
        out_mem_readout = out_mem_seq[:, readout_start:]
        spike_count = spike_readout.sum(dim=1)
        if self.readout == "membrane":
            logits = out_mem_readout.mean(dim=1)
        elif self.readout == "logit_mean":
            logits = raw_logit_readout.mean(dim=1)
        elif self.readout == "logit_sum":
            logits = raw_logit_readout.sum(dim=1)
        else:
            logits = spike_count
        stats = {
            f"{name}_firing_rate": firing_sums[name] / max(firing_counts[name], 1)
            for name in firing_sums
        }
        stats["output_spike_mean"] = spike_count.detach().mean()
        return logits, stats


class TacSpikeStreamingLiteSCNN(nn.Module):
    """Stateful Lite-SCNN that emits one logit per input millisecond."""

    def __init__(
        self,
        input_channels: int = 2,
        num_classes: int = 2,
        beta: float = 0.85,
        threshold: float = 0.1,
        surrogate_alpha: float = 2.0,
        hidden: int = 64,
        conv1_channels: int = 16,
        conv2_channels: int = 32,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.conv1_channels = int(conv1_channels)
        self.conv2_channels = int(conv2_channels)
        self.hidden = int(hidden)
        self.conv1 = nn.Conv2d(input_channels, self.conv1_channels, kernel_size=5, stride=1, padding=2, bias=False)
        self.lif1 = LIFCell(beta=beta, threshold=threshold, surrogate_alpha=surrogate_alpha)
        self.conv2 = nn.Conv2d(self.conv1_channels, self.conv2_channels, kernel_size=3, stride=2, padding=1, bias=False)
        self.lif2 = LIFCell(beta=beta, threshold=threshold, surrogate_alpha=surrogate_alpha)
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.fc1 = nn.Linear(self.conv2_channels * 4 * 4, self.hidden, bias=False)
        self.lif3 = LIFCell(beta=beta, threshold=threshold, surrogate_alpha=surrogate_alpha)
        self.dropout = nn.Dropout(float(dropout))
        self.fc2 = nn.Linear(self.hidden, num_classes)

    def step(
        self,
        x_t: torch.Tensor,
        state: Tuple[LIFState | None, LIFState | None, LIFState | None] | None = None,
    ) -> Tuple[torch.Tensor, Tuple[LIFState, LIFState, LIFState], Dict[str, torch.Tensor]]:
        if state is None:
            state1 = state2 = state3 = None
        else:
            state1, state2, state3 = state
        z = self.conv1(x_t)
        s1, state1 = self.lif1(z, state1)
        z = self.conv2(s1)
        s2, state2 = self.lif2(z, state2)
        z = self.pool(s2).flatten(1)
        z = self.fc1(z)
        s3, state3 = self.lif3(z, state3)
        logits = self.fc2(self.dropout(s3))
        stats = {
            "lif1_firing_rate": s1.detach().mean(),
            "lif2_firing_rate": s2.detach().mean(),
            "lif3_firing_rate": s3.detach().mean(),
        }
        return logits, (state1, state2, state3), stats

    def forward(
        self,
        x: torch.Tensor,
        state: Tuple[LIFState | None, LIFState | None, LIFState | None] | None = None,
        return_state: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]] | Tuple[
        torch.Tensor,
        Dict[str, torch.Tensor],
        Tuple[LIFState, LIFState, LIFState],
    ]:
        if x.ndim != 5:
            raise ValueError(f"Expected x shape [B, T, C, H, W], got {tuple(x.shape)}")
        logits = []
        firing_totals = {
            "lif1_firing_rate": x.new_tensor(0.0),
            "lif2_firing_rate": x.new_tensor(0.0),
            "lif3_firing_rate": x.new_tensor(0.0),
        }
        for t in range(x.shape[1]):
            logit_t, state, stats = self.step(x[:, t], state)
            logits.append(logit_t)
            for key, value in stats.items():
                firing_totals[key] = firing_totals[key] + value
        logit_seq = torch.stack(logits, dim=1)
        stats = {key: value / max(x.shape[1], 1) for key, value in firing_totals.items()}
        if return_state:
            return logit_seq, stats, state
        return logit_seq, stats


class TacSpikeMultiTauStreamingSCNN(nn.Module):
    """Multi-timescale streaming SCNN with parallel LIF branches."""

    def __init__(
        self,
        input_channels: int = 2,
        num_classes: int = 2,
        betas: Iterable[float] = (0.65, 0.85, 0.95),
        threshold: float = 0.1,
        surrogate_alpha: float = 2.0,
        hidden: int = 64,
        conv1_channels: int = 16,
        conv2_channels: int = 32,
        dropout: float = 0.0,
        fusion: str = "mean",
    ) -> None:
        super().__init__()
        if fusion not in {"mean", "linear"}:
            raise ValueError(f"Unsupported fusion={fusion!r}")
        self.fusion = fusion
        self.betas = tuple(float(beta) for beta in betas)
        if not self.betas:
            raise ValueError("betas must contain at least one value")
        self.branches = nn.ModuleList(
            [
                TacSpikeStreamingLiteSCNN(
                    input_channels=input_channels,
                    num_classes=num_classes,
                    beta=beta,
                    threshold=threshold,
                    surrogate_alpha=surrogate_alpha,
                    hidden=hidden,
                    conv1_channels=conv1_channels,
                    conv2_channels=conv2_channels,
                    dropout=dropout,
                )
                for beta in self.betas
            ]
        )
        self.fusion_layer = nn.Linear(len(self.betas) * num_classes, num_classes) if fusion == "linear" else None

    def forward(
        self,
        x: torch.Tensor,
        state: Tuple[Tuple[LIFState | None, LIFState | None, LIFState | None] | None, ...] | None = None,
        return_state: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]] | Tuple[
        torch.Tensor,
        Dict[str, torch.Tensor],
        Tuple[Tuple[LIFState, LIFState, LIFState], ...],
    ]:
        branch_logits = []
        next_states = []
        stats: Dict[str, torch.Tensor] = {}
        if state is None:
            state = tuple(None for _ in self.branches)
        for idx, (branch, branch_state) in enumerate(zip(self.branches, state)):
            logits_i, stats_i, state_i = branch(x, state=branch_state, return_state=True)
            branch_logits.append(logits_i)
            next_states.append(state_i)
            for key, value in stats_i.items():
                stats[f"branch{idx}_{key}"] = value
        stacked = torch.stack(branch_logits, dim=2)
        if self.fusion == "mean":
            logits = stacked.mean(dim=2)
        else:
            logits = self.fusion_layer(stacked.flatten(2))
        if return_state:
            return logits, stats, tuple(next_states)
        return logits, stats


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
