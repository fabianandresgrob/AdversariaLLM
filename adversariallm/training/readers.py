"""Detector readers for the cooperative pipeline.

A reader maps a target-model hidden-state tensor to a harmful/benign score. v1 is a
linear probe on the residual-stream activation at the last prompt token — the same
readout position the vendored activation detector uses, so it reuses build_detector_batch
unchanged. Richer readers (MLP, the gemma detector) slot in behind the same interface.

Label convention (from the predecessor detector): logits column 0 = harmful ("yes"),
column 1 = benign ("no").
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn


def readout_index(target_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Index of the last prompt token per row (the token before the response begins).

    Right-padded layout, matching build_detector_batch:
        target_ids:     [ 0 … 0 | response ids | 0 … 0 (pad) ]
        attention_mask: [ 1 … 1 | 1 …………… 1 | 0 … 0 (pad) ]
    The response starts at the first nonzero target id, so the last prompt token is one
    before it. Deriving the position from target_ids/attention_mask (never hidden[:, -1])
    keeps it correct under padding. A prompt-only row (no response region) falls back to
    its last attended token.
    """
    resp = target_ids != 0                                  # (B, T)
    has_resp = resp.any(dim=1)                               # (B,)
    first_resp = resp.float().argmax(dim=1)                  # (B,) 0 if no response
    last_real = attention_mask.long().sum(dim=1) - 1         # (B,)
    return torch.where(has_resp, first_resp - 1, last_real).clamp_min(0)


class Reader(ABC):
    """Scores a prompt as harmful/benign from target-model hidden states.

    logits(hidden, target_ids, attention_mask) -> (B, 2); column 0 = harmful.
    hidden is (B, T, D) at a single chosen layer (the loop selects the layer).
    """

    @abstractmethod
    def logits(self, hidden: torch.Tensor, target_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        ...

    def p_harmful(self, hidden: torch.Tensor, target_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """P(harmful) per row = softmax(logits)[:, 0]."""
        return torch.softmax(self.logits(hidden, target_ids, attention_mask).float(), dim=-1)[:, 0]


class LinearProbe(Reader, nn.Module):
    """Linear probe on the last-prompt-token activation.

    The readout vector is taken in fp32 and unit-normed: activation norms drift during
    fine-tuning, so the probe reads direction, not scale (and stays fp32 even when the
    target model runs in bf16).
    """

    def __init__(self, input_dim: int, eps: float = 1e-6):
        nn.Module.__init__(self)
        self.eps = eps
        self.linear = nn.Linear(input_dim, 2)  # fp32 params

    def readout(self, hidden: torch.Tensor, target_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        idx = readout_index(target_ids, attention_mask)
        rows = torch.arange(hidden.size(0), device=hidden.device)
        h = hidden[rows, idx].float()                        # (B, D)
        return h / h.norm(dim=-1, keepdim=True).clamp_min(self.eps)

    def logits(self, hidden: torch.Tensor, target_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self.linear(self.readout(hidden, target_ids, attention_mask))


def build_reader(cfg: dict | None, input_dim: int) -> Reader:
    """Construct the reader from config. v1 supports "linear"; MLP/gemma slot in here."""
    kind = (cfg or {}).get("type", "linear")
    if kind == "linear":
        return LinearProbe(input_dim)
    raise ValueError(f"unknown reader type: {kind!r}")
