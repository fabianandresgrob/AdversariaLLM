"""Loss primitives for the cooperative pipeline (design doc §4, §5).

The model loss and detector loss are assembled in coop_loop.py, which owns the model
forwards and the two-timescale schedule. This module holds the reusable pieces:

    model loss  = lambda_kl * utility_kl                         (from losses.py, UltraChat)
                + lambda_beh * behavior_gate * per_example_ce(y_safe | x')
                + lambda_rep * rep_gate * detector_ce(reader(h(x')), harmful)
    detector loss = detector_ce over every example (attacked -> harmful, benign -> benign)

per_example_ce is per-row (not batch-mean) so a gate can scale each example before the
mean. There is no "away" term — harmful-side pressure is the gated representation term.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def per_example_ce(logits: torch.Tensor, targets: torch.Tensor, ignore_index: int = -100) -> torch.Tensor:
    """Per-example mean cross-entropy over each row's target tokens. Returns (B,)."""
    logp = F.log_softmax(logits, dim=-1)
    tok = logp.gather(-1, targets.clamp_min(0).unsqueeze(-1)).squeeze(-1)  # (B, T)
    mask = (targets != ignore_index).float()
    return -(tok * mask).sum(-1) / mask.sum(-1).clamp_min(1)


def detector_ce(reader_logits: torch.Tensor, labels: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
    """Detector cross-entropy. labels (B,) long: 0 = harmful, 1 = benign (column-0 =
    harmful convention). reduction='none' returns per-example loss so the representation
    term can be gated per example."""
    return F.cross_entropy(reader_logits, labels, reduction=reduction)
