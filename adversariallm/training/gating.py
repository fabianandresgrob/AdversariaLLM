"""Soft gating for the cooperative pipeline (design doc §3, §6).

Two per-example failure weights, both stop-gradient, and two floored gates built from
them. The gates scale the model-loss terms so each component is trained hardest exactly
where the *other* one fails:

    behavior gate = eps + (1-eps) * w_miss     (teach refusal, scaled by detector failure)
    rep gate      = delta + (1-delta) * w_harm (reshape activations, scaled by model failure)

The four corners of (w_harm, w_miss) are the four cases; everything between is interpolated.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def avg_logprob(logits: torch.Tensor, targets: torch.Tensor, ignore_index: int = -100) -> torch.Tensor:
    """Length-normalized (per-token mean) log-prob of the target tokens. Returns (B,).

    Length normalization matters: y_harmful and y_safe differ in length, and summed
    log-probs would mostly measure length (§14.3). clamp_min guards the ignore_index
    positions from an out-of-bounds gather.
    """
    logp = F.log_softmax(logits, dim=-1)
    tok = logp.gather(-1, targets.clamp_min(0).unsqueeze(-1)).squeeze(-1)  # (B, T)
    mask = (targets != ignore_index).float()
    return (tok * mask).sum(-1) / mask.sum(-1).clamp_min(1)


def w_harm(logp_harm: torch.Tensor, logp_safe: torch.Tensor, tau: float = 1.0) -> torch.Tensor:
    """Soft "the model failed": sigmoid of the length-normalized margin by which the model
    prefers the harmful completion over the safe one. Inputs are (B,) avg log-probs.

    Stop-gradient: this is a gate, not a loss. tau sets sharpness (small -> hard switch)."""
    return torch.sigmoid((logp_harm - logp_safe) / tau).detach()


def w_miss(p_harmful: torch.Tensor) -> torch.Tensor:
    """Soft "the detector failed": the detector's own P(benign) = 1 - P(harmful). (B,).

    Stop-gradient (§14.9): without it the optimizer could shrink the behavior term by
    making activations more detectable (reducing w_miss) instead of by learning to refuse.
    sg freezes the weight for the backward pass, leaving only the intended route."""
    return (1.0 - p_harmful).detach()


def behavior_gate(w_miss_val: torch.Tensor, epsilon: float) -> torch.Tensor:
    """Refusal-teaching weight, floored at epsilon so the model never fully offloads
    safety onto the detector (§7). epsilon=0 -> pure coupled; epsilon=1 -> always teach."""
    return epsilon + (1.0 - epsilon) * w_miss_val


def rep_gate(w_harm_val: torch.Tensor, delta: float) -> torch.Tensor:
    """Activation-reshaping weight, floored at delta. delta=0 switches reshaping off when
    the model already refuses (the case-C handshake update, §6/§12); delta=1 always reshapes."""
    return delta + (1.0 - delta) * w_harm_val


def w_refuse(logp_refuse: torch.Tensor, logp_help: torch.Tensor, tau: float = 1.0) -> torch.Tensor:
    """Soft "the model over-refused a BENIGN prompt": sigmoid of the length-normalized margin
    by which the model prefers the canned refusal over the helpful answer. Inputs are (B,) avg
    log-probs. Stop-gradient (a gate, not a loss). Benign twin of w_harm, but callers pass a
    SEPARATE tau_b: a short canned refusal vs a long free-form answer does not share the harmful
    gate's calibration (§8 length asymmetry)."""
    return torch.sigmoid((logp_refuse - logp_help) / tau).detach()
