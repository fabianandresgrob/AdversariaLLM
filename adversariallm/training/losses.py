from __future__ import annotations

import torch
import torch.nn.functional as F


def _token_ce(logits: torch.Tensor, targets: torch.Tensor, ignore_index: int = -100) -> torch.Tensor:
    """Mean cross-entropy over target tokens. logits (B,T,V), targets (B,T)."""
    return F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1),
        ignore_index=ignore_index,
    )


def toward_benign(logits: torch.Tensor, targets: torch.Tensor, ignore_index: int = -100) -> torch.Tensor:
    """Standard CE: minimizing makes y_benign more likely."""
    return _token_ce(logits, targets, ignore_index)


def away_from_harmful(
    logits: torch.Tensor, targets: torch.Tensor, variant: str = "ce", ignore_index: int = -100, eps: float = 1e-6
) -> torch.Tensor:
    """Push the model away from y_harmful.
    variant="ce": -CE (unbounded gradient ascent).
    variant="ul": unlikelihood -mean(log(1 - p(target))) (bounded)."""
    if variant == "ce":
        return -_token_ce(logits, targets, ignore_index)
    if variant == "ul":
        logp = F.log_softmax(logits, dim=-1)
        p_target = logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1).exp()  # (B,T)
        mask = targets != ignore_index
        ul = -torch.log((1.0 - p_target).clamp_min(eps))
        ul = (ul * mask).sum() / mask.sum().clamp_min(1)
        return ul
    raise ValueError(f"unknown away variant: {variant}")


def utility_kl(model_logits: torch.Tensor, ref_logits: torch.Tensor) -> torch.Tensor:
    """KL(model || ref) averaged over tokens. Both (B,T,V)."""
    logp = F.log_softmax(model_logits, dim=-1)
    logq = F.log_softmax(ref_logits, dim=-1)
    p = logp.exp()
    return (p * (logp - logq)).sum(-1).mean()
