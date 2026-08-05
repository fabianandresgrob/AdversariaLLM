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
        mask = targets != ignore_index
        # Clamp before gather: ignore_index (-100) is not a valid vocab index and
        # would be an out-of-bounds gather. mask zeroes those positions afterwards.
        p_target = logp.gather(-1, targets.clamp_min(0).unsqueeze(-1)).squeeze(-1).exp()  # (B,T)
        ul = -torch.log((1.0 - p_target).clamp_min(eps))
        ul = (ul * mask).sum() / mask.sum().clamp_min(1)
        return ul
    raise ValueError(f"unknown away variant: {variant}")


def utility_kl(
    model_logits: torch.Tensor, ref_logits: torch.Tensor, attention_mask: torch.Tensor | None = None
) -> torch.Tensor:
    """KL(model || ref) averaged over tokens. Both (B,T,V).
    If attention_mask (B,T) is given, only attended positions are averaged so
    right-padding doesn't leak into the utility term."""
    logp = F.log_softmax(model_logits, dim=-1)
    logq = F.log_softmax(ref_logits, dim=-1)
    p = logp.exp()
    kl_tok = (p * (logp - logq)).sum(-1)  # (B, T)
    if attention_mask is not None:
        m = attention_mask.to(kl_tok.dtype)
        return (kl_tok * m).sum() / m.sum().clamp_min(1)
    return kl_tok.mean()


def sequence_logprob(logits: torch.Tensor, targets: torch.Tensor, ignore_index: int = -100) -> torch.Tensor:
    """Sum of log p(target_token) over the response tokens. Returns (B,)."""
    logp = F.log_softmax(logits, dim=-1)
    tok_logp = logp.gather(-1, targets.clamp_min(0).unsqueeze(-1)).squeeze(-1)  # (B,T)
    mask = targets != ignore_index
    return (tok_logp * mask).sum(-1)


def ipo_preference(
    pi_chosen: torch.Tensor,
    pi_rejected: torch.Tensor,
    ref_chosen: torch.Tensor,
    ref_rejected: torch.Tensor,
    beta: float = 0.1,
) -> torch.Tensor:
    """IPO loss (Azar et al.): (h - 1/(2*beta))^2, h = (pi_c-pi_r) - (ref_c-ref_r).
    chosen = y_benign, rejected = y_harmful. Inputs are sequence log-probs (B,)."""
    h = (pi_chosen - pi_rejected) - (ref_chosen - ref_rejected)
    return ((h - 1.0 / (2.0 * beta)) ** 2).mean()
