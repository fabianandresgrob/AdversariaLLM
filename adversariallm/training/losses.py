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


def soft_floor(loss: torch.Tensor, cutoff: float | None) -> torch.Tensor:
    """CAT's loss cutoff (Xhonneux et al. 2024): below `cutoff` the loss is replaced by
    cutoff + 0.001 * loss, so it keeps a thousandth of its gradient. None = no cutoff."""
    if cutoff is None:
        return loss
    return torch.where(loss < cutoff, cutoff + 1e-3 * loss, loss)


def toward_benign(logits: torch.Tensor, targets: torch.Tensor, ignore_index: int = -100,
                  cutoff: float | None = None) -> torch.Tensor:
    """Standard CE: minimizing makes y_benign more likely. cutoff: CAT's toward cutoff (0.5)."""
    return soft_floor(_token_ce(logits, targets, ignore_index), cutoff)


def away_from_harmful(
    logits: torch.Tensor, targets: torch.Tensor, variant: str = "ce", ignore_index: int = -100, eps: float = 1e-6,
    cutoff: float | None = None,
) -> torch.Tensor:
    """Push the model away from y_harmful.
    variant="ce": -CE (unbounded gradient ascent unless cut off).
    variant="ul": unlikelihood -mean(log(1 - p(target))) (bounded).
    cutoff (CAT's away cutoff, e.g. -5): for "ce" the soft floor on -CE; for "ul" tokens whose
    log p(target) is already below it drop out of the loss (still counted in the mean), as in CAT."""
    if variant == "ce":
        return soft_floor(-_token_ce(logits, targets, ignore_index), cutoff)
    if variant == "ul":
        logp = F.log_softmax(logits, dim=-1)
        mask = targets != ignore_index
        # Clamp before gather: ignore_index (-100) is not a valid vocab index and
        # would be an out-of-bounds gather. mask zeroes those positions afterwards.
        logp_target = logp.gather(-1, targets.clamp_min(0).unsqueeze(-1)).squeeze(-1)  # (B,T)
        ul = -torch.log((1.0 - logp_target.exp()).clamp_min(eps))
        keep = mask if cutoff is None else mask & (logp_target >= cutoff)
        ul = (ul * keep).sum() / mask.sum().clamp_min(1)
        return ul
    raise ValueError(f"unknown away variant: {variant}")


def utility_kl(
    model_logits: torch.Tensor, ref_logits: torch.Tensor, attention_mask: torch.Tensor | None = None,
    n_tokens: torch.Tensor | None = None,
) -> torch.Tensor:
    """KL(model || ref) averaged over tokens. Both (B,T,V).
    If attention_mask (B,T) is given, only attended positions are averaged so
    right-padding doesn't leak into the utility term. n_tokens: divide by this (the whole batch's
    attended tokens) instead, so the parts of a batch split into chunks sum to the batch's KL."""
    logp = F.log_softmax(model_logits, dim=-1)
    logq = F.log_softmax(ref_logits, dim=-1)
    p = logp.exp()
    kl_tok = (p * (logp - logq)).sum(-1)  # (B, T)
    if attention_mask is not None:
        m = attention_mask.to(kl_tok.dtype)
        return (kl_tok * m).sum() / (m.sum() if n_tokens is None else n_tokens).clamp_min(1)
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
