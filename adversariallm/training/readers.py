"""Detector readers for the cooperative pipeline.

A reader maps a target-model hidden-state tensor to a harmful/benign score. v1 is a
linear probe on the residual-stream activation at the last prompt token — the same
readout position the activation detector uses, so it reuses build_detector_batch
unchanged. Richer readers (MLP, the gemma detector) slot in behind the same interface.

Which position the probe reads decides what it can see, because the forward is causal:
the last prompt token cannot attend to the response, so it scores the prompt alone, while
any response position scores prompt *and* response. `readout` picks between them:

    prompt_last     last prompt token (default; the original v1 behaviour)
    stream_last     last real token of the sequence — prompt + the whole response
    response_mean   mean over the first `readout_k` response tokens — prompt + the
                    opening of the response. Training targets run 15-34 tokens, so this
                    window covers the whole target; note that it therefore cannot learn
                    what harmful content looks like deep in a long generation (in PAIR
                    outputs the instructions start at a median of 74 tokens) — that case
                    is what stream_last covers

Every mode derives its positions from target_ids/attention_mask, so all of them are
invariant to right padding and to variable prompt/response lengths.

Label convention: logits column 0 = harmful ("yes"),
column 1 = benign ("no").
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import nn

HARMFUL_COL = 0  # logits column convention (see module docstring): 0 = harmful, 1 = benign
BENIGN_COL = 1

READOUT_MODES = ("prompt_last", "stream_last", "response_mean")
DEFAULT_READOUT = "prompt_last"
# p90 of the adv_training target lengths (median 18, max 34), so the window covers the whole
# teacher-forced target for ~90% of rows instead of truncating it mid-target.
DEFAULT_READOUT_K = 24


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
    resp = target_ids != 0  # (B, T)
    has_resp = resp.any(dim=1)  # (B,)
    first_resp = resp.float().argmax(dim=1)  # (B,) 0 if no response
    last_real = attention_mask.long().sum(dim=1) - 1  # (B,)
    return torch.where(has_resp, first_resp - 1, last_real).clamp_min(0)


def stream_last_index(target_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Index of the last real (non-pad) token per row — the end of the response when there is
    one, the end of the prompt when there is not. target_ids is unused; it is in the signature
    so every position helper takes the same arguments."""
    del target_ids
    return (attention_mask.long().sum(dim=1) - 1).clamp_min(0)


def response_mean_mask(
    target_ids: torch.Tensor, attention_mask: torch.Tensor, k: int = DEFAULT_READOUT_K
) -> torch.Tensor:
    """Boolean (B, T) mask over the first `k` response tokens of each row.

    A row with no response region falls back to its last attended token, so prompt-only
    batches (and the loop's benign free-generation checks) still produce a vector."""
    resp = target_ids != 0  # (B, T)
    has_resp = resp.any(dim=1)  # (B,)
    first_resp = resp.float().argmax(dim=1)  # (B,) 0 if no response
    last_real = (attention_mask.long().sum(dim=1) - 1).clamp_min(0)  # (B,)
    end = torch.minimum(last_real, first_resp + (k - 1))
    pos = torch.arange(target_ids.size(1), device=target_ids.device).unsqueeze(0)  # (1, T)
    span = (pos >= first_resp.unsqueeze(1)) & (pos <= end.unsqueeze(1)) & attention_mask.bool()
    fallback = pos == last_real.unsqueeze(1)
    return torch.where(has_resp.unsqueeze(1), span, fallback)


def probe_readout(
    hidden: torch.Tensor,
    target_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    eps: float = 1e-6,
    readout: str = DEFAULT_READOUT,
    readout_k: int = DEFAULT_READOUT_K,
) -> torch.Tensor:
    """Unit-normed fp32 activation at the readout position(s) — the reader's input feature.
    Norms drift during fine-tuning, so the probe reads direction, not scale (§14.6).
    `readout` selects the position (see module docstring); the output dim is D in every mode."""
    if readout == "response_mean":
        mask = response_mean_mask(target_ids, attention_mask, readout_k).unsqueeze(-1).float()
        h = (hidden.float() * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
    else:
        if readout == "prompt_last":
            idx = readout_index(target_ids, attention_mask)
        elif readout == "stream_last":
            idx = stream_last_index(target_ids, attention_mask)
        else:
            raise ValueError(f"unknown readout mode: {readout!r} (expected one of {READOUT_MODES})")
        rows = torch.arange(hidden.size(0), device=hidden.device)
        h = hidden[rows, idx].float()
    return h / h.norm(dim=-1, keepdim=True).clamp_min(eps)


class Reader(ABC):
    """Scores a prompt as harmful/benign from target-model hidden states.

    logits(hidden, target_ids, attention_mask) -> (B, 2); column 0 = harmful.
    hidden is (B, T, D) at a single chosen layer (the loop selects the layer).
    """

    @abstractmethod
    def logits(self, hidden: torch.Tensor, target_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor: ...

    def p_harmful(self, hidden: torch.Tensor, target_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """P(harmful) per row = softmax(logits)[:, HARMFUL_COL]."""
        return torch.softmax(self.logits(hidden, target_ids, attention_mask).float(), dim=-1)[:, HARMFUL_COL]

    def evasion_loss(
        self,
        hidden: torch.Tensor,
        target_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        reduction: str = "mean",
    ) -> torch.Tensor:
        """Loss a detector-aware attacker minimizes to make this reader predict benign — the
        detector-side term of the Stage C embedding attack. CE toward BENIGN_COL.

        reduction="none" returns the per-example (B,) loss, which a discrete attacker needs to
        rank candidate substitutions against each other."""
        logits = self.logits(hidden, target_ids, attention_mask)
        target = torch.full((logits.size(0),), BENIGN_COL, dtype=torch.long, device=logits.device)
        return nn.functional.cross_entropy(logits, target, reduction=reduction)


class LinearProbe(Reader, nn.Module):
    """Linear probe on the activation at the configured readout position.

    The readout vector is taken in fp32 and unit-normed: activation norms drift during
    fine-tuning, so the probe reads direction, not scale (and stays fp32 even when the
    target model runs in bf16). `readout` only changes which position is read, so the
    parameter shape — and therefore every saved state_dict — is the same in all modes.
    """

    def __init__(
        self,
        input_dim: int,
        eps: float = 1e-6,
        readout: str = DEFAULT_READOUT,
        readout_k: int = DEFAULT_READOUT_K,
    ):
        nn.Module.__init__(self)
        if readout not in READOUT_MODES:
            raise ValueError(f"unknown readout mode: {readout!r} (expected one of {READOUT_MODES})")
        self.eps = eps
        self.readout_mode = readout
        self.readout_k = int(readout_k)
        self.linear = nn.Linear(input_dim, 2)  # fp32 params

    def readout(self, hidden: torch.Tensor, target_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return probe_readout(hidden, target_ids, attention_mask, self.eps, self.readout_mode, self.readout_k)

    def logits(self, hidden: torch.Tensor, target_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self.linear(self.readout(hidden, target_ids, attention_mask))


def build_reader(cfg: dict | None, input_dim: int) -> Reader:
    """Construct the reader from config. v1 supports "linear"; MLP/gemma slot in here.

    cfg keys: type, readout (see READOUT_MODES), readout_k (response_mean window)."""
    cfg = cfg or {}
    kind = cfg.get("type", "linear")
    if kind == "linear":
        return LinearProbe(
            input_dim,
            readout=cfg.get("readout") or DEFAULT_READOUT,
            readout_k=cfg.get("readout_k") or DEFAULT_READOUT_K,
        )
    raise ValueError(f"unknown reader type: {kind!r}")
