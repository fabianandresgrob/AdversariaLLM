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

A second reader type, DualProbe (type "dual"), keeps the prompt_last probe unchanged and adds a
separate response channel with its own weights; see its docstring.

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
# DualProbe response channel: tokens per rolling window before the max. Long enough that one odd
# token cannot fire the channel, short enough that a harmful paragraph deep in a refusal-prefixed
# answer still dominates its window.
DEFAULT_RESPONSE_WINDOW = 16


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


def window_max(values: torch.Tensor, mask: torch.Tensor, window: int) -> torch.Tensor:
    """Max over rolling means of `window` consecutive masked positions, per row: (B, T) -> (B,).

    The masked region is assumed contiguous (a response followed by right padding). A row
    shorter than `window` gets the mean over the whole region; a row with no masked position
    gets -inf, so callers must handle it (DualProbe treats it as "no response")."""
    mask = mask.bool()
    v = values.float() * mask
    n = mask.float()
    zero = torch.zeros_like(v[:, :1])
    cs = torch.cat([zero, v.cumsum(dim=1)], dim=1)  # (B, T+1)
    cn = torch.cat([zero, n.cumsum(dim=1)], dim=1)
    end = torch.arange(1, v.size(1) + 1, device=v.device)  # window ends (exclusive), (T,)
    start = (end - window).clamp_min(0)
    win_sum = cs[:, end] - cs[:, start]
    win_n = cn[:, end] - cn[:, start]
    need = torch.minimum(n.sum(dim=1, keepdim=True), torch.tensor(float(window), device=v.device))
    valid = mask & (win_n >= need) & (win_n > 0)
    means = win_sum / win_n.clamp_min(1)
    return means.masked_fill(~valid, float("-inf")).max(dim=1).values


class DualProbe(Reader, nn.Module):
    """prompt_last probe + an independent response channel; fires if EITHER channel fires.

        p_harmful = max( p_prompt , p_response )

    The prompt channel is a LinearProbe at the last prompt token -- the readout that catches
    optimized suffixes, even adaptively. The response channel scores every response token with
    its own linear head (unit-normed activations, as the prompt channel) and pools the per-token
    margins as the max over rolling windows (window_max), so harmful content anywhere in a long
    answer can fire it. Taking the max of the two probabilities, rather than one linear head over
    both, means adding the response channel can never lower the prompt channel's score: a
    confident "benign" response cannot talk the probe out of a suffix it caught. A row with no
    response scores on the prompt channel alone.

    The response head starts at p_response ~ 0 (zero weights, benign-leaning bias), so an
    untrained DualProbe scores exactly like its prompt channel.
    """

    readout_mode = "dual"

    def __init__(self, input_dim: int, eps: float = 1e-6, response_window: int = DEFAULT_RESPONSE_WINDOW):
        nn.Module.__init__(self)
        self.eps = eps
        self.response_window = int(response_window)
        self.prompt = LinearProbe(input_dim, eps=eps, readout="prompt_last")
        self.response = nn.Linear(input_dim, 2)
        with torch.no_grad():
            self.response.weight.zero_()
            self.response.bias.copy_(torch.tensor([-10.0, 10.0]))  # [harmful, benign]

    def response_margin(
        self, hidden: torch.Tensor, target_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """(pooled harmful-minus-benign logit margin (B,), has_response (B,) bool)."""
        region = (target_ids != 0) & attention_mask.bool()
        h = hidden.float()
        h = h / h.norm(dim=-1, keepdim=True).clamp_min(self.eps)
        logits = self.response(h)  # (B, T, 2)
        margins = logits[..., HARMFUL_COL] - logits[..., BENIGN_COL]
        has = region.any(dim=1)
        pooled = window_max(margins, region, self.response_window)
        return torch.where(has, pooled, torch.zeros_like(pooled)), has

    def p_harmful(self, hidden: torch.Tensor, target_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        p_prompt = self.prompt.p_harmful(hidden, target_ids, attention_mask)
        margin, has = self.response_margin(hidden, target_ids, attention_mask)
        p_resp = torch.where(has, torch.sigmoid(margin), torch.zeros_like(margin))
        return torch.maximum(p_prompt, p_resp)

    def logits(self, hidden: torch.Tensor, target_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Log-probabilities in the (B, 2) logits layout, so softmax(logits)[:, 0] == p_harmful."""
        p = self.p_harmful(hidden, target_ids, attention_mask).clamp(1e-7, 1 - 1e-7)
        out = torch.stack([p.log(), (-p).log1p()], dim=-1)
        return out if HARMFUL_COL == 0 else out.flip(-1)


def load_reader(checkpoint_path: str, readout: str | None = None, readout_k: int | None = None) -> Reader:
    """Build a reader from a pair checkpoint ({"reader", "cfg"}) or a bare LinearProbe state_dict.

    The type and readout come from the checkpoint's cfg.reader, so a probe is scored where it was
    trained. `readout`/`readout_k` override the LinearProbe position (an ablation); a DualProbe has
    fixed positions and refuses them. Returned on CPU in eval mode."""
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    is_pair = isinstance(ckpt, dict) and "reader" in ckpt
    state = ckpt["reader"] if is_pair else ckpt
    trained = ((ckpt.get("cfg") or {}).get("reader") or {}) if is_pair else {}
    if trained.get("type") == "dual":
        if readout or readout_k:
            raise ValueError("a dual probe reads fixed positions; readout overrides do not apply")
        reader: Reader = DualProbe(
            state["prompt.linear.weight"].shape[1],
            response_window=trained.get("response_window") or DEFAULT_RESPONSE_WINDOW,
        )
    else:
        reader = LinearProbe(  # (2, input_dim)
            state["linear.weight"].shape[1],
            readout=readout or trained.get("readout") or DEFAULT_READOUT,
            readout_k=readout_k or trained.get("readout_k") or DEFAULT_READOUT_K,
        )
    reader.load_state_dict(state)
    return reader.eval()


def build_reader(cfg: dict | None, input_dim: int) -> Reader:
    """Construct the reader from config. v1 supports "linear"; MLP/gemma slot in here.

    cfg keys: type ("linear" | "dual"), readout (see READOUT_MODES), readout_k (response_mean
    window), response_window (dual only)."""
    cfg = cfg or {}
    kind = cfg.get("type", "linear")
    if kind == "linear":
        return LinearProbe(
            input_dim,
            readout=cfg.get("readout") or DEFAULT_READOUT,
            readout_k=cfg.get("readout_k") or DEFAULT_READOUT_K,
        )
    if kind == "dual":
        return DualProbe(input_dim, response_window=cfg.get("response_window") or DEFAULT_RESPONSE_WINDOW)
    raise ValueError(f"unknown reader type: {kind!r}")
