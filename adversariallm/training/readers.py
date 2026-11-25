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
    response_window max over windows of `window` consecutive tokens within the first `readout_k`
                    response tokens: each window's mean activation is scored, the most harmful window
                    counts. A long harmless frame (role-play, disclaimers) cannot dilute a harmful middle
                    the way it dilutes response_mean
    response_mean   mean over the first `readout_k` response tokens — prompt + the
                    opening of the response. Training targets run 15-34 tokens, so this
                    window covers the whole target; note that it therefore cannot learn
                    what harmful content looks like deep in a long generation (in PAIR
                    outputs the instructions start at a median of 74 tokens) — that case
                    is what stream_last covers

A second reader type, DualProbe (type "dual"), keeps the prompt_last probe unchanged and adds a
separate response channel with its own weights; see its docstring.

A third type, PromptMLPProbe (type "prompt_mlp"), is EuroSafeGuard's prompt probe: an MLP on the outputs of
several layers, scoring every prompt token. It reads `layer: all` (or a list of layers); see select_hidden.

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

READOUT_MODES = ("prompt_last", "stream_last", "response_mean", "response_window")
DEFAULT_READOUT = "prompt_last"
# p90 of the adv_training target lengths (median 18, max 34), so the window covers the whole
# teacher-forced target for ~90% of rows instead of truncating it mid-target.
DEFAULT_READOUT_K = 24
# DualProbe response channel: tokens per rolling window before the max. Long enough that one odd
# token cannot fire the channel, short enough that a harmful paragraph deep in a refusal-prefixed
# answer still dominates its window.
DEFAULT_RESPONSE_WINDOW = 16


def parse_layer(layer, num_layers: int):
    """The reader config's `layer` as select_hidden takes it: an int, or a list of ints for "all" (the output of
    every block, hidden_states[1..num_layers], as EuroSafeGuard's probe reads) or a list of layers."""
    if layer is None:
        return -1
    if isinstance(layer, str):
        if layer != "all":
            raise ValueError(f"layer must be an int, a list of ints or 'all', got {layer!r}")
        return list(range(1, num_layers + 1))
    if isinstance(layer, int):
        return layer
    return [int(i) for i in layer]


def select_hidden(hidden_states, layer) -> torch.Tensor:
    """What a reader is given from a forward with output_hidden_states: one layer (B, T, D) for an int, the layers
    concatenated along the feature axis (B, T, len(layer) * D) for a list. Gradients flow in both cases."""
    if isinstance(layer, int):
        return hidden_states[layer]
    return torch.cat([hidden_states[i] for i in layer], dim=-1)


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

    def example_loss(self, hidden: torch.Tensor, target_ids: torch.Tensor, attention_mask: torch.Tensor,
                     labels: torch.Tensor) -> torch.Tensor:
        """The reader's own training loss per example, (B,). labels: 0 = harmful, 1 = benign. Cross-entropy on
        logits() unless a reader has its own loss (PromptMLPProbe)."""
        return nn.functional.cross_entropy(self.logits(hidden, target_ids, attention_mask), labels, reduction="none")

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
        window: int = DEFAULT_RESPONSE_WINDOW,
    ):
        nn.Module.__init__(self)
        if readout not in READOUT_MODES:
            raise ValueError(f"unknown readout mode: {readout!r} (expected one of {READOUT_MODES})")
        self.eps = eps
        self.readout_mode = readout
        self.readout_k = int(readout_k)
        self.window = int(window)  # response_window only
        self.linear = nn.Linear(input_dim, 2)  # fp32 params

    def _windows(self, hidden, target_ids, attention_mask):
        """Unit-normed mean activation of every window of `window` consecutive answer tokens (within the first
        readout_k), and which windows are valid: (B, T, D) features, (B, T) mask. A window ends at each answer
        position; answers shorter than `window` get one window over the whole answer."""
        mask = response_mean_mask(target_ids, attention_mask, self.readout_k)  # prompt-only row: its last token
        h = hidden.float() * mask.unsqueeze(-1)
        zero = torch.zeros_like(h[:, :1])
        cs = torch.cat([zero, h.cumsum(dim=1)], dim=1)  # (B, T+1, D)
        cn = torch.cat([torch.zeros_like(mask[:, :1], dtype=torch.float), mask.float().cumsum(dim=1)], dim=1)
        end = torch.arange(1, h.size(1) + 1, device=h.device)
        start = (end - self.window).clamp_min(0)
        n = cn[:, end] - cn[:, start]  # (B, T)
        feats = (cs[:, end] - cs[:, start]) / n.clamp_min(1).unsqueeze(-1)
        need = torch.minimum(mask.float().sum(dim=1, keepdim=True), torch.tensor(float(self.window), device=h.device))
        valid = mask & (n >= need) & (n > 0)
        return feats / feats.norm(dim=-1, keepdim=True).clamp_min(self.eps), valid

    def _top_window(self, hidden, target_ids, attention_mask):
        """Per row, the logits and feature of the window the probe finds most harmful."""
        feats, valid = self._windows(hidden, target_ids, attention_mask)
        logits = self.linear(feats)  # (B, T, 2)
        margin = (logits[..., 0] - logits[..., 1]).masked_fill(~valid, float("-inf"))
        idx = margin.argmax(dim=1)
        rows = torch.arange(hidden.size(0), device=hidden.device)
        return logits[rows, idx], feats[rows, idx]

    def readout(self, hidden: torch.Tensor, target_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if self.readout_mode == "response_window":
            return self._top_window(hidden, target_ids, attention_mask)[1]
        return probe_readout(hidden, target_ids, attention_mask, self.eps, self.readout_mode, self.readout_k)

    def logits(self, hidden: torch.Tensor, target_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if self.readout_mode == "response_window":
            return self._top_window(hidden, target_ids, attention_mask)[0]
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


class PromptMLPProbe(Reader, nn.Module):
    """EuroSafeGuard's prompt probe (Waibl et al. 2026; github.com/EuroSafeAI/EuroSafeGuard, Apache-2.0, commit
    2a69489: eurosafeguard/guard.py MLPProbe and ema_max, training/probe_lib.py swim and probe_loss).

        z_t    = w2 . gelu(W1 (scale * psi_t) + b1) + b2     one logit per prompt token, psi_t = the outputs of
                                                             several layers at token t, concatenated
        score  = max_t EMA_t(z), alpha = 2 / (M + 1)         the prompt's score when the probe is used
        loss   = sum_t softmax_t(zbar_t) * BCE(y, zbar_t)    training; zbar_t = mean of the last M logits

    Scored positions are the prompt up to and including the assistant header: every attended token before the
    answer (target_ids == 0), so the score does not change with what follows the prompt. `scale` is a fixed
    1/RMS per feature, set once from unlabelled data (set_scale). Parameter names are theirs, so the released
    probe.safetensors loads with load_state_dict.

    In the (B, 2) logits layout the score is the harmful column and the benign column is 0, so
    softmax(logits)[:, 0] = sigmoid(score), CE toward harmful = softplus(-score), and evasion_loss = softplus(score).
    """

    readout_mode = "prompt_tokens"
    reads_answer = False

    def __init__(self, input_dim: int, hidden: int = 256, window: int = DEFAULT_RESPONSE_WINDOW, tau: float = 1.0):
        nn.Module.__init__(self)
        self.window = int(window)  # M
        self.tau = float(tau)
        self.W1 = nn.Linear(input_dim, hidden)
        self.w2 = nn.Linear(hidden, 1)
        self.register_buffer("scale", torch.ones(input_dim))
        nn.init.normal_(self.W1.weight, std=1.0 / input_dim ** 0.5)  # their training initialisation
        nn.init.zeros_(self.w2.weight)

    def set_scale(self, scale: torch.Tensor) -> None:
        self.scale.copy_(scale.to(self.scale))

    @staticmethod
    def prompt_mask(target_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """(B, T) bool: attended positions up to the last prompt token. A prompt-only row: every attended token."""
        pos = torch.arange(target_ids.size(1), device=target_ids.device).unsqueeze(0)
        return (pos <= readout_index(target_ids, attention_mask).unsqueeze(1)) & attention_mask.bool()

    def token_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        """(B, T) logit per token from (B, T, input_dim) features, in fp32 after the first layer."""
        w = (self.W1.weight * self.scale).to(hidden.dtype)
        return self.w2(nn.functional.gelu((hidden @ w.T).float() + self.W1.bias)).squeeze(-1)

    def score(self, hidden: torch.Tensor, target_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """(B,) max over prompt tokens of the moving average of the token logits."""
        z, mask = self.token_logits(hidden), self.prompt_mask(target_ids, attention_mask)
        a = 2.0 / (self.window + 1)
        state = torch.zeros_like(z[:, 0])
        started = torch.zeros_like(mask[:, 0])
        best = torch.full_like(z[:, 0], -1e9)
        for t in range(z.size(1)):
            m = mask[:, t]
            state = torch.where(m, torch.where(started, (1 - a) * state + a * z[:, t], z[:, t]), state)
            started = started | m
            best = torch.where(m, torch.maximum(best, state), best)
        return best

    def logits(self, hidden: torch.Tensor, target_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        s = self.score(hidden, target_ids, attention_mask)
        cols = [s, torch.zeros_like(s)]
        return torch.stack(cols if HARMFUL_COL == 0 else cols[::-1], dim=-1)

    def example_loss(self, hidden: torch.Tensor, target_ids: torch.Tensor, attention_mask: torch.Tensor,
                     labels: torch.Tensor) -> torch.Tensor:
        z, mask = self.token_logits(hidden), self.prompt_mask(target_ids, attention_mask)
        M = self.window
        zc = torch.cumsum(z * mask, dim=1)
        n = torch.cumsum(mask.to(z.dtype), dim=1)
        pad = nn.functional.pad
        zbar = (zc - pad(zc, (M, 0))[:, :-M]) / (n - pad(n, (M, 0))[:, :-M]).clamp(min=1)
        valid = mask & (n >= M)
        short = mask.sum(dim=1) < M  # fewer than M prompt tokens: one prediction from their mean, at the last one
        if short.any():
            last = (torch.cumsum(mask.long(), dim=1) * mask).argmax(dim=1)
            valid = valid.clone()
            valid[short] = False
            valid[short, last[short]] = True
        y = (labels == HARMFUL_COL).to(zbar.dtype)[:, None].expand_as(zbar)
        per = nn.functional.binary_cross_entropy_with_logits(zbar, y, reduction="none")
        w = torch.softmax((zbar / self.tau).masked_fill(~valid, -1e9), dim=1)
        return (w * per.masked_fill(~valid, 0.0)).sum(dim=1)


def load_reader(checkpoint_path: str, readout: str | None = None, readout_k: int | None = None,
                window: int | None = None) -> Reader:
    """Build a reader from a pair checkpoint ({"reader", "cfg"}) or a bare LinearProbe state_dict.

    The type and readout come from the checkpoint's cfg.reader, so a probe is scored where it was
    trained. `readout`/`readout_k` override the LinearProbe position (an ablation); a DualProbe has
    fixed positions and refuses them. Returned on CPU in eval mode."""
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    is_pair = isinstance(ckpt, dict) and "reader" in ckpt
    state = ckpt["reader"] if is_pair else ckpt
    trained = ((ckpt.get("cfg") or {}).get("reader") or {}) if is_pair else {}
    if trained.get("type") == "prompt_mlp":
        reader: Reader = PromptMLPProbe(state["W1.weight"].shape[1], hidden=state["W1.weight"].shape[0],
                                        window=trained.get("window") or DEFAULT_RESPONSE_WINDOW)
    elif trained.get("type") == "dual":
        if readout or readout_k:
            raise ValueError("a dual probe reads fixed positions; readout overrides do not apply")
        reader = DualProbe(
            state["prompt.linear.weight"].shape[1],
            response_window=trained.get("response_window") or DEFAULT_RESPONSE_WINDOW,
        )
    else:
        reader = LinearProbe(  # (2, input_dim)
            state["linear.weight"].shape[1],
            readout=readout or trained.get("readout") or DEFAULT_READOUT,
            readout_k=readout_k or trained.get("readout_k") or DEFAULT_READOUT_K,
            window=window or trained.get("window") or DEFAULT_RESPONSE_WINDOW,
        )
    reader.load_state_dict(state)
    return reader.eval()


def build_reader(cfg: dict | None, input_dim: int, num_layers: int | None = None) -> Reader:
    """Construct the reader from config. input_dim is the model's hidden size.

    cfg keys: type ("linear" | "dual" | "prompt_mlp"), layer (int, list or "all"; a list or "all" multiplies the
    reader's input by the number of layers, and "all" needs num_layers), readout (see READOUT_MODES), readout_k
    (response_mean window), response_window (dual only), window and hidden (prompt_mlp)."""
    cfg = cfg or {}
    kind = cfg.get("type", "linear")
    layer = parse_layer(cfg.get("layer", -1), num_layers or 0)
    if kind == "prompt_mlp":
        if isinstance(layer, int):
            layer = [layer]
        if not layer:
            raise ValueError("reader.layer='all' needs the model's number of layers")
        return PromptMLPProbe(input_dim * len(layer), hidden=int(cfg.get("hidden") or 256),
                              window=cfg.get("window") or DEFAULT_RESPONSE_WINDOW)
    if not isinstance(layer, int):
        raise ValueError(f"a {kind} reader reads one layer, got layer={cfg.get('layer')!r}")  # noqa: TRY004 (a config value)
    if kind == "linear":
        return LinearProbe(
            input_dim,
            readout=cfg.get("readout") or DEFAULT_READOUT,
            readout_k=cfg.get("readout_k") or DEFAULT_READOUT_K,
            window=cfg.get("window") or DEFAULT_RESPONSE_WINDOW,
        )
    if kind == "dual":
        return DualProbe(input_dim, response_window=cfg.get("response_window") or DEFAULT_RESPONSE_WINDOW)
    raise ValueError(f"unknown reader type: {kind!r}")
