"""EuroSafeGuard's activation probe (EuroSafeAI/EuroSafeGuard-v1, probe/), scoring prompts as their loader does.

Copied from github.com/EuroSafeAI/EuroSafeGuard, eurosafeguard/guard.py (commit 2a69489, Apache-2.0): MLPProbe and
ema_max unchanged; probe_scores split into prompt_ids (their template: the tokenizer's own chat template, user turn
only, add_generation_prompt, cut from the left to max_len) and score_ids (forward, all hidden_states in
probe_config["layers"] concatenated, per-token logit, EMA over every prompt token, max). Their gate and exchange
classifiers (Qwen3.5-9B) are not loaded.

The probe reads base Llama-3.1-8B-Instruct (probe_config["base_model"]). Its score is a logit; the paper z-scores it
on WildChat-clean (thresholds/seed{k}.json: probe_mu, probe_sd; full_seed{k}.json: probe_escalation_z = the
WildChat-clean 90th percentile).
"""

from __future__ import annotations

import json
import os

import torch
import torch.nn.functional as F
from torch import nn

REPO_ID = "EuroSafeAI/EuroSafeGuard-v1"


class MLPProbe(nn.Module):
    """z_t = w2 . gelu(W1 (scale * psi_t) + b1) + b2, per token."""

    def __init__(self, dim: int, hidden: int = 256):
        super().__init__()
        self.W1 = nn.Linear(dim, hidden)
        self.w2 = nn.Linear(hidden, 1)
        self.register_buffer("scale", torch.ones(dim))

    def forward(self, psi):
        w = (self.W1.weight * self.scale).to(psi.dtype)
        return self.w2(F.gelu((psi @ w.T).float() + self.W1.bias)).squeeze(-1)


def ema_max(z: torch.Tensor, mask: torch.Tensor, M: int = 16) -> torch.Tensor:
    """Streaming score: EMA of per-token logits (alpha = 2/(M+1)) over scored positions; returns the max per row."""
    a = 2.0 / (M + 1)
    state = torch.zeros(z.shape[0], dtype=z.dtype, device=z.device)
    started = torch.zeros(z.shape[0], dtype=torch.bool, device=z.device)
    best = torch.full((z.shape[0],), -1e9, dtype=z.dtype, device=z.device)
    for t in range(z.shape[1]):
        m = mask[:, t]
        new = torch.where(started, (1 - a) * state + a * z[:, t], z[:, t])
        state = torch.where(m, new, state)
        started = started | m
        best = torch.where(m, torch.maximum(best, state), best)
    return best


def snapshot(repo_or_path: str = REPO_ID) -> str:
    """Local folder of the released probe (the hub cache; jobs run offline)."""
    if os.path.isdir(repo_or_path):
        return repo_or_path
    from huggingface_hub import snapshot_download

    return snapshot_download(repo_or_path, allow_patterns=["probe/*", "thresholds/*", "*.json"])


def load_probe(root: str, device) -> tuple[MLPProbe, dict]:
    from safetensors.torch import load_file

    with open(os.path.join(root, "probe", "probe_config.json")) as f:
        cfg = json.load(f)
    probe = MLPProbe(cfg["feature_dim"], cfg["hidden"])
    probe.load_state_dict(load_file(os.path.join(root, "probe", "probe.safetensors")))
    return probe.to(device).eval(), cfg


def load_thresholds(root: str, seed: int = 0) -> dict:
    """probe_mu, probe_sd (alert mode) and probe_escalation_z (full mode) of gate run `seed`."""
    out = {}
    for name in (f"seed{seed}.json", f"full_seed{seed}.json"):
        path = os.path.join(root, "thresholds", name)
        if os.path.exists(path):
            with open(path) as f:
                out[name.removesuffix(".json")] = json.load(f)
    return out


def prompt_ids(tokenizer, prompt: str, max_len: int) -> list[int]:
    """Their template: the tokenizer's chat template on the user turn, up to the assistant header."""
    text = tokenizer.apply_chat_template([{"role": "user", "content": str(prompt)}], tokenize=False,
                                         add_generation_prompt=True)
    return tokenizer(text, add_special_tokens=False)["input_ids"][-max_len:]


@torch.no_grad()
def score_ids(model, probe, ids: list[list[int]], layers: list[int], M: int, pad_id: int) -> torch.Tensor:
    """EMA-max probe logit of each token sequence (right-padded batch, every real token scored)."""
    device = next(probe.parameters()).device
    T = max(map(len, ids))
    inp = torch.full((len(ids), T), pad_id, dtype=torch.long)
    att = torch.zeros((len(ids), T), dtype=torch.long)
    for k, x in enumerate(ids):
        inp[k, : len(x)] = torch.tensor(x)
        att[k, : len(x)] = 1
    hs = model(input_ids=inp.to(device), attention_mask=att.to(device), output_hidden_states=True).hidden_states
    psi = torch.cat([hs[layer] for layer in layers], dim=-1)
    return ema_max(probe(psi), att.bool().to(device), M).float().cpu()
