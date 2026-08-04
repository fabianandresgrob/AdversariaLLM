"""Offline pretraining of the cooperative detector's linear probe.

Fits our LinearProbe on the FROZEN base model over a diverse harmful-vs-benign set, then
reports recall@1%FPR CROSS-DISTRIBUTION (train on one benign source, evaluate on held-out
sources). This is sanity-gate item 4 — a probe that generalizes before any coupling — and
the fix for the coop_11_control collapse (the co-trained probe over-fit UltraChat-only
benign). The saved checkpoint is loaded in the coop loop via `probe_init`.

Diverse benign comes from AdversariaLLM's own datasets (alpaca / or_bench / xs_test); the
probe reads the generation-onset position (build_prompt_only), matching the in-loop readout.
"""
from __future__ import annotations

import logging
import os

import torch

from .readers import LinearProbe
from .coop_metrics import recall_at_fpr
from .data import build_prompt_only, load_dataset_prompts, pad_collate

log = logging.getLogger(__name__)


@torch.no_grad()
def _features(model, tokenizer, template_id, prompts, layer, device, batch_size=16):
    """Readout features (unit-normed, fp32) at the generation onset for each prompt."""
    probe_readout = LinearProbe(1)  # borrow its readout() only; linear unused here
    feats = []
    for start in range(0, len(prompts), batch_size):
        chunk = prompts[start:start + batch_size]
        items = [dict(zip(("d_ids", "d_targetids", "d_attn"),
                          build_prompt_only(p, tokenizer, template_id))) for p in chunk]
        batch = pad_collate(items, ["d_ids", "d_targetids", "d_attn"], pad_id=0)
        ids = batch["d_ids"].to(device)
        tgt = batch["d_targetids"].to(device)
        attn = batch["d_attn"].to(device)
        hidden = model(input_ids=ids, attention_mask=attn, output_hidden_states=True).hidden_states[layer]
        feats.append(probe_readout.readout(hidden, tgt, attn).cpu())
    return torch.cat(feats, dim=0)


def run_pretrain_probe(cfg):
    from omegaconf import OmegaConf
    from ..io_utils import load_model_and_tokenizer
    from ..defenses.monitors._activation_detector_model import get_chat_template

    container = OmegaConf.to_container(cfg, resolve=True)
    model_params = cfg.models[cfg.model]
    template_id = cfg.chat_template_id
    model, tokenizer = load_model_and_tokenizer(model_params)
    model.eval()
    device = next(model.parameters()).device
    get_chat_template(template_id)  # fail fast if the template id is unsupported
    layer = int(cfg.reader.layer)

    # harmful prompts (from a behaviors CSV) + diverse benign (from AdversariaLLM datasets)
    import pandas as pd
    beh = pd.read_csv(os.path.join(cfg.data.dir, cfg.data.behaviors))
    harmful = beh["Behavior"].tolist()[: cfg.data.n_harmful]

    benign = {}  # source name -> list of prompts
    for name in cfg.data.benign_sources:
        prompts, _ = load_dataset_prompts(cfg.datasets, name, n=cfg.data.n_benign_per_source, seed=cfg.data.seed)
        benign[name] = prompts
        log.info(f"benign source {name}: {len(prompts)} prompts")

    # feature extraction
    h_feat = _features(model, tokenizer, template_id, harmful, layer, device)
    b_feat = {name: _features(model, tokenizer, template_id, ps, layer, device) for name, ps in benign.items()}

    # cross-distribution split: train benign = the first source; eval = every source held out
    train_src = cfg.data.benign_sources[0]
    nh = int(len(h_feat) * cfg.data.train_frac)
    h_tr, h_ev = h_feat[:nh], h_feat[nh:]
    b_tr = b_feat[train_src][: int(len(b_feat[train_src]) * cfg.data.train_frac)]

    probe = LinearProbe(h_feat.size(-1)).to(device)
    opt = torch.optim.Adam(probe.parameters(), lr=cfg.train.lr)
    X = torch.cat([h_tr, b_tr], dim=0).to(device)
    y = torch.cat([torch.zeros(len(h_tr), dtype=torch.long), torch.ones(len(b_tr), dtype=torch.long)]).to(device)
    for step in range(cfg.train.steps):
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(probe.linear(X), y)
        loss.backward()
        opt.step()
    log.info(f"probe train loss={loss.item():.4f}")

    # cross-distribution recall@1%FPR: harmful (held-out) vs each benign source (held-out)
    def p_harm(feat):
        return torch.softmax(probe.linear(feat.to(device)).float(), dim=-1)[:, 0]
    h_scores = p_harm(h_ev)
    for name, feat in b_feat.items():
        ev = feat[int(len(feat) * cfg.data.train_frac):] if name == train_src else feat
        r = recall_at_fpr(p_harm(ev), h_scores, fpr=0.01)
        tag = "(train src)" if name == train_src else "(held-out)"
        log.info(f"recall@1%FPR vs {name} {tag}: {r:.3f}")

    out_dir = os.path.join(cfg.output.probe_path, container.get("name") or "probe")
    os.makedirs(out_dir, exist_ok=True)
    torch.save({"state": probe.state_dict(), "cfg": container, "input_dim": h_feat.size(-1), "layer": layer},
               os.path.join(out_dir, "probe.pt"))
    log.info(f"saved probe to {out_dir}/probe.pt")
    return out_dir
