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

    # harmful prompts (shuffled from a behaviors CSV) + diverse benign (AdversariaLLM datasets)
    import pandas as pd
    import random
    beh = pd.read_csv(os.path.join(cfg.data.dir, cfg.data.behaviors))
    harmful_all = beh[cfg.data.harmful_col].dropna().tolist()
    random.Random(cfg.data.seed).shuffle(harmful_all)
    harmful = harmful_all[: cfg.data.n_harmful]

    train_benign = list(cfg.data.train_benign_sources)
    all_benign = list(dict.fromkeys(train_benign + list(cfg.data.get("eval_only_sources", []))))
    benign = {}
    for name in all_benign:
        prompts, _ = load_dataset_prompts(cfg.datasets, name, n=cfg.data.n_benign_per_source, seed=cfg.data.seed)
        benign[name] = prompts
        log.info(f"benign source {name}: {len(prompts)} prompts")

    # features
    h_feat = _features(model, tokenizer, template_id, harmful, layer, device)
    b_feat = {name: _features(model, tokenizer, template_id, ps, layer, device) for name, ps in benign.items()}

    # per-source train/eval split; train the probe on the MIX of train_benign_sources
    frac = cfg.data.train_frac
    nh = int(len(h_feat) * frac)
    h_tr, h_ev = h_feat[:nh], h_feat[nh:]
    b_ev = {}
    b_tr_parts = []
    for name, feat in b_feat.items():
        k = int(len(feat) * frac)
        b_ev[name] = feat[k:]                       # held-out portion for eval
        if name in train_benign:
            b_tr_parts.append(feat[:k])             # train portion into the mix
    b_tr = torch.cat(b_tr_parts, dim=0)

    probe = LinearProbe(h_feat.size(-1)).to(device)
    opt = torch.optim.Adam(probe.parameters(), lr=cfg.train.lr)
    X = torch.cat([h_tr, b_tr], dim=0).to(device)
    y = torch.cat([torch.zeros(len(h_tr), dtype=torch.long), torch.ones(len(b_tr), dtype=torch.long)]).to(device)
    for step in range(cfg.train.steps):
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(probe.linear(X), y)
        loss.backward()
        opt.step()
    log.info(f"probe train loss={loss.item():.4f} (trained on benign mix: {train_benign})")

    # recall@1%FPR on held-out harmful vs each benign source's held-out portion
    def p_harm(feat):
        return torch.softmax(probe.linear(feat.to(device)).float(), dim=-1)[:, 0]
    h_scores = p_harm(h_ev)
    for name, feat in b_ev.items():
        tag = "(in mix)" if name in train_benign else "(eval-only OOD)"
        log.info(f"recall@1%FPR vs {name} {tag}: {recall_at_fpr(p_harm(feat), h_scores, fpr=0.01):.3f}")

    out_dir = os.path.join(cfg.output.probe_path, container.get("name") or "probe")
    os.makedirs(out_dir, exist_ok=True)
    torch.save({"state": probe.state_dict(), "cfg": container, "input_dim": h_feat.size(-1), "layer": layer},
               os.path.join(out_dir, "probe.pt"))
    log.info(f"saved probe to {out_dir}/probe.pt")
    return out_dir
