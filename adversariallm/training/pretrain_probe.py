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

from .coop_metrics import recall_at_fpr
from .data import build_prompt_only, load_dataset_prompts, pad_collate
from .readers import LinearProbe, probe_readout

log = logging.getLogger(__name__)


@torch.no_grad()
def _features(model, tokenizer, template_id, prompts, layer, device, batch_size=16):
    """Readout features (unit-normed, fp32) at the generation onset for each prompt."""
    feats = []
    for start in range(0, len(prompts), batch_size):
        chunk = prompts[start : start + batch_size]
        items = [
            dict(zip(("d_ids", "d_targetids", "d_attn"), build_prompt_only(p, tokenizer, template_id))) for p in chunk
        ]
        batch = pad_collate(items, ["d_ids", "d_targetids", "d_attn"], pad_id=0)
        ids = batch["d_ids"].to(device)
        tgt = batch["d_targetids"].to(device)
        attn = batch["d_attn"].to(device)
        hidden = model(input_ids=ids, attention_mask=attn, output_hidden_states=True).hidden_states[layer]
        feats.append(probe_readout(hidden, tgt, attn).cpu())
    return torch.cat(feats, dim=0)


def run_pretrain_probe(cfg):
    from omegaconf import OmegaConf

    from ..defenses.monitors._activation_detector_model import get_chat_template
    from ..io_utils import load_model_and_tokenizer

    container = OmegaConf.to_container(cfg, resolve=True)
    model_params = cfg.models[cfg.model]
    template_id = cfg.chat_template_id
    model, tokenizer = load_model_and_tokenizer(model_params)
    model.eval()
    device = next(model.parameters()).device
    get_chat_template(template_id)  # fail fast if the template id is unsupported
    layer = int(cfg.reader.layer)

    # harmful = the TRAIN behavior set (advbench), disjoint from the HarmBench test set.
    import random

    import pandas as pd

    beh = pd.read_csv(os.path.join(cfg.data.dir, cfg.data.behaviors))
    harmful_all = beh[cfg.data.harmful_col].dropna().tolist()
    random.Random(cfg.data.seed).shuffle(harmful_all)
    harmful = harmful_all[: cfg.data.n_harmful]

    # benign: fit on the TRAIN window of each source, evaluate the diagnostic on the VAL
    # window (canonical split, conf/splits.yaml) — disjoint from coop-val and final-test.
    train_benign = list(cfg.data.train_benign_sources)
    b_tr_parts, b_ev = [], {}
    for name in train_benign:
        tr_prompts, _ = load_dataset_prompts(cfg.datasets, name, window=cfg.splits[name].train, seed=cfg.data.seed)
        val_prompts, _ = load_dataset_prompts(cfg.datasets, name, window=cfg.splits[name].val, seed=cfg.data.seed)
        b_tr_parts.append(_features(model, tokenizer, template_id, tr_prompts, layer, device))
        b_ev[name] = _features(model, tokenizer, template_id, val_prompts, layer, device)
        log.info(f"benign {name}: {len(tr_prompts)} train / {len(val_prompts)} val prompts")

    # harmful train/eval split (both from advbench; HarmBench test untouched)
    h_feat = _features(model, tokenizer, template_id, harmful, layer, device)
    nh = int(len(h_feat) * cfg.data.train_frac)
    h_tr, h_ev = h_feat[:nh], h_feat[nh:]
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

    def p_harm(feat):
        return torch.softmax(probe.linear(feat.to(device)).float(), dim=-1)[:, 0]

    # Honest diagnostic: report recall vs both the in-distribution harmful (advbench, easy —
    # the base model's refusal signal is trivially readable) AND an OOD harmful benchmark
    # (jbb, disjoint from advbench-train and harmbench-test). A big in-dist>OOD gap means the
    # probe is reading the refusal direction, not a robust harm representation. NOTE: both are
    # CLEAN prompts — the real test is ATTACKED harmful (the attack suppresses the refusal
    # signal), which only the in-loop / eval-time attacked recall measures.
    h_scores = p_harm(h_ev)
    ood_src = cfg.data.eval_harmful_source
    eh_prompts, _ = load_dataset_prompts(cfg.datasets, ood_src, window=cfg.data.eval_harmful_window, seed=cfg.data.seed)
    eh_scores = p_harm(_features(model, tokenizer, template_id, eh_prompts, layer, device))
    for name, feat in b_ev.items():
        bs = p_harm(feat)
        log.info(
            f"recall@1%FPR vs {name}: advbench(in-dist)={recall_at_fpr(bs, h_scores, fpr=0.01):.3f}"
            f"  {ood_src}(OOD)={recall_at_fpr(bs, eh_scores, fpr=0.01):.3f}"
        )

    out_dir = os.path.join(cfg.output.probe_path, container.get("name") or "probe")
    os.makedirs(out_dir, exist_ok=True)
    torch.save(
        {"state": probe.state_dict(), "cfg": container, "input_dim": h_feat.size(-1), "layer": layer},
        os.path.join(out_dir, "probe.pt"),
    )
    log.info(f"saved probe to {out_dir}/probe.pt")
    return out_dir
