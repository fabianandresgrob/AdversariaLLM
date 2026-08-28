"""Over-refusal eval: base vs coop (vs CAT) on held-out benign sets.

The utility/helpfulness axis of the trade-off (pairs with attack-harness ASR for robustness).
Mirrors coop's _coop_validate over-refusal path EXACTLY — same chat template, greedy generation,
and coop_metrics.refusal_rate — so numbers are parity-comparable to the training `model/
refusal_rate_ood` log. The only generalization: runs base + every coop checkpoint, on the held-out
window (test, or 'all' since xs_test is never trained on), not just 16 val prompts of one model.

    cd ~/projects/AdversariaLLM
    pixi run python run_overrefusal_eval.py \
        +checkpoints.coop_C=checkpoints_coop/coop_v2_C_targets_eps005_s0/final_adapter \
        +checkpoints.coop_D=checkpoints_coop/coop_v2_D_targets_magpie_eps005_s0/final_adapter

    # add datasets explicitly if wanted (large sets need a limit):
    #   datasets_eval=[xs_test,or_bench]        datasets_eval=[xs_test,alpaca] limit=200
"""
from __future__ import annotations

import csv
import json
import logging
import os

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

log = logging.getLogger(__name__)


def _prompts(cfg, ds, which, limit):
    """User prompts for dataset `ds` over the split `which` (train|val|test|all)."""
    from adversariallm.dataset import PromptDataset

    if which == "all":
        idx = None                                     # xs_test is held out entirely -> all rows fair
    else:
        s, e = cfg.splits[ds][which]
        idx = f"list(range({int(s)},{int(e)}))"
    node = OmegaConf.merge(cfg.datasets[ds], {"seed": cfg.val_seed, "idx": idx})
    dataset = PromptDataset.from_name(ds)(node)
    out = []
    for conv in dataset:
        u = next((m["content"] for m in conv if m["role"] == "user"), None)
        if u:
            out.append(u)
    return out[:limit] if limit else out


@torch.no_grad()
def _refusal_on(model, tokenizer, template_id, prompts, max_new_tokens):
    """Greedy-generate on each prompt, count refusals — identical to _coop_validate."""
    from adversariallm.defenses.monitors._activation_detector_model import get_chat_template
    from adversariallm.training.coop_metrics import refusal_rate

    first_user_msg, _, response_key, _, _ = get_chat_template(template_id)
    device = next(model.parameters()).device
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    gens = []
    for p in prompts:
        enc = tokenizer(first_user_msg.format(instruction=p) + response_key, return_tensors="pt").to(device)
        out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=pad_id)
        gens.append(tokenizer.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True))
    return refusal_rate(gens), gens


@hydra.main(version_base=None, config_path="conf", config_name="overrefusal")
def main(cfg: DictConfig) -> None:
    from adversariallm.io_utils import load_model_and_tokenizer

    datasets = list(cfg.datasets_eval)
    prompts = {ds: _prompts(cfg, ds, cfg.window, cfg.limit) for ds in datasets}
    for ds in datasets:
        log.info(f"{ds}: {len(prompts[ds])} prompts ({cfg.window} window)")

    os.makedirs(cfg.out, exist_ok=True)
    results = {}
    for name, spec in cfg.checkpoints.items():
        log.info(f"=== {name} ({spec}) ===")
        model_params = OmegaConf.to_container(cfg.models[cfg.model], resolve=True)  # plain dict (unlock struct)
        if spec != "base":
            model_params["adapter_path"] = spec
        model, tok = load_model_and_tokenizer(model_params)
        results[name] = {}
        for ds in datasets:
            rate, gens = _refusal_on(model, tok, cfg.chat_template_id, prompts[ds], int(cfg.max_new_tokens))
            results[name][ds] = rate
            log.info(f"  {ds:10s} refusal={rate:.3f} (n={len(prompts[ds])})")
            if cfg.dump_gens:
                with open(os.path.join(cfg.out, f"gens_{name}_{ds}.json"), "w") as f:
                    json.dump([{"prompt": p, "generation": g} for p, g in zip(prompts[ds], gens)], f, indent=2)
        del model
        torch.cuda.empty_cache()

    lines = ["", f"# over-refusal rate ({cfg.window} window; lower = more helpful)",
             f"{'checkpoint':16s} " + " ".join(f"{ds:>10s}" for ds in datasets)]
    for name in cfg.checkpoints:
        lines.append(f"{name:16s} " + " ".join(f"{results[name][ds]:>10.3f}" for ds in datasets))
    print("\n".join(lines))

    with open(os.path.join(cfg.out, "overrefusal.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["checkpoint"] + datasets)
        for name in cfg.checkpoints:
            w.writerow([name] + [f"{results[name][ds]:.4f}" for ds in datasets])
    with open(os.path.join(cfg.out, "overrefusal.json"), "w") as f:
        json.dump({"window": cfg.window, "results": results}, f, indent=2)
    log.info(f"wrote {cfg.out}/overrefusal.csv + .json")


if __name__ == "__main__":
    main()
