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


@torch.no_grad()
def _judge_all(cfg, prompts, gens_by, datasets):
    """Load the judge once (after eval models are freed), score every checkpoint x dataset.

    Replaces the string-match refusal_rate with the xstest 3/4-class judge + degeneration
    detector: malformed generations count as refusals, not compliance."""
    from adversariallm.io_utils import load_model_and_tokenizer
    from adversariallm.lm_utils.text_generation import LocalTextGenerator, generate_from_prompts
    from adversariallm.training.xstest_judge import (
        build_judge_prompt, compute_quality_metrics, detect_degeneration, parse_judgment,
    )

    jid = cfg.judge.model
    judge_params = {
        "id": jid, "tokenizer_id": jid, "short_name": jid.split("/")[-1],
        "developer_name": jid.split("/")[0] if "/" in jid else jid,
        "compile": False, "dtype": cfg.judge.dtype, "chat_template": None, "trust_remote_code": False,
    }
    log.info(f"loading judge {jid} ({cfg.judge.prompt_mode})")
    judge_model, judge_tok = load_model_and_tokenizer(judge_params)
    judge_gen = LocalTextGenerator(judge_model, judge_tok)

    deg, mode = cfg.degeneration, cfg.judge.prompt_mode
    out = {}
    for name in cfg.checkpoints:
        out[name] = {}
        for ds in datasets:
            gens = gens_by[(name, ds)]
            jprompts = [build_judge_prompt(p, g, mode) for p, g in zip(prompts[ds], gens)]
            raw = generate_from_prompts(judge_gen, jprompts, max_new_tokens=int(cfg.judge.max_new_tokens), temperature=0.0)
            parsed = [parse_judgment(r, mode) for r in raw]
            reasons = [detect_degeneration(g, min_length=deg.min_length, max_alnum_ratio=deg.max_alnum_ratio,
                       repeated_char_threshold=deg.repeated_char_threshold,
                       control_character_threshold=deg.control_character_threshold)
                       if deg.enabled else [] for g in gens]
            m = compute_quality_metrics(parsed, reasons)
            out[name][ds] = {"metrics": m, "judgments": parsed, "degeneration_reasons": reasons, "raw_judge": raw}
            log.info(f"  [judge] {name:16s} {ds}: adj_refusal={m['degeneration_adjusted_refusal_rate']:.3f} "
                     f"coherent_compliance={m['coherent_compliance_rate']:.3f} degen={m['degeneration_rate']:.3f}")
    del judge_model, judge_gen
    torch.cuda.empty_cache()
    return out


@hydra.main(version_base=None, config_path="conf", config_name="overrefusal")
def main(cfg: DictConfig) -> None:
    from adversariallm.io_utils import load_model_and_tokenizer

    datasets = list(cfg.datasets_eval)
    prompts = {ds: _prompts(cfg, ds, cfg.window, cfg.limit) for ds in datasets}
    for ds in datasets:
        log.info(f"{ds}: {len(prompts[ds])} prompts ({cfg.window} window)")

    os.makedirs(cfg.out, exist_ok=True)
    results = {}
    gens_by = {}                                       # (name, ds) -> generations, reused by the judge
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
            gens_by[(name, ds)] = gens
            log.info(f"  {ds:10s} refusal={rate:.3f} (n={len(prompts[ds])})")
            if cfg.dump_gens:
                with open(os.path.join(cfg.out, f"gens_{name}_{ds}.json"), "w") as f:
                    json.dump([{"prompt": p, "generation": g} for p, g in zip(prompts[ds], gens)], f, indent=2)
        del model
        torch.cuda.empty_cache()

    judged = _judge_all(cfg, prompts, gens_by, datasets) if cfg.judge.enabled else {}

    lines = ["", f"# string-match refusal rate ({cfg.window} window; lower = more helpful)",
             f"{'checkpoint':16s} " + " ".join(f"{ds:>10s}" for ds in datasets)]
    for name in cfg.checkpoints:
        lines.append(f"{name:16s} " + " ".join(f"{results[name][ds]:>10.3f}" for ds in datasets))
    if judged:                                         # judge over single-dataset runs -> compact table
        for ds in datasets:
            lines += ["", f"# judge metrics on {ds} (adj_refusal = degeneration-adjusted; degen counts as refusal)",
                      f"{'checkpoint':16s} {'str_refusal':>12s} {'adj_refusal':>12s} {'coh_comply':>12s} {'partial':>12s} {'degen':>12s}"]
            for name in cfg.checkpoints:
                m = judged[name][ds]["metrics"]
                lines.append(f"{name:16s} {results[name][ds]:>12.3f} {m['degeneration_adjusted_refusal_rate']:>12.3f} "
                             f"{m['coherent_compliance_rate']:>12.3f} {m['coherent_partial_rate']:>12.3f} {m['degeneration_rate']:>12.3f}")
    print("\n".join(lines))

    with open(os.path.join(cfg.out, "overrefusal.csv"), "w", newline="") as f:
        w = csv.writer(f)
        header = ["checkpoint"] + datasets
        if judged:
            header += [f"{ds}_{k}" for ds in datasets for k in ("adj_refusal", "coh_comply", "partial", "degen")]
        w.writerow(header)
        for name in cfg.checkpoints:
            row = [name] + [f"{results[name][ds]:.4f}" for ds in datasets]
            if judged:
                for ds in datasets:
                    m = judged[name][ds]["metrics"]
                    row += [f"{m['degeneration_adjusted_refusal_rate']:.4f}", f"{m['coherent_compliance_rate']:.4f}",
                            f"{m['coherent_partial_rate']:.4f}", f"{m['degeneration_rate']:.4f}"]
            w.writerow(row)
    with open(os.path.join(cfg.out, "overrefusal.json"), "w") as f:
        payload = {"window": cfg.window, "results": results}
        if judged:
            payload["judge"] = {"model": cfg.judge.model, "prompt_mode": cfg.judge.prompt_mode,
                                "metrics": {n: {ds: judged[n][ds]["metrics"] for ds in datasets} for n in cfg.checkpoints}}
        json.dump(payload, f, indent=2)
    if judged:                                         # per-example judgments for manual inspection
        for name in cfg.checkpoints:
            for ds in datasets:
                jd = judged[name][ds]
                with open(os.path.join(cfg.out, f"judged_{name}_{ds}.json"), "w") as f:
                    json.dump([{"prompt": p, "generation": g, "judgment": pj, "raw_judge": rj,
                                "degeneration_reasons": dr}
                               for p, g, pj, rj, dr in zip(prompts[ds], gens_by[(name, ds)],
                                                           jd["judgments"], jd["raw_judge"], jd["degeneration_reasons"])],
                              f, indent=2)
    log.info(f"wrote {cfg.out}/overrefusal.csv + .json" + (" + judged_*.json" if judged else ""))


if __name__ == "__main__":
    main()
