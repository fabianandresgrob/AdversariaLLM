"""Cross-attack eval: test a FINISHED coop checkpoint's model+detector pair under an attack
condition that may differ from what it trained under (native / model-only / detector-aware).

Post-hoc only -- run after training, never during it. Reuses _coop_validate exactly as training
does, so the numbers are apples-to-apples with the training-time validation logs; the only thing
that changes is which attack drives the harmful batches. The point: training only ever tests a
checkpoint against its OWN attack, so a low comply_rate under a detector-aware attack could mean
the model genuinely hardened, or just that the attacker's per-example budget is split between
eliciting harm and evading the detector. Running the SAME checkpoint under a plain model-only
attack (eval_use_detector: false) separates the two.

    cd ~/projects/AdversariaLLM
    pixi run python run_cross_attack_eval.py \
        +checkpoints.eps005_s0.adapter=checkpoints_coop/coop-magpie-eps005-s0/final_adapter \
        +checkpoints.eps005_det6.adapter=checkpoints_coop/coop-magpie-eps005-detaware-ndet6-s0/final_adapter

    # force a single eval condition instead of the native+true+false sweep:
    #   eval_use_detector=[false]
"""
from __future__ import annotations

import json
import logging
import os

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

log = logging.getLogger(__name__)


def _reader_path(adapter_path: str) -> str:
    """<tag>_adapter/ -> sibling <tag>_reader.pt (the _save_pair naming convention)."""
    d, tag_adapter = os.path.split(adapter_path.rstrip("/"))
    tag = tag_adapter[: -len("_adapter")] if tag_adapter.endswith("_adapter") else tag_adapter
    return os.path.join(d, f"{tag}_reader.pt")


def _load_checkpoint(cfg, name, spec):
    """Load model(+adapter) and reader(+its own training cfg, if a checkpoint was saved)."""
    from adversariallm.io_utils import load_model_and_tokenizer
    from adversariallm.training.readers import build_reader

    model_params = OmegaConf.to_container(cfg.models[cfg.model], resolve=True)
    adapter = spec.get("adapter")
    if adapter:
        model_params["adapter_path"] = adapter
    model, tok = load_model_and_tokenizer(model_params)
    device = next(model.parameters()).device
    hidden_dim = model.get_input_embeddings().weight.shape[-1]

    reader_cfg = OmegaConf.to_container(cfg.reader, resolve=True)
    reader = build_reader(reader_cfg, hidden_dim).to(device)
    reader_path = spec.get("reader") or (_reader_path(adapter) if adapter else None)
    train_cfg = None
    if reader_path and os.path.exists(reader_path):
        ckpt = torch.load(reader_path, map_location=device)
        reader.load_state_dict(ckpt["reader"])
        train_cfg = ckpt.get("cfg")
        log.info(f"[{name}] loaded reader from {reader_path}")
    else:
        log.warning(f"[{name}] no reader checkpoint at {reader_path!r}; using an untrained reader")
    return model, tok, reader, train_cfg


def _native_use_detector(train_cfg) -> bool:
    if not train_cfg:
        return False
    return bool((train_cfg.get("attack") or {}).get("use_detector", False))


def _case_frequencies(model, reader, layer, attack, harmful_batches, use_detector, tau=1.0):
    """pipeline/case_A..D on THIS eval's held-out harmful batches, under the given attack
    condition. Same definition as training (w_harm/w_miss at a fixed 0.5 threshold, via
    gating.py), but computed directly here rather than accumulated from training-step history --
    NOT numerically comparable to a run's own training-log pipeline/case_*, only internally
    consistent across the conditions this script runs (all on the same held-out data)."""
    from adversariallm.training.coop_loop import _adv_embeds, _hidden_and_logits
    from adversariallm.training.coop_metrics import four_case_frequencies
    from adversariallm.training.gating import avg_logprob, w_harm as w_harm_fn, w_miss as w_miss_fn
    from adversariallm.training.loop import _benign_under_adv_prompt

    has_feats = hasattr(reader, "readout") and hasattr(reader, "linear")
    wh_all, wm_all = [], []
    for b in harmful_batches:
        adv_embeds = _adv_embeds(attack, model, b, reader=reader, use_detector=use_detector)
        with torch.no_grad():
            h_hidden, logits_h = _hidden_and_logits(model, layer, inputs_embeds=adv_embeds, attention_mask=b["h_attn"])
            be, b_attn, b_labels = _benign_under_adv_prompt(model, adv_embeds, b)
            logits_s = model(inputs_embeds=be, attention_mask=b_attn).logits
            if has_feats:
                f = reader.readout(h_hidden, b["h_targetids"], b["h_attn"])
                p_harm = torch.softmax(reader.linear(f).float(), dim=-1)[:, 0]
            else:
                p_harm = reader.p_harmful(h_hidden, b["h_targetids"], b["h_attn"])
            lp_h = avg_logprob(logits_h[:, :-1], b["h_labels"][:, 1:])
            lp_s = avg_logprob(logits_s[:, :-1], b_labels[:, 1:])
            wh_all.append(w_harm_fn(lp_h, lp_s, tau=tau))
            wm_all.append(w_miss_fn(p_harm))
    return four_case_frequencies(torch.cat(wh_all), torch.cat(wm_all))


@hydra.main(version_base=None, config_path="conf", config_name="cross_attack_eval")
def main(cfg: DictConfig) -> None:
    from torch.utils.data import DataLoader

    from adversariallm.defenses.monitors._activation_detector_model import get_chat_template
    from adversariallm.training.attacks import ContinuousEmbeddingAttack
    from adversariallm.training.coop_loop import _coop_validate
    from adversariallm.training.data import (
        AdvTupleStream, BenignStream, collate_adv, collate_benign, load_dataset_prompts, split_adv_stream,
    )
    from adversariallm.training.loop import _to_device

    os.makedirs(cfg.out, exist_ok=True)
    layer = int(cfg.reader.layer)

    results = {}
    for name, spec in cfg.checkpoints.items():
        log.info(f"=== {name} ===")
        model, tok, reader, train_cfg = _load_checkpoint(cfg, name, spec)
        model.requires_grad_(False)  # eval-only: freeze adapter+base so the attack's backward
        device = next(model.parameters()).device  # only ever builds a graph for the perturbation

        # held-out harmful val batches: identical construction to training (deterministic split)
        adv_ds = AdvTupleStream(
            data_dir=cfg.harmful.data_dir, behaviors_csv=cfg.harmful.behaviors,
            targets_json=cfg.harmful.targets, safe_csv=cfg.harmful.safe,
            tokenizer=tok, model_name=cfg.chat_template_id,
        )
        _, adv_val_ds = split_adv_stream(adv_ds, val_size=int(cfg.harmful.val_size), seed=int(cfg.harmful.val_seed))
        harmful_batches = [
            _to_device(b, device)
            for b in DataLoader(adv_val_ds, batch_size=int(cfg.harmful.batch_size), shuffle=False, collate_fn=collate_adv)
        ]

        # pinned calibration benign (sets the 1%-FPR threshold) + held-out xs_test, same pools as training
        calib_name = cfg.benign.calibration_benign
        calib_p, calib_r = load_dataset_prompts(cfg.datasets, calib_name, window=cfg.splits[calib_name].val, seed=int(cfg.val_seed))
        calib_ds = BenignStream(list(zip(calib_p, calib_r)), tok, cfg.chat_template_id)
        calib_batches = [
            _to_device(b, device)
            for b in DataLoader(calib_ds, batch_size=int(cfg.harmful.batch_size), shuffle=False, collate_fn=collate_benign)
        ]
        xs_p, xs_r = load_dataset_prompts(cfg.datasets, "xs_test", window=cfg.splits.xs_test.val, seed=int(cfg.val_seed))
        xs_ds = BenignStream(list(zip(xs_p, xs_r)), tok, cfg.chat_template_id)
        xs_batches = [
            _to_device(b, device)
            for b in DataLoader(xs_ds, batch_size=int(cfg.harmful.batch_size), shuffle=False, collate_fn=collate_benign)
        ]
        xs_prompts = [p for p, _ in xs_ds.rows[: int(cfg.benign.benign_gen_n)]]

        _, _, response_key, _, _ = get_chat_template(cfg.chat_template_id)
        attack = ContinuousEmbeddingAttack(
            model.get_input_embeddings().weight, response_key, tok,
            iters=int(cfg.attack.iters), eps=float(cfg.attack.eps), lr=float(cfg.attack.lr),
            detector_loss_coeff=float(cfg.attack.detector_loss_coeff), detector_layer=layer,
        )

        native = _native_use_detector(train_cfg)
        conditions = sorted({native if c == "native" else bool(c) for c in cfg.eval_use_detector})

        results[name] = {}
        for use_det in conditions:
            tag = "detector_aware" if use_det else "model_only"
            log.info(f"[{name}] evaluating under {tag} attack (native={native})")
            metrics = _coop_validate(
                model, reader, layer, harmful_batches, calib_batches, xs_batches, xs_prompts,
                [],  # easy_help_batches: w_mb sanity check not needed for this eval
                tok, cfg.chat_template_id, int(cfg.benign.max_new_tokens), attack, 1.0,
                [], cfg.out, 0, use_detector=use_det,
            )
            cases = _case_frequencies(model, reader, layer, attack, harmful_batches, use_det)
            results[name][tag] = {
                "native": use_det == native,
                "comply": metrics["model/comply_rate"],
                "recall": metrics["detector/recall@1fpr"],
                "asr": metrics["pipeline/asr"],
                "saved": metrics["pipeline/detector_saved"],
                "case_A": cases["A"], "case_B": cases["B"], "case_C": cases["C"], "case_D": cases["D"],
            }
            log.info(f"  comply={metrics['model/comply_rate']:.3f} recall={metrics['detector/recall@1fpr']:.3f} "
                     f"asr={metrics['pipeline/asr']:.3f} saved={metrics['pipeline/detector_saved']:.3f} "
                     f"cases=A{cases['A']:.2f}/B{cases['B']:.2f}/C{cases['C']:.2f}/D{cases['D']:.2f}")
        del model
        torch.cuda.empty_cache()

    lines = ["", "# cross-attack eval (native = the attack condition the checkpoint trained under)",
             "# case_A..D computed on THIS eval's held-out data (not the training log's pipeline/case_*)",
             f"{'checkpoint':28s} {'eval_attack':15s} {'native':7s} {'comply':>8s} {'recall':>8s} {'asr':>8s} "
             f"{'saved':>8s} {'A':>6s} {'B':>6s} {'C':>6s} {'D':>6s}"]
    for name, conds in results.items():
        for tag, m in conds.items():
            lines.append(f"{name:28s} {tag:15s} {str(m['native']):7s} "
                        f"{m['comply']:8.3f} {m['recall']:8.3f} {m['asr']:8.3f} {m['saved']:8.3f} "
                        f"{m['case_A']:6.3f} {m['case_B']:6.3f} {m['case_C']:6.3f} {m['case_D']:6.3f}")
    print("\n".join(lines))

    with open(os.path.join(cfg.out, "cross_attack_eval.json"), "w") as f:
        json.dump(results, f, indent=2)
    log.info(f"wrote {cfg.out}/cross_attack_eval.json")


if __name__ == "__main__":
    main()
