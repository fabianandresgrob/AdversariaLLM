"""Calibrate a coop probe's 1%-FPR threshold for the monitor defense.

Loads the co-trained model (base + adapter) + probe, scores the pinned easy-benign
calibration set through LinearProbeMonitor (the exact eval path), and writes the
(1-fpr) quantile of P(harmful) as the operating threshold — the number to put in
conf/defenses coop_probe.threshold so the pipeline matches coop validation.

A response readout (readers.READOUT_MODES) scores the generation, not just the prompt, so for
those probes the benign responses are generated here first. Scoring them with an empty response
would read the prompt-only fallback position — a feature distribution the probe never trained on
and never meets at eval, where the monitor always runs after generation — and the resulting
threshold would not transfer. Prompt-only probes keep the cheap empty-response path, so their
thresholds stay bit-identical to earlier calibrations.
"""
from __future__ import annotations

import json
import logging
import os

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from adversariallm.training.generation import generate_responses  # noqa: F401 (re-export)

log = logging.getLogger(__name__)


@torch.no_grad()
@hydra.main(version_base=None, config_path="conf", config_name="calibrate_probe")
def main(cfg: DictConfig) -> None:
    from adversariallm.defenses.monitors.linear_probe import LinearProbeMonitor
    from adversariallm.io_utils import load_model_and_tokenizer
    from adversariallm.training.coop_metrics import threshold_at_fpr
    from adversariallm.training.data import load_dataset_prompts

    model_params = OmegaConf.to_container(cfg.models[cfg.model], resolve=True)  # plain dict (unlock struct)
    model_params["adapter_path"] = cfg.adapter_path
    model, tokenizer = load_model_and_tokenizer(model_params)

    monitor = LinearProbeMonitor.from_config(
        {
            "checkpoint_path": cfg.checkpoint_path,
            "target_model_id": cfg.chat_template_id,
            "index_hidden_layer_detector": cfg.layer,
            "batch_size": cfg.batch_size,
        }
    )

    window = cfg.splits[cfg.calibration_benign][cfg.calibration_window]
    prompts, _ = load_dataset_prompts(cfg.datasets, cfg.calibration_benign, window=window, seed=cfg.val_seed)
    if monitor.reads_response(model):
        log.info(f"response readout: generating {len(prompts)} benign responses to calibrate on")
        responses = generate_responses(model, tokenizer, prompts, int(cfg.max_new_tokens), int(cfg.batch_size))
    else:
        responses = [""] * len(prompts)  # prompt-only readout: the response is not read
    scores = monitor.score(prompts, responses, target_model=model, target_tokenizer=tokenizer)
    tau = threshold_at_fpr(scores, fpr=float(cfg.fpr))

    out = {
        "threshold": tau,
        "fpr": float(cfg.fpr),
        "n_benign": len(scores),
        "calibration_benign": cfg.calibration_benign,
        "calibration_window": cfg.calibration_window,
        "window": list(window),
        "checkpoint_path": cfg.checkpoint_path,
        "adapter_path": cfg.adapter_path,
        "calibrated_on": "generated_responses" if any(responses) else "prompts_only",
    }
    # val keeps the historical name; other windows get a suffix so both operating points can coexist
    suffix = "" if cfg.calibration_window == "val" else f"_{cfg.calibration_window}"
    out_path = os.path.join(
        os.path.dirname(cfg.checkpoint_path), f"threshold_{int(float(cfg.fpr) * 100)}pct{suffix}.json"
    )
    with open(out_path, "w") as fh:
        json.dump(out, fh, indent=2)
    log.info(
        f"{float(cfg.fpr):.0%}-FPR threshold tau={tau:.6f} on {len(scores)} benign "
        f"({cfg.calibration_benign} {cfg.calibration_window} {list(window)}); wrote {out_path}"
    )


if __name__ == "__main__":
    main()
