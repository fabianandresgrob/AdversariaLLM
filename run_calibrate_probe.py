"""Calibrate a coop probe's 1%-FPR threshold for the monitor defense.

Loads the co-trained model (base + adapter) + probe, scores the pinned easy-benign
calibration set through LinearProbeMonitor (the exact eval path), and writes the
(1-fpr) quantile of P(harmful) as the operating threshold — the number to put in
conf/defenses coop_probe.threshold so the pipeline matches coop validation.
"""
from __future__ import annotations

import json
import logging
import os

import hydra
from omegaconf import DictConfig, OmegaConf

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="conf", config_name="calibrate_probe")
def main(cfg: DictConfig) -> None:
    from adversariallm.defenses.monitors.linear_probe import LinearProbeMonitor
    from adversariallm.io_utils import load_model_and_tokenizer
    from adversariallm.training.coop_metrics import threshold_at_fpr
    from adversariallm.training.data import load_dataset_prompts

    model_params = OmegaConf.merge(cfg.models[cfg.model], {"adapter_path": cfg.adapter_path})
    model, tokenizer = load_model_and_tokenizer(model_params)

    monitor = LinearProbeMonitor.from_config(
        {
            "checkpoint_path": cfg.checkpoint_path,
            "target_model_id": cfg.chat_template_id,
            "index_hidden_layer_detector": cfg.layer,
            "batch_size": cfg.batch_size,
        }
    )

    window = cfg.splits[cfg.calibration_benign].val
    prompts, _ = load_dataset_prompts(cfg.datasets, cfg.calibration_benign, window=window, seed=cfg.val_seed)
    scores = monitor.score(prompts, [""] * len(prompts), target_model=model, target_tokenizer=tokenizer)
    tau = threshold_at_fpr(scores, fpr=float(cfg.fpr))

    out = {
        "threshold": tau,
        "fpr": float(cfg.fpr),
        "n_benign": len(scores),
        "calibration_benign": cfg.calibration_benign,
        "checkpoint_path": cfg.checkpoint_path,
        "adapter_path": cfg.adapter_path,
    }
    out_path = os.path.join(os.path.dirname(cfg.checkpoint_path), f"threshold_{int(float(cfg.fpr) * 100)}pct.json")
    with open(out_path, "w") as fh:
        json.dump(out, fh, indent=2)
    log.info(f"1%-FPR threshold tau={tau:.6f} on {len(scores)} benign ({cfg.calibration_benign}); wrote {out_path}")


if __name__ == "__main__":
    main()
