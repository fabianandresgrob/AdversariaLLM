from __future__ import annotations

import glob
import json
import logging
import os
from typing import Any

from ..lm_utils import LocalTextGenerator
from ..lm_utils.text_generation import GenerationResult, RetryOverrides
from ..types import Conversation
from .base import DefenseDecision, TargetSystem
from .monitors import build_monitor

log = logging.getLogger(__name__)


def _calibrated_threshold(monitor_cfg, fallback, calibration_path=None):
    """Prefer the calibrated operating point written next to the checkpoint by
    run_calibrate_probe.py; the config value is only a fallback.

    run_calibrate_probe.py names its output for the FPR it calibrated at
    (threshold_1pct.json, threshold_5pct.json, ...), so match the whole family
    rather than assuming the fpr=0.01 default. An explicit calibration_path in
    the defense config wins over the search."""
    if calibration_path:
        path = str(calibration_path)
        if not os.path.exists(path):
            raise FileNotFoundError(f"calibration_path={path} does not exist")
    else:
        ckpt = (monitor_cfg or {}).get("checkpoint_path")
        if not ckpt:
            return fallback
        pattern = os.path.join(os.path.dirname(str(ckpt)), "threshold_*pct.json")
        found = sorted(glob.glob(pattern))
        if not found:
            log.warning(f"no calibration matching {pattern}; using threshold={fallback} (NOT a calibrated point)")
            return fallback
        if len(found) > 1:
            raise ValueError(
                f"{len(found)} calibrations match {pattern}: {found}. "
                "Set defense.calibration_path to pick the operating point explicitly."
            )
        path = found[0]
    with open(path) as fh:
        calib = json.load(fh)
    thr = float(calib["threshold"])
    fpr = calib.get("fpr")
    at_fpr = f"{float(fpr):.1%} FPR" if isinstance(fpr, (int, float)) else "unrecorded FPR"
    log.info(f"threshold={thr:.6f} ({at_fpr}, from {path})")
    return thr


def _last_user_content(conv: Conversation) -> str:
    for msg in reversed(conv):
        if msg.get("role") == "user":
            return str(msg.get("content", ""))
    return ""


class MonitorDefense(TargetSystem):
    NAME = "monitor"

    def __init__(self, model, tokenizer, monitor, threshold, refusal_text, default_generate_kwargs=None):
        super().__init__(model, tokenizer)
        self.monitor = monitor
        self.threshold = threshold
        self.refusal_text = refusal_text
        self.text_generator = LocalTextGenerator(
            model, tokenizer, default_generate_kwargs=default_generate_kwargs
        )

    @classmethod
    def from_config(cls, cfg, *, model, tokenizer, default_generate_kwargs=None) -> "MonitorDefense":
        monitor_cfg = dict(cfg["monitor"])
        monitor = build_monitor(monitor_cfg)
        calibration_path = cfg.get("calibration_path")
        return cls(
            model=model,
            tokenizer=tokenizer,
            monitor=monitor,
            threshold=_calibrated_threshold(monitor_cfg, float(cfg.get("threshold", 0.5)), calibration_path),
            refusal_text=cfg["refusal_text"],
            default_generate_kwargs=default_generate_kwargs,
        )

    def generate(
        self,
        convs: list[Conversation],
        num_return_sequences: int | None = None,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        filters: list[dict] | None = None,
        retry_overrides: RetryOverrides | None = None,
        **kwargs: Any,
    ) -> GenerationResult:
        base = self.text_generator.generate(
            convs,
            num_return_sequences=num_return_sequences,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            filters=filters,
            retry_overrides=retry_overrides,
            **kwargs,
        )
        if not base.gen:
            return base

        prompts = [_last_user_content(conv) for conv in convs]
        flat_prompts: list[str] = []
        flat_outputs: list[str] = []
        row_lengths: list[int] = []
        for prompt, outputs in zip(prompts, base.gen):
            row_lengths.append(len(outputs))
            flat_prompts.extend([prompt] * len(outputs))
            flat_outputs.extend(outputs)

        scores = self.monitor.score(
            flat_prompts,
            flat_outputs,
            target_model=self.model,
            target_tokenizer=self.tokenizer,
        )

        flat_decisions = [
            DefenseDecision(
                output_text=self.refusal_text if score >= self.threshold else response,
                metadata={"score": float(score), "applied": bool(score >= self.threshold)},
            )
            for response, score in zip(flat_outputs, scores)
        ]

        defended: list[list[str]] = []
        decisions: list[list[dict[str, Any]]] = []
        idx = 0
        for n in row_lengths:
            defended.append([d.output_text for d in flat_decisions[idx : idx + n]])
            decisions.append([d.metadata for d in flat_decisions[idx : idx + n]])
            idx += n

        return GenerationResult(
            gen=defended,
            input_ids=base.input_ids,
            raw_gen=base.gen,
            defense_decisions=decisions,
        )
