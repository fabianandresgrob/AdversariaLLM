from __future__ import annotations

from typing import Any

import torch

from ..lm_utils import LocalTextGenerator
from ..lm_utils.text_generation import GenerationResult, RetryOverrides
from ..types import Conversation
from .base import DefenseDecision, TargetSystem
from .monitors import build_monitor


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
        monitor = build_monitor(dict(cfg["monitor"]))
        return cls(
            model=model,
            tokenizer=tokenizer,
            monitor=monitor,
            threshold=float(cfg.get("threshold", 0.5)),
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
