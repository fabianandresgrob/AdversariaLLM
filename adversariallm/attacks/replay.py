"""Transfer eval: replay a prior attack's adversarial prompts through a (defended) target.

Loads the adversarial prompts a white-box attack (e.g. GCG) found against the raw model, then
sends them through whatever target is configured — including a monitor defense — so a
NON-adaptive transfer attack can be scored against the pipeline. Keeps original_prompt = the
clean behavior so the judge scores the true goal, not the suffix. Aligns phase-1 to phase-2 by
behavior text, so the two runs need not share row order.
"""

import copy
import glob
import json
import logging
import os
import time
from dataclasses import dataclass, field

import torch

from ..defenses import TargetSystem
from ..types import Conversation
from .attack import Attack, AttackResult, AttackStepResult, GenerationConfig, SingleAttackRunResult


@dataclass
class ReplayConfig:
    name: str = "replay"
    type: str = "discrete"
    version: str = ""
    generation_config: GenerationConfig = field(default_factory=GenerationConfig)
    seed: int = 0
    source: str = "???"  # phase-1 attack results dir holding <idx>/run.json


def _user_content(conv):
    return next((m["content"] for m in conv if m.get("role") == "user"), None)


def _pick_adv_prompt(steps):
    """User prompt of the committed adversarial step: min-loss step with a model_input, else the
    last such step (the attack's final suffix)."""
    cands = [s for s in steps if s.get("model_input")]
    if not cands:
        return None
    losses = [s.get("loss") for s in cands]
    best = min(cands, key=lambda s: s["loss"]) if all(x is not None for x in losses) else cands[-1]
    return _user_content(best["model_input"])


def _load_adv_prompts(source):
    """behavior text -> adversarial prompt, from a prior attack's per-run run.json files."""
    mapping = {}
    for rj in sorted(glob.glob(os.path.join(source, "*", "run.json"))):
        with open(rj) as f:
            data = json.load(f)
        for run in data.get("runs", []):
            behavior = _user_content(run.get("original_prompt", []))
            adv = _pick_adv_prompt(run.get("steps", []))
            if behavior and adv:
                mapping[behavior] = adv
    return mapping


class ReplayAttack(Attack):
    def __init__(self, config):
        super().__init__(config)

    @torch.no_grad
    def run(self, target: TargetSystem, dataset) -> AttackResult:
        t0 = time.time()
        adv = _load_adv_prompts(self.config.source)

        original_conversations: list[Conversation] = []
        generation_conversations: list[Conversation] = []
        n_hit = 0
        for conversation in dataset:
            behavior = _user_content(conversation)
            adv_prompt = adv.get(behavior)
            n_hit += adv_prompt is not None
            adv_prompt = adv_prompt if adv_prompt is not None else behavior  # fall back to clean behavior
            original_conversations.append(conversation)
            gen_conv = copy.deepcopy(conversation)
            for m in gen_conv:
                if m["role"] == "user":
                    m["content"] = adv_prompt
                elif m["role"] == "assistant":
                    m["content"] = ""
            generation_conversations.append(gen_conv)

        logging.info(
            f"Replay: matched {n_hit}/{len(original_conversations)} behaviors to adversarial "
            f"prompts from {self.config.source}"
        )

        B = len(original_conversations)
        result = target.generate(
            generation_conversations,
            max_new_tokens=self.config.generation_config.max_new_tokens,
            temperature=self.config.generation_config.temperature,
            top_p=self.config.generation_config.top_p,
            top_k=self.config.generation_config.top_k,
            num_return_sequences=self.config.generation_config.num_return_sequences,
            initial_batch_size=B * self.config.generation_config.num_return_sequences,
        )
        completions = result.gen
        input_ids = result.require_input_ids("Replay", expected_len=B)
        t1 = time.time()

        runs = []
        for i in range(B):
            step = AttackStepResult(
                step=0,
                model_completions=completions[i],
                model_completions_raw=result.raw_for(i),
                time_taken=(t1 - t0) / B,
                loss=None,
                flops=0,
                model_input=generation_conversations[i],  # the adversarial prompt actually sent
                model_input_tokens=input_ids[i],
                defense_metadata=result.defense_metadata_for(i),  # probe score + fired?
            )
            runs.append(
                SingleAttackRunResult(
                    original_prompt=original_conversations[i],  # clean behavior -> judge scores the true goal
                    steps=[step],
                    total_time=t1 - t0,
                )
            )
        logging.info(f"Replay attack completed. Total time: {t1 - t0:.2f}s")
        return AttackResult(runs=runs)
