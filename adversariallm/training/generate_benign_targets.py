"""Offline generation of hard-benign helpful targets (y_gen) for the cooperative pipeline (§4).

Greedy-decode the FROZEN base model on the train windows of the hard-benign sources, then keep
only the compliant generations as y_gen; non-compliant (refused/hedged) prompts are kept with
y_gen=null so they still feed the detector as the hardest negatives, and their count is the
base's pre-existing over-refusal rate. THE FILTER IS NOT TRUSTED: hand spot-check a sample of
the kept targets before training (hedges / "can't do X but here's Y" / lecture-then-answer pass
markers and are worse than clean refusals). Output: JSON list of {"prompt", "y_gen"}."""

from __future__ import annotations

import json
import logging
import os

from .coop_metrics import _REFUSAL_MARKERS

log = logging.getLogger(__name__)


def is_compliant(text: str, min_len: int = 20) -> bool:
    """A usable helpful answer: not a refusal marker and not too short. Length is a second,
    weak signal for partial refusals/hedges (they run short); it does NOT replace the hand
    spot-check the spec requires."""
    if not isinstance(text, str):
        return False
    t = text.strip()
    if len(t) < min_len:
        return False
    low = t.lower()
    if low.startswith("sorry") or any(m in low for m in _REFUSAL_MARKERS):
        return False
    return True


def partition_targets(prompts, gens):
    """Zip prompts with generations into JSON rows, nulling non-compliant y_gen. Returns
    (rows, n_refused)."""
    rows, n_refused = [], 0
    for p, g in zip(prompts, gens):
        if is_compliant(g):
            rows.append({"prompt": p, "y_gen": g})
        else:
            rows.append({"prompt": p, "y_gen": None})
            n_refused += 1
    return rows, n_refused


def _generate(model, tokenizer, template_id, prompts, max_new_tokens, device):
    from ..defenses.monitors._activation_detector_model import get_chat_template

    first_user_msg, _, response_key, _, _ = get_chat_template(template_id)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    gens = []
    for p in prompts:
        enc = tokenizer(first_user_msg.format(instruction=p) + response_key, return_tensors="pt").to(device)
        out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=pad_id)
        gens.append(tokenizer.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True))
    return gens


def run_generate_benign_targets(cfg):
    import torch

    from ..io_utils import load_model_and_tokenizer
    from .data import load_dataset_prompts

    model, tokenizer = load_model_and_tokenizer(cfg.models[cfg.model])
    device = next(model.parameters()).device
    model.eval()

    all_rows, total_refused = [], 0
    for name in cfg.hard_benign_sources:
        prompts, _ = load_dataset_prompts(cfg.datasets, name, window=cfg.splits[name].train, seed=cfg.seed)
        with torch.no_grad():
            gens = _generate(model, tokenizer, cfg.chat_template_id, prompts, int(cfg.max_new_tokens), device)
        rows, n_refused = partition_targets(prompts, gens)
        total_refused += n_refused
        all_rows += rows
        log.info(f"{name}: {len(rows)} prompts, {n_refused} base-refused "
                 f"({n_refused / max(len(rows), 1):.1%} pre-existing over-refusal)")

    os.makedirs(os.path.dirname(cfg.out_path), exist_ok=True)
    with open(cfg.out_path, "w") as fh:
        json.dump(all_rows, fh, indent=2)
    log.info(f"wrote {len(all_rows)} rows ({total_refused} base-refused) to {cfg.out_path}. "
             f"HAND SPOT-CHECK a sample of the compliant targets before training.")
