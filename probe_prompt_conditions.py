"""Does a probe read harm from the answer, or from the prompt in front of it? Score one fixed set of answers with a
model + its probe under different prompts, changing only the prompt:

    pixi run --frozen python probe_prompt_conditions.py model=W-coop-s0-ema

Conditions (each row of `answers`, a jsonl with behavior, answer, optional pair_prompt, source):
  pair               PAIR's own (role-play) prompt + the answer; rows without a pair_prompt skip it
  behavior           the plain behavior as the prompt + the answer (no role-play framing in the activations)
  behavior_attacked  the behavior prompt with a training-style embedding perturbation on the user message (the model
                     -only attack of the coop training, eps / iters / lr as configured, target = the answer) + the answer
  neutral            a fixed prompt with no request in it (`neutral_prompt`) + the answer: what the probe reads from the
                     answer itself, with nothing harmful or disguising in front of it
Same answer, same model, same probe across conditions, so a difference comes from the prompt alone: pair vs behavior
tests whether the framing hides harm from the probe, behavior vs behavior_attacked whether the probe keys on the
perturbation itself. Answers are cut to `max_answer_tokens` in every condition.

Each answer is scored with the probe's P(harmful) and flagged at its 1% Alpaca threshold (threshold_1pct_calib.json
next to the reader, the pipeline's operating point). Writes <out>/<model>.jsonl (one row per answer x condition) and
<out>/<model>.summary.json (flag rate per source x condition); `tag` is appended to both names, so a run of other
conditions does not overwrite an earlier one.
"""

import json
import logging
import os
from collections import defaultdict
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

log = logging.getLogger(__name__)
CONDITIONS = ("pair", "behavior", "behavior_attacked", "neutral")


def example(prompt, answer, tokenizer, max_answer_tokens):
    """(ids, target_ids, attn, user mask) for prompt + answer, the answer cut to max_answer_tokens."""
    from adversariallm.training.data import build_example_full, user_token_mask

    prompt = prompt.strip()  # the chat template strips the user message (PAIR prompts start with a space)
    ids, _, tgt, attn = build_example_full(prompt, answer, tokenizer)
    start = int((tgt != 0).nonzero()[0]) if (tgt != 0).any() else ids.numel()
    end = min(ids.numel(), start + max_answer_tokens)
    ids, tgt, attn = ids[:end], tgt[:end], attn[:end]
    return {"h_ids": ids, "h_targetids": tgt, "h_attn": attn, "h_perturb_mask": user_token_mask(tokenizer, prompt, end)}


def collate(items, device):
    from adversariallm.training.data import pad_collate

    batch = pad_collate(items, ["h_ids", "h_targetids", "h_attn", "h_perturb_mask"], pad_id=0)
    return {k: v.to(device) for k, v in batch.items()}


@torch.no_grad()
def probe_scores(model, reader, layer, batch, inputs_embeds=None):
    if inputs_embeds is None:
        out = model(input_ids=batch["h_ids"], attention_mask=batch["h_attn"], output_hidden_states=True)
    else:
        out = model(inputs_embeds=inputs_embeds, attention_mask=batch["h_attn"], output_hidden_states=True)
    return reader.p_harmful(out.hidden_states[layer], batch["h_targetids"], batch["h_attn"]).float().cpu().tolist()


@hydra.main(version_base=None, config_path="conf", config_name="probe_prompt_conditions")
def main(cfg: DictConfig) -> None:
    from adversariallm.io_utils import load_model_and_tokenizer
    from adversariallm.training.attacks import ContinuousEmbeddingAttack, embedding_scale
    from adversariallm.training.data import generation_prefix
    from adversariallm.training.readers import load_reader

    entry = OmegaConf.to_container(cfg.models[cfg.model], resolve=True)
    model, tokenizer = load_model_and_tokenizer(entry)
    model.eval().requires_grad_(False)
    device = next(model.parameters()).device
    reader = load_reader(entry["reader_path"]).to(device)
    threshold = json.loads((Path(entry["reader_path"]).parent / "threshold_1pct_calib.json").read_text())["threshold"]
    attack = ContinuousEmbeddingAttack(model.get_input_embeddings().weight, generation_prefix(tokenizer), tokenizer,
                                       iters=int(cfg.attack.iters), eps=float(cfg.attack.eps), lr=float(cfg.attack.lr),
                                       target_eot=False, perturb="user", embed_scale=embedding_scale(model))
    torch.manual_seed(int(cfg.seed))

    rows = [json.loads(line) for line in open(cfg.answers)]
    out_rows = []
    for cond in cfg.conditions:
        if cond not in CONDITIONS:
            raise ValueError(f"unknown condition {cond!r}; choose from {CONDITIONS}")
        todo = [r for r in rows if cond != "pair" or r.get("pair_prompt")]
        for i in range(0, len(todo), int(cfg.batch_size)):
            chunk = todo[i: i + int(cfg.batch_size)]
            prompt = {"pair": lambda r: r["pair_prompt"], "neutral": lambda r: str(cfg.neutral_prompt)}.get(
                cond, lambda r: r["behavior"])
            batch = collate([example(prompt(r), r["answer"], tokenizer, int(cfg.max_answer_tokens)) for r in chunk], device)
            embeds = attack.attack(model, batch) if cond == "behavior_attacked" else None
            for r, p in zip(chunk, probe_scores(model, reader, int(cfg.layer), batch, embeds)):
                out_rows.append({"id": r["id"], "source": r["source"], "condition": cond, "p_probe": p,
                                 "flagged": p > threshold, "p_harmful_judge": r.get("p_harmful")})
        log.info(f"{cfg.model} {cond}: {len(todo)} answers scored")

    os.makedirs(cfg.out, exist_ok=True)
    stem = f"{cfg.model}{cfg.get('tag') or ''}"
    with open(os.path.join(cfg.out, f"{stem}.jsonl"), "w") as fh:
        fh.writelines(json.dumps(r) + "\n" for r in out_rows)
    groups = defaultdict(list)
    for r in out_rows:
        groups[(r["source"], r["condition"])].append(r)
    summary = {"model": cfg.model, "threshold": threshold, "flag_rate": {
        f"{s}|{c}": {"n": len(v), "flagged": round(sum(x["flagged"] for x in v) / len(v), 4),
                     "p_probe_median": round(sorted(x["p_probe"] for x in v)[len(v) // 2], 4)}
        for (s, c), v in sorted(groups.items())}}
    Path(cfg.out, f"{stem}.summary.json").write_text(json.dumps(summary, indent=1))
    log.info(json.dumps(summary["flag_rate"], indent=1))


if __name__ == "__main__":
    main()
