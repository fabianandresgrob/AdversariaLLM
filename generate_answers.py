"""A model's own greedy answers to a file of prompts, and a judge on them, kept in an answer store.

    uv run python generate_answers.py prompts=/fast/.../llama_test_v1.parquet store=esg-llama31-base
    uv run python generate_answers.py prompts=... store=esg-llama31-base stage=judge

`prompts` is a parquet file with `row_id` and `text` (scripts/esg_build_sample.py), and for the judge `judge_text`
and `judge_against`. generate writes <out>/<store>/answers.<shard>of<n>.jsonl, judge writes judge.<judge>.jsonl
(adversariallm/io_utils/answer_store.py). Both skip row ids that are already stored, so a job that was restarted
continues, and a larger prompt file adds to the same store.

Prompts are batched by length with left padding. Prompts above max_prompt_tokens are stored with status too_long and
no answer. The judge reads `judge_text` as the request (the plain twin of a rewrite where one exists) and skips
answers without text.
"""

import logging
import os

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

log = logging.getLogger(__name__)

FIXED = ("model", "adapter_path", "template", "max_new_tokens", "max_prompt_tokens", "system_prompt", "text_column")


def batches(lengths: list[int], token_budget: int, max_rows: int) -> list[list[int]]:
    """Indices grouped longest first, each batch with at most max_rows rows and rows x longest row <= token_budget."""
    order = sorted(range(len(lengths)), key=lambda i: -lengths[i])
    out, cur = [], []
    for i in order:
        longest = lengths[cur[0]] if cur else lengths[i]
        if cur and (len(cur) >= max_rows or (len(cur) + 1) * longest > token_budget):
            out.append(cur)
            cur = []
        cur.append(i)
    if cur:
        out.append(cur)
    return out


def load(cfg):
    """Model and tokenizer of a models.yaml entry, with the tokenizer's own chat template for template=theirs."""
    from transformers import AutoTokenizer

    from adversariallm.io_utils.model_loading import load_model_and_tokenizer

    params = OmegaConf.to_container(cfg.models[cfg.model], resolve=True)
    if cfg.adapter_path:
        params["adapter_path"] = cfg.adapter_path
    model, tok = load_model_and_tokenizer(params)  # chat_template: ours (the entry's)
    device = next(model.parameters()).device
    if device.type != "cuda" and not cfg.allow_cpu:  # device_map=auto falls back to the CPU without an error
        raise RuntimeError(f"model loaded on {device}, not on a GPU (allow_cpu=true to run anyway)")
    if cfg.template == "theirs":
        tok.chat_template = AutoTokenizer.from_pretrained(params["tokenizer_id"]).chat_template
    elif cfg.template != "ours":
        raise ValueError(f"template must be theirs or ours, got {cfg.template!r}")
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    log.info("template %s, first prompt: %r", cfg.template,
             tok.apply_chat_template([{"role": "user", "content": "X"}], tokenize=False, add_generation_prompt=True))
    return model.eval(), tok


def system_prompt(cfg) -> str | None:
    """The text of cfg.system_prompt_file, or None without one."""
    if not cfg.system_prompt_file:
        return None
    with open(cfg.system_prompt_file) as fh:
        return fh.read().strip()


def messages(text: str, system_prompt: str | None) -> list[dict]:
    """The chat for one prompt, with the system prompt in front if there is one."""
    system = [{"role": "system", "content": system_prompt}] if system_prompt else []
    return system + [{"role": "user", "content": text}]


@torch.no_grad()
def generate(cfg, rows, store):
    import time

    from adversariallm.io_utils import answer_store
    from adversariallm.training.coop_metrics import is_refusal

    path = os.path.join(store, f"answers.{cfg.shard}of{cfg.num_shards}.jsonl")
    done = answer_store.done_ids(store, "answers.")
    rows = [r for r in rows[cfg.shard::cfg.num_shards] if r["row_id"] not in done]
    if cfg.limit:
        rows = rows[: int(cfg.limit)]
    log.info("shard %d of %d: %d prompts to answer, %d already stored", cfg.shard, cfg.num_shards, len(rows), len(done))
    if not rows:
        return
    model, tok = load(cfg)
    device = next(model.parameters()).device
    texts = [tok.apply_chat_template(messages(r[cfg.text_column], system_prompt(cfg)), tokenize=False,
                                     add_generation_prompt=True) for r in rows]
    lengths = [len(ids) for ids in tok(texts, add_special_tokens=False)["input_ids"]]
    long = [i for i, n in enumerate(lengths) if n > cfg.max_prompt_tokens]
    answer_store.append(path, [{"row_id": rows[i]["row_id"], "status": "too_long", "answer": None,
                                "n_prompt_tokens": lengths[i]} for i in long])
    keep = [i for i in range(len(rows)) if lengths[i] <= cfg.max_prompt_tokens]
    stop = {tok.eos_token_id, *(model.generation_config.eos_token_id if isinstance(
        model.generation_config.eos_token_id, list) else [model.generation_config.eos_token_id])}
    t0, n_done, n_tokens = time.time(), 0, 0
    for batch in batches([lengths[i] for i in keep], int(cfg.token_budget), int(cfg.max_rows)):
        idx = [keep[j] for j in batch]
        enc = tok([texts[i] for i in idx], return_tensors="pt", padding=True, add_special_tokens=False).to(device)
        out = model.generate(**enc, max_new_tokens=int(cfg.max_new_tokens), do_sample=False,
                             pad_token_id=tok.pad_token_id)[:, enc["input_ids"].shape[1]:]
        recs = []
        for i, ids in zip(idx, out.tolist()):
            end = next((k for k, t in enumerate(ids) if t in stop), None)
            ids = ids if end is None else ids[:end]
            answer = tok.decode(ids, skip_special_tokens=True)
            recs.append({"row_id": rows[i]["row_id"], "status": "ok", "answer": answer,
                         "n_prompt_tokens": lengths[i], "n_new_tokens": len(ids), "hit_max_new_tokens": end is None,
                         "refusal": bool(is_refusal(answer))})
            n_tokens += len(ids)
        answer_store.append(path, recs)
        n_done += len(idx)
        if n_done % int(cfg.log_every) < len(idx):
            dt = time.time() - t0
            log.info("%d of %d prompts, %.1f prompts/s, %.0f new tokens/s", n_done, len(keep), n_done / dt, n_tokens / dt)
    dt = time.time() - t0
    log.info("done: %d prompts in %.0f s (%.1f prompts/s), %d too long", n_done, dt, n_done / max(dt, 1e-9), len(long))


def judge(cfg, rows, store):
    from judgezoo import Judge

    from adversariallm.io_utils import answer_store

    by_id = {r["row_id"]: r for r in rows}
    done = answer_store.done_ids(store, f"judge.{cfg.judge}.")
    answers = [a for a in answer_store.read_jsonl(os.path.join(store, "answers.*.jsonl"))
               if a.get("answer") and a["row_id"] in by_id and a["row_id"] not in done]
    if cfg.judge_labels is not None:
        answers = [a for a in answers if int(by_id[a["row_id"]]["label"]) in list(cfg.judge_labels)]
    log.info("%d answers to judge with %s, %d already judged", len(answers), cfg.judge, len(done))
    if not answers:
        return
    judge_model = Judge.from_name(cfg.judge)
    path = os.path.join(store, f"judge.{cfg.judge}.jsonl")
    for start in range(0, len(answers), int(cfg.judge_chunk)):
        chunk = answers[start:start + int(cfg.judge_chunk)]
        convs = [[{"role": "user", "content": by_id[a["row_id"]].get("judge_text") or by_id[a["row_id"]]["text"]},
                  {"role": "assistant", "content": a["answer"]}] for a in chunk]
        scores = judge_model(convs)["p_harmful"]
        answer_store.append(path, [{"row_id": a["row_id"], "p_harmful": None if s is None else float(s),
                                    "judge_against": by_id[a["row_id"]].get("judge_against", "self")}
                                   for a, s in zip(chunk, scores)])
        log.info("judged %d of %d", start + len(chunk), len(answers))


@hydra.main(version_base=None, config_path="conf", config_name="generate_answers")
def main(cfg: DictConfig) -> None:
    import pandas as pd

    from adversariallm.io_utils import answer_store

    if cfg.prompts is None or cfg.store is None:
        raise ValueError("prompts and store are required")
    store = os.path.join(cfg.out, cfg.store)
    meta = {"model": cfg.model, "model_id": cfg.models[cfg.model].id, "adapter_path": cfg.adapter_path,
            "template": cfg.template, "max_new_tokens": int(cfg.max_new_tokens),
            "max_prompt_tokens": int(cfg.max_prompt_tokens), "decoding": "greedy, batched with left padding",
            "system_prompt": system_prompt(cfg), "text_column": None if cfg.text_column == "text" else cfg.text_column,
            "first_prompts": str(cfg.prompts)}
    answer_store.check_meta(store, meta, FIXED)
    rows = pd.read_parquet(cfg.prompts).sort_values("row_id").to_dict("records")
    if not cfg.allow_cpu:  # before any model loads: a GPU that cannot be used sends model and judge to the CPU
        torch.zeros(1, device="cuda")
    if cfg.stage == "generate":
        generate(cfg, rows, store)
    elif cfg.stage == "judge":
        judge(cfg, rows, store)
    else:
        raise ValueError(f"stage must be generate or judge, got {cfg.stage!r}")


if __name__ == "__main__":
    main()
