"""EuroSafeGuard's released prompt probe on stored attack prompts and benign prompt sets (round 1, R1.2 and R1.3):

    uv run python score_prompt_probe.py source=pair                       # every stored PAIR attempt
    uv run python score_prompt_probe.py source=xs_test
    uv run python score_prompt_probe.py source=gcg shard=0 num_shards=4
    uv run python score_prompt_probe.py model=W-coop-s0-ema template=ours prompt_system=W-coop-s0-ema \
        'sources=[xs_test,or_bench,alpaca,pair,jailbreak_r1,gcg]'          # R1.3: the probe reading a trained model

Sources: an attack name (the user message of every attempt in outputs/<attack>__*/**/run.json; the probe reads
the prompt only, so attempts on different systems that sent the same prompt are scored once) or a benign set
(xs_test, or_bench, alpaca). By default the probe reads base Llama-3.1-8B-Instruct under their template
(adversariallm/defenses/monitors/eurosafeguard_probe.py), as in their release (R1.2). For R1.3, `model` names a
models.yaml entry (with its LoRA adapter) whose hidden states the probe reads instead, `template` picks their template
(the tokenizer's own) or ours (the entry's chat_template), and `prompt_system` restricts attack prompts to the attempts
sent to one system (outputs/<attack>__none__<system>/). `sources` scores several sources with one model load.

Writes <out>/<source>.<shard>-of-<num_shards>.jsonl, one row per distinct prompt: sha1 of the prompt text, score (the
probe logit), n_tokens, truncated, and for benign sets the dataset label. Rows are appended and flushed after every
batch, and a restart skips the prompts already in the file. The analysis joins the scores back to the attempts by
sha1(prompt text).
"""

import glob
import hashlib
import json
import logging
import os
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

log = logging.getLogger(__name__)
BENIGN = ("xs_test", "or_bench", "alpaca")


def sha1(text: str) -> str:
    return hashlib.sha1(text.encode()).hexdigest()


def attack_prompts(outputs_dir: str, attack: str, system: str | None = None) -> dict[str, dict]:
    """{sha1: row} of the distinct user messages sent in every stored attempt of `attack` (against `system` alone,
    if given)."""
    rows = {}
    cells = f"{attack}__none__{system}" if system else f"{attack}__*"
    files = sorted(glob.glob(os.path.join(outputs_dir, cells, "**", "run.json"), recursive=True))
    for path in files:
        with open(path) as f:
            runs = json.load(f)["runs"]
        for run in runs:
            for step in run["steps"]:
                user = [m["content"] for m in step.get("model_input") or [] if m["role"] == "user"]
                if len(user) != 1:
                    raise ValueError(f"{path}: expected one user message, got {len(user)}")
                rows.setdefault(sha1(user[0]), {"prompt": user[0]})
    log.info("%s: %d distinct prompts in %d run files", attack, len(rows), len(files))
    return rows


def benign_prompts(name: str, alpaca_n: int, seed: int) -> dict[str, dict]:
    """The same loads as adversariallm/dataset/*.py, all rows (alpaca: a seeded sample of alpaca_n)."""
    from datasets import load_dataset

    if name == "xs_test":
        ds = load_dataset("walledai/XSTest")["test"]
        items = [(r["prompt"], {"label": r["label"], "type": r["type"]}) for r in ds]
    elif name == "or_bench":
        ds = load_dataset("bench-llm/or-bench", "or-bench-hard-1k")["train"]
        items = [(r["prompt"], {"category": r["category"]}) for r in ds]
    elif name == "alpaca":
        ds = load_dataset("tatsu-lab/alpaca")["train"].shuffle(seed=seed).select(range(alpaca_n))
        items = [(r["instruction"], {}) for r in ds]  # instruction only, as AlpacaDataset
    else:
        raise ValueError(f"unknown benign set {name!r}; known: {BENIGN}")
    return {sha1(p): {"prompt": p, **meta} for p, meta in items}


def shard_of(rows: dict[str, dict], shard: int, num_shards: int) -> list[str]:
    return [h for h in sorted(rows) if int(h, 16) % num_shards == shard]


def done_hashes(path: Path) -> set[str]:
    if not path.exists():
        return set()
    out = set()
    for line in path.read_text().splitlines():
        try:
            out.add(json.loads(line)["sha1"])
        except (json.JSONDecodeError, KeyError):
            pass  # a line cut by a kill: scored again
    return out


def batches(order: list[str], lengths: dict[str, int], token_budget: int, max_rows: int):
    """Length-sorted batches with rows x longest row <= token_budget."""
    cur, longest = [], 0
    for h in sorted(order, key=lengths.get):
        n = lengths[h]
        if cur and (len(cur) >= max_rows or (len(cur) + 1) * max(longest, n) > token_budget):
            yield cur
            cur, longest = [], 0
        cur.append(h)
        longest = max(longest, n)
    if cur:
        yield cur


def load_scorer(cfg, pcfg, device):
    """(model, tokenizer) the probe reads: the probe's base model (model=null, R1.2) or a models.yaml entry with its
    adapter (R1.3); the tokenizer's chat template set to theirs or ours."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    base = cfg.base_model or pcfg["base_model"]
    theirs = AutoTokenizer.from_pretrained(base).chat_template
    if cfg.model is None:
        log.info("probe trained on %s, reading %s", pcfg["base_model"], base)
        tok = AutoTokenizer.from_pretrained(base)
        model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16).to(device).eval()
        if cfg.template != "theirs":
            raise ValueError("model=null reads the probe's base model under their template only")
    else:
        from adversariallm.io_utils.model_loading import load_model_and_tokenizer

        params = cfg.models[cfg.model]
        log.info("probe trained on %s, reading models.yaml %s (%s, adapter %s)", pcfg["base_model"], cfg.model,
                 params.id, params.get("adapter_path"))
        model, tok = load_model_and_tokenizer(params)  # chat_template: ours (the entry's)
        if cfg.template == "theirs":
            tok.chat_template = theirs
        elif cfg.template != "ours":
            raise ValueError(f"template must be theirs or ours, got {cfg.template!r}")
    log.info("template %s, first prompt: %r", cfg.template,
             tok.apply_chat_template([{"role": "user", "content": "X"}], tokenize=False, add_generation_prompt=True))
    return model, tok


@hydra.main(config_path="conf", config_name="score_prompt_probe", version_base="1.3")
def main(cfg: DictConfig) -> None:
    from adversariallm.defenses.monitors.eurosafeguard_probe import load_probe, prompt_ids, score_ids, snapshot

    log.info("config:\n%s", OmegaConf.to_yaml(cfg))
    torch.manual_seed(cfg.seed)
    sources = list(cfg.sources) if cfg.sources else [cfg.source]
    work = []
    for source in sources:
        if source in BENIGN:
            rows = benign_prompts(source, cfg.alpaca_n, cfg.seed)
        else:
            rows = attack_prompts(cfg.outputs_dir, source, cfg.prompt_system)
        out = Path(cfg.out) / f"{source}.{cfg.shard}-of-{cfg.num_shards}.jsonl"
        out.parent.mkdir(parents=True, exist_ok=True)
        done = done_hashes(out)
        todo = [h for h in shard_of(rows, cfg.shard, cfg.num_shards) if h not in done]
        log.info("%s shard %d/%d: %d prompts left (%d already in %s)", source, cfg.shard, cfg.num_shards, len(todo),
                 len(done), out)
        if todo:
            work.append((source, rows, todo, out))
    if not work:
        return

    if not torch.cuda.is_available() and not cfg.allow_cpu:
        raise RuntimeError("no GPU visible (allow_cpu=true to score on the CPU anyway)")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("device %s (%s)", device, torch.cuda.get_device_name(0) if device == "cuda" else "cpu")
    probe, pcfg = load_probe(snapshot(cfg.probe_repo), device)
    model, tok = load_scorer(cfg, pcfg, device)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    for source, rows, todo, out in work:
        full_len, ids = {}, {}
        for h in todo:
            x = prompt_ids(tok, rows[h]["prompt"], 10**9)
            full_len[h], ids[h] = len(x), x[-pcfg["max_len"]:]
        lengths = {h: len(x) for h, x in ids.items()}
        n = 0
        with out.open("a") as f:
            for batch in batches(todo, lengths, cfg.token_budget, cfg.max_rows):
                scores = score_ids(model, probe, [ids[h] for h in batch], pcfg["layers"], pcfg["M"], pad)
                for h, s in zip(batch, scores.tolist()):
                    meta = {k: v for k, v in rows[h].items() if k != "prompt"}
                    f.write(json.dumps({"sha1": h, "score": s, "n_tokens": lengths[h],
                                        "truncated": full_len[h] > lengths[h], **meta}) + "\n")
                f.flush()
                n += len(batch)
                if n // cfg.log_every != (n - len(batch)) // cfg.log_every:
                    log.info("%s: %d / %d prompts", source, n, len(todo))
        log.info("done: %s, %d prompts -> %s", source, n, out)


if __name__ == "__main__":
    main()
