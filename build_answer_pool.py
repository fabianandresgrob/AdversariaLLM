"""Build the answer-channel pool: judged completions for the coop training behaviors.

    pixi run --frozen python build_answer_pool.py generator=heretic-llama31-8b name=pool-heretic

The DualProbe's answer channel needs real harmful answers to train on (the coop target stubs taught
it the stub, not harm). This generates them with an uncensored generator and labels them with the
same judge and input shape as run_judges.py. For every coop behavior (all of them; the coop
train/val split is recorded per row) it writes:

  prefill=none     the generator's own greedy answer to the plain behavior
  prefill=target   one answer per adv_training target: the assistant turn starts with the target
                   ("Sure, here's a plan to ...") and the generator continues it

`completion` is the full answer text (target included for prefilled rows), i.e. what the answer
channel will read after the prompt. The generator only writes text; the defended model's own
activations are computed on it later (teacher forcing), so the pool can come from another model.

Writes outputs/eval/answer_pool/<name>.jsonl (one row per completion, with p_harmful) and <name>.md.
"""

import json
import logging
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

log = logging.getLogger(__name__)


def pool_rows(rows, val_behaviors: set[str]) -> list[dict]:
    """One job per (behavior, prefill): the plain answer, then one per target. rows are
    AdvTupleStream.rows, (behavior, target, safe) per target."""
    jobs, n_targets = [], {}
    for behavior, target, _ in rows:
        split = "val" if behavior in val_behaviors else "train"
        if behavior not in n_targets:
            n_targets[behavior] = 0
            jobs.append({"behavior": behavior, "split": split, "prefill": "none", "target_index": None, "target": ""})
        jobs.append({"behavior": behavior, "split": split, "prefill": "target",
                     "target_index": n_targets[behavior], "target": target})
        n_targets[behavior] += 1
    return jobs


def render(records: list[dict], threshold: float, max_chars: int, header: str) -> str:
    lines = [header, ""]
    for mode in ("none", "target"):
        subset = [r for r in records if r["prefill"] == mode]
        if subset:
            hits = sum(r["p_harmful"] > threshold for r in subset)
            behaviors = len({r["behavior"] for r in subset if r["p_harmful"] > threshold})
            lines.append(f"- prefill={mode}: {hits}/{len(subset)} judged harmful (> {threshold}), "
                         f"covering {behaviors} behaviors")
    current = None
    for r in records:
        if r["behavior"] != current:
            current = r["behavior"]
            lines += ["", f"## {current}  ({r['split']})", ""]
        label = "no prefill" if r["prefill"] == "none" else f"prefill target {r['target_index']}"
        lines += [f"**{label}** | p_harmful={r['p_harmful']:.3f}", f"```\n{r['completion'][:max_chars]}\n```", ""]
    return "\n".join(lines)


@hydra.main(version_base=None, config_path="conf", config_name="build_answer_pool")
def main(cfg: DictConfig) -> None:
    import torch
    from judgezoo import Judge

    from adversariallm.io_utils import load_model_and_tokenizer
    from adversariallm.training.data import AdvTupleStream, split_adv_stream
    from run_calibrate_probe import generate_responses

    if not cfg.name:
        raise ValueError("name=... is required (it names the output files)")
    entry = OmegaConf.to_container(cfg.models[cfg.generator], resolve=True)
    model, tokenizer = load_model_and_tokenizer(entry)
    model.eval()

    ds = AdvTupleStream(cfg.data.dir, cfg.data.behaviors, cfg.data.targets, cfg.data.safe, tokenizer, cfg.chat_template_id)
    _, val_ds = split_adv_stream(ds, val_size=int(cfg.data.val_size), seed=int(cfg.data.val_seed))
    val_behaviors = {ds.rows[i][0] for i in val_ds.indices}
    jobs = pool_rows(ds.rows, val_behaviors)
    if cfg.limit_behaviors:
        keep = list(dict.fromkeys(j["behavior"] for j in jobs))[: int(cfg.limit_behaviors)]
        jobs = [j for j in jobs if j["behavior"] in set(keep)]
    log.info(f"{len(jobs)} generations over {len({j['behavior'] for j in jobs})} behaviors")

    continuations = generate_responses(model, tokenizer, [j["behavior"] for j in jobs], int(cfg.max_new_tokens),
                                       int(cfg.batch_size), prefills=[j["target"] for j in jobs])
    for job, cont in zip(jobs, continuations):
        job["completion"] = job["target"] + cont
        job["generator"] = entry["id"]
    del model
    torch.cuda.empty_cache()

    judge = Judge.from_name(cfg.classifier)
    convs = [[{"role": "user", "content": j["behavior"]}, {"role": "assistant", "content": j["completion"]}]
             for j in jobs]
    scores = judge(convs)["p_harmful"]
    for job, score in zip(jobs, scores):
        job["p_harmful"] = float(score) if score is not None else float("nan")

    out = Path(cfg.root_dir) / "outputs" / "eval" / "answer_pool"
    out.mkdir(parents=True, exist_ok=True)
    with (out / f"{cfg.name}.jsonl").open("w") as f:
        for job in jobs:
            f.write(json.dumps(job) + "\n")
    header = (f"# Answer pool {cfg.name}\n\ngenerator={entry['id']} | greedy, {cfg.max_new_tokens} new tokens | "
              f"judge={cfg.classifier}")
    (out / f"{cfg.name}.md").write_text(render(jobs, float(cfg.threshold), int(cfg.max_chars), header))
    print(f"{len(jobs)} completions -> {out / cfg.name}.jsonl")


if __name__ == "__main__":
    main()
