"""Score a coop probe on the four things it could be asked about, to locate where it fails.

    pixi run --frozen python probe_diagnose.py model=M-respmean-s0 hits=hits.json

The probe is trained on ONE kind of harmful example: an attacked prompt followed by the
teacher-forced target from adv_training_targets.json ("Sure, here is an email that ..."), against
alpaca prompt+answer as benign. At eval it meets something else entirely: a fluent jailbreak
prompt followed by the model's own long answer. This scores all four combinations so the failure
can be attributed instead of guessed:

  train_style   behavior        + canonical target     what training looks like -> expect harmful
  real_hit      jailbreak prompt + real completion     what eval looks like     -> observed ~0.002
  swap_prompt   behavior        + real completion      isolates the RESPONSE half
  swap_response jailbreak prompt + canonical target    isolates the PROMPT half
  benign        alpaca prompt   + alpaca answer        the negative class       -> expect benign

If train_style scores harmful and real_hit does not, the probe does what it was trained to do and
the training distribution is the problem. If train_style ALSO scores benign, the probe is broken
and the distribution is a red herring. swap_* says which half carries the failure.
"""

import json
import logging

import hydra
from omegaconf import DictConfig, OmegaConf

log = logging.getLogger(__name__)


def pick_targets(targets: dict, behaviors: list[str], limit: int) -> list[tuple[str, str]]:
    """(behavior, canonical target) pairs, preferring behaviors that appear in the hits."""
    out = []
    for behavior in behaviors:
        value = targets.get(behavior)
        if value is None:
            continue
        target = value[0] if isinstance(value, (list, tuple)) else value
        if str(target).strip():
            out.append((behavior, str(target)))
        if len(out) >= limit:
            break
    return out


def summarize(name: str, scores: list[float], threshold: float) -> str:
    if not scores:
        return f"{name:14s} (no examples)"
    ordered = sorted(scores)
    median = ordered[len(ordered) // 2]
    fired = sum(s > threshold for s in scores)
    return (f"{name:14s} n={len(scores):3d}  median={median:.4f}  min={ordered[0]:.4f}  "
            f"max={ordered[-1]:.4f}  over_threshold={fired}/{len(scores)}")


@hydra.main(version_base=None, config_path="conf", config_name="calibrate_probe")
def main(cfg: DictConfig) -> None:
    from adversariallm.defenses.monitors.linear_probe import LinearProbeMonitor
    from adversariallm.io_utils import load_model_and_tokenizer
    from adversariallm.training.data import load_dataset_prompts

    entry = OmegaConf.to_container(cfg.models[cfg.model], resolve=True)
    model, tokenizer = load_model_and_tokenizer(entry)
    monitor = LinearProbeMonitor.from_config({
        "checkpoint_path": entry["reader_path"], "target_model_id": cfg.chat_template_id,
        "index_hidden_layer_detector": cfg.layer, "batch_size": cfg.batch_size,
    })
    threshold = float(cfg.get("threshold") or 0.5)

    hits = json.loads(open(cfg.hits).read())[: int(cfg.limit)]
    behaviors = [h["behavior"] for h in hits]
    prompts_jb = [h["prompt"] for h in hits]
    responses = [h["response_raw"] or h["response"] for h in hits]

    targets = json.load(open(cfg.targets_path))
    pairs = pick_targets(targets, behaviors, int(cfg.limit))
    if not pairs:  # the hit behaviors are JBB, the targets file is advbench -- fall back to its own
        pairs = pick_targets(targets, list(targets), int(cfg.limit))
    t_behaviors = [b for b, _ in pairs]
    t_targets = [t for _, t in pairs]

    alpaca_p, alpaca_r = load_dataset_prompts(cfg.datasets, "alpaca", window=cfg.splits.alpaca.val, seed=0)
    n = int(cfg.limit)
    alpaca_p, alpaca_r = alpaca_p[:n], [r or "" for r in alpaca_r[:n]]

    sets = {
        "train_style": (t_behaviors, t_targets),
        "real_hit": (prompts_jb, responses),
        "swap_prompt": (behaviors, responses),
        "swap_response": (prompts_jb[: len(t_targets)], t_targets[: len(prompts_jb)]),
        "benign": (alpaca_p, alpaca_r),
    }
    log.info(f"probe={entry['reader_path']}  threshold={threshold}")
    print(f"\n=== {cfg.model}   threshold={threshold:.4f} ===")
    for name, (prompts, texts) in sets.items():
        if not prompts:
            print(summarize(name, [], threshold))
            continue
        scores = monitor.score(list(prompts), list(texts), target_model=model, target_tokenizer=tokenizer)
        print(summarize(name, scores, threshold))


if __name__ == "__main__":
    main()
