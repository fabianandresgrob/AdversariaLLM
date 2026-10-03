"""Refusals (y_safe) for training behaviors that have none, in the style of the existing ones.

    pixi run --frozen python generate_safe_responses.py name=adv_training_varied

The current behaviors' refusals (safe_responses/adv_training_behaviors_safe_responses_longer.csv) are
Llama-2-7b-chat's own answers ("I cannot fulfill your request. I'm just an AI, ..."). New behaviors get theirs the
same way: greedy, no system prompt, trimmed to the last complete sentence. An answer that is not a refusal falls
back to the loader's default ("I can't help with that."), so no compliance ever becomes a y_safe target.

Writes data/safe_responses/<name>_safe_responses.csv: the existing rows, then one row per new behavior.
"""

import csv
import logging
import re
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

log = logging.getLogger(__name__)
REFUSAL = re.compile(r"^\s*(I cannot|I can't|I can not|I apologize|I'm sorry|I am sorry|I won't|I will not|"
                     r"As a responsible|I'm just an AI)", re.I)
DEFAULT = "I can't help with that."


def trim(text: str) -> str:
    """Cut at the last sentence end, so a length-capped answer does not stop mid-sentence."""
    text = text.strip()
    ends = [m.end() for m in re.finditer(r"[.!?](\s|$)", text)]
    return text[: ends[-1]].strip() if ends else text


def finalize(answers: list[str]) -> tuple[list[str], int]:
    """Trimmed refusals; non-refusals replaced by DEFAULT. Returns (responses, number replaced)."""
    out = [trim(a) if REFUSAL.match(a) else DEFAULT for a in answers]
    return out, sum(o == DEFAULT for o in out)


@hydra.main(version_base=None, config_path="conf", config_name="generate_safe_responses")
def main(cfg: DictConfig) -> None:
    from adversariallm.io_utils import load_model_and_tokenizer
    from adversariallm.training.generation import generate_responses

    data = Path(cfg.data_dir)
    with open(data / cfg.behaviors, newline="") as fh:
        behaviors = [r["Behavior"] for r in csv.DictReader(fh)]
    with open(data / cfg.existing, newline="") as fh:
        existing = list(csv.DictReader(fh))
    have = {r["Behavior"] for r in existing}
    missing = [b for b in behaviors if b not in have]
    log.info(f"{len(behaviors)} behaviors, {len(have)} with refusals, generating {len(missing)}")

    entry = OmegaConf.to_container(cfg.models[cfg.generator], resolve=True)
    model, tok = load_model_and_tokenizer(entry)
    model.eval()
    responses, n_default = finalize(generate_responses(model, tok, missing, int(cfg.max_new_tokens), int(cfg.batch_size)))
    log.info(f"{n_default} of {len(missing)} answers were not refusals and got the default")

    out = data / "safe_responses" / f"{cfg.name}_safe_responses.csv"
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["Behavior", "Safe_Response"])
        w.writeheader()
        w.writerows([{"Behavior": r["Behavior"], "Safe_Response": r["Safe_Response"]} for r in existing])
        w.writerows([{"Behavior": b, "Safe_Response": s} for b, s in zip(missing, responses)])
    print(f"wrote {out}: {len(existing)} existing + {len(missing)} new ({n_default} default)")


if __name__ == "__main__":
    main()
