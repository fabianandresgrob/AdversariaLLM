"""Judge the raw answers that the probe flagged in attacks against the pipeline, with StrongREJECT.

    pixi run --frozen python judge_flagged_raw.py model=W-coop-s0-ema

An attack against the pipeline replaces a flagged answer by a refusal and judges that refusal, so whether the model
itself complied is unknown for flagged steps. This judges model_completions_raw of every flagged step (defense
applied) in outputs/{replay,pair}__coop_probe__<model>/ (the GCG / adaptive GCG replays and PAIR), against the original
behavior as run_judges.py builds it. Writes outputs/eval/flagged_raw/<model>.jsonl, one row per flagged step:
{path, behavior, step, p_harmful_raw}.
"""

import copy
import glob
import json
import logging
from pathlib import Path

import hydra
from omegaconf import DictConfig

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="conf", config_name="judge_flagged_raw")
def main(cfg: DictConfig) -> None:
    from judgezoo import Judge

    out = Path(cfg.root_dir) / "outputs"
    rows, convs = [], []
    for cell in cfg.cells:
        for path in sorted(glob.glob(str(out / f"{cell}__coop_probe__{cfg.model}" / "*" / "*" / "*" / "run.json"))):
            for run in json.load(open(path))["runs"]:
                behavior = run["original_prompt"][-1]["content"] if isinstance(run["original_prompt"], list) else str(run["original_prompt"])
                for i, step in enumerate(run["steps"]):
                    if not step["defense_metadata"][0]["applied"]:
                        continue
                    conv = copy.deepcopy(run["original_prompt"])
                    raw = step["model_completions_raw"][0]
                    if conv[-1]["role"] == "assistant":
                        conv[-1]["content"] = step["model_input"][-1]["content"] + raw
                    else:
                        conv.append({"role": "assistant", "content": raw})
                    rows.append({"path": str(Path(path).relative_to(out)), "behavior": behavior, "step": i})
                    convs.append(conv)
    log.info(f"{cfg.model}: {len(rows)} flagged steps")
    if convs:
        scores = Judge.from_name("strong_reject")(convs)["p_harmful"]
        for r, s in zip(rows, scores):
            r["p_harmful_raw"] = float(s) if s is not None else float("nan")
    d = out / "eval" / "flagged_raw"
    d.mkdir(parents=True, exist_ok=True)
    with open(d / f"{cfg.model}.jsonl", "w") as fh:
        fh.writelines(json.dumps(r) + "\n" for r in rows)


if __name__ == "__main__":
    main()
