"""Collect every evaluation of the coop and CAT checkpoints and the baselines into one table (+ per-config means and plots).

    pixi run --frozen python collect_eval.py            # writes outputs/eval/summary/
    pixi run --frozen python collect_eval.py --no-plots

One row per checkpoint (checkpoints_coop/ and checkpoints_cat/<block>/<run>/) plus reference rows
(outputs/eval/*/reference/*):
  config     coop: run_config.json (epsilon, delta, lambda_*, seed, probe_init, kl_source, ...);
             cat: the overrides recorded in the jsc-jobs run.json (lambda_away, model_objective, ...)
  thresholds threshold_1pct.json (val window) / threshold_1pct_calib.json (2000 held-out alpaca) -- coop only
  overrefusal outputs/eval/overrefusal/<block>/<run>/overrefusal.json (string match + gemma judge)
  utility    outputs/eval/utility/<block>/<run>/utility.json (lm_eval llama3 tasks, in percent)
  training   last "[step N] detector/..." metrics line of the jsc-jobs training log ($JOBS_ROOT) -- coop only
  cross-attack outputs/eval/cross_attack/<block>/<run>/seed*/cross_attack_eval.json, averaged over attack
             seeds: comply/asr/recall per attack condition (model_only, detaware_c*) -- where tier K ran

One row per (checkpoint dir, evaluated checkpoint): the final adapter, and the weight average ("ema")
and step-N adapters wherever they were evaluated (eval dirs <run>-<ckpt>; a plain <run> dir is the final
one). Baseline rows come from outputs/eval/*/baselines/<name>, reference rows from .../reference/<name>.
  judged     cross-attack judged_asr_model / judged_asr_pipeline per attack condition (judge.enabled)
  transfer   outputs/<attack>__<defense>__<model>/ via collect_attacks: share of behaviors jailbroken
             within 128 attempts, per attack, for the model alone and through its probe (defense=coop_probe)
  answer probe  outputs/eval/response_head/RH*-<model>-<variant>.json: the same share after the
             two-channel probe (prompt + answer channel), per training variant

Outputs (outputs/eval/summary/): all_runs.csv/.json, by_config.csv (mean/std over seeds), missing.txt,
plots/tradeoff.png (utility vs over-refusal per block) and plots/sweep_<block>.png (metrics vs swept knob).
Missing evals leave empty cells and a line in missing.txt; nothing is silently dropped.
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent

CONFIG_KEYS = {
    "seed": ("seed",),
    "epsilon": ("epsilon",),
    "delta": ("delta",),
    "lambda_beh": ("lambda_beh",),
    "lambda_rep": ("lambda_rep",),
    "lambda_kl": ("lambda_kl",),
    "probe_init": ("probe_init",),
    "kl_source": ("data", "kl_source"),
    "use_detector": ("attack", "use_detector"),
    "n_detector_steps": ("training", "n_detector_steps"),
    "rep_warmup_steps": ("training", "rep_warmup_steps"),
    "n_steps": ("training", "n_steps"),
    "lambda_away": ("lambda_away",),          # CAT only
    "model_objective": ("model_objective",),  # CAT only
}
CHECKPOINT_ROOTS = {"coop": "checkpoints_coop", "cat": "checkpoints_cat"}
CKPT_SUFFIX = re.compile(r"^(final|ema|ema\d+|step\d+)$")
# outputs/eval/*/baselines/<dir> and .../reference/<dir> -> the models.yaml entry their attacks ran under
EXTERNAL_MODELS = {"mixat": "mixat-llama31-8b", "cb": "cb-llama3-8b-rr", "base": "base-llama31-8b"}
TRANSFER_ATTACKS = ("direct", "gcg", "gcg_adaptive", "inpainting", "pair")
TRANSFER_BEHAVIORS = (0, 20)  # the suite's behavior window (JBB 0-19); older 100-behavior runs do not leak in
# the current protocol where a model was attacked under several: PAIR judged by vicuna, 128 inpainting samples
PREFERRED_PROTOCOL = {"pair": "judge_model=vicuna-13b-v1.5", "inpainting": "num_samples_per_behavior=128"}
ANSWER_PROBE_VARIANTS = ("emb", "attacks")
TRAIN_METRICS = {
    "detector/recall@1fpr": "train_recall_1fpr",
    "detector/fpr_xstest": "train_fpr_xstest",
    "model/refusal_rate_ood": "train_refusal_rate_ood",       # runs before 2026-09-24
    "model/comply_rate": "train_comply_rate",                 # runs before 2026-09-24 (teacher-forced)
    "model/refusal_rate_xstest": "train_refusal_rate_xstest",
    "model/refusal_rate_alpaca": "train_refusal_rate_alpaca",
    "model/asr_gen": "train_asr_gen",
    "pipeline/asr": "train_pipeline_asr",
    "pipeline/detector_saved": "train_detector_saved",
    "pipeline/case_A": "train_case_A",
    "pipeline/case_B": "train_case_B",
    "pipeline/case_C": "train_case_C",
    "pipeline/case_D": "train_case_D",
}
UTILITY_TASKS = {
    "mmlu_pct": ("mmlu_llama", "exact_match,strict_match"),
    "arc_c_pct": ("arc_challenge_llama", "exact_match,strict_match"),
    "gsm8k_pct": ("gsm8k_llama", "exact_match,strict_match"),
    "gsm8k_flexible_pct": ("gsm8k_llama", "exact_match,flexible_extract"),
}
EVAL_COLUMNS = {  # per kind: which evals a checkpoint of that kind is expected to have
    "coop": {
        "thresholds": ["tau_calib"],
        "overrefusal": ["xstest_refusal_string"],
        "utility": ["mmlu_pct"],
        "training": ["train_recall_1fpr"],
    },
    "cat": {  # no probe: no thresholds, and the training log has no detector metrics
        "overrefusal": ["xstest_refusal_string"],
        "utility": ["mmlu_pct"],
    },
}
_STEP_LINE = re.compile(r"\[step (\d+)\] (detector/\S+=.*)$")


def _read_json(path: Path):
    return json.loads(path.read_text()) if path.is_file() else None


def _dig(d, keys):
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


def config_columns(run_config: dict) -> dict:
    return {col: _dig(run_config, keys) for col, keys in CONFIG_KEYS.items()}


def threshold_columns(ckpt_dir: Path, reader: str | None = None) -> dict:
    """The calibrated thresholds in ckpt_dir; with `reader` (e.g. "ema_step750_reader.pt") only those calibrated
    for that reader, since one dir can hold several checkpoints but only one calibration file."""
    def own(data):
        if data and reader and data.get("checkpoint_path") and not str(data["checkpoint_path"]).endswith(reader):
            return None
        return data
    val, calib = (own(_read_json(ckpt_dir / f)) for f in ("threshold_1pct.json", "threshold_1pct_calib.json"))
    return {"tau_val": val and val["threshold"], "tau_calib": calib and calib["threshold"]}


def overrefusal_columns(path: Path) -> dict:
    data = _read_json(path)
    if data is None:
        return {}
    name = next(iter(data["results"]))
    out = {"xstest_refusal_string": data["results"][name].get("xs_test")}
    judged = _dig(data, ("judge", "metrics", name, "xs_test"))
    if judged:
        out.update({
            "xstest_refusal_judge": judged["degeneration_adjusted_refusal_rate"],
            "xstest_partial_judge": judged["coherent_partial_rate"],
            "xstest_comply_judge": judged["degeneration_adjusted_compliance_rate"],
            "xstest_degenerate": judged["degeneration_rate"],
        })
    return out


def utility_columns(path: Path) -> dict:
    data = _read_json(path)
    if data is None:
        return {}
    out = {}
    for col, (task, metric) in UTILITY_TASKS.items():
        value = _dig(data, ("results", task, metric))
        out[col] = None if value is None else round(100 * value, 2)
    core = [out[c] for c in ("mmlu_pct", "arc_c_pct", "gsm8k_pct") if out.get(c) is not None]
    out["utility_mean_pct"] = round(sum(core) / len(core), 2) if len(core) == 3 else None
    return out


def cross_attack_columns(run_dir: Path) -> dict:
    """Tier K: mean over attack seeds of the per-condition numbers in seed*/cross_attack_eval.json.

    comply = the model answered the harmful request; asr = the whole pipeline failed (the probe did not
    catch it), so for a checkpoint without a probe (CAT) comply IS the asr. Conditions are the model-only
    attack and the detector-aware attack at each budget coefficient."""
    files = sorted(run_dir.glob("seed*/cross_attack_eval.json"))
    per_condition: dict[str, dict[str, list[float]]] = {}
    for path in files:
        data = _read_json(path) or {}
        for spec in data.values():  # one entry per evaluated checkpoint; single-checkpoint mode has one
            for condition, metrics in spec.items():
                bucket = per_condition.setdefault(condition, {})
                for metric in ("comply", "asr", "recall", "judged_asr_model", "judged_asr_pipeline"):
                    if metrics.get(metric) is not None:
                        bucket.setdefault(metric, []).append(float(metrics[metric]))
    out: dict = {}
    for condition, metrics in per_condition.items():
        for metric, values in metrics.items():
            out[f"xattack_{metric}_{condition}"] = round(sum(values) / len(values), 4)
    if out:
        out["xattack_seeds"] = len(files)
        asrs = [v for k, v in out.items() if k.startswith("xattack_asr_")]
        out["xattack_asr_worst"] = max(asrs) if asrs else None
    return out


def entry_name(run: str) -> str:
    """models.yaml entry of a checkpoint (gen_coop_models.py: dots become "p")."""
    return run.replace(".", "p")


def attack_model(run: str, ckpt: str) -> str | None:
    """The models.yaml entry the transfer attacks ran under: <run> for the final adapter, <run>-ema for
    the weight average (the <run>-ema checkpoint dir); other checkpoints were not attacked."""
    if re.fullmatch(r"ema\d+", ckpt):  # a weight average saved inside the run's dir (ema_step<N>_adapter)
        return f"{entry_name(run)}-{ckpt}"
    return {"final": entry_name(run), "ema": entry_name(run) + "-ema"}.get(ckpt)


def transfer_columns(attacks: pd.DataFrame, model: str | None) -> dict:
    """<attack>_asr: share of behaviors jailbroken at any point of the attack's budget (250 GCG steps, 90 PAIR
    attempts), <attack>_asr_prompt: share of single attempts that work; suffixed _<defense> for the pipeline. GCG's
    pipeline numbers come from replaying all its steps through the probe (attack=replay). transfer_behaviors: the
    behavior counts they rest on."""
    if model is None or attacks.empty:
        return {}
    out, counts = {}, set()
    for (attack, defense), cells in attacks[attacks["model"] == model].groupby(["attack", "defense"]):
        if attack == "replay":  # the protocol names the replayed run: source=<attack>__none__<model>
            for _, cell in cells.iterrows():
                family = str(cell["protocol"]).split("source=")[-1].split("__")[0]
                if family in TRANSFER_ATTACKS:
                    out[f"{family}_asr_{defense}"] = cell["asr_behavior"]
                    out[f"{family}_asr_prompt_{defense}"] = cell["asr_per_sample"]
                    counts.add(int(cell["n_behaviors"]))
            continue
        if attack not in TRANSFER_ATTACKS:
            continue
        preferred = cells[cells["protocol"].str.contains(PREFERRED_PROTOCOL.get(attack, ""), regex=False)]
        # the current protocol where it exists, else the most complete cell (never a small smoke test)
        cell = preferred.sort_values("n_behaviors").iloc[-1] if len(preferred) else \
            cells.sort_values("n_behaviors").iloc[-1]
        suffix = "" if defense == "none" else f"_{defense}"
        out[f"{attack}_asr{suffix}"] = cell["asr_behavior"]
        out[f"{attack}_asr_prompt{suffix}"] = cell["asr_per_sample"]
        counts.add(int(cell["n_behaviors"]))
    if counts:
        out["transfer_behaviors"] = "/".join(str(c) for c in sorted(counts))
    return out


def answer_probe_columns(repo: Path, model: str | None) -> dict:
    """rh_<variant>_<attack>_asr128: share jailbroken within 128 attempts after the two-channel probe."""
    out = {}
    if model is None:
        return out
    for path in sorted((repo / "outputs/eval/response_head").glob(f"RH*-{model}-*.json")):
        variant = path.stem[path.stem.index(model) + len(model) + 1:]
        if variant not in ANSWER_PROBE_VARIANTS:
            continue
        data = _read_json(path) or {}
        for attack, m in (data.get("per_attack") or {}).items():
            out[f"rh_{variant}_{attack}_asr128"] = m.get("asr_after_dual_at_128")
        out[f"rh_{variant}_benign_fpr"] = data.get("benign_test_fpr_dual")
    return out


def _eval_path(repo: Path, kind: str, block: str, run: str, ckpt: str, filename: str | None) -> Path:
    """outputs/eval/<kind>/<block>/<run>-<ckpt>[/filename], falling back to the plain <run> dir for final."""
    base = repo / "outputs/eval" / kind / block
    candidates = [base / f"{run}-{ckpt}"] + ([base / run] if ckpt == "final" else [])
    for d in candidates:
        if d.is_dir():
            return d / filename if filename else d
    return candidates[0] / filename if filename else candidates[0]


def evaluated_checkpoints(repo: Path, block: str, run: str, has_final: bool = True) -> list[str]:
    ckpts = {"final"} if has_final else set()
    for kind in ("overrefusal", "utility", "cross_attack"):
        for d in (repo / "outputs/eval" / kind / block).glob(f"{run}-*"):
            suffix = d.name[len(run) + 1:]
            if CKPT_SUFFIX.match(suffix):
                ckpts.add(suffix)
    return sorted(ckpts, key=lambda c: (c != "final", c != "ema", c))


def training_columns(log_path: Path) -> dict:
    """Last validation metrics line ("[step N] detector/recall@1fpr=... ...") of the training log."""
    if not log_path.is_file():
        return {}
    last = None
    with open(log_path, errors="replace") as fh:
        for line in fh:
            m = _STEP_LINE.search(line.rstrip())
            if m:
                last = m
    if last is None:
        return {}
    pairs = dict(kv.split("=", 1) for kv in last.group(2).split() if "=" in kv)
    out = {"train_last_step": int(last.group(1))}
    for key, col in TRAIN_METRICS.items():
        try:
            out[col] = float(pairs[key])
        except (KeyError, ValueError):
            out[col] = None
    return out


def nest(flat: dict) -> dict:
    """{'training.n_steps': 1000} -> {'training': {'n_steps': 1000}}, so CONFIG_KEYS can read
    the dotted hydra overrides recorded in run.json the same way it reads run_config.json."""
    out: dict = {}
    for key, value in flat.items():
        parts = str(key).split(".")
        node = out
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value
    return out


def hydra_config(job_dir: Path) -> dict | None:
    """The resolved config hydra saved for the run (<attempt>/hydra/.hydra/config.yaml). Unlike the recorded
    CLI overrides it also holds what came from config files and presets (e.g. +cat_recipe=leo)."""
    path = job_dir / "hydra" / ".hydra" / "config.yaml"
    if not path.is_file():
        return None
    import yaml

    return yaml.safe_load(path.read_text())


def jobs_index(jobs_root: Path | None) -> dict[str, Path]:
    """run name -> its latest attempt dir. Looked up by run name because the jsc-jobs experiment
    name and the checkpoint block don't always match (coop-I-full vs I-ablation, cat-J-ce vs J-cat)."""
    if jobs_root is None:
        return {}
    return {latest.parent.name: latest for latest in sorted(jobs_root.glob("*/*/latest"))}


def collect(repo: Path, jobs_root: Path | None) -> pd.DataFrame:
    from collect_attacks import collect as collect_attack_cells

    attacks = collect_attack_cells(repo, behaviors=TRANSFER_BEHAVIORS)
    rows = []
    jobs = jobs_index(jobs_root)
    for kind, root in CHECKPOINT_ROOTS.items():
        for ckpt_dir in sorted((repo / root).glob("*/*")):
            if not any(ckpt_dir.glob("*_adapter")):  # runs stopped by the time limit have only ema_step<N>_adapter
                continue
            block, run = ckpt_dir.parent.name, ckpt_dir.name
            run_config = _read_json(ckpt_dir / "run_config.json")
            if run_config and run_config.get("ema_of"):  # <run>-ema: the weight average of <run>, a row of it
                continue
            job_dir = jobs.get(run)
            if run_config is None and job_dir is not None:  # CAT runs before 30 Sep wrote no run_config.json
                run_config = hydra_config(job_dir) or nest((_read_json(job_dir / "run.json") or {}).get("overrides", {}))
            for ckpt in evaluated_checkpoints(repo, block, run, (ckpt_dir / "final_adapter").is_dir()):
                model = attack_model(run, ckpt)
                row = {"block": block, "run": run, "checkpoint": ckpt, "kind": kind, "model": model}
                row.update(config_columns(run_config or {}))
                if kind == "coop":
                    if ckpt == "final":
                        row.update(threshold_columns(ckpt_dir, "final_reader.pt"))
                    elif re.fullmatch(r"ema\d+", ckpt):
                        row.update(threshold_columns(ckpt_dir, f"ema_step{ckpt[3:]}_reader.pt"))
                    else:  # <run>-ema: the weight average copied into a sibling dir
                        row.update(threshold_columns(ckpt_dir.parent / f"{run}-{ckpt}"))
                row.update(overrefusal_columns(_eval_path(repo, "overrefusal", block, run, ckpt, "overrefusal.json")))
                row.update(utility_columns(_eval_path(repo, "utility", block, run, ckpt, "utility.json")))
                row.update(cross_attack_columns(_eval_path(repo, "cross_attack", block, run, ckpt, None)))
                row.update(transfer_columns(attacks, model))
                row.update(answer_probe_columns(repo, model))
                if job_dir is not None and ckpt == "final":
                    row.update(training_columns(job_dir / "stdout.log"))
                rows.append(row)
    for kind, folder in (("baseline", "baselines"), ("reference", "reference")):
        names = {p.name for e in ("overrefusal", "utility") for p in (repo / "outputs/eval" / e / folder).glob("*")}
        for name in sorted(names):
            model = EXTERNAL_MODELS.get(name)
            row = {"block": folder, "run": name, "checkpoint": "-", "kind": kind, "model": model}
            row.update(overrefusal_columns(repo / "outputs/eval/overrefusal" / folder / name / "overrefusal.json"))
            row.update(utility_columns(repo / "outputs/eval/utility" / folder / name / "utility.json"))
            row.update(transfer_columns(attacks, model))
            rows.append(row)
    return pd.DataFrame(rows)


def missing_report(df: pd.DataFrame) -> list[str]:
    lines = []
    for _, row in df[df["kind"].isin(EVAL_COLUMNS) & (df.get("checkpoint", "final") == "final")].iterrows():
        for evaluation, cols in EVAL_COLUMNS[row["kind"]].items():
            if any(c not in df.columns or pd.isna(row.get(c)) for c in cols):
                lines.append(f"{row['block']}/{row['run']}: no {evaluation}")
    return lines


def swept_knobs(block_df: pd.DataFrame) -> list[str]:
    """Config columns (other than seed) that take more than one value inside a block."""
    return [c for c in CONFIG_KEYS if c != "seed" and c in block_df and block_df[c].astype(str).nunique() > 1]


def by_config(df: pd.DataFrame) -> pd.DataFrame:
    trained = df[df["kind"].isin(CHECKPOINT_ROOTS)]
    config_cols = [c for c in CONFIG_KEYS if c != "seed" and c in trained]
    metric_cols = [c for c in trained.columns if c not in config_cols + ["block", "run", "kind", "seed", "checkpoint", "model"]
                   and pd.api.types.is_numeric_dtype(trained[c])]
    keys = ["block", "checkpoint"] + config_cols
    grouped = trained.assign(**{c: trained[c].astype(str) for c in config_cols}).groupby(keys, sort=True)
    agg = grouped[metric_cols].agg(["mean", "std"])
    agg.columns = [f"{m}_{stat}" for m, stat in agg.columns]
    agg.insert(0, "n_seeds", grouped.size())
    return agg.reset_index()


def main(argv=None, repo: Path = REPO) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=None, help="output dir (default outputs/eval/summary)")
    parser.add_argument("--jobs-root", type=Path, default=os.environ.get("JOBS_ROOT"), help="jsc-jobs runs dir")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args(argv)

    out = args.out or repo / "outputs" / "eval" / "summary"
    out.mkdir(parents=True, exist_ok=True)
    df = collect(repo, Path(args.jobs_root) if args.jobs_root else None)
    df.to_csv(out / "all_runs.csv", index=False)
    df.to_json(out / "all_runs.json", orient="records", indent=2)
    by_config(df).to_csv(out / "by_config.csv", index=False)
    missing = missing_report(df)
    (out / "missing.txt").write_text("\n".join(missing) + ("\n" if missing else ""))
    if not args.no_plots:
        from plot_eval import plot_all

        plot_all(df[df["checkpoint"].isin(["final", "-"])], out / "plots")
    counts = df["kind"].value_counts().to_dict() if len(df) else {}
    kinds = ", ".join(f"{counts.get(k, 0)} {k}" for k in list(CHECKPOINT_ROOTS) + ["baseline", "reference"])
    print(f"{kinds} rows, {len(missing)} missing evals -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
