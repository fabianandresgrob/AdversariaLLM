"""Collect every evaluation of the coop checkpoints into one table (+ per-config means and plots).

    pixi run --frozen python collect_eval.py            # writes outputs/eval/summary/
    pixi run --frozen python collect_eval.py --no-plots

One row per checkpoint (checkpoints_coop/<block>/<run>/) plus reference rows (outputs/eval/*/reference/*):
  config     run_config.json: epsilon, delta, lambda_*, seed, probe_init, kl_source, use_detector, ...
  thresholds threshold_1pct.json (val window) / threshold_1pct_calib.json (2000 held-out alpaca)
  overrefusal outputs/eval/overrefusal/<block>/<run>/overrefusal.json (string match + gemma judge)
  utility    outputs/eval/utility/<block>/<run>/utility.json (lm_eval llama3 tasks, in percent)
  training   last "[step N] detector/..." metrics line of the jsc-jobs training log ($JOBS_ROOT)

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
}
TRAIN_METRICS = {
    "detector/recall@1fpr": "train_recall_1fpr",
    "detector/fpr_xstest": "train_fpr_xstest",
    "model/refusal_rate_ood": "train_refusal_rate_ood",
    "model/comply_rate": "train_comply_rate",
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
EVAL_COLUMNS = {
    "thresholds": ["tau_calib"],
    "overrefusal": ["xstest_refusal_string"],
    "utility": ["mmlu_pct"],
    "training": ["train_recall_1fpr"],
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


def threshold_columns(ckpt_dir: Path) -> dict:
    val, calib = _read_json(ckpt_dir / "threshold_1pct.json"), _read_json(ckpt_dir / "threshold_1pct_calib.json")
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


def collect(repo: Path, jobs_root: Path | None) -> pd.DataFrame:
    rows = []
    for run_config_path in sorted((repo / "checkpoints_coop").glob("*/*/run_config.json")):
        ckpt = run_config_path.parent
        if not (ckpt / "final_adapter").is_dir():
            continue
        block, run = ckpt.parent.name, ckpt.name
        row = {"block": block, "run": run, "kind": "coop"}
        row.update(config_columns(json.loads(run_config_path.read_text())))
        row.update(threshold_columns(ckpt))
        row.update(overrefusal_columns(repo / "outputs/eval/overrefusal" / block / run / "overrefusal.json"))
        row.update(utility_columns(repo / "outputs/eval/utility" / block / run / "utility.json"))
        if jobs_root is not None:
            row.update(training_columns(jobs_root / f"coop-{block}" / run / "latest" / "stdout.log"))
        rows.append(row)
    references = {p.name for kind in ("overrefusal", "utility") for p in (repo / "outputs/eval" / kind / "reference").glob("*")}
    for name in sorted(references):
        row = {"block": "reference", "run": name, "kind": "reference"}
        row.update(overrefusal_columns(repo / "outputs/eval/overrefusal/reference" / name / "overrefusal.json"))
        row.update(utility_columns(repo / "outputs/eval/utility/reference" / name / "utility.json"))
        rows.append(row)
    return pd.DataFrame(rows)


def missing_report(df: pd.DataFrame) -> list[str]:
    lines = []
    for _, row in df[df["kind"] == "coop"].iterrows():
        for evaluation, cols in EVAL_COLUMNS.items():
            if any(c not in df.columns or pd.isna(row.get(c)) for c in cols):
                lines.append(f"{row['block']}/{row['run']}: no {evaluation}")
    return lines


def swept_knobs(block_df: pd.DataFrame) -> list[str]:
    """Config columns (other than seed) that take more than one value inside a block."""
    return [c for c in CONFIG_KEYS if c != "seed" and c in block_df and block_df[c].astype(str).nunique() > 1]


def by_config(df: pd.DataFrame) -> pd.DataFrame:
    coop = df[df["kind"] == "coop"]
    config_cols = [c for c in CONFIG_KEYS if c != "seed" and c in coop]
    metric_cols = [c for c in coop.columns if c not in config_cols + ["block", "run", "kind", "seed"]
                   and pd.api.types.is_numeric_dtype(coop[c])]
    keys = ["block"] + config_cols
    grouped = coop.assign(**{c: coop[c].astype(str) for c in config_cols}).groupby(keys, sort=True)
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

        plot_all(df, out / "plots")
    n_coop = int((df["kind"] == "coop").sum()) if len(df) else 0
    print(f"{n_coop} checkpoints, {len(df) - n_coop} reference rows, {len(missing)} missing evals -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
