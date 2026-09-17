import json

import pytest

pytest.importorskip("pandas")
pytest.importorskip("matplotlib")

import pandas as pd  # noqa: E402

from collect_eval import main  # noqa: E402

TRAIN_LOG = (
    "[2026-09-17 03:49:24][coop_loop][INFO] - [step 999] beh=0.0077 total=0.0123\n"
    "[2026-09-17 03:50:22][coop_loop][INFO] - [step 974] detector/recall@1fpr=0.5 model/refusal_rate_ood=0.1\n"
    "[2026-09-17 03:50:22][coop_loop][INFO] - [step 999] detector/recall@1fpr=1.0000 detector/fpr_xstest=0.0500 "
    "model/refusal_rate_ood=0.0625 pipeline/asr=0.0000 pipeline/case_A=0.8061\n"
)


def _write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data) if not isinstance(data, str) else data)


def _checkpoint(repo, jobs, block, run, eps, seed, evals=True):
    ckpt = repo / "checkpoints_coop" / block / run
    (ckpt / "final_adapter").mkdir(parents=True)
    _write(ckpt / "run_config.json", {"name": run, "seed": seed, "epsilon": eps, "delta": 0.0, "lambda_rep": 1.0,
                                      "data": {"kl_source": "magpie"}, "training": {"n_detector_steps": 1}})
    _write(ckpt / "threshold_1pct.json", {"threshold": 0.008})
    _write(jobs / f"coop-{block}" / run / "latest" / "stdout.log", TRAIN_LOG)
    if not evals:
        return
    _write(ckpt / "threshold_1pct_calib.json", {"threshold": 0.021})
    _write(repo / "outputs/eval/overrefusal" / block / run / "overrefusal.json", {
        "results": {"model": {"xs_test": 0.064 + eps / 10}},
        "judge": {"metrics": {"model": {"xs_test": {"degeneration_adjusted_refusal_rate": 0.076,
                                                    "coherent_partial_rate": 0.156,
                                                    "degeneration_adjusted_compliance_rate": 0.768,
                                                    "degeneration_rate": 0.0}}}},
    })
    _write(repo / "outputs/eval/utility" / block / run / "utility.json", {"results": {
        "mmlu_llama": {"exact_match,strict_match": 0.6926},
        "arc_challenge_llama": {"exact_match,strict_match": 0.8309},
        "gsm8k_llama": {"exact_match,strict_match": 0.8544 - eps, "exact_match,flexible_extract": 0.8597},
    }})


@pytest.fixture
def tree(tmp_path):
    repo, jobs = tmp_path / "repo", tmp_path / "runs"
    for eps in (0.0, 0.05):
        for seed in (0, 1):
            _checkpoint(repo, jobs, "A-eps-sweep", f"A-eps{eps}-s{seed}", eps, seed)
    _checkpoint(repo, jobs, "G-delta", "G-delta1-s0", 0.05, 0, evals=False)
    _write(repo / "outputs/eval/utility/reference/base/utility.json", {"results": {
        "mmlu_llama": {"exact_match,strict_match": 0.69}, "arc_challenge_llama": {"exact_match,strict_match": 0.83},
        "gsm8k_llama": {"exact_match,strict_match": 0.85, "exact_match,flexible_extract": 0.86}}})
    return repo, jobs


def test_collects_all_sources_into_one_row_per_checkpoint(tree):
    repo, jobs = tree
    assert main(["--jobs-root", str(jobs)], repo=repo) == 0
    df = pd.read_csv(repo / "outputs/eval/summary/all_runs.csv")
    assert len(df[df.kind == "coop"]) == 5 and list(df[df.kind == "reference"].run) == ["base"]
    row = df[df.run == "A-eps0.05-s1"].iloc[0]
    assert (row.block, row.seed, row.epsilon, row.kl_source, row.n_detector_steps) == ("A-eps-sweep", 1, 0.05, "magpie", 1)
    assert (row.tau_val, row.tau_calib) == (0.008, 0.021)
    assert row.xstest_refusal_string == pytest.approx(0.069) and row.xstest_refusal_judge == 0.076
    assert (row.mmlu_pct, row.arc_c_pct, row.gsm8k_pct) == (69.26, 83.09, 80.44)
    assert row.utility_mean_pct == pytest.approx(round((69.26 + 83.09 + 80.44) / 3, 2))
    assert (row.train_last_step, row.train_recall_1fpr, row.train_pipeline_asr) == (999, 1.0, 0.0)


def test_by_config_averages_seeds_and_missing_lists_gaps(tree):
    repo, jobs = tree
    main(["--jobs-root", str(jobs)], repo=repo)
    by_cfg = pd.read_csv(repo / "outputs/eval/summary/by_config.csv")
    a = by_cfg[by_cfg.block == "A-eps-sweep"].set_index("epsilon")
    assert list(a.n_seeds) == [2, 2] and a.loc[0.05, "gsm8k_pct_mean"] == pytest.approx(80.44)
    missing = (repo / "outputs/eval/summary/missing.txt").read_text().splitlines()
    assert missing == ["G-delta/G-delta1-s0: no thresholds", "G-delta/G-delta1-s0: no overrefusal",
                       "G-delta/G-delta1-s0: no utility"]


def test_plots_are_written_for_swept_blocks_only(tree):
    repo, jobs = tree
    main(["--jobs-root", str(jobs)], repo=repo)
    plots = sorted(p.name for p in (repo / "outputs/eval/summary/plots").iterdir())
    assert plots == ["sweep_A-eps-sweep.png", "tradeoff.png"]
