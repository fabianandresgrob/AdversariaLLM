import json

import pytest

pytest.importorskip("pandas")
pytest.importorskip("matplotlib")

import pandas as pd  # noqa: E402

from collect_attacks import collect, main, parse_dir_name, run_scores  # noqa: E402


def _write_run(repo, attack, defense, model, behaviors, scored=True):
    path = repo / "outputs" / f"{attack}__{defense}__{model}" / "2026-09-19" / "10-00-00" / "0" / "run.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    runs = []
    for scores in behaviors:
        steps = [{"model_completions": ["x"],
                  "scores": {"strong_reject": {"p_harmful": [s]}} if scored else {}} for s in scores]
        runs.append({"original_prompt": [{"role": "user", "content": "b"}], "steps": steps})
    path.write_text(json.dumps({"config": {}, "runs": runs}))


def test_parse_dir_name():
    assert parse_dir_name("gcg__none__E-nd6-s0") == ("gcg", "none", "E-nd6-s0")
    assert parse_dir_name("not-an-attack-dir") is None


def test_run_scores_flattens_steps_in_order():
    run = {"steps": [{"scores": {"strong_reject": {"p_harmful": [0.1, 0.2]}}},
                     {"scores": {"strong_reject": {"p_harmful": [0.9]}}},
                     {"scores": {}}]}
    assert run_scores(run, "strong_reject") == [0.1, 0.2, 0.9]


def test_per_sample_and_budget_metrics(tmp_path):
    # behavior 1: 4 completions, the 3rd succeeds; behavior 2: nothing succeeds
    _write_run(tmp_path, "inpainting", "none", "E-nd6-s0", [[0.1, 0.2, 0.9, 0.4], [0.0, 0.1, 0.2, 0.3]])
    df = collect(tmp_path)
    row = df.iloc[0]
    assert (row.attack, row.defense, row.model) == ("inpainting", "none", "E-nd6-s0")
    assert (row.n_behaviors, row.n_completions) == (2, 8)
    assert row.asr_per_sample == pytest.approx(1 / 8)
    assert row.asr_at_1 == 0.0          # first completion of neither behavior succeeds
    assert row.asr_at_10 == 0.5         # within 10 tries, behavior 1 falls
    assert row.asr_behavior == 0.5  # 1 of 2 behaviors jailbroken at any budget


def test_replay_shares_the_panel_of_the_attack_it_replays(tmp_path):
    _write_run(tmp_path, "gcg", "none", "E-nd6-s0", [[0.9]])
    _write_run(tmp_path, "replay", "coop_probe", "E-nd6-s0", [[0.01]])
    df = collect(tmp_path).set_index("attack")
    assert df.loc["gcg", "family"] == "gcg" and df.loc["replay", "family"] == "gcg"
    assert collect(tmp_path, replay_source="pair").set_index("attack").loc["replay", "family"] == "pair"


def test_defended_and_undefended_are_separate_rows(tmp_path):
    _write_run(tmp_path, "gcg", "none", "E-nd6-s0", [[0.9]])
    _write_run(tmp_path, "gcg", "coop_probe", "E-nd6-s0", [[0.01]])
    df = collect(tmp_path).set_index("defense")
    assert df.loc["none", "asr_behavior"] == 1.0 and df.loc["coop_probe", "asr_behavior"] == 0.0


def test_unscored_runs_are_skipped_rather_than_counted_as_zero(tmp_path):
    _write_run(tmp_path, "pair", "none", "L-cold-s0", [[0.9]], scored=False)
    assert collect(tmp_path).empty


def test_main_writes_table_and_plots(tmp_path):
    _write_run(tmp_path, "gcg", "none", "E-nd6-s0", [[0.9, 0.1], [0.2, 0.3]])
    _write_run(tmp_path, "gcg", "coop_probe", "E-nd6-s0", [[0.01, 0.0], [0.0, 0.0]])
    assert main([], repo=tmp_path) == 0
    out = tmp_path / "outputs" / "eval" / "attacks"
    assert (out / "attacks.csv").is_file() and (out / "attacks.json").is_file()
    assert sorted(p.name for p in (out / "plots").iterdir()) == ["asr_by_attack.png", "budget_curve.png"]
    assert len(pd.read_csv(out / "attacks.csv")) == 2


def test_asr_at_128_reads_a_larger_run_at_the_agreed_budget(tmp_path):
    # behavior 1 falls on completion 120 (inside 128), behavior 2 only on 300 (outside)
    early = [0.0] * 119 + [0.9] + [0.0] * 200
    late = [0.0] * 299 + [0.9] + [0.0] * 20
    _write_run(tmp_path, "inpainting", "none", "E-nd6-s0", [early, late])
    row = collect(tmp_path).iloc[0]
    assert row.asr_at_100 == 0.0        # neither behavior has fallen by 100 tries
    assert row.asr_at_128 == 0.5        # behavior 1 has, behavior 2 has not
    assert row.asr_behavior == 1.0      # both fall eventually, at the full 1024 budget
