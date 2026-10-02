import json

from rescore_answer_probe import collect_answers


def _write(repo, defense, step):
    d = repo / "outputs" / f"pair__{defense}__m" / "2026-10-01" / "00-00-00" / "0"
    d.mkdir(parents=True)
    run = {"original_prompt": [{"role": "user", "content": "b"}], "steps": [step]}
    (d / "run.json").write_text(json.dumps({"runs": [run]}))


def test_collect_answers_defended_reads_raw_answers_and_recorded_scores(tmp_path):
    step = {"model_input": [{"role": "user", "content": "p"}], "model_completions": ["Sorry", "ok"],
            "model_completions_raw": ["harm", "ok"], "scores": {"strong_reject": {"p_harmful": [0.1, 0.2]}},
            "defense_metadata": [{"score": 0.9, "applied": True}, {"score": 0.1, "applied": False}]}
    _write(tmp_path, "coop_probe", step)
    _write(tmp_path, "none", {**step, "defense_metadata": None})
    rows = collect_answers(tmp_path, "m", ["pair"], defense="coop_probe")
    assert [r["answer"] for r in rows] == ["harm", "ok"]
    assert [(r["recorded"], r["applied"]) for r in rows] == [(0.9, True), (0.1, False)]
    plain = collect_answers(tmp_path, "m", ["pair"])
    assert [r["answer"] for r in plain] == ["Sorry", "ok"] and plain[0]["recorded"] is None


def test_collect_answers_reads_every_behavior_of_only_the_newest_run(tmp_path):
    def write(day, idx, answer):
        d = tmp_path / "outputs" / "pair__none__m" / day / "00-00-00" / str(idx)
        d.mkdir(parents=True)
        step = {"model_input": [{"role": "user", "content": "p"}], "model_completions": [answer],
                "scores": {"strong_reject": {"p_harmful": [0.9]}}}
        run = {"original_prompt": [{"role": "user", "content": f"b{idx}"}], "steps": [step]}
        (d / "run.json").write_text(json.dumps({"runs": [run]}))

    for idx in (0, 1, 10, 2):
        write("2026-10-01", idx, f"old{idx}")
        write("2026-10-02", idx, f"new{idx}")
    assert [r["answer"] for r in collect_answers(tmp_path, "m", ["pair"])] == ["new0", "new1", "new2", "new10"]
