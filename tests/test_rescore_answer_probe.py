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
