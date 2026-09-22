import json

import pytest

from inspect_attacks import hits, last_user, main


def _write(repo, attack="inpainting", defense="coop_probe", model="M-respmean-s0",
           scores=(0.9, 0.1), meta=None, raw=None):
    path = repo / "outputs" / f"{attack}__{defense}__{model}" / "2026-09-22" / "10-00-00" / "3" / "run.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    step = {
        "step": 7,
        "model_completions": ["HARMFUL TEXT", "a refusal"],
        "model_completions_raw": list(raw) if raw else ["HARMFUL TEXT", "a refusal"],
        "model_input": [{"role": "user", "content": "the inpainted prompt"}],
        "scores": {"strong_reject": {"p_harmful": list(scores)}},
    }
    if meta is not None:
        step["defense_metadata"] = list(meta)
    path.write_text(json.dumps({"config": {}, "runs": [
        {"original_prompt": [{"role": "user", "content": "the behavior"},
                             {"role": "assistant", "content": "Sure, here"}],
         "steps": [step]}]}))
    return path


def test_only_completions_above_the_threshold_are_reported(tmp_path):
    _write(tmp_path, scores=(0.9, 0.1))
    found = hits(tmp_path)
    assert len(found) == 1
    assert (found[0]["p_harmful"], found[0]["completion"], found[0]["step"]) == (0.9, 0, 7)


def test_the_locator_identifies_the_exact_record(tmp_path):
    _write(tmp_path)
    hit = hits(tmp_path)[0]
    assert hit["cell"] == "inpainting__coop_probe__M-respmean-s0"
    assert hit["behavior_dir"] == "inpainting__coop_probe__M-respmean-s0/2026-09-22/10-00-00/3"
    assert hit["behavior"] == "the behavior"        # the last USER turn, not the affirmative target
    assert hit["prompt"] == "the inpainted prompt"


def test_the_probe_verdict_travels_with_a_defended_hit(tmp_path):
    # a hit where the probe did not fire is the interesting case: the defense let it through
    _write(tmp_path, meta=[{"score": 0.004, "applied": False}, {"score": 0.9, "applied": True}])
    hit = hits(tmp_path)[0]
    assert (hit["probe_score"], hit["probe_fired"]) == (0.004, False)


def test_an_undefended_run_has_no_probe_verdict(tmp_path):
    _write(tmp_path, defense="none", meta=None)
    assert hits(tmp_path)[0]["probe_fired"] is None


def test_the_pre_defense_response_is_kept_only_when_it_differs(tmp_path):
    _write(tmp_path, raw=["THE MODEL'S REAL ANSWER", "a refusal"])
    assert hits(tmp_path)[0]["response_raw"] == "THE MODEL'S REAL ANSWER"
    _write(tmp_path)  # raw == defended
    assert hits(tmp_path)[0]["response_raw"] is None


def test_filters_select_one_cell(tmp_path):
    _write(tmp_path, model="M-respmean-s0")
    _write(tmp_path, model="E-nd6-s0")
    assert len(hits(tmp_path)) == 2
    assert len(hits(tmp_path, model="E-nd6-s0")) == 1
    assert hits(tmp_path, attack="pair") == []


def test_last_user_ignores_assistant_turns():
    assert last_user([{"role": "user", "content": "u"}, {"role": "assistant", "content": "a"}]) == "u"
    assert last_user(None) == ""


def test_main_writes_markdown_and_json(tmp_path):
    _write(tmp_path, meta=[{"score": 0.004, "applied": False}, {"score": 0.9, "applied": True}])
    out, js = tmp_path / "hits.md", tmp_path / "hits.json"
    assert main(["--out", str(out), "--json", str(js)], repo=tmp_path) == 0
    text = out.read_text()
    assert "1 completion(s) scored above 0.5" in text and "HARMFUL TEXT" in text
    assert "fired=False" in text
    assert json.loads(js.read_text())[0]["p_harmful"] == 0.9
