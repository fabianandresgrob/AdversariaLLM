import pytest

from probe_diagnose import one_per_behavior, summarize


def _hit(behavior, completion):
    return {"behavior": behavior, "completion": completion}


def test_one_hit_per_behavior_not_the_first_n():
    # 3 completions of "a" then one each of "b"/"c": naively taking 2 would give two "a" rows,
    # and a prompt-position probe would score them identically
    hits = [_hit("a", 0), _hit("a", 1), _hit("a", 2), _hit("b", 0), _hit("c", 0)]
    picked = one_per_behavior(hits, limit=3)
    assert [h["behavior"] for h in picked] == ["a", "b", "c"]
    assert picked[0]["completion"] == 0  # the first hit of that behavior


def test_the_limit_caps_behaviors_and_missing_behaviors_do_not_crash():
    assert len(one_per_behavior([_hit(str(i), 0) for i in range(10)], limit=4)) == 4
    assert one_per_behavior([], limit=5) == []
    assert len(one_per_behavior([{"completion": 0}, {"completion": 1}], limit=5)) == 1  # both ""


def test_summarize_reports_the_spread_and_how_many_cross_the_threshold():
    line = summarize("real_hit", [0.001, 0.002, 0.9], threshold=0.02)
    assert "n=  3" in line and "median=0.0020" in line and "over_threshold=1/3" in line
    assert summarize("empty", [], 0.5).endswith("(no examples)")
