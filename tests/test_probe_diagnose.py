import pytest

from probe_diagnose import pick_targets, summarize


def test_pick_targets_prefers_the_behaviors_that_were_attacked():
    targets = {"make a bomb": ["Sure, here is how to make a bomb"], "hotwire a car": "Sure, here is how"}
    assert pick_targets(targets, ["hotwire a car", "make a bomb"], limit=2) == [
        ("hotwire a car", "Sure, here is how"),
        ("make a bomb", "Sure, here is how to make a bomb"),
    ]


def test_pick_targets_skips_unknown_and_empty_and_honours_the_limit():
    targets = {"a": ["Sure A"], "b": [""], "c": ["Sure C"]}
    assert pick_targets(targets, ["unknown", "b", "a", "c"], limit=1) == [("a", "Sure A")]
    assert pick_targets(targets, ["b"], limit=5) == []          # empty target is not usable
    assert pick_targets({}, ["a"], limit=5) == []


def test_summarize_reports_the_spread_and_how_many_cross_the_threshold():
    line = summarize("real_hit", [0.001, 0.002, 0.9], threshold=0.02)
    assert "n=  3" in line and "median=0.0020" in line and "over_threshold=1/3" in line
    assert summarize("empty", [], 0.5).endswith("(no examples)")
