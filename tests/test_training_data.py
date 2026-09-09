from __future__ import annotations

import pytest
import torch
from adversariallm.training.data import build_supervised_example, split_adv_stream


class _FakeTok:
    def __call__(self, text, **kw):
        return {"input_ids": [ord(c) for c in text]}


def test_prompt_tokens_are_masked_in_labels(monkeypatch):
    import adversariallm.training.data as d
    monkeypatch.setattr(d, "get_chat_template",
                        lambda m: ("U{instruction}", "R{target}E", "R", "U{instruction}", ""))
    ids, labels = build_supervised_example("ab", "xy", _FakeTok(), "meta-llama/Llama-3.1-8B-Instruct")
    # full = "Uab" + "RxyE"; prompt_with_key = "UabR" -> prompt_len 4
    assert ids.tolist() == [ord(c) for c in "UabRxyE"]
    assert labels.tolist() == [-100, -100, -100, -100, ord("x"), ord("y"), ord("E")]


class _FakeAdvStream:
    """Stands in for AdvStream: rows are (behavior, target, safe_response) and a behavior
    may carry several targets, which is exactly what the split has to keep together."""

    def __init__(self, n_behaviors=20, targets_per_behavior=2):
        self.rows = [
            (f"behavior_{b}", f"target_{b}_{t}", "I can't help with that.")
            for b in range(n_behaviors)
            for t in range(targets_per_behavior)
        ]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        return self.rows[i]


def _behaviors(ds, subset):
    return {ds.rows[i][0] for i in subset.indices}


def test_val_split_is_behavior_level_and_leak_free():
    ds = _FakeAdvStream(n_behaviors=20, targets_per_behavior=2)
    train, val = split_adv_stream(ds, val_size=5, seed=0)

    # val keeps ONE row per held-out behavior, so validation cost does not scale with targets
    assert len(val) == 5
    assert len(_behaviors(ds, val)) == 5
    # every row of a non-val behavior stays in train: 15 behaviors x 2 targets
    assert len(train) == 30
    assert set(train.indices).isdisjoint(val.indices)
    # the leakage guard: no behavior may straddle the split
    assert _behaviors(ds, train).isdisjoint(_behaviors(ds, val))
    assert _behaviors(ds, train) | _behaviors(ds, val) == {f"behavior_{b}" for b in range(20)}


def test_val_split_is_stable_across_runs():
    ds = _FakeAdvStream()
    _, val = split_adv_stream(ds, val_size=5, seed=0)
    # same seed -> same split, so best-checkpoint selection is comparable across runs
    assert split_adv_stream(ds, val_size=5, seed=0)[1].indices == val.indices
    assert split_adv_stream(ds, val_size=5, seed=1)[1].indices != val.indices


def test_val_split_rejects_degenerate_sizes():
    ds = _FakeAdvStream(n_behaviors=20, targets_per_behavior=2)
    with pytest.raises(ValueError):
        split_adv_stream(ds, val_size=0)
    with pytest.raises(ValueError):
        split_adv_stream(ds, val_size=20)
    # val_size is counted in behaviors, not rows: 25 < 40 rows but exceeds the 20 behaviors
    with pytest.raises(ValueError):
        split_adv_stream(ds, val_size=25)
