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


def test_val_split_is_disjoint_stable_and_covers_dataset():
    ds = list(range(50))
    train, val = split_adv_stream(ds, val_size=10, seed=0)

    assert len(val) == 10
    assert len(train) == 40
    assert set(train.indices).isdisjoint(val.indices), "val behaviors leaked into training"
    assert set(train.indices) | set(val.indices) == set(range(50))

    # same seed -> same split, so best-checkpoint selection is comparable across runs
    train2, val2 = split_adv_stream(ds, val_size=10, seed=0)
    assert val2.indices == val.indices
    assert split_adv_stream(ds, val_size=10, seed=1)[1].indices != val.indices


def test_val_split_rejects_degenerate_sizes():
    ds = list(range(50))
    with pytest.raises(ValueError):
        split_adv_stream(ds, val_size=0)
    with pytest.raises(ValueError):
        split_adv_stream(ds, val_size=50)
