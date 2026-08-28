from __future__ import annotations

import json

import torch

from adversariallm.training.data import (
    AdvTupleStream, HardBenignStream, build_kl_stream, collate_hard_benign,
    load_benign_targets, split_adv_stream,
)


class _FakeTok:
    def __call__(self, text, **kw):
        return {"input_ids": [ord(c) for c in text]}


_TEMPLATE = ("U{instruction}", "R{target}E", "R", "U{instruction}", "")


def test_load_benign_targets_counts_the_refused_bucket(tmp_path):
    p = tmp_path / "y.json"
    p.write_text(json.dumps([
        {"prompt": "a", "y_gen": "helpful"},
        {"prompt": "b", "y_gen": None},          # base refused
        {"prompt": "c", "y_gen": "also helpful"},
    ]))
    rows, n_total, n_refused = load_benign_targets(str(p))
    assert n_total == 3
    assert n_refused == 1
    assert rows[1] == ("b", None)


def test_hard_benign_stream_masks_base_refused(monkeypatch):
    import adversariallm.training.data as d
    monkeypatch.setattr(d, "get_chat_template", lambda m: _TEMPLATE)
    rows = [("a", "hi"), ("b", None)]          # one with target, one base-refused
    ds = HardBenignStream(rows, _FakeTok(), "m", refusal="no")
    with_target, refused = ds[0], ds[1]
    assert with_target["has_target"].item() == 1.0
    assert refused["has_target"].item() == 0.0
    # base-refused row still builds a valid (dummy) help forward from the refusal string
    assert refused["g_ids"].numel() > 0
    batch = collate_hard_benign([with_target, refused])
    assert batch["has_target"].tolist() == [1.0, 0.0]
    assert batch["g_ids"].shape[0] == 2 and batch["r_ids"].shape[0] == 2
    assert batch["prompt"] == ["a", "b"]
    assert batch["y_help_text"] == ["hi", "no"]   # target text (dummy=refusal for base-refused)


def _write_adv_data(tmp_path, targets):
    """Write minimal behaviors.csv / targets.json / safe.csv for AdvTupleStream."""
    (tmp_path / "beh.csv").write_text(
        "Behavior,BehaviorID\npromptA,b1\npromptB,b2\n"
    )
    (tmp_path / "safe.csv").write_text(
        "Behavior,Safe_Response\npromptA,safeA\npromptB,safeB\n"
    )
    (tmp_path / "tgt.json").write_text(json.dumps(targets))
    return AdvTupleStream(str(tmp_path), "beh.csv", "tgt.json", "safe.csv", _FakeTok(), "m")


def test_adv_tuple_stream_explodes_targets_and_drops_empty(tmp_path):
    ds = _write_adv_data(tmp_path, {"b1": ["A1", "A2", "  "], "b2": ["B1"]})
    # 2 non-empty targets for A (empty dropped) + 1 for B = 3 rows
    assert len(ds.rows) == 3
    a_rows = [r for r in ds.rows if r[0] == "promptA"]
    assert [r[1] for r in a_rows] == ["A1", "A2"]        # both A targets, empty dropped
    assert all(r[2] == "safeA" for r in a_rows)          # all share A's single y_safe
    assert ("promptB", "B1", "safeB") in ds.rows


def test_split_adv_stream_is_behavior_level(tmp_path):
    ds = _write_adv_data(tmp_path, {"b1": ["A1", "A2"], "b2": ["B1", "B2", "B3"]})
    train, val = split_adv_stream(ds, val_size=1, seed=0)
    train_beh = {ds.rows[i][0] for i in train.indices}
    val_beh = {ds.rows[i][0] for i in val.indices}
    assert len(val_beh) == 1                              # exactly one behavior held out
    assert len(val.indices) == 1                          # deduped to one row per val behavior
    assert train_beh.isdisjoint(val_beh)                  # a behavior never straddles the split
    assert train_beh | val_beh == {"promptA", "promptB"}


def test_build_kl_stream_routes_by_source(monkeypatch):
    import adversariallm.training.data as d

    seen = {}

    class _RecStream:  # capture what UtilityStream is built with
        def __init__(self, tok, mn, window=None, fraction=0.01, rows=None, max_length=None):
            seen.clear()
            seen.update(window=window, fraction=fraction, rows=rows, max_length=max_length)

    monkeypatch.setattr(d, "UtilityStream", _RecStream)
    # ultrachat -> built-in path: window/fraction, no injected rows
    d.build_kl_stream({}, "ultrachat", None, "m", window=[0, 10], fraction=0.5, max_length=99)
    assert seen["rows"] is None and seen["window"] == [0, 10] and seen["max_length"] == 99
    # registry source -> rows via load_dataset_prompts, response-less rows dropped
    monkeypatch.setattr(
        d, "load_dataset_prompts",
        lambda cfg, name, window, seed=0: (["p1", "p2", "p3"], ["r1", None, "r3"]),
    )
    d.build_kl_stream({}, "magpie", None, "m", window=[0, 3], max_length=42)
    assert seen["rows"] == [("p1", "r1"), ("p3", "r3")]    # None response dropped
    assert seen["max_length"] == 42
