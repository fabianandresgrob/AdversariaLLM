from __future__ import annotations

import json

import torch

from adversariallm.training.data import (
    HardBenignStream, collate_hard_benign, load_benign_targets,
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
    ds = HardBenignStream(rows, _FakeTok(), "m", refusal_opener="no", compliance_opener="yes")
    with_target, refused = ds[0], ds[1]
    assert with_target["has_target"].item() == 1.0
    assert refused["has_target"].item() == 0.0
    # base-refused row still builds a valid (dummy) help forward from the compliance opener
    assert refused["g_ids"].numel() > 0
    batch = collate_hard_benign([with_target, refused])
    assert batch["has_target"].tolist() == [1.0, 0.0]
    assert batch["g_ids"].shape[0] == 2 and batch["r_ids"].shape[0] == 2 and batch["c_ids"].shape[0] == 2
    assert batch["prompt"] == ["a", "b"]
    assert batch["y_help_text"] == ["hi", "yes"]   # dummy y_help = compliance opener for base-refused
