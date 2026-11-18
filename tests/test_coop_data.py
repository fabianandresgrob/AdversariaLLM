from __future__ import annotations

import json

import torch

from adversariallm.training.data import (
    AdvTupleStream, build_kl_stream, collate_adv,
    split_adv_stream, user_token_mask,
)


class _FakeTok:
    """Tokenizer stub so these tests run offline (the real chat template needs the gated
    Llama tokenizer). Arbitrary shape: user -> "U<content>", assistant -> "R<content>E",
    generation prompt -> "R"."""

    def __call__(self, text, return_offsets_mapping=False, **kw):
        out = {"input_ids": [ord(c) for c in text]}
        if return_offsets_mapping:
            out["offset_mapping"] = [(i, i + 1) for i in range(len(text))]
        return out

    def apply_chat_template(self, conv, tokenize=False, add_generation_prompt=False, **kw):
        s = ""
        for m in conv:
            if m["role"] == "user":
                s += "U" + m["content"]
            elif m["role"] == "assistant":
                s += "R" + m["content"] + "E"
        return s + ("R" if add_generation_prompt else "")


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


def test_split_adv_stream_keeps_up_to_val_targets_rows_per_val_behavior(tmp_path):
    ds = _write_adv_data(tmp_path, {"b1": ["A1", "A2"], "b2": ["B1", "B2", "B3"]})
    _, val = split_adv_stream(ds, val_size=1, seed=0, val_targets=2)
    rows = [ds.rows[i] for i in val.indices]
    assert len({r[0] for r in rows}) == 1 and len(rows) == 2   # one behavior, its first two targets
    _, val_all = split_adv_stream(ds, val_size=1, seed=0, val_targets=10)
    assert len(val_all.indices) == len([r for r in ds.rows if r[0] == rows[0][0]])  # capped at what exists


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


def test_user_token_mask_covers_exactly_the_user_message():
    # _FakeTok renders "U<content>R": template U ... R, one token per char
    mask = user_token_mask(_FakeTok(), "abc", length=8)
    assert mask.tolist() == [False, True, True, True, False, False, False, False]


def test_user_token_mask_can_narrow_to_a_span_of_the_message():
    mask = user_token_mask(_FakeTok(), "abc xyz", span=(3, 7))   # the " xyz" suffix
    assert mask.tolist() == [False, False, False, False, True, True, True, True, False]


def test_adv_batches_carry_the_user_message_mask(tmp_path):
    ds = _write_adv_data(tmp_path, {"b1": ["A1"], "b2": ["Blonger"]})
    batch = collate_adv([ds[0], ds[1]])
    # promptA -> "UpromptAR" + target: the 7 prompt chars are perturbable, nothing else (pads False)
    assert batch["h_perturb_mask"].shape == batch["h_ids"].shape
    assert batch["h_perturb_mask"].sum(dim=1).tolist() == [7, 7]
    assert batch["h_perturb_mask"][0, 1:8].all() and not batch["h_perturb_mask"][0, 0]


def test_utility_stream_marks_the_user_message():
    from adversariallm.training.data import UtilityStream, collate_util

    ds = UtilityStream(_FakeTok(), "m", rows=[("abc", "xy"), ("de", "zzz")])
    item = ds[0]
    # "UabcRxyE": U template, abc user, R xy E answer
    assert item["perturb_mask"].tolist() == [False, True, True, True, False, False, False, False]
    batch = collate_util([ds[0], ds[1]])
    assert batch["perturb_mask"].shape == batch["input_ids"].shape


def test_utility_stream_mask_survives_truncation_into_the_prompt():
    from adversariallm.training.data import UtilityStream

    item = UtilityStream(_FakeTok(), "m", rows=[("abcdef", "xy")], max_length=3)[0]
    assert item["perturb_mask"].tolist() == [False, True, True]


def test_answer_pool_keeps_target_prefilled_harmful_answers(tmp_path):
    from adversariallm.training.data import load_answer_pool

    rows = [{"behavior": "promptA", "target": "A1", "prefill": "target", "completion": "A1 harm", "p_harmful": 0.9},
            {"behavior": "promptA", "target": "A2", "prefill": "target", "completion": "A2 meh", "p_harmful": 0.2},
            {"behavior": "promptA", "target": "", "prefill": "none", "completion": "harm", "p_harmful": 0.9}]
    (tmp_path / "pool.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    assert load_answer_pool(tmp_path / "pool.jsonl") == {("promptA", "A1"): "A1 harm"}


def test_attached_answer_is_read_in_full_but_only_the_target_is_elicited(tmp_path):
    ds = _write_adv_data(tmp_path, {"b1": ["A1", "A2"], "b2": ["B1"]})
    assert ds.attach_answers({("promptA", "A1"): "A1 then harm follows"}, max_tokens=6) == 1
    item = ds[0]  # promptA / A1: "UpromptA" + "R" + answer, answer capped at 6 tokens
    prompt_len = len("UpromptAR")
    assert item["h_ids"].tolist() == [ord(c) for c in "UpromptAR" + "A1 the"]
    assert (item["h_targetids"][prompt_len:] > 0).all()  # the probe reads the whole (capped) answer
    elicit = item["h_elicit_mask"]
    assert elicit.nonzero().squeeze(-1).tolist() == [prompt_len, prompt_len + 1]  # "A1", no end-of-turn
    assert (item["h_labels"] != -100).nonzero().squeeze(-1).tolist() == [prompt_len, prompt_len + 1]
    plain = ds[1]  # promptA / A2 has no answer: unchanged, the whole target (with its end-of-turn) is elicited
    assert plain["h_ids"].tolist() == [ord(c) for c in "UpromptAR" + "A2E"]
    assert torch.equal(plain["h_elicit_mask"], plain["h_targetids"] > 0)
    batch = collate_adv([item, plain])
    assert batch["h_elicit_mask"].shape == batch["h_ids"].shape


def test_val_pool_keeps_the_original_held_out_behaviors_when_the_set_grows():
    from types import SimpleNamespace

    from adversariallm.training.data import split_adv_stream

    original = [(f"b{i}", f"t{i}", "s") for i in range(10)]
    extended = original + [(f"new{i}", f"n{i}", "s") for i in range(30)]
    _, val_orig = split_adv_stream(SimpleNamespace(rows=original), val_size=3, seed=0)
    train_ext, val_ext = split_adv_stream(SimpleNamespace(rows=extended), val_size=3, seed=0,
                                          val_pool={p for p, _, _ in original})
    held_out = lambda ds: {ds.dataset.rows[i][0] for i in ds.indices}
    assert held_out(val_ext) == held_out(val_orig)
    assert {f"new{i}" for i in range(30)} <= held_out(train_ext)
