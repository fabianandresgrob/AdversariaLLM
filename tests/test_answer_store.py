import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest

from adversariallm.io_utils import answer_store

ROOT = Path(__file__).resolve().parents[1]


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_append_and_resume_after_a_cut_line(tmp_path):
    path = tmp_path / "answers.0of1.jsonl"
    answer_store.append(str(path), [{"row_id": "a", "answer": "x"}])
    with open(path, "a") as fh:  # a job killed in the middle of a line
        fh.write('{"row_id": "b", "ans')
    answer_store.append(str(path), [{"row_id": "c", "answer": "z"}])
    assert answer_store.done_ids(str(tmp_path), "answers.") == {"a", "c"}
    assert [r["row_id"] for r in answer_store.read_jsonl(str(tmp_path / "answers.*.jsonl"))] == ["a", "c"]


def test_done_ids_find_the_judge_file(tmp_path):
    """The judge writes judge.<name>.jsonl with no shard part; a restart must see its rows as done."""
    answer_store.append(str(tmp_path / "judge.strong_reject.jsonl"), [{"row_id": "a", "p_harmful": 0.9}])
    answer_store.append(str(tmp_path / "answers.0of2.jsonl"), [{"row_id": "b", "answer": "x"}])
    assert answer_store.done_ids(str(tmp_path), "judge.strong_reject.") == {"a"}
    assert answer_store.done_ids(str(tmp_path), "answers.") == {"b"}


def test_meta_is_fixed_by_the_first_run(tmp_path):
    store = str(tmp_path / "s")
    answer_store.check_meta(store, {"model": "m", "max_new_tokens": 256, "note": 1}, ("model", "max_new_tokens"))
    answer_store.check_meta(store, {"model": "m", "max_new_tokens": 256, "note": 2}, ("model", "max_new_tokens"))
    assert json.loads((Path(store) / "meta.json").read_text())["note"] == 1
    with pytest.raises(ValueError):
        answer_store.check_meta(store, {"model": "m", "max_new_tokens": 512}, ("model", "max_new_tokens"))


def test_batches_respect_rows_and_token_budget():
    gen = _load("generate_answers.py", "generate_answers")
    lengths = [10, 500, 20, 480, 30, 15]
    out = gen.batches(lengths, token_budget=1000, max_rows=3)
    assert sorted(i for b in out for i in b) == list(range(len(lengths)))
    for b in out:
        assert len(b) <= 3 and len(b) * max(lengths[i] for i in b) <= 1000


def test_sample_strata_twins_and_judge_text():
    mod = _load("scripts/esg_build_sample.py", "esg_build_sample")
    probe = pd.DataFrame([
        {"row_id": "p1", "input_id": "g1", "source": "plain", "label": "1", "category": "Cyber", "text": "plain one"},
        {"row_id": "p2", "input_id": "g1", "source": "redact_jb", "label": "1", "category": "Cyber", "text": "wrapped one"},
        {"row_id": "p3", "input_id": "g2", "source": "redact_jb", "label": "1", "category": "Cyber", "text": "wrapped two"},
        {"row_id": "p4", "input_id": "g3", "source": "obf_rot13_h", "label": "1", "category": "Cyber", "text": "cynva"},
        {"row_id": "p5", "input_id": "g4", "source": "dan_b", "label": "0", "category": "Benign", "text": "benign wrapped"},
        {"row_id": "p6", "input_id": "g5", "source": "civil_pos", "label": "1", "category": "Hate Speech", "text": "toxic"},
        {"row_id": "p7", "input_id": "g6", "source": "jf", "label": "1", "category": "Cyber", "text": ""},
    ])
    gate = pd.DataFrame([
        {"input_id": "g1", "source": "plain", "entry_type": "harmful", "text": "plain one"},
        {"input_id": "g2", "source": "plain", "entry_type": "dual_use_harmful", "text": "plain two"},
    ])
    out = mod.build(probe, gate, per_stratum=5, seed=0).set_index("row_id")
    assert out.stratum.to_dict() == {"p1": "twin_plain", "p2": "twin_rewrite", "p3": "h_redact_jb", "p4": "h_obf_h",
                                     "p5": "b_dan_b"}
    assert out.loc["p2", "judge_text"] == "plain one" and out.loc["p2", "judge_against"] == "plain_twin"
    assert out.loc["p3", "judge_text"] == "plain two" and out.loc["p3", "entry_type"] == "dual_use_harmful"
    assert out.loc["p4", "judge_against"] == "self" and out.loc["p1", "judge_against"] == "self"


def test_sample_takes_the_wildjailbreak_plain_request():
    mod = _load("scripts/esg_build_sample.py", "esg_build_sample")
    probe = pd.DataFrame([
        {"row_id": "w1", "input_id": "g1", "source": "wjb_adversarial_harmful", "label": "1", "category": "Cyber",
         "text": " a long role play "},
        {"row_id": "w2", "input_id": "g2", "source": "wjb_vanilla_harmful", "label": "1", "category": "Cyber",
         "text": "plain request"},
    ])
    gate = pd.DataFrame(columns=["input_id", "source", "entry_type", "text"])
    wjb = pd.DataFrame([{"vanilla": "plain request", "adversarial": "a long role play"},
                        {"vanilla": "plain request", "adversarial": ""}])
    out = mod.build(probe, gate, per_stratum=5, seed=0, wildjailbreak=wjb).set_index("row_id")
    assert out.loc["w1", "judge_text"] == "plain request" and out.loc["w1", "judge_against"] == "plain_twin"
    assert out.loc["w2", "judge_against"] == "self"


def test_rest_sample_is_strictly_harmful_and_new():
    mod = _load("scripts/esg_build_sample.py", "esg_build_sample")
    row = {"label": "1", "category": "Cyber"}
    probe = pd.DataFrame([
        {"row_id": "p1", "input_id": "g1", "source": "redact_jb", "text": "wrapped one", **row},
        {"row_id": "p2", "input_id": "g1", "source": "jf", "text": "already answered", **row},
        {"row_id": "p3", "input_id": "g2", "source": "dan_h", "text": "dual use", **row},
        {"row_id": "p4", "input_id": "g3", "source": "civil_pos", "text": "toxic comment", **row},
        {"row_id": "p5", "input_id": "g4", "source": "wjb_vanilla_harmful", "text": "no type", **row},
        {"row_id": "p6", "input_id": "g1", "source": "dan_b", "text": "benign", "label": "0", "category": "Benign"},
    ])
    gate = pd.DataFrame([
        {"input_id": "g1", "source": "plain", "entry_type": "harmful", "text": "plain one"},
        {"input_id": "g2", "source": "plain", "entry_type": "dual_use_harmful", "text": "plain two"},
        {"input_id": "g3", "source": "civil_pos", "entry_type": "harmful", "text": "toxic comment"},
    ])
    out = mod.build_rest(probe, gate, exclude={"p2"}).set_index("row_id")
    assert list(out.index) == ["p1"]
    assert out.loc["p1", "stratum"] == "rest_redact_jb" and out.loc["p1", "judge_text"] == "plain one"


def test_refusal_prompts_are_the_strictly_harmful_broken_ones():
    mod = _load("scripts/esg_build_refusal_prompts.py", "esg_build_refusal_prompts")
    sample = pd.DataFrame([
        {"row_id": "a", "label": 1, "entry_type": "harmful"},           # broken: kept
        {"row_id": "b", "label": 1, "entry_type": "harmful"},           # judged harmless
        {"row_id": "c", "label": 1, "entry_type": "dual_use_harmful"},  # broken, but dual-use
        {"row_id": "d", "label": 1, "entry_type": "harmful"},           # too long, no answer
        {"row_id": "a", "label": 1, "entry_type": "harmful"},           # the same row in a second sample
    ])
    answers = pd.DataFrame([{"row_id": r, "status": "ok"} for r in "abc"] + [{"row_id": "d", "status": "too_long"}])
    scores = pd.DataFrame([{"row_id": "a", "p_harmful": 0.8}, {"row_id": "b", "p_harmful": 0.2},
                           {"row_id": "c", "p_harmful": 0.9}])
    assert list(mod.build(sample, answers, scores).row_id) == ["a"]


def test_messages_put_the_system_prompt_first():
    mod = _load("generate_answers.py", "generate_answers")
    assert mod.messages("hi", None) == [{"role": "user", "content": "hi"}]
    assert [m["role"] for m in mod.messages("hi", "decline")] == ["system", "user"]


def test_triples_keep_broken_prompts_with_a_working_refusal():
    mod = _load("scripts/esg_build_triples.py", "esg_build_triples")
    row = {"label": 1, "entry_type": "harmful", "category": "Cyber", "judge_against": "self"}
    sample = pd.DataFrame([
        {"row_id": "a", "input_id": "g1", "source": "redact_jb", "text": "wrapped", "judge_text": "plain one",
         **row, "judge_against": "plain_twin"},
        {"row_id": "b", "input_id": "g2", "source": "plain", "text": "plain two", "judge_text": "plain two", **row},
        {"row_id": "c", "input_id": "g3", "source": "dan_h", "text": "refusal fails", "judge_text": "x", **row},
        {"row_id": "d", "input_id": "g4", "source": "plain", "text": "model refused", "judge_text": "x", **row},
        {"row_id": "e", "input_id": "g5", "source": "plain", "text": "dual use", "judge_text": "x", **row,
         "entry_type": "dual_use_harmful"},
    ])
    ans = lambda ids, text, phrase: pd.DataFrame(
        [{"row_id": i, "status": "ok", "answer": text, "refusal": phrase, "hit_max_new_tokens": False} for i in ids])
    answers, refusals = ans("abcde", "harmful text", False), ans("abce", "I cannot help", True)
    scores = pd.DataFrame([{"row_id": i, "p_harmful": s} for i, s in zip("abcde", [0.8, 0.6, 0.9, 0.0, 0.9])])
    refusal_scores = pd.DataFrame([{"row_id": i, "p_harmful": s} for i, s in zip("abce", [0.0, 0.0, 0.7, 0.0])])
    out = mod.build(sample, answers, scores, refusals, refusal_scores, val_groups={"g2"}).set_index("row_id")
    assert list(out.index) == ["a", "b"]
    assert out.loc["a", "plain_request"] == "plain one" and pd.isna(out.loc["b", "plain_request"])
    assert out.split.to_dict() == {"a": "train", "b": "val"}
    assert out.loc["a", "prompt"] == "wrapped" and out.loc["a", "refusal"] == "I cannot help"


def test_triples_hold_out_the_groups_of_the_probe_pretraining():
    mod = _load("scripts/esg_build_triples.py", "esg_build_triples")
    pre = _load("pretrain_prompt_probe.py", "pretrain_prompt_probe")
    ids = [f"g{i % 37}" for i in range(200)]
    train, val, test = pre.split_three(ids, 0.1, 0.2, 3)
    val_groups, test_groups = mod.held_out_groups(ids, 0.1, 3, 0.2)
    assert val_groups == {ids[i] for i in val} and test_groups == {ids[i] for i in test}
    assert len(val_groups) == 4 and len(test_groups) == 7
    assert not {ids[i] for i in train} & (val_groups | test_groups) and not val_groups & test_groups
    assert pre.split_groups(ids, 0.1, 3)[1] == val  # val is the same with and without a test split
