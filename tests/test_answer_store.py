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
