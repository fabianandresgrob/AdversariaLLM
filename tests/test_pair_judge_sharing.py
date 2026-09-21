"""The judge's weights: whose they are, and how many copies of them exist.

PAIR already holds an attacker (vicuna-13b, ~26GB) beside the target, so pointing judge_model at
the attacker's id must reuse those weights rather than load a second copy -- on an 80GB card the
difference is OOM -- and must carry the attacker's tokenizer, which is where the chat template
lives (JudgeModelConfig.chat_template defaults to None).
"""

from types import SimpleNamespace

import pytest

pytest.importorskip("torch")

from adversariallm.attacks.pair import select_judge  # noqa: E402

VICUNA = "lmsys/vicuna-13b-v1.5"
TARGET = "meta-llama/Llama-3.1-8B-Instruct"


class _Model:
    def __init__(self, name):
        self.name_or_path = name


@pytest.fixture
def loads(monkeypatch):
    """Record every model load select_judge triggers."""
    seen = []

    def fake_load(cfg):
        seen.append(cfg.id)
        return _Model(cfg.id), f"tokenizer:{cfg.id}"

    monkeypatch.setattr("adversariallm.attacks.pair.load_model_and_tokenizer", fake_load)
    return seen


def _config(judge_id, attack_id=VICUNA):
    return SimpleNamespace(attack_model=SimpleNamespace(id=attack_id),
                           judge_model=SimpleNamespace(id=judge_id))


def _target():
    return SimpleNamespace(model=_Model(TARGET), tokenizer=f"tokenizer:{TARGET}")


def test_judge_shares_the_attacker_weights_when_the_ids_match(loads):
    attacker, attack_tok = _Model(VICUNA), f"tokenizer:{VICUNA}"
    judge, judge_tok = select_judge(_config(VICUNA), _target(), attacker, attack_tok)
    assert judge is attacker, "judge must reuse the attacker's weights, not load a second copy"
    assert judge_tok is attack_tok, "judge must inherit the attacker's tokenizer (chat template)"
    assert loads == [], "no model should be loaded for a judge that shares the attacker"


def test_null_judge_means_the_target_judges_itself(loads):
    target = _target()
    judge, judge_tok = select_judge(_config(None), target, _Model(VICUNA), f"tokenizer:{VICUNA}")
    assert judge is target.model and judge_tok is target.tokenizer
    assert loads == []


def test_a_distinct_judge_id_is_loaded_separately(loads):
    attacker = _Model(VICUNA)
    judge, judge_tok = select_judge(_config("google/gemma-3-12b-it"), _target(), attacker, "t")
    assert judge is not attacker and judge.name_or_path == "google/gemma-3-12b-it"
    assert loads == ["google/gemma-3-12b-it"]
