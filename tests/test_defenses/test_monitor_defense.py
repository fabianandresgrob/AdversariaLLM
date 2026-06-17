from __future__ import annotations

import torch

from adversariallm.defenses.monitors import activation_llm as act_mod
from adversariallm.defenses.monitors.activation_llm import ActivationLLMMonitor


class _FakeHidden:
    def __init__(self, hidden):
        self.hidden_states = [hidden, hidden]  # index -1 valid


class _FakeTargetModel:
    def get_input_embeddings(self):
        class _Emb:
            def __call__(self, input_ids):
                return input_ids.unsqueeze(-1).float()
        return _Emb()

    def parameters(self):
        yield torch.zeros(1)

    def __call__(self, inputs_embeds, attention_mask, output_hidden_states):
        assert output_hidden_states is True
        B, L = attention_mask.shape
        return _FakeHidden(torch.zeros(B, L, 4))


class _FakeDetector:
    def __call__(self, hidden_states, target_ids, attention_mask):
        B = hidden_states.shape[0]
        # logits_yes_no: column 0 = yes(harmful). Make row 0 harmful, others safe.
        logits = torch.tensor([[5.0, 0.0]] + [[0.0, 5.0]] * (B - 1))
        return None, logits


def test_activation_monitor_score_returns_p_yes(monkeypatch):
    monkeypatch.setattr(
        act_mod, "build_detector_batch",
        lambda prompts, responses, tokenizer, model_name: (
            torch.zeros(len(prompts), 3, dtype=torch.long),
            torch.zeros(len(prompts), 3, dtype=torch.long),
            torch.ones(len(prompts), 3, dtype=torch.long),
        ),
    )
    monitor = ActivationLLMMonitor.__new__(ActivationLLMMonitor)
    monitor.index_hidden_layer_detector = -1
    monitor.target_model_id = "meta-llama/Llama-3.1-8B-Instruct"
    monitor.batch_size = 8
    monitor._detector = _FakeDetector()

    scores = monitor.score(
        ["bad", "good"], ["r1", "r2"],
        target_model=_FakeTargetModel(),
        target_tokenizer=object(),
    )
    assert scores[0] > 0.9   # harmful
    assert scores[1] < 0.1   # safe


from adversariallm.defenses.monitor_defense import MonitorDefense
from adversariallm.lm_utils.text_generation import GenerationResult


class _RaggedGenerator:
    def generate(self, convs, **kwargs):
        return GenerationResult(
            gen=[["harmful-text", "safe-text"], ["safe-text"]],
            input_ids=[[10], [20]],
        )


class _ScriptedMonitor:
    def score(self, prompts, responses, *, target_model, target_tokenizer):
        # harmful only when the response contains "harmful"
        return [0.99 if "harmful" in r else 0.01 for r in responses]


def test_monitor_defense_overwrites_above_threshold():
    defense = object.__new__(MonitorDefense)
    defense.text_generator = _RaggedGenerator()
    defense.monitor = _ScriptedMonitor()
    defense.threshold = 0.5
    defense.refusal_text = "REFUSED"
    defense.model = object()
    defense.tokenizer = object()

    result = defense.generate(
        [
            [{"role": "user", "content": "p0"}],
            [{"role": "user", "content": "p1"}],
        ],
        num_return_sequences=2,
    )

    assert result.gen == [["REFUSED", "safe-text"], ["safe-text"]]
    assert result.raw_gen == [["harmful-text", "safe-text"], ["safe-text"]]
    assert result.defense_decisions[0][0]["applied"] is True
    assert result.defense_decisions[0][1]["applied"] is False
    assert result.input_ids == [[10], [20]]


def test_monitor_registered_and_buildable(monkeypatch):
    from adversariallm.defenses import DEFENSE_COMPATIBLE_ATTACKS, build_target_system
    from adversariallm.defenses import monitor_defense as md_mod

    # NB: DEFENSE_COMPATIBLE_ATTACKS lists *attacks*, not defenses. A monitor defense
    # works with any attack already in that set; just sanity-check the set is intact.
    assert "pair" in DEFENSE_COMPATIBLE_ATTACKS

    class _StubMonitor:
        def score(self, prompts, responses, *, target_model, target_tokenizer):
            return [0.0 for _ in prompts]

    monkeypatch.setattr(md_mod, "build_monitor", lambda _cfg: _StubMonitor())
    monkeypatch.setattr(
        md_mod, "LocalTextGenerator",
        lambda _m, _t, default_generate_kwargs=None: object(),
    )

    defended = build_target_system(
        {
            "name": "monitor",
            "threshold": 0.5,
            "refusal_text": "REFUSED",
            "monitor": {"name": "activation_llm"},
        },
        model=object(),
        tokenizer=object(),
    )
    assert isinstance(defended, md_mod.MonitorDefense)
    assert defended.threshold == 0.5
