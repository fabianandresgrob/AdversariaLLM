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
