from __future__ import annotations

import torch

from adversariallm.defenses.monitors.tokenization import build_detector_batch


class _FakeTokenizer:
    """Char-level tokenizer where enc(a + b) == enc(a) + enc(b)."""

    def __call__(self, text, **kwargs):
        return {"input_ids": [ord(c) for c in text]}


def _patch_template(monkeypatch):
    from adversariallm.defenses.monitors import tokenization as tok_mod

    def fake_template(_model_name):
        # first_user_msg, response_template, response_key, first_part, second_part
        return ("U{instruction}", "R{target}E", "R", "U{instruction}", "")

    monkeypatch.setattr(tok_mod, "get_chat_template", fake_template)


def test_target_ids_mask_prompt_region(monkeypatch):
    _patch_template(monkeypatch)
    tok = _FakeTokenizer()
    input_ids, target_ids, attention_mask = build_detector_batch(
        prompts=["ab"],
        responses=["xy"],
        tokenizer=tok,
        model_name="meta-llama/Llama-3.1-8B-Instruct",
    )
    # full = "U" + "ab" + "R" + "xy" + "E"  -> 6 tokens
    assert input_ids.shape == (1, 6)
    # prompt_with_key = "U" + "ab" + "R" -> prompt_len = 4
    # target_ids: first 4 zeroed, rest = response-region token ids
    assert target_ids[0].tolist() == [0, 0, 0, 0, ord("y"), ord("E")]
    assert attention_mask[0].tolist() == [1, 1, 1, 1, 1, 1]


def test_batch_is_right_padded_with_zero(monkeypatch):
    _patch_template(monkeypatch)
    tok = _FakeTokenizer()
    input_ids, target_ids, attention_mask = build_detector_batch(
        prompts=["a", "abc"],
        responses=["y", "y"],
        tokenizer=tok,
        model_name="meta-llama/Llama-3.1-8B-Instruct",
    )
    assert input_ids.shape[0] == 2
    # shorter row padded on the right with 0; its attention_mask padded with 0
    assert input_ids[0, -1].item() == 0
    assert attention_mask[0, -1].item() == 0
