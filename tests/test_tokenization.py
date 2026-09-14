from __future__ import annotations

import torch

from adversariallm.defenses.monitors.tokenization import build_detector_batch


class _FakeTokenizer:
    """Char-level tokenizer (enc(a + b) == enc(a) + enc(b)) with a stub chat template:
    user -> "U<content>", assistant -> "R<content>E", generation prompt -> "R"."""

    def __call__(self, text, **kwargs):
        return {"input_ids": [ord(c) for c in text]}

    def apply_chat_template(self, conv, tokenize=False, add_generation_prompt=False, **kw):
        s = ""
        for m in conv:
            if m["role"] == "user":
                s += "U" + m["content"]
            elif m["role"] == "assistant":
                s += "R" + m["content"] + "E"
        return s + ("R" if add_generation_prompt else "")


def test_target_ids_mask_prompt_region():
    tok = _FakeTokenizer()
    input_ids, target_ids, attention_mask = build_detector_batch(
        prompts=["ab"],
        responses=["xy"],
        tokenizer=tok,
    )
    # full = "U" + "ab" + "R" + "xy" + "E"  -> "UabRxyE" -> 7 tokens
    assert input_ids.shape == (1, 7)
    # prompt_with_key = "U" + "ab" + "R" -> prompt_len = 4
    # target_ids: first 4 zeroed, rest = response-region token ids ("x", "y", "E")
    assert target_ids[0].tolist() == [0, 0, 0, 0, ord("x"), ord("y"), ord("E")]
    assert attention_mask[0].tolist() == [1, 1, 1, 1, 1, 1, 1]


def test_batch_is_right_padded_with_zero():
    tok = _FakeTokenizer()
    input_ids, target_ids, attention_mask = build_detector_batch(
        prompts=["a", "abc"],
        responses=["y", "y"],
        tokenizer=tok,
    )
    assert input_ids.shape[0] == 2
    # shorter row padded on the right with 0; its attention_mask padded with 0
    assert input_ids[0, -1].item() == 0
    assert attention_mask[0, -1].item() == 0
