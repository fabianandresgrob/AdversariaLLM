from __future__ import annotations

import torch
from adversariallm.training.data import build_supervised_example


class _FakeTok:
    def __call__(self, text, **kw):
        return {"input_ids": [ord(c) for c in text]}


def test_prompt_tokens_are_masked_in_labels(monkeypatch):
    import adversariallm.training.data as d
    monkeypatch.setattr(d, "get_chat_template",
                        lambda m: ("U{instruction}", "R{target}E", "R", "U{instruction}", ""))
    ids, labels = build_supervised_example("ab", "xy", _FakeTok(), "meta-llama/Llama-3.1-8B-Instruct")
    # full = "Uab" + "RxyE"; prompt_with_key = "UabR" -> prompt_len 4
    assert ids.tolist() == [ord(c) for c in "UabRxyE"]
    assert labels.tolist() == [-100, -100, -100, -100, ord("x"), ord("y"), ord("E")]
