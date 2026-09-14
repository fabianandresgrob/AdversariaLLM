"""Turn-boundary invariants: where each builder puts the prompt/response split, and where
the reader consequently reads.

These guard a failure mode that produces no error — if the builders disagree about the
boundary, the probe trains at one position and reads at another, and the only symptom is a
detector that looks weak. The chat template is stubbed so the tests run offline.
"""
from __future__ import annotations

import torch

from adversariallm.training.data import (
    build_example_full, build_prompt_only, generation_prefix, render_full, render_prompt,
)
from adversariallm.training.readers import readout_index


class _FakeTok:
    """Char-level tokenizer with an explicit multi-turn template. Turn markers are distinct
    so a dropped terminator (the real Llama-3.1 registry bug) would fail these tests."""

    USER_OPEN, USER_CLOSE = "[U]", "[/U]"
    ASST_OPEN, ASST_CLOSE = "[A]", "[/A]"

    def __call__(self, text, **kw):
        return {"input_ids": [ord(c) for c in text]}

    def apply_chat_template(self, conv, tokenize=False, add_generation_prompt=False, **kw):
        s = ""
        for m in conv:
            if m["role"] == "user":
                s += self.USER_OPEN + m["content"] + self.USER_CLOSE
            elif m["role"] == "assistant":
                s += self.ASST_OPEN + m["content"] + self.ASST_CLOSE
        return s + (self.ASST_OPEN if add_generation_prompt else "")


def test_user_turn_is_closed_before_the_assistant_header():
    """The regression that motivated this file: the user turn must be terminated before the
    assistant header, in both renders."""
    tok = _FakeTok()
    assert render_prompt(tok, "ab") == "[U]ab[/U][A]"
    assert render_full(tok, "ab", "xy") == "[U]ab[/U][A]xy[/A]"
    assert generation_prefix(tok) == "[A]"


def test_masked_prompt_region_ends_exactly_at_the_generation_prefix():
    tok = _FakeTok()
    ids, labels, _, _ = build_example_full("ab", "xy", tok)
    n_masked = int((labels == -100).sum())
    prompt_region = "".join(chr(i) for i in ids[:n_masked].tolist())
    target_region = "".join(chr(i) for i in ids[n_masked:].tolist())
    assert prompt_region == "[U]ab[/U][A]"       # user turn closed, header included
    assert target_region == "xy[/A]"             # response only — no scaffold leaks in
    assert prompt_region.endswith(generation_prefix(tok))


def test_readout_position_is_identical_for_prompt_only_and_prompt_plus_response():
    """build_prompt_only's docstring promises this; a probe trained on one and applied to the
    other only transfers if it holds."""
    tok = _FakeTok()
    _, _, tgt_full, attn_full = build_example_full("ab", "xy", tok)
    ids_p, tgt_p, attn_p = build_prompt_only("ab", tok)

    i_full = readout_index(tgt_full.unsqueeze(0), attn_full.unsqueeze(0))
    i_prompt = readout_index(tgt_p.unsqueeze(0), attn_p.unsqueeze(0))
    assert i_full.tolist() == i_prompt.tolist()

    # and it is the LAST token of the generation prefix, not the first response token
    idx = int(i_full[0])
    assert chr(int(ids_p[idx])) == generation_prefix(tok)[-1]


def test_readout_position_survives_a_longer_prompt():
    """Guards an off-by-one that only shows up when prompt and response lengths differ."""
    tok = _FakeTok()
    for prompt, response in [("a", "xyz"), ("abcdef", "x"), ("ab", "xy")]:
        _, labels, tgt, attn = build_example_full(prompt, response, tok)
        idx = int(readout_index(tgt.unsqueeze(0), attn.unsqueeze(0))[0])
        n_masked = int((labels == -100).sum())
        assert idx == n_masked - 1, f"readout must sit on the last masked token ({prompt!r})"
