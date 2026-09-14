from __future__ import annotations

import torch
from torch.nn.utils.rnn import pad_sequence

_SENTINEL = "\x00__CONTENT__\x00"


def split_user_turn(tokenizer, prompt):
    """Split the rendered user turn at its content: returns (prefix, suffix).

    prefix ends with `prompt`; suffix is the turn terminator plus the assistant header. Callers
    that need to splice embeddings into the middle of a prompt (the activation detector) encode
    the two halves separately. Derived from the tokenizer's chat template by rendering a
    sentinel and partitioning on it, so it is correct for any model without a per-model table.
    """
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": _SENTINEL}], tokenize=False, add_generation_prompt=True
    )
    head, found, tail = rendered.partition(_SENTINEL)
    if not found:
        raise ValueError("chat template dropped the content sentinel; cannot split the user turn")
    return head + prompt, tail


def build_detector_inputs_single(prompt, response, tokenizer):
    """(input_ids, target_ids, attention_mask) for one (prompt, response).

    target_ids is input_ids with the prompt region zeroed — the 0-masked convention the
    readers use to locate the readout position. add_special_tokens=False throughout because
    the chat template already emits bos_token.
    """
    full = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}, {"role": "assistant", "content": response}],
        tokenize=False,
        add_generation_prompt=False,
    )
    prompt_only = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
    )
    input_ids = torch.tensor(tokenizer(full, add_special_tokens=False)["input_ids"], dtype=torch.long)
    prompt_len = len(tokenizer(prompt_only, add_special_tokens=False)["input_ids"])
    attention_mask = torch.ones_like(input_ids)
    target_ids = input_ids.clone()
    target_ids[:prompt_len] = 0
    return input_ids, target_ids, attention_mask


def build_detector_batch(prompts, responses, tokenizer):
    """Returns right-padded (input_ids, target_ids, attention_mask), padding_value=0."""
    triples = [build_detector_inputs_single(p, r, tokenizer) for p, r in zip(prompts, responses)]
    input_ids = pad_sequence([t[0] for t in triples], batch_first=True, padding_value=0)
    target_ids = pad_sequence([t[1] for t in triples], batch_first=True, padding_value=0)
    attention_mask = pad_sequence([t[2] for t in triples], batch_first=True, padding_value=0)
    return input_ids, target_ids, attention_mask
