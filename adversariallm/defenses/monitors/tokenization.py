from __future__ import annotations

import torch
from torch.nn.utils.rnn import pad_sequence

from ._activation_detector_model import get_chat_template


def build_detector_inputs_single(prompt, response, tokenizer, model_name):
    """Reproduce adversarial_detector/src/data.py formatting for one (prompt, response)."""
    first_user_msg, response_template, response_key, _, _ = get_chat_template(model_name)

    full = first_user_msg.format(instruction=prompt) + response_template.format(target=response)
    input_ids = torch.tensor(tokenizer(full)["input_ids"], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)

    prompt_with_key = first_user_msg.format(instruction=prompt) + response_key
    prompt_len = len(tokenizer(prompt_with_key)["input_ids"])

    target_ids = input_ids.clone()
    target_ids[:prompt_len] = 0
    return input_ids, target_ids, attention_mask


def build_detector_batch(prompts, responses, tokenizer, model_name):
    """Returns right-padded (input_ids, target_ids, attention_mask), padding_value=0."""
    triples = [
        build_detector_inputs_single(p, r, tokenizer, model_name)
        for p, r in zip(prompts, responses)
    ]
    input_ids = pad_sequence([t[0] for t in triples], batch_first=True, padding_value=0)
    target_ids = pad_sequence([t[1] for t in triples], batch_first=True, padding_value=0)
    attention_mask = pad_sequence([t[2] for t in triples], batch_first=True, padding_value=0)
    return input_ids, target_ids, attention_mask
