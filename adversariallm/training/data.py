from __future__ import annotations

import json
import os
import torch
import pandas as pd
from torch.utils.data import Dataset

from ..defenses.monitors._activation_detector_model import get_chat_template


def build_supervised_example(prompt, response, tokenizer, model_name):
    """Return (input_ids, labels) with prompt tokens set to -100 in labels."""
    first_user_msg, response_template, response_key, _, _ = get_chat_template(model_name)
    full = first_user_msg.format(instruction=prompt) + response_template.format(target=response)
    input_ids = torch.tensor(tokenizer(full)["input_ids"], dtype=torch.long)
    prompt_len = len(tokenizer(first_user_msg.format(instruction=prompt) + response_key)["input_ids"])
    labels = input_ids.clone()
    labels[:prompt_len] = -100
    return input_ids, labels


def build_example_full(prompt, response, tokenizer, model_name):
    """Return (input_ids, labels, target_ids, attn).

    - labels: prompt region set to -100 (CE convention used by the losses).
    - target_ids: clone of input_ids with the prompt region set to 0 (0-masked
      convention used by ContinuousEmbeddingAttack: 0 = prompt/pad, real id = response).
    - attn: all-ones attention mask (per-example; padding handled by the collate).
    """
    first_user_msg, response_template, response_key, _, _ = get_chat_template(model_name)
    full = first_user_msg.format(instruction=prompt) + response_template.format(target=response)
    input_ids = torch.tensor(tokenizer(full)["input_ids"], dtype=torch.long)
    prompt_len = len(tokenizer(first_user_msg.format(instruction=prompt) + response_key)["input_ids"])
    labels = input_ids.clone()
    labels[:prompt_len] = -100
    target_ids = input_ids.clone()
    target_ids[:prompt_len] = 0
    attn = torch.ones_like(input_ids)
    return input_ids, labels, target_ids, attn


class AdvTupleStream(Dataset):
    """(x, y_harmful, y_benign) per behavior, tokenized for the model."""

    def __init__(self, data_dir, behaviors_csv, targets_json, safe_csv, tokenizer, model_name):
        beh = pd.read_csv(os.path.join(data_dir, behaviors_csv), usecols=["Behavior", "BehaviorID"])
        targets = pd.read_json(os.path.join(data_dir, targets_json), typ="series").reset_index()
        targets.columns = ["BehaviorID", "target"]
        safe = pd.read_csv(os.path.join(data_dir, safe_csv))  # columns: Behavior, Safe_Response
        df = beh.merge(targets, on="BehaviorID")
        safe_map = dict(zip(safe["Behavior"], safe["Safe_Response"]))

        def _first_target(t):
            # adv_training_targets.json stores a list of targets per behavior;
            # advbench_targets.json stores a single string. Use the first either way.
            return t[0] if isinstance(t, (list, tuple)) else t

        self.rows = [(r["Behavior"], _first_target(r["target"]), safe_map.get(r["Behavior"], "I can't help with that."))
                     for _, r in df.iterrows()]
        self.tokenizer, self.model_name = tokenizer, model_name

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        x, y_h, y_b = self.rows[i]
        h_ids, h_lab, h_tgt, h_attn = build_example_full(x, y_h, self.tokenizer, self.model_name)
        b_ids, b_lab, b_tgt, b_attn = build_example_full(x, y_b, self.tokenizer, self.model_name)
        return {
            "prompt": x,
            "h_ids": h_ids, "h_labels": h_lab, "h_targetids": h_tgt, "h_attn": h_attn,
            "b_ids": b_ids, "b_labels": b_lab, "b_targetids": b_tgt, "b_attn": b_attn,
        }


class UtilityStream(Dataset):
    """UltraChat (x, y) for the KL term: first user turn + first model reply."""

    def __init__(self, tokenizer, model_name, fraction=0.01):
        from datasets import load_dataset
        ds = load_dataset("stingning/ultrachat", split="train")
        if 0 < fraction < 1.0:
            ds = ds.select(range(int(fraction * len(ds))))
        self.rows = [(d["data"][0], d["data"][1]) for d in ds if len(d["data"]) >= 2]
        self.tokenizer, self.model_name = tokenizer, model_name

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        x, y = self.rows[i]
        ids, lab = build_supervised_example(x, y, self.tokenizer, self.model_name)
        attn = torch.ones_like(ids)
        return {"input_ids": ids, "labels": lab, "attn": attn}


def pad_collate(batch, keys, pad_id=0):
    from torch.nn.utils.rnn import pad_sequence
    out = {}
    for k in keys:
        seqs = [b[k] for b in batch]
        out[k] = pad_sequence(seqs, batch_first=True, padding_value=(-100 if "labels" in k else pad_id))
    return out


def collate_adv(batch):
    """Collate AdvTupleStream items.

    ids/targetids/attn are padded with 0; labels are padded with -100. The string
    `prompt` field is passed through as a list.
    """
    tensor_keys = [
        "h_ids", "h_labels", "h_targetids", "h_attn",
        "b_ids", "b_labels", "b_targetids", "b_attn",
    ]
    out = pad_collate(batch, tensor_keys, pad_id=0)
    out["prompt"] = [b["prompt"] for b in batch]
    return out


def collate_util(batch):
    """Collate UtilityStream items: input_ids/attn padded with 0, labels with -100."""
    return pad_collate(batch, ["input_ids", "labels", "attn"], pad_id=0)
