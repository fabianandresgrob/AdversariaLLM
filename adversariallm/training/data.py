from __future__ import annotations

import json
import os
import torch
import pandas as pd
from torch.utils.data import Dataset, Subset

from ..defenses.monitors._activation_detector_model import get_chat_template


def split_adv_stream(dataset, val_size, seed=0):
    """Split adversarial behaviors into disjoint (train, val) subsets.

    The split is behavior-level and seeded, so the held-out behaviors used for
    best-checkpoint selection stay fixed across runs.
    """
    if not 0 < val_size < len(dataset):
        raise ValueError(f"val_size must be in (0, {len(dataset)}), got {val_size}")
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(dataset), generator=g).tolist()
    return Subset(dataset, perm[val_size:]), Subset(dataset, perm[:val_size])


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


def build_prompt_only(prompt, tokenizer, model_name):
    """Prompt + response-key scaffold, no completion. target_ids all-zero, so the reader's
    readout falls back to the last real token — the last response-key token, i.e. the
    generation-onset position. This is the SAME readout position as build_example_full
    (which reads the token just before the response), so a probe trained on these transfers
    to the in-loop harmful examples (prompt + response). Returns (input_ids, target_ids, attn)."""
    first_user_msg, _, response_key, _, _ = get_chat_template(model_name)
    text = first_user_msg.format(instruction=prompt) + response_key
    ids = torch.tensor(tokenizer(text)["input_ids"], dtype=torch.long)
    return ids, torch.zeros_like(ids), torch.ones_like(ids)


def load_dataset_prompts(datasets_cfg, name, window, seed=0):
    """Pull (user prompts, assistant responses) from a registered AdversariaLLM dataset
    (alpaca, or_bench, xs_test, ...) over a fixed index window (start, end) — the canonical
    split (conf/splits.yaml). responses are None for prompt-only datasets."""
    from omegaconf import OmegaConf
    from ..dataset.prompt_dataset import PromptDataset

    start, end = int(window[0]), int(window[1])
    node = OmegaConf.merge(datasets_cfg[name], {"seed": seed, "idx": f"list(range({start},{end}))"})
    ds = PromptDataset.from_name(name)(node)
    prompts, responses = [], []
    for i in range(len(ds)):
        conv = ds[i]
        user = next((m["content"] for m in conv if m["role"] == "user"), None)
        asst = next((m["content"] for m in conv if m["role"] == "assistant"), None)
        if user:
            prompts.append(user)
            responses.append(asst)
    return prompts, responses


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
    """UltraChat (x, y) for the KL term: first user turn + first model reply.
    window=(start, end) selects a fixed index slice (the canonical split, conf/splits.yaml)."""

    def __init__(self, tokenizer, model_name, window=None, fraction=0.01):
        from datasets import load_dataset
        ds = load_dataset("stingning/ultrachat", split="train")
        if window is not None:
            ds = ds.select(range(int(window[0]), int(window[1])))       # canonical split (coop)
        elif 0 < fraction < 1.0:
            ds = ds.select(range(int(fraction * len(ds))))              # fraction (model-CAT)
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


class OODBenignStream(Dataset):
    """Out-of-distribution benign (instruction, output) pairs for the over-refusal and FPR
    metrics — a DIFFERENT distribution from the UltraChat KL leash. In-distribution benign
    (UltraChat) is misleadingly optimistic because the leash preserves it by construction;
    OOD benign measures whether utility generalizes. Formatted like the adv stream (0-masked
    target_ids) so the reader reads the same last-prompt-token position for both classes.
    """

    def __init__(self, rows, tokenizer, model_name):
        # rows = list of (prompt, response) from load_dataset_prompts, so the ordering is
        # the canonical split (shared with pretraining and final eval).
        self.rows = [(p, r or "") for p, r in rows if p]
        self.tokenizer, self.model_name = tokenizer, model_name

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        x, y = self.rows[i]
        ids, _, tgt, attn = build_example_full(x, y, self.tokenizer, self.model_name)
        return {"d_ids": ids, "d_targetids": tgt, "d_attn": attn, "prompt": x}


def collate_ood(batch):
    """Collate OODBenignStream items; pass prompts through for free-generation refusal checks."""
    out = pad_collate(batch, ["d_ids", "d_targetids", "d_attn"], pad_id=0)
    out["prompt"] = [b["prompt"] for b in batch]
    return out
