from __future__ import annotations

import os

import pandas as pd
import torch
from torch.utils.data import Dataset, Subset

def render_prompt(tokenizer, prompt):
    """Prompt rendered up to the generation onset (assistant header, no content).

    Uses the tokenizer's own chat template — set from models.yaml `chat_template` by
    load_model_and_tokenizer — so training, attacks and eval all agree."""
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
    )


def render_full(tokenizer, prompt, response):
    """Prompt + assistant response, terminated. Pairs with render_prompt for label masking."""
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}, {"role": "assistant", "content": response}],
        tokenize=False,
        add_generation_prompt=False,
    )


def user_token_mask(tokenizer, prompt, length=None, span=None):
    """Bool mask over the tokens of render_prompt(tokenizer, prompt) that carry the user message --
    everything else (system block, role headers, generation prefix) is template. `span` (char
    offsets into `prompt`) narrows it to part of the message, e.g. an appended suffix. Right-padded
    with False to `length` (a full prompt+response sequence). Uses the tokenizer's char offsets, so
    it is exact for any chat template, including tokens that straddle the message boundary."""
    rendered = render_prompt(tokenizer, prompt)
    start = rendered.rfind(prompt)
    if start < 0:
        raise ValueError("the chat template altered the user message; cannot locate it")
    lo, hi = span if span is not None else (0, len(prompt))
    lo, hi = start + lo, start + hi
    offsets = tokenizer(rendered, add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"]
    mask = torch.tensor([s < hi and e > lo for s, e in offsets], dtype=torch.bool)
    if length is not None:
        mask = torch.cat([mask, torch.zeros(length - mask.numel(), dtype=torch.bool)])
    return mask


def generation_prefix(tokenizer):
    """The assistant-header scaffold the template appends at generation onset (the old
    registry's `response_key`). Derived by diffing the same conversation rendered with and
    without add_generation_prompt, so it works for any model's template."""
    conv = [{"role": "user", "content": "x"}]
    without = tokenizer.apply_chat_template(conv, tokenize=False, add_generation_prompt=False)
    with_ = tokenizer.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)
    return with_[len(without):]


def _encode(tokenizer, text):
    """Tokenize a rendered chat string. add_special_tokens=False because the jinja already
    emits bos_token — letting the tokenizer add another double-prepends BOS."""
    return tokenizer(text, add_special_tokens=False)["input_ids"]


def split_adv_stream(dataset, val_size, seed=0):
    """Split adversarial behaviors into disjoint (train, val) subsets.

    Behavior-level (not row-level): a behavior's multiple targets never straddle the
    split. Seeded, so the held-out behaviors stay fixed across runs. Val keeps one row
    per behavior (first target) so validation cost is independent of targets/behavior.
    """
    behaviors = list(dict.fromkeys(p for p, _, _ in dataset.rows))  # unique, first-appearance order
    if not 0 < val_size < len(behaviors):
        raise ValueError(f"val_size must be in (0, {len(behaviors)}), got {val_size}")
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(behaviors), generator=g).tolist()
    val_beh = {behaviors[i] for i in perm[:val_size]}
    train_idx = [i for i, r in enumerate(dataset.rows) if r[0] not in val_beh]
    val_idx, seen = [], set()  # one row per val behavior
    for i, r in enumerate(dataset.rows):
        if r[0] in val_beh and r[0] not in seen:
            val_idx.append(i)
            seen.add(r[0])
    return Subset(dataset, train_idx), Subset(dataset, val_idx)


def build_supervised_example(prompt, response, tokenizer):
    """Return (input_ids, labels) with prompt tokens set to -100 in labels."""
    input_ids = torch.tensor(_encode(tokenizer, render_full(tokenizer, prompt, response)), dtype=torch.long)
    prompt_len = len(_encode(tokenizer, render_prompt(tokenizer, prompt)))
    labels = input_ids.clone()
    labels[:prompt_len] = -100
    return input_ids, labels


def build_example_full(prompt, response, tokenizer):
    """Return (input_ids, labels, target_ids, attn).

    - labels: prompt region set to -100 (CE convention used by the losses).
    - target_ids: clone of input_ids with the prompt region set to 0 (0-masked
      convention used by ContinuousEmbeddingAttack: 0 = prompt/pad, real id = response).
    - attn: all-ones attention mask (per-example; padding handled by the collate).
    """
    input_ids = torch.tensor(_encode(tokenizer, render_full(tokenizer, prompt, response)), dtype=torch.long)
    prompt_len = len(_encode(tokenizer, render_prompt(tokenizer, prompt)))
    labels = input_ids.clone()
    labels[:prompt_len] = -100
    target_ids = input_ids.clone()
    target_ids[:prompt_len] = 0
    attn = torch.ones_like(input_ids)
    return input_ids, labels, target_ids, attn


def build_prompt_only(prompt, tokenizer):
    """Prompt up to the generation onset, no completion. target_ids all-zero, so the reader's
    readout falls back to the last real token — the last assistant-header token, i.e. the
    generation-onset position. This is the SAME readout position as build_example_full
    (which reads the token just before the response), so a probe trained on these transfers
    to the in-loop harmful examples (prompt + response). Returns (input_ids, target_ids, attn)."""
    ids = torch.tensor(_encode(tokenizer, render_prompt(tokenizer, prompt)), dtype=torch.long)
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

        def _targets(t):
            # adv_training_targets.json = list per behavior; advbench_targets.json = single string.
            ts = list(t) if isinstance(t, (list, tuple)) else [t]
            return [s for s in ts if str(s).strip()]  # drop empty targets

        # one row per (behavior, target); all targets of a behavior share its y_safe.
        self.rows = [
            (r["Behavior"], tgt, safe_map.get(r["Behavior"], "I can't help with that."))
            for _, r in df.iterrows()
            for tgt in _targets(r["target"])
        ]
        self.tokenizer, self.model_name = tokenizer, model_name

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        x, y_h, y_b = self.rows[i]
        h_ids, h_lab, h_tgt, h_attn = build_example_full(x, y_h, self.tokenizer)
        b_ids, b_lab, b_tgt, b_attn = build_example_full(x, y_b, self.tokenizer)
        return {
            "prompt": x,
            "h_perturb_mask": user_token_mask(self.tokenizer, x, h_ids.numel()),  # user message only
            "h_ids": h_ids,
            "h_labels": h_lab,
            "h_targetids": h_tgt,
            "h_attn": h_attn,
            "b_ids": b_ids,
            "b_labels": b_lab,
            "b_targetids": b_tgt,
            "b_attn": b_attn,
        }


class UtilityStream(Dataset):
    """(x, y) supervised pairs for the KL term. Default source = UltraChat (first user turn +
    first model reply); pass rows=[(prompt, response), ...] to use any other source (e.g. a
    registered dataset via load_dataset_prompts). window/fraction select the UltraChat slice.
    max_length truncates long examples (e.g. Magpie responses run to ~90k chars)."""

    def __init__(self, tokenizer, model_name, window=None, fraction=0.01, rows=None, max_length=None):
        if rows is not None:
            self.rows = rows
        else:
            from datasets import load_dataset

            ds = load_dataset("stingning/ultrachat", split="train")
            if window is not None:
                ds = ds.select(range(int(window[0]), int(window[1])))  # canonical split (coop)
            elif 0 < fraction < 1.0:
                ds = ds.select(range(int(fraction * len(ds))))  # fraction (model-CAT)
            self.rows = [(d["data"][0], d["data"][1]) for d in ds if len(d["data"]) >= 2]
        self.tokenizer, self.model_name, self.max_length = tokenizer, model_name, max_length

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        x, y = self.rows[i]
        ids, lab = build_supervised_example(x, y, self.tokenizer)
        if self.max_length is not None and ids.numel() > self.max_length:
            ids, lab = ids[: self.max_length], lab[: self.max_length]  # cap runaway lengths
        attn = torch.ones_like(ids)
        return {"input_ids": ids, "labels": lab, "attn": attn}


def build_kl_stream(datasets_cfg, kl_source, tokenizer, model_name,
                    window=None, fraction=0.01, max_length=None, seed=0):
    """KL-leash UtilityStream from a named source. 'ultrachat' = the built-in loader (window
    or fraction); any other name pulls (prompt, response) via the dataset registry."""
    if kl_source == "ultrachat":
        return UtilityStream(tokenizer, model_name, window=window, fraction=fraction, max_length=max_length)
    prompts, responses = load_dataset_prompts(datasets_cfg, kl_source, window=window, seed=seed)
    rows = [(p, r) for p, r in zip(prompts, responses) if r]  # KL needs a reference response
    return UtilityStream(tokenizer, model_name, rows=rows, max_length=max_length)


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
        "h_ids",
        "h_labels",
        "h_targetids",
        "h_attn",
        "h_perturb_mask",
        "b_ids",
        "b_labels",
        "b_targetids",
        "b_attn",
    ]
    out = pad_collate(batch, tensor_keys, pad_id=0)
    out["prompt"] = [b["prompt"] for b in batch]
    return out


def collate_util(batch):
    """Collate UtilityStream items: input_ids/attn padded with 0, labels with -100."""
    return pad_collate(batch, ["input_ids", "labels", "attn"], pad_id=0)


class BenignStream(Dataset):
    """Benign (prompt, response) pairs in reader format (0-masked target_ids; the reader reads
    the last prompt token, so the response is irrelevant and may be None). Rows come from
    load_dataset_prompts, so ordering is the canonical split. Serves both roles: the detector's
    benign training class (a diverse mix of sources' train windows) and OOD over-refusal / FPR
    validation. `prompt` is kept for free-generation refusal checks."""

    def __init__(self, rows, tokenizer, model_name):
        self.rows = [(p, r or "") for p, r in rows if p]
        self.tokenizer, self.model_name = tokenizer, model_name

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        x, y = self.rows[i]
        ids, _, tgt, attn = build_example_full(x, y, self.tokenizer)
        return {"d_ids": ids, "d_targetids": tgt, "d_attn": attn, "prompt": x}


def collate_benign(batch):
    out = pad_collate(batch, ["d_ids", "d_targetids", "d_attn"], pad_id=0)
    out["prompt"] = [b["prompt"] for b in batch]
    return out


class HelpRefusePairStream(Dataset):
    """Benign prompts paired with a helpful target y_help and a fixed canned refusal. Provides
    two model forwards per row — help (prompt+y_help) and refuse (prompt+refusal) — which is
    what the w_M^b benign-gate diagnostic needs. Rows with y_help None carry has_target=0 and
    use the refusal string as a dummy so collation stays uniform."""

    def __init__(self, rows, tokenizer, model_name, refusal="I can't help with that."):
        self.rows = [(p, y) for p, y in rows if p]
        self.tokenizer, self.model_name, self.refusal = tokenizer, model_name, refusal

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        x, y = self.rows[i]
        has_target = y is not None
        y_help = y if has_target else self.refusal  # dummy for targetless rows (masked out)
        g_ids, g_lab, g_tgt, g_attn = build_example_full(x, y_help, self.tokenizer)
        r_ids, r_lab, _, r_attn = build_example_full(x, self.refusal, self.tokenizer)
        return {
            "prompt": x,
            "y_help_text": y_help,
            "has_target": torch.tensor(1.0 if has_target else 0.0),
            "g_ids": g_ids, "g_labels": g_lab, "g_targetids": g_tgt, "g_attn": g_attn,
            "r_ids": r_ids, "r_labels": r_lab, "r_attn": r_attn,
        }


def collate_help_pair(batch):
    """Collate HelpRefusePairStream items: ids/targetids/attn padded with 0, labels with -100."""
    out = pad_collate(
        batch,
        ["g_ids", "g_labels", "g_targetids", "g_attn", "r_ids", "r_labels", "r_attn"],
        pad_id=0,
    )
    out["has_target"] = torch.stack([b["has_target"] for b in batch])
    out["prompt"] = [b["prompt"] for b in batch]
    out["y_help_text"] = [b["y_help_text"] for b in batch]
    return out
