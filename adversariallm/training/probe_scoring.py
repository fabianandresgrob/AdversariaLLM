"""Answer probes on fixed text: score (prompt, answer) pairs, and the pipeline's refusals (model refuses OR probe flags).

Shared by rescore_answer_probe.py (attack answers) and run_overrefusal_eval.py (xs_test answers)."""

import json
import os
from pathlib import Path

import torch


def score_pairs(model, tokenizer, readers: dict, pairs, layer: int = -1, max_answer_tokens: int = 512,
                batch_size: int = 8) -> dict[str, list[float]]:
    """p_harmful of every reader on each (prompt, answer), teacher-forced through the model; the answer is cut to
    max_answer_tokens (no reader reads further). Returns {reader name: [score per pair]}."""
    from .data import build_example_full, pad_collate

    device = next(model.parameters()).device
    out = {name: [] for name in readers}
    keys = ["ids", "tgt", "attn"]
    for start in range(0, len(pairs), batch_size):
        items = []
        for prompt, answer in pairs[start:start + batch_size]:
            ids, _, tgt, attn = build_example_full(prompt, answer, tokenizer)
            p_len = int((tgt > 0).float().argmax()) if (tgt > 0).any() else ids.numel()
            items.append(dict(zip(keys, (t[: p_len + max_answer_tokens] for t in (ids, tgt, attn)))))
        batch = {k: v.to(device) for k, v in pad_collate(items, keys, pad_id=0).items()}
        with torch.no_grad():
            hidden = model(input_ids=batch["ids"], attention_mask=batch["attn"],
                           output_hidden_states=True).hidden_states[layer]
            for name, reader in readers.items():
                out[name] += reader.p_harmful(hidden, batch["tgt"], batch["attn"]).tolist()
    return out


def calibrated_threshold(reader_path: str) -> float:
    """The coop_probe defense's threshold for this reader: threshold_1pct_calib.json next to it (1% false positives on
    2000 own Alpaca answers). One run dir holds one calibration file, so it must name this reader."""
    path = Path(reader_path).parent / "threshold_1pct_calib.json"
    data = json.loads(path.read_text())
    if os.path.normpath(str(data["checkpoint_path"])) != os.path.normpath(str(reader_path)):
        raise ValueError(f"{path} calibrates {data['checkpoint_path']}, not {reader_path}")
    return float(data["threshold"])


def pipeline_refusals(refused, scores, threshold: float) -> list[bool]:
    """Per answer: the pipeline refuses if the model refused or the probe flags the answer (score above threshold)."""
    if len(refused) != len(scores):
        raise ValueError("one refusal flag per probe score")
    return [bool(r) or s > threshold for r, s in zip(refused, scores)]
