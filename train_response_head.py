"""Fit a DualProbe's response channel offline on a model's own attack completions.

    pixi run --frozen python train_response_head.py model=E-nd6-s0 name=RH-E-nd6-all
    pixi run --frozen python train_response_head.py model=E-nd6-s0 name=RH-E-nd6-nopair \
        'train_attacks=[inpainting,gcg]'

The coop probe's harmful class is an attacked prompt followed by a hand-written target ("Sure,
here is ..."), which no real jailbroken answer looks like (probe_diagnose: 0/40 real hits caught).
This trains the response channel on what the model actually says instead, the recipe of Libon et
al.: the model's completions under attack, labelled by the judge that scored the attack run.

  harmful   completions with p_harmful > pos_threshold  (the attack's hits)
  harmless  completions to the SAME attacked prompts with p_harmful < neg_threshold -- mostly
            refusals, the negatives the coop probe never sees -- plus the model's own greedy
            answers to alpaca prompts

The model and the prompt channel are frozen: the prompt channel is the model's co-trained probe,
copied unchanged, so the DualProbe keeps everything it already catches (p = max of the channels).

Held out: every behavior with dataset index < test_below (the 20-behavior eval set). train_attacks
restricts which attacks train the head; every attack is always evaluated, so leaving one out of
training measures transfer to an attack family the head never saw.

Writes checkpoints_probe/RH/<name>/final_reader.pt (a pair-style checkpoint LinearProbeMonitor
loads) and metrics.json, also copied to outputs/eval/response_head/<name>.json.
"""

import json
import logging
import random
import shutil
from collections import defaultdict
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from inspect_attacks import last_user, parse_dir_name

log = logging.getLogger(__name__)
SCORE_KEY = "p_harmful"


def collect_examples(repo: Path, model: str, attacks: list[str], defense: str = "none",
                     classifier: str = "strong_reject", pos_threshold: float = 0.5,
                     neg_threshold: float = 0.1) -> list[dict]:
    """Every judged completion of `model` under `attacks`, labelled 1 (hit) or 0 (clear miss).

    Completions between the two thresholds are dropped: the judge is unsure about them and they
    would only add label noise. idx is the behavior's dataset index (each run.json holds one)."""
    out = []
    for run_json in sorted((repo / "outputs").glob("*__*__*/*/*/*/run.json")):
        parsed = parse_dir_name(run_json.parents[3].name)
        if parsed is None:
            continue
        attack, cell_defense, cell_model = parsed
        if attack not in attacks or cell_defense != defense or cell_model != model:
            continue
        try:
            data = json.loads(run_json.read_text())
        except (OSError, ValueError):
            continue
        idx = ((data.get("config") or {}).get("dataset_params") or {}).get("idx")
        if not isinstance(idx, list) or len(idx) != 1:
            continue  # cannot place the behavior in a split -- never guess
        for run in data.get("runs", []):
            behavior = last_user(run.get("original_prompt"))
            for step in run.get("steps", []):
                scores = (step.get("scores", {}).get(classifier) or {}).get(SCORE_KEY) or []
                completions = step.get("model_completions") or []
                prompt = last_user(step.get("model_input"))
                for score, completion in zip(scores, completions):
                    score = float(score)
                    if score > pos_threshold:
                        label = 1
                    elif score < neg_threshold:
                        label = 0
                    else:
                        continue
                    out.append({"idx": int(idx[0]), "attack": attack, "behavior": behavior,
                                "prompt": prompt, "response": completion, "p_harmful": score,
                                "label": label})
    return out


def split_examples(examples: list[dict], test_below: int, train_attacks: list[str],
                   cap: int, seed: int = 0) -> tuple[list[dict], list[dict]]:
    """Behavior-disjoint split, capped per (behavior, attack, label).

    The cap keeps one attack (inpainting: ~1000 samples per behavior) or one easy behavior from
    dominating. Test positives are NOT capped: every hit of a held-out behavior is scored, so the
    post-defense behavior ASR counts all of them. Train keeps only train_attacks."""
    rng = random.Random(seed)
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for ex in examples:
        groups[(ex["idx"], ex["attack"], ex["label"])].append(ex)
    train, test = [], []
    for (idx, attack, label), group in sorted(groups.items()):
        group = list(group)
        rng.shuffle(group)
        if idx < test_below:
            test += group if label == 1 else group[:cap]
        elif attack in train_attacks:
            train += group[:cap]
    return train, test


def threshold_at(scores: list[float], fpr: float) -> float:
    """The (1 - fpr) quantile of benign scores: flag above it."""
    ordered = sorted(scores)
    return ordered[min(len(ordered) - 1, int((1 - fpr) * len(ordered)))]


def evaluate(test: list[dict], p_prompt: list[float], p_resp: list[float], tau_prompt: float,
             tau_dual: float) -> dict:
    """Per attack: how many held-out hits each channel flags, and what the pipeline ASR would be.

    asr_after counts a behavior as jailbroken iff one of its hits goes unflagged -- a flagged
    completion is replaced by a refusal, so it no longer counts. tpr_prompt is the co-trained
    probe alone at its own 1% threshold, i.e. what the pipeline catches today."""
    per_attack: dict[str, dict] = {}
    for attack in sorted({ex["attack"] for ex in test}):
        rows = [i for i, ex in enumerate(test) if ex["attack"] == attack]
        pos = [i for i in rows if test[i]["label"] == 1]
        neg = [i for i in rows if test[i]["label"] == 0]
        behaviors = {test[i]["idx"] for i in rows}
        dual = {i: max(p_prompt[i], p_resp[i]) > tau_dual for i in rows}
        prompt_only = {i: p_prompt[i] > tau_prompt for i in rows}

        def asr(flagged):
            return len({test[i]["idx"] for i in pos if not flagged[i]}) / max(len(behaviors), 1)

        per_attack[attack] = {
            "n_behaviors": len(behaviors), "n_hits": len(pos), "n_misses": len(neg),
            "tpr_prompt": _rate(prompt_only, pos), "tpr_dual": _rate(dual, pos),
            "flag_rate_misses_dual": _rate(dual, neg),
            "asr_before": len({test[i]["idx"] for i in pos}) / max(len(behaviors), 1),
            "asr_after_prompt": asr(prompt_only), "asr_after_dual": asr(dual),
        }
    return per_attack


def _rate(flags: dict, rows: list[int]) -> float | None:
    return round(sum(flags[i] for i in rows) / len(rows), 4) if rows else None


def extract(model, tokenizer, reader, prompts, responses, batch_size, max_response_tokens, layer):
    """One forward per example: the frozen prompt channel's p_harmful, and the response tokens'
    unit-normed activations (fp16, CPU, at most max_response_tokens) for training the head."""
    import torch

    from adversariallm.defenses.monitors.tokenization import build_detector_batch

    device = next(model.parameters()).device
    p_prompt, feats = [], []
    for start in range(0, len(prompts), batch_size):
        ids, tgt, attn = build_detector_batch(prompts[start:start + batch_size],
                                              responses[start:start + batch_size], tokenizer)
        ids, tgt, attn = ids.to(device), tgt.to(device), attn.to(device)
        with torch.no_grad():
            hidden = model(input_ids=ids, attention_mask=attn, output_hidden_states=True).hidden_states[layer]
            p_prompt += reader.prompt.p_harmful(hidden, tgt, attn).tolist()
            h = hidden.float()
            h = h / h.norm(dim=-1, keepdim=True).clamp_min(reader.eps)
            region = (tgt != 0) & attn.bool()
            for row in range(h.size(0)):
                feats.append(h[row][region[row]][:max_response_tokens].half().cpu())
    return p_prompt, feats


def response_scores(reader, feats, batch_size, device) -> list[float]:
    import torch

    out = []
    with torch.no_grad():
        for start in range(0, len(feats), batch_size):
            hidden, tgt, attn = pad_feats(feats[start:start + batch_size], device)
            margin, has = reader.response_margin(hidden, tgt, attn)
            out += torch.where(has, torch.sigmoid(margin), torch.zeros_like(margin)).tolist()
    return out


def pad_feats(feats, device):
    """Stored response activations -> (hidden, target_ids, attention_mask) with the whole stored
    span marked as response, so training runs the exact DualProbe.response_margin code path."""
    import torch

    width = max(max(f.size(0) for f in feats), 1)
    dim = feats[0].size(1)
    hidden = torch.zeros(len(feats), width, dim, device=device)
    mask = torch.zeros(len(feats), width, dtype=torch.long, device=device)
    for row, f in enumerate(feats):
        hidden[row, : f.size(0)] = f.to(device).float()
        mask[row, : f.size(0)] = 1
    return hidden, mask, mask


def train_head(reader, feats, labels, cfg, device) -> list[float]:
    """Only reader.response trains. Balanced BCE on the pooled margin."""
    import torch

    params = list(reader.response.parameters())
    opt = torch.optim.AdamW(params, lr=float(cfg.lr), weight_decay=float(cfg.weight_decay))
    y_all = torch.tensor(labels, dtype=torch.float)
    pos_weight = torch.tensor((y_all == 0).sum().item() / max((y_all == 1).sum().item(), 1), device=device)
    order = list(range(len(feats)))
    rng = random.Random(int(cfg.seed))
    losses = []
    for epoch in range(int(cfg.epochs)):
        rng.shuffle(order)
        total = 0.0
        for start in range(0, len(order), int(cfg.train_batch_size)):
            batch = order[start:start + int(cfg.train_batch_size)]
            hidden, tgt, attn = pad_feats([feats[i] for i in batch], device)
            margin, _ = reader.response_margin(hidden, tgt, attn)
            y = y_all[batch].to(device)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(margin, y, pos_weight=pos_weight)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += loss.item() * len(batch)
        losses.append(total / len(order))
        log.info(f"epoch {epoch}: loss {losses[-1]:.4f}")
    return losses


@hydra.main(version_base=None, config_path="conf", config_name="train_response_head")
def main(cfg: DictConfig) -> None:
    import torch

    from adversariallm.defenses.monitors.linear_probe import LinearProbeMonitor
    from adversariallm.io_utils import load_model_and_tokenizer
    from adversariallm.training.data import load_dataset_prompts
    from adversariallm.training.readers import DualProbe, load_reader
    from run_calibrate_probe import generate_responses

    random.seed(int(cfg.seed))
    torch.manual_seed(int(cfg.seed))
    repo = Path(cfg.root_dir)
    entry = OmegaConf.to_container(cfg.models[cfg.model], resolve=True)
    model, tokenizer = load_model_and_tokenizer(entry)
    model.eval()
    device = next(model.parameters()).device
    layer = int(cfg.layer)

    base = load_reader(entry["reader_path"])
    if getattr(base, "readout_mode", None) != "prompt_last":
        raise ValueError(f"{entry['reader_path']} is not a prompt_last probe; the prompt channel must be one")
    reader = DualProbe(base.linear.in_features, response_window=int(cfg.response_window))
    reader.prompt.load_state_dict(base.state_dict())
    reader.to(device)

    examples = collect_examples(repo, cfg.model, list(cfg.attacks), cfg.defense, cfg.classifier,
                                float(cfg.pos_threshold), float(cfg.neg_threshold))
    train, test = split_examples(examples, int(cfg.test_below), list(cfg.train_attacks),
                                 int(cfg.cap_per_group), int(cfg.seed))
    log.info(f"{len(examples)} labelled completions -> train {len(train)}, test {len(test)}")
    if not train or not test:
        raise RuntimeError("empty train or test split -- check model/attacks/defense")

    # benign: the model's own greedy answers to alpaca prompts (train window trains, calib
    # window sets the 1% threshold, test window reports FPR)
    benign = {}
    for window in ("train", "calib", "test"):
        prompts, _ = load_dataset_prompts(cfg.datasets, "alpaca", window=cfg.splits.alpaca[window], seed=0)
        prompts = prompts[: int(cfg.benign_n[window])]
        responses = generate_responses(model, tokenizer, prompts, int(cfg.max_new_tokens), int(cfg.gen_batch_size))
        benign[window] = (prompts, responses)
        log.info(f"alpaca {window}: {len(prompts)} generated")

    bs = int(cfg.batch_size)
    mrt = int(cfg.max_response_tokens)
    tr_pp, tr_feats = extract(model, tokenizer, reader, [e["prompt"] for e in train],
                              [e["response"] for e in train], bs, mrt, layer)
    b_pp, b_feats = extract(model, tokenizer, reader, *benign["train"], bs, mrt, layer)
    losses = train_head(reader, tr_feats + b_feats, [e["label"] for e in train] + [0] * len(b_feats), cfg, device)
    del tr_feats, b_feats

    reader.eval()
    scored = {}
    for window in ("calib", "test"):
        pp, feats = extract(model, tokenizer, reader, *benign[window], bs, mrt, layer)
        scored[window] = (pp, response_scores(reader, feats, bs, device))
    te_pp, te_feats = extract(model, tokenizer, reader, [e["prompt"] for e in test],
                              [e["response"] for e in test], bs, mrt, layer)
    te_pr = response_scores(reader, te_feats, bs, device)

    fpr = float(cfg.fpr)
    c_pp, c_pr = scored["calib"]
    tau_prompt = threshold_at(c_pp, fpr)
    tau_dual = threshold_at([max(a, b) for a, b in zip(c_pp, c_pr)], fpr)
    t_pp, t_pr = scored["test"]
    metrics = {
        "name": cfg.name, "model": cfg.model, "base_reader": entry["reader_path"],
        "train_attacks": list(cfg.train_attacks), "attacks": list(cfg.attacks),
        "n_train": len(train), "n_train_benign": len(benign["train"][0]),
        "n_train_hits": sum(e["label"] for e in train), "train_loss": [round(x, 4) for x in losses],
        "fpr_target": fpr, "tau_prompt": tau_prompt, "tau_dual": tau_dual,
        "benign_test_fpr_prompt": round(sum(p > tau_prompt for p in t_pp) / len(t_pp), 4),
        "benign_test_fpr_dual": round(sum(max(a, b) > tau_dual for a, b in zip(t_pp, t_pr)) / len(t_pp), 4),
        "per_attack": evaluate(test, te_pp, te_pr, tau_prompt, tau_dual),
    }

    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = out_dir / "final_reader.pt"
    reader_cfg = {"type": "dual", "response_window": int(cfg.response_window), "layer": layer}
    torch.save({"reader": reader.cpu().state_dict(),
                "cfg": {"reader": reader_cfg, "name": cfg.name, "base_reader": entry["reader_path"],
                        "train": OmegaConf.to_container(cfg, resolve=True)},
                "step": len(losses)}, ckpt)
    reader.to(device)

    # the saved checkpoint through the real eval path must reproduce the offline scores
    n = min(16, len(test))
    monitor = LinearProbeMonitor.from_config({"checkpoint_path": str(ckpt), "target_model_id": entry["id"],
                                              "index_hidden_layer_detector": layer, "batch_size": bs})
    live = monitor.score([e["prompt"] for e in test[:n]], [e["response"] for e in test[:n]],
                         target_model=model, target_tokenizer=tokenizer)
    offline = [max(a, b) for a, b in zip(te_pp[:n], te_pr[:n])]
    metrics["monitor_max_abs_diff"] = max(abs(a - b) for a, b in zip(live, offline))
    if metrics["monitor_max_abs_diff"] > 0.05:
        log.warning(f"monitor scores diverge from offline scores by {metrics['monitor_max_abs_diff']:.3f}")

    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    eval_dir = repo / "outputs" / "eval" / "response_head"
    eval_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(out_dir / "metrics.json", eval_dir / f"{cfg.name}.json")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
