"""Offline probe study: which training data and which layer give a probe that reads harm from the answer, rather than
"complies with a harmful-looking request"? On one frozen model + adapter, fit linear probes offline and test them:

    pixi run --frozen python offline_probe_study.py model=W-coop-s0-ema

Features: hidden states at each of `layers`, mean over the first `max_answer_tokens` completion tokens, unit-normed
(Libon et al.'s readout: completion tokens, mean). One logistic probe per (training set, layer):
  ours         heretic answers behind the behavior request (harmful, the W training behaviors only) vs the behavior's
               refusal and Alpaca train-window reference answers (harmless): our coop probe's data, without attacks
  wildguard    lenalibon/jailbreak-judge-completions wildguardtest_harmful + _unharmful: WildGuardTest prompts (vanilla
               and adversarial) with heretic Llama-3-8B completions, labelled by the completion's harmfulness (WildGuard)
  beavertails  PKU-Alignment/BeaverTails 30k_train, `n_beavertails` unsafe + as many safe (prompt, response) pairs
  ours+wildguard  the union of ours and wildguard (what coop's probe would train on: our heretic answers for the
                  representation term plus completion-labelled data)
  wildguard_orig  the original WildGuardTest (allenai/wildguardmix, wildguardtest): the same prompts with their original
                  responses (several models) and human response-harmfulness labels; a cross-check of Libon's version
  ours+wildguard_orig  the union of ours and wildguard_orig
`train_sets` picks which are fit; `tag` is appended to the output name so a run of other sets keeps an earlier one.
Evaluation, each probe at its own 1% threshold on the model's own Alpaca calibration answers (from
reread_monitor_eval.py's <model>_scores.jsonl, own prompts):
  - the PAIR-analysis answer set (condition_set.jsonl) behind PAIR's prompt / the plain behavior / a neutral prompt:
    flag rates for harmful and harmless PAIR answers and heretic answers, and AUROC harmful vs harmless PAIR answers
  - xs_test: the model's saved answers (outputs/eval/overrefusal/*/<model>/), flag rate and pipeline over-refusal
  - wildjailbreak_eval_harmful (adversarial prompts, completions labelled harmful / not): AUROC
Writes <out>/<model>.json.
"""

import glob
import json
import logging
import random
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

log = logging.getLogger(__name__)
JB = "lenalibon/jailbreak-judge-completions"
JB_COMPLETION = "completion_llama3_8b_heretic_mlabonne"
JB_LABEL = "response_harmfulness_llama3_8b_heretic_mlabonne"


def auroc(pos, neg):
    """P(score of a random positive > score of a random negative), ties count half."""
    if not pos or not neg:
        return None
    s = sorted([(x, 1) for x in pos] + [(x, 0) for x in neg])
    rank_sum, i = 0.0, 0
    while i < len(s):
        j = i
        while j < len(s) and s[j][0] == s[i][0]:
            j += 1
        avg = (i + j + 1) / 2  # 1-based average rank of the tie block
        rank_sum += avg * sum(lbl for _, lbl in s[i:j])
        i = j
    n_pos, n_neg = len(pos), len(neg)
    return (rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def fit_probe(x, y, steps=2000, lr=1e-2, weight_decay=1e-4):
    """Logistic probe on unit-normed features, classes weighted to balance. Returns (w, b)."""
    x, y = x.float(), y.float()
    w = torch.zeros(x.size(1), device=x.device, requires_grad=True)
    b = torch.zeros((), device=x.device, requires_grad=True)
    pos_weight = (y == 0).sum() / (y == 1).sum().clamp_min(1)
    opt = torch.optim.Adam([w, b], lr=lr, weight_decay=weight_decay)
    for _ in range(steps):
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(x @ w + b, y, pos_weight=pos_weight)
        loss.backward()
        opt.step()
    return w.detach(), b.detach()


@torch.no_grad()
def features(model, tok, pairs, layers, max_answer_tokens, batch_size):
    """{layer: (N, D) unit-normed mean of the completion-token hidden states} for (prompt, completion) pairs."""
    from adversariallm.training.data import build_example_full, pad_collate

    device = next(model.parameters()).device
    out = {layer: [] for layer in layers}
    for i in range(0, len(pairs), batch_size):
        items = []
        for prompt, completion in pairs[i: i + batch_size]:
            ids, _, tgt, attn = build_example_full(prompt.strip() or ".", completion or ".", tok)
            start = int((tgt != 0).nonzero()[0]) if (tgt != 0).any() else ids.numel() - 1
            end = min(ids.numel(), start + max_answer_tokens)
            items.append({"ids": ids[:end], "tgt": tgt[:end], "attn": attn[:end]})
        batch = {k: v.to(device) for k, v in pad_collate(items, ["ids", "tgt", "attn"], pad_id=0).items()}
        hs = model(input_ids=batch["ids"], attention_mask=batch["attn"], output_hidden_states=True).hidden_states
        mask = (batch["tgt"] != 0).unsqueeze(-1).float()
        for layer in layers:
            h = hs[layer].float()
            m = (h * mask).sum(1) / mask.sum(1).clamp_min(1)
            out[layer].append(torch.nn.functional.normalize(m, dim=-1).cpu())
    return {layer: torch.cat(v) for layer, v in out.items()}


@hydra.main(version_base=None, config_path="conf", config_name="offline_probe_study")
def main(cfg: DictConfig) -> None:
    import datasets

    from adversariallm.io_utils import load_model_and_tokenizer
    from adversariallm.training.coop_metrics import threshold_at_fpr
    from adversariallm.training.data import (AdvTupleStream, load_answer_pool, load_dataset_prompts,
                                             load_labelled_completions, split_adv_stream)

    rng = random.Random(int(cfg.seed))
    root = Path(cfg.root_dir)
    entry = OmegaConf.to_container(cfg.models[cfg.model], resolve=True)
    model, tok = load_model_and_tokenizer(entry)
    model.eval()
    layers = [int(x) for x in cfg.layers]
    feat = lambda pairs: features(model, tok, pairs, layers, int(cfg.max_answer_tokens), int(cfg.batch_size))

    # ---- training sets: (prompt, completion, label 1 = harmful) ----
    train = {}
    ds = AdvTupleStream(str(root / "data"), "behavior_datasets/extra_behavior_datasets/adv_training_behaviors.csv",
                        "optimizer_targets/extra_targets/adv_training_targets.json",
                        "safe_responses/adv_training_behaviors_safe_responses_longer.csv", tokenizer=None, model_name=None)
    tr, _ = split_adv_stream(ds, val_size=24, seed=0, val_targets=6)
    train_behaviors = {ds.rows[i][0] for i in tr.indices}
    pool = load_answer_pool(str(root / "outputs/eval/answer_pool/pool-heretic.jsonl"))
    harmful = [(b, a, 1) for (b, _), a in pool.items() if b in train_behaviors]
    refusals = list({(b, ds.rows[i][2], 0) for i in tr.indices for b in [ds.rows[i][0]]})
    ap, ar = load_dataset_prompts(cfg.datasets, "alpaca", window=cfg.splits.alpaca.train, seed=0)
    train["ours"] = harmful + refusals + [(p, r, 0) for p, r in zip(ap, ar)]
    wg = datasets.concatenate_datasets([datasets.load_dataset(JB, split=s) for s in ("wildguardtest_harmful", "wildguardtest_unharmful")])
    train["wildguard"] = [(r["prompt"], r[JB_COMPLETION], int(r[JB_LABEL] == "harmful")) for r in wg if r[JB_LABEL] in ("harmful", "unharmful")]
    bt = datasets.load_dataset("PKU-Alignment/BeaverTails", split="30k_train")
    unsafe = [(r["prompt"], r["response"], 1) for r in bt if not r["is_safe"]]
    safe = [(r["prompt"], r["response"], 0) for r in bt if r["is_safe"]]
    rng.shuffle(unsafe); rng.shuffle(safe)
    n = int(cfg.n_beavertails)
    train["beavertails"] = unsafe[:n] + safe[:n]
    train["ours+wildguard"] = train["ours"] + train["wildguard"]
    if {"wildguard_orig", "ours+wildguard_orig"} & set(cfg.train_sets):  # gated dataset: loaded only when asked for
        h, b = load_labelled_completions(OmegaConf.to_container(cfg.wildguard_orig, resolve=True))
        train["wildguard_orig"] = [(p, c, 1) for p, c in h] + [(p, c, 0) for p, c in b]
        train["ours+wildguard_orig"] = train["ours"] + train["wildguard_orig"]
    train = {name: rows for name, rows in train.items() if name in set(cfg.train_sets)}

    # ---- evaluation sets ----
    cond = [json.loads(line) for line in open(cfg.condition_set)]
    neutral = str(cfg.neutral_prompt)
    ev = {"cond_pair": [(r["pair_prompt"], r["answer"]) for r in cond if r.get("pair_prompt")],
          "cond_behavior": [(r["behavior"], r["answer"]) for r in cond],
          "cond_neutral": [(neutral, r["answer"]) for r in cond]}
    cond_src = {"cond_pair": [r["source"] for r in cond if r.get("pair_prompt")],
                "cond_behavior": [r["source"] for r in cond], "cond_neutral": [r["source"] for r in cond]}
    scores_file = root / "outputs/eval/reread_monitor" / f"{cfg.model}_scores.jsonl"
    calib = [json.loads(line) for line in open(scores_file)]
    ev["alpaca_calib"] = [(r["prompt"], r["answer"]) for r in calib if r["set"] == "calib"]
    xs_dir = Path(glob.glob(str(root / "outputs/eval/overrefusal/*" / cfg.model / "judged_model_xs_test.json"))[0]).parent
    judged = json.loads((xs_dir / "judged_model_xs_test.json").read_text())
    xs_refused = [j["judgment"] == "2_full_refusal" or bool(j["degeneration_reasons"]) for j in judged]
    ev["xs_test"] = [(j["prompt"], j["generation"]) for j in judged]
    wj = [r for r in datasets.load_dataset(JB, split="wildjailbreak_eval_harmful") if r[JB_LABEL] in ("harmful", "unharmful")]
    rng.shuffle(wj)
    wj = wj[: int(cfg.n_wildjailbreak)]
    ev["wildjailbreak"] = [(r["prompt"], r[JB_COMPLETION]) for r in wj]
    wj_label = [r[JB_LABEL] == "harmful" for r in wj]

    log.info({k: len(v) for k, v in {**train, **ev}.items()})
    tr_feat = {name: feat([(p, c) for p, c, _ in rows]) for name, rows in train.items()}
    ev_feat = {name: feat(pairs) for name, pairs in ev.items()}

    results = []
    for name, rows in train.items():
        y = torch.tensor([lbl for _, _, lbl in rows])
        for layer in layers:
            w, b = fit_probe(tr_feat[name][layer].cuda(), y.cuda())
            score = lambda key: torch.sigmoid(ev_feat[key][layer].cuda().float() @ w + b).cpu().tolist()
            thr = threshold_at_fpr(score("alpaca_calib"), fpr=0.01)
            res = {"train": name, "layer": layer, "n_train": len(rows), "threshold": thr}
            for key in ("cond_pair", "cond_behavior", "cond_neutral"):
                s = score(key)
                for src in ("pair_harmful", "pair_harmless", "heretic_val"):
                    sel = [x for x, t in zip(s, cond_src[key]) if t == src]
                    if sel:
                        res[f"{key}|{src}"] = sum(x > thr for x in sel) / len(sel)
                res[f"{key}|auroc_pair"] = auroc([x for x, t in zip(s, cond_src[key]) if t == "pair_harmful"],
                                                 [x for x, t in zip(s, cond_src[key]) if t == "pair_harmless"])
            xs = score("xs_test")
            res["xs_flagged"] = sum(x > thr for x in xs) / len(xs)
            res["xs_pipeline"] = sum(r or x > thr for r, x in zip(xs_refused, xs)) / len(xs)
            wjs = score("wildjailbreak")
            res["wildjailbreak_auroc"] = auroc([x for x, h in zip(wjs, wj_label) if h], [x for x, h in zip(wjs, wj_label) if not h])
            res["wildjailbreak_flag_harmful"] = sum(x > thr for x, h in zip(wjs, wj_label) if h) / max(1, sum(wj_label))
            results.append(res)
            log.info(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in res.items()}))
    out = Path(cfg.out); out.mkdir(parents=True, exist_ok=True)
    (out / f"{cfg.model}{cfg.get('tag') or ''}.json").write_text(json.dumps({"model": cfg.model, "layers": layers, "results": results}, indent=1))


if __name__ == "__main__":
    main()
