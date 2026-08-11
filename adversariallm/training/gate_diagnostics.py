"""Offline comparison of benign-gate variants (gate redesign, design doc §3/§8).

Scores the FROZEN base model and dumps every variant's per-example logprobs + gate weights so
we can pick the least-confounded benign over-refusal gate. Three sets:

  harmful : w_harm = sigmoid((lp̄(y_harm) − lp̄(y_safe))/τ)   (a designed-to-separate reference)
  alpaca  : all comply → every benign gate should sit LOW (false-positive test)
  or_bench: re-split by the fixed is_compliant(raw) into answered / refused

Benign variants share the refusal opener as the negative side:
  open  : vs the compliance opener            (current; needs no target)
  full  : vs the full answer                  (old; alpaca response / or_bench raw)
  trunc : vs the answer truncated to k tokens (length-matched to the refusal opener)
"""

from __future__ import annotations

import json
import logging
import os
import statistics as st

from .gating import avg_logprob

log = logging.getLogger(__name__)


def truncate_completion(tokenizer, text: str, k: int) -> str:
    """Decode the first k tokens of text — the length-matched positive side."""
    ids = tokenizer(text)["input_ids"][:k]
    return tokenizer.decode(ids, skip_special_tokens=True)


def variant_stats(ws) -> dict:
    """Summary of a gate-weight list: n, mean, median, std, frac>0.5."""
    ws = [w for w in ws if w == w]  # drop nan
    if not ws:
        return {"n": 0}
    return {
        "n": len(ws),
        "mean": st.mean(ws),
        "median": st.median(ws),
        "std": st.pstdev(ws),
        "frac_gt0.5": sum(w > 0.5 for w in ws) / len(ws),
    }


def _score(model, tokenizer, model_name, pairs, device, batch_size=8):
    """Length-normalized log-prob of each completion given its prompt. nan for empty completion."""
    import torch
    from torch.nn.utils.rnn import pad_sequence

    from .data import build_supervised_example

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    out = []
    for i in range(0, len(pairs), batch_size):
        chunk = pairs[i : i + batch_size]
        valid = [(p, c) for p, c in chunk if c and c.strip()]
        exs = [build_supervised_example(p, c, tokenizer, model_name) for p, c in valid]
        if exs:
            ids = pad_sequence([e[0] for e in exs], batch_first=True, padding_value=pad_id).to(device)
            lab = pad_sequence([e[1] for e in exs], batch_first=True, padding_value=-100).to(device)
            attn = pad_sequence([torch.ones(len(e[0]), dtype=torch.long) for e in exs], batch_first=True).to(device)
            with torch.no_grad():
                logits = model(input_ids=ids, attention_mask=attn).logits
            scores = avg_logprob(logits[:, :-1], lab[:, 1:]).tolist()
        it = iter(scores) if exs else iter(())
        out += [next(it) if (c and c.strip()) else float("nan") for _, c in chunk]
    return out


def _gate(lp_pos, lp_neg, tau):
    """w = sigmoid((lp_neg − lp_pos)/τ): high when the model prefers the negative (refuse) side."""
    import math

    return [1 / (1 + math.exp(-(n - p) / tau)) if (p == p and n == n) else float("nan")
            for p, n in zip(lp_pos, lp_neg)]


def _stats_by_tau(lp_pos, lp_neg, taus):
    """Sweep τ on the same margins (free — no extra forward passes)."""
    return {str(t): variant_stats(_gate(lp_pos, lp_neg, t)) for t in taus}


def run_gate_diagnostics(cfg):
    from ..io_utils import load_model_and_tokenizer
    from .data import AdvTupleStream, load_dataset_prompts
    from .generate_benign_targets import is_compliant

    model, tokenizer = load_model_and_tokenizer(cfg.models[cfg.model])
    device = next(model.parameters()).device
    model.eval()
    mn = cfg.chat_template_id
    taus = [float(t) for t in cfg.taus]
    primary = taus[len(taus) // 2]  # τ used for the per-example w in the dumped rows
    refuse, comply = cfg.refusal_opener, cfg.compliance_opener
    k = len(tokenizer(refuse)["input_ids"])
    report = {"taus": taus, "primary_tau": primary, "k": k,
              "refusal_opener": refuse, "compliance_opener": comply}

    def benign_rows(prompts, answers):
        lp_refuse = _score(model, tokenizer, mn, [(p, refuse) for p in prompts], device)
        lp_comply = _score(model, tokenizer, mn, [(p, comply) for p in prompts], device)
        lp_full = _score(model, tokenizer, mn, list(zip(prompts, answers)), device)
        lp_trunc = _score(model, tokenizer, mn,
                          [(p, truncate_completion(tokenizer, a, k) if a else "") for p, a in zip(prompts, answers)], device)
        w_open = _gate(lp_comply, lp_refuse, primary)
        w_full = _gate(lp_full, lp_refuse, primary)
        w_trunc = _gate(lp_trunc, lp_refuse, primary)
        rows = [{"prompt": p, "lp_refuse": r, "lp_comply": c, "lp_full": f, "lp_trunc": t,
                 "w_open": wo, "w_full": wf, "w_trunc": wt}
                for p, r, c, f, t, wo, wf, wt in
                zip(prompts, lp_refuse, lp_comply, lp_full, lp_trunc, w_open, w_full, w_trunc)]
        stats_by_tau = {"open": _stats_by_tau(lp_comply, lp_refuse, taus),
                        "full": _stats_by_tau(lp_full, lp_refuse, taus),
                        "trunc": _stats_by_tau(lp_trunc, lp_refuse, taus)}
        return {"stats_by_tau": stats_by_tau, "rows": rows}

    # harmful reference: w_harm should separate (model prefers y_harm over the safe refusal)
    adv = AdvTupleStream(cfg.data.dir, cfg.data.behaviors, cfg.data.targets, cfg.data.safe, tokenizer, mn).rows
    adv = adv[: int(cfg.n_harmful)]
    xh = [x for x, _, _ in adv]
    lp_harm = _score(model, tokenizer, mn, [(x, yh) for x, yh, _ in adv], device)
    lp_safe = _score(model, tokenizer, mn, [(x, ys) for x, _, ys in adv], device)
    w_harm = _gate(lp_safe, lp_harm, primary)  # sigmoid((lp_harm − lp_safe)/τ)
    report["harmful"] = {
        "stats_by_tau": {"harm": _stats_by_tau(lp_safe, lp_harm, taus)},
        "rows": [{"prompt": x, "lp_harm": h, "lp_safe": s, "w_harm": w}
                 for (x, _, _), h, s, w in zip(adv, lp_harm, lp_safe, w_harm)],
    }

    # alpaca: all comply → every gate should be LOW
    a_prompts, a_answers = load_dataset_prompts(cfg.datasets, "alpaca", window=cfg.splits.alpaca.train, seed=cfg.seed)
    n = int(cfg.n_alpaca)
    report["alpaca"] = benign_rows(a_prompts[:n], [a or "" for a in a_answers[:n]])

    # or_bench: re-split by the FIXED is_compliant(raw) so a stale ygen file can't mislabel
    with open(cfg.benign_targets_path) as fh:
        ob = json.load(fh)
    answered = [(r["prompt"], r.get("raw", "")) for r in ob if is_compliant(r.get("raw", ""))]
    refused = [r["prompt"] for r in ob if not is_compliant(r.get("raw", ""))]
    report["orbench_answered"] = benign_rows([p for p, _ in answered], [a for _, a in answered])
    # refused rows have no answer → only the target-free opener gate applies (want HIGH here)
    lp_r = _score(model, tokenizer, mn, [(p, refuse) for p in refused], device)
    lp_c = _score(model, tokenizer, mn, [(p, comply) for p in refused], device)
    w_open_ref = _gate(lp_c, lp_r, primary)
    report["orbench_refused"] = {
        "stats_by_tau": {"open": _stats_by_tau(lp_c, lp_r, taus)},
        "rows": [{"prompt": p, "lp_refuse": r, "lp_comply": c, "w_open": w}
                 for p, r, c, w in zip(refused, lp_r, lp_c, w_open_ref)],
    }

    os.makedirs(os.path.dirname(cfg.out_path), exist_ok=True)
    with open(cfg.out_path, "w") as fh:
        json.dump(report, fh, indent=2)
    _log_summary(report)
    log.info(f"wrote gate diagnostics to {cfg.out_path}")


def _log_summary(report):
    taus = report["taus"]

    def line(name, by_tau):
        cells = " | ".join(f"τ={t}: mean={by_tau[str(t)].get('mean', float('nan')):.3f} "
                           f"f>.5={by_tau[str(t)].get('frac_gt0.5', float('nan')):.2f}" for t in taus)
        return f"  {name:8} n={by_tau[str(taus[0])].get('n', 0):4}  {cells}"

    log.info(f"=== gate diagnostics (τ sweep {taus}, k={report['k']}) ===")
    log.info("HARMFUL ref (want HIGH — model prefers y_harm over refusal):")
    log.info(line("w_harm", report["harmful"]["stats_by_tau"]["harm"]))
    log.info("ALPACA (all comply — want LOW):")
    for v in ("open", "full", "trunc"):
        log.info(line(f"w_{v}", report["alpaca"]["stats_by_tau"][v]))
    log.info("OR_BENCH answered (want LOW):")
    for v in ("open", "full", "trunc"):
        log.info(line(f"w_{v}", report["orbench_answered"]["stats_by_tau"][v]))
    log.info("OR_BENCH refused (want HIGH — opener gate only):")
    log.info(line("w_open", report["orbench_refused"]["stats_by_tau"]["open"]))
