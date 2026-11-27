"""The training file for coop on EuroSafeGuard prompts: one row per triple of prompt, harmful answer and refusal.

Run on a login node of the MPI cluster, from the AdversariaLLM folder (pandas only, no GPU):

    PYTHONPATH=. .venv/bin/python scripts/esg_build_triples.py \\
        --samples /fast/fgrob/coop-detect/esg_data/samples/llama_test_v1_twins.parquet \\
                  /fast/fgrob/coop-detect/esg_data/samples/llama_rest_v1.parquet \\
        --out /fast/fgrob/coop-detect/esg_data/coop_triples_v1.parquet

A prompt becomes a triple when all of this holds:

    it is a strictly harmful request (label 1, entry_type "harmful")
    base Llama answered it and the judge scored the answer above --harmful-above   (the harmful answer)
    the refusal generated with the system prompt has a refusal phrase and scores below --refusal-below

`plain_request` is the plain request behind a rewritten prompt where one is known, else empty. The refusal was
generated for the plain request where one is known, else for the prompt itself. `split` is "val" for the input_id
groups that pretrain_prompt_probe.py holds out (same val_frac and split_seed), so the probe has not seen them.
"""

import argparse
import random

import pandas as pd

from adversariallm.io_utils import answer_store

DATA = "/fast/fgrob/coop-detect/esg_data"
COLUMNS = ["row_id", "input_id", "source", "category", "split", "prompt", "plain_request", "harmful_answer",
           "harmful_score", "harmful_answer_cut", "refusal", "refusal_score"]


def held_out_groups(input_ids, val_frac: float, seed: int) -> set:
    """The input_id groups pretrain_prompt_probe.split_groups holds out, for the same ids, val_frac and seed."""
    groups = sorted(set(input_ids))
    random.Random(seed).shuffle(groups)
    return set(groups[: round(val_frac * len(groups))])


def build(sample: pd.DataFrame, answers: pd.DataFrame, scores: pd.DataFrame, refusals: pd.DataFrame,
          refusal_scores: pd.DataFrame, held_out: set, harmful_above: float = 0.5,
          refusal_below: float = 0.1) -> pd.DataFrame:
    """sample: row_id, input_id, source, category, label, entry_type, text, judge_text, judge_against.
    answers, refusals: row_id, status, answer, refusal (the phrase flag), hit_max_new_tokens.
    scores, refusal_scores: row_id, p_harmful."""
    d = sample.drop_duplicates("row_id").set_index("row_id")
    d = d[(d.label.astype(int) == 1) & (d.entry_type == "harmful")]
    a = answers[answers.status == "ok"].drop_duplicates("row_id").set_index("row_id")
    r = refusals[refusals.status == "ok"].drop_duplicates("row_id").set_index("row_id")
    d = d.join(a[["answer", "hit_max_new_tokens"]].rename(columns={"answer": "harmful_answer",
                                                                    "hit_max_new_tokens": "harmful_answer_cut"}))
    d = d.join(scores.drop_duplicates("row_id").set_index("row_id").p_harmful.rename("harmful_score"))
    d = d.join(r[["answer", "refusal"]].rename(columns={"answer": "refusal", "refusal": "refusal_phrase"}))
    d = d.join(refusal_scores.drop_duplicates("row_id").set_index("row_id").p_harmful.rename("refusal_score"))
    keep = ((d.harmful_score > harmful_above) & d.refusal.notna() & d.refusal_phrase.fillna(False).astype(bool)
            & (d.refusal_score < refusal_below))
    d = d[keep].rename(columns={"text": "prompt"}).reset_index()
    d["plain_request"] = d.judge_text.where(d.judge_against == "plain_twin")
    d["split"] = d.input_id.isin(held_out).map({True: "val", False: "train"})
    d["harmful_answer_cut"] = d.harmful_answer_cut.astype(bool)
    return d[COLUMNS].sort_values("row_id").reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--samples", nargs="+", required=True)
    ap.add_argument("--answers", default="outputs/answers/esg-llama31-base")
    ap.add_argument("--refusals", default="outputs/answers/esg-llama31-base-refusals")
    ap.add_argument("--probe", default=f"{DATA}/probe_train_full.parquet", help="the probe's training file, for the split")
    ap.add_argument("--out", required=True)
    ap.add_argument("--judge", default="strong_reject")
    ap.add_argument("--harmful-above", type=float, default=0.5)
    ap.add_argument("--refusal-below", type=float, default=0.1)
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--split-seed", type=int, default=0)
    a = ap.parse_args()
    sample = pd.concat([pd.read_parquet(p) for p in a.samples], ignore_index=True)
    probe = pd.read_parquet(a.probe, columns=["input_id", "text"])
    probe = probe[probe.text.fillna("").str.strip() != ""]  # as pretrain_prompt_probe.py reads it
    held = held_out_groups(probe.input_id.tolist(), a.val_frac, a.split_seed)
    read = lambda store, name: pd.DataFrame(answer_store.read_jsonl(f"{store}/{name}"))
    out = build(sample, read(a.answers, "answers.*.jsonl"), read(a.answers, f"judge.{a.judge}.jsonl"),
                read(a.refusals, "answers.*.jsonl"), read(a.refusals, f"judge.{a.judge}.jsonl"), held,
                a.harmful_above, a.refusal_below)
    out.to_parquet(a.out, index=False)
    print(pd.crosstab(out.source, out.split, margins=True).to_string())
    print("requests (input_id):", out.input_id.nunique(), "| with a plain request:", int(out.plain_request.notna().sum()),
          "| harmful answer cut at the token limit:", int(out.harmful_answer_cut.sum()))
    print(len(out), "triples ->", a.out)


if __name__ == "__main__":
    main()
