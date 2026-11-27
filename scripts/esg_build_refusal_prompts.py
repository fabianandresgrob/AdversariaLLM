"""Prompts that need a generated refusal: strictly harmful prompts whose stored answer the judge scored as harmful.

Run on a login node of the MPI cluster, from the AdversariaLLM folder (pandas only, no GPU):

    .venv/bin/python scripts/esg_build_refusal_prompts.py --store outputs/answers/esg-llama31-base \\
        --samples /fast/fgrob/coop-detect/esg_data/samples/llama_test_v1_twins.parquet \\
                  /fast/fgrob/coop-detect/esg_data/samples/llama_rest_v1.parquet \\
        --out /fast/fgrob/coop-detect/esg_data/samples/refusal_prompts_v1.parquet

The output keeps the sample's columns. generate_answers.py answers its `judge_text` under a system prompt
(experiments/esg-refusals.yaml), so a rewrite gets the refusal to its plain request where one is known.
"""

import argparse

import pandas as pd

from adversariallm.io_utils import answer_store


def build(sample: pd.DataFrame, answers: pd.DataFrame, scores: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    """sample: row_id, label, entry_type, ... answers: row_id, status. scores: row_id, p_harmful."""
    ok = set(answers.loc[answers.status == "ok", "row_id"])
    harmful = set(scores.loc[scores.p_harmful > threshold, "row_id"])
    sample = sample.drop_duplicates("row_id")
    keep = (sample.label.astype(int) == 1) & (sample.entry_type == "harmful") & sample.row_id.isin(ok & harmful)
    return sample[keep].sort_values("row_id").reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--store", required=True)
    ap.add_argument("--samples", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--judge", default="strong_reject")
    ap.add_argument("--threshold", type=float, default=0.5)
    a = ap.parse_args()
    sample = pd.concat([pd.read_parquet(p) for p in a.samples], ignore_index=True)
    answers = pd.DataFrame(answer_store.read_jsonl(f"{a.store}/answers.*.jsonl"))
    scores = pd.DataFrame(answer_store.read_jsonl(f"{a.store}/judge.{a.judge}.jsonl")).drop_duplicates("row_id")
    out = build(sample, answers, scores, a.threshold)
    out.to_parquet(a.out, index=False)
    print(out.groupby("stratum").size().to_string())
    print("plain request known for", int((out.judge_against == "plain_twin").sum()), "of", len(out), "->", a.out)


if __name__ == "__main__":
    main()
