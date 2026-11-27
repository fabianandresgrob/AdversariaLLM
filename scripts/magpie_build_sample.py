"""Prompt file of the Magpie rows that coop training uses for the KL, for generate_answers.py.

Run on a login node of the MPI cluster, from the AdversariaLLM folder (reads the dataset from the Hugging Face cache):

    PYTHONPATH=. HF_HUB_OFFLINE=1 .venv/bin/python scripts/magpie_build_sample.py \\
        --out /fast/fgrob/coop-detect/esg_data/samples/magpie_train_v1.parquet

The rows are the `train` window of magpie in conf/splits.yaml, read the same way as build_kl_stream does (seed 0), so
row i here is row i of the KL stream. Base Llama then answers them (experiments/llama-magpie-answers.yaml), and the KL
is taken on its own answers in place of Magpie's.
"""

import argparse

import pandas as pd


def rows(prompts: list[str], split: str) -> pd.DataFrame:
    """One benign row per prompt, with the columns generate_answers.py and the ESG samples use."""
    ids = [f"magpie_{split}_{i:05d}" for i in range(len(prompts))]
    return pd.DataFrame({"row_id": ids, "input_id": ids, "source": "magpie", "label": 0, "category": "",
                         "entry_type": "", "stratum": "b_magpie", "text": prompts, "judge_text": prompts,
                         "judge_against": "self"})


def main():
    from omegaconf import OmegaConf

    from adversariallm.training.data import load_dataset_prompts

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--seed", type=int, default=0, help="data.val_seed of the coop runs")
    a = ap.parse_args()
    datasets = OmegaConf.load("conf/datasets/datasets.yaml")
    window = OmegaConf.load("conf/splits.yaml").splits.magpie[a.split]
    prompts, _ = load_dataset_prompts(datasets, "magpie", window=window, seed=a.seed)
    out = rows(prompts, a.split)
    out.to_parquet(a.out, index=False)
    print(len(out), "prompts, median", int(out.text.str.len().median()), "chars ->", a.out)


if __name__ == "__main__":
    main()
