"""Stratified sample of the EuroSafeGuard probe prompts for generate_answers.py (the "Llama test").

Run on a login node of the MPI cluster (pandas only, no GPU):

    .venv/bin/python scripts/esg_build_sample.py --out /fast/fgrob/coop-detect/esg_data/samples/llama_test_v1.parquet

Reads the filled probe and gate files (context/results/esg_data_counts/fill_withheld.py). Strata:

    twin_plain, twin_rewrite   every harmful plain prompt and its rewrites in the probe set
    h_<source>                 --per-stratum harmful prompts of each source without a plain twin in the probe set
    b_<source>                 --per-stratum benign prompts of each wrapped or plain benign source

Civil Comments and ToxicChat are left out: toxic text, not requests. Every row keeps the dataset's `row_id`, the key
of the answer store, so a later sample can add rows to the same store. `judge_text` is what the judge reads as the
request: the plain twin where one exists (probe set first, then gate set; for WildJailbreak's adversarial prompts the
`vanilla` column of its own train file), else the prompt itself (`judge_against`).
`entry_type` comes from the gate set by `input_id` (empty where the gate set has no such group).
"""

import argparse
import re

import pandas as pd

DATA = "/fast/fgrob/coop-detect/esg_data"
HARMFUL_SOURCES = ["redact_jb", "jf", "dan_h", "obf_h", "bon_h", "wjb_adversarial_harmful", "wjb_vanilla_harmful",
                   "wildguardmix_train", "aegis2_train"]
BENIGN_SOURCES = ["plain", "redact_jb", "jf", "dan_b", "obf_b", "wjb_adversarial_benign"]


def source_group(source: str) -> str:
    """obf_<transform>_h -> obf_h, obf_<transform>_b -> obf_b; every other source as it is."""
    m = re.fullmatch(r"obf_.*_([bh])", source)
    return f"obf_{m.group(1)}" if m else source


def build(probe: pd.DataFrame, gate: pd.DataFrame, per_stratum: int, seed: int,
          wildjailbreak: pd.DataFrame | None = None) -> pd.DataFrame:
    """probe: row_id, input_id, source, label, category, text. gate: input_id, source, entry_type, text.
    wildjailbreak: its train file with `vanilla` (the plain request) and `adversarial` (the rewrite)."""
    probe = probe[probe.text.fillna("").str.strip() != ""].copy()
    probe["label"] = probe.label.astype(int)
    probe["group"] = probe.source.map(source_group)
    gate = gate[gate.text.fillna("").str.strip() != ""]

    twin_ids = set(probe.loc[(probe.source == "plain") & (probe.label == 1), "input_id"])
    in_twin = probe.input_id.isin(twin_ids) & (probe.label == 1)
    parts = [probe[in_twin & (probe.source == "plain")].assign(stratum="twin_plain"),
             probe[in_twin & (probe.source != "plain")].assign(stratum="twin_rewrite")]
    rest = probe[~in_twin]
    for label, prefix, sources in ((1, "h", HARMFUL_SOURCES), (0, "b", BENIGN_SOURCES)):
        for src in sources:
            pool = rest[(rest.label == label) & (rest.group == src)].sort_values("row_id")
            take = pool.sample(n=min(per_stratum, len(pool)), random_state=seed)
            parts.append(take.assign(stratum=f"{prefix}_{src}"))
    out = pd.concat(parts, ignore_index=True)
    assert out.row_id.is_unique

    plain = pd.concat([probe.loc[probe.source == "plain", ["input_id", "text"]],
                       gate.loc[gate.source == "plain", ["input_id", "text"]]]).drop_duplicates("input_id")
    twin_text = out.input_id.map(plain.set_index("input_id").text)
    if wildjailbreak is not None:
        wjb = wildjailbreak[wildjailbreak.vanilla.str.strip() != ""].assign(key=lambda d: d.adversarial.str.strip())
        vanilla = wjb.drop_duplicates("key").set_index("key").vanilla.str.strip()
        is_wjb = out.source.str.startswith("wjb_adversarial")
        twin_text = twin_text.where(~is_wjb, out.text.str.strip().map(vanilla))
    has_twin = twin_text.notna() & (out.source != "plain")
    out["judge_text"] = out.text.where(~has_twin, twin_text)
    out["judge_against"] = has_twin.map({True: "plain_twin", False: "self"})
    out["entry_type"] = out.input_id.map(gate.drop_duplicates("input_id").set_index("input_id").entry_type).fillna("")
    cols = ["row_id", "input_id", "source", "label", "category", "entry_type", "stratum", "text", "judge_text",
            "judge_against"]
    return out[cols].sort_values("row_id").reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--probe", default=f"{DATA}/probe_train_full.parquet")
    ap.add_argument("--gate", default=f"{DATA}/gate_train_full.parquet")
    ap.add_argument("--wildjailbreak", default=None, help="allenai/wildjailbreak train/train.tsv (optional)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-stratum", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    probe = pd.read_parquet(a.probe, columns=["row_id", "input_id", "source", "label", "category", "text"])
    gate = pd.read_parquet(a.gate, columns=["input_id", "source", "entry_type", "text"])
    wjb = None
    if a.wildjailbreak:
        wjb = pd.read_csv(a.wildjailbreak, sep="\t", dtype=str, keep_default_na=False, on_bad_lines="skip",
                          usecols=["vanilla", "adversarial"])
    out = build(probe, gate, a.per_stratum, a.seed, wjb)
    out.to_parquet(a.out, index=False)
    table = out.groupby("stratum").agg(rows=("row_id", "size"), judged_against_twin=("judge_against", lambda s: int((s == "plain_twin").sum())),
                                       strictly_harmful=("entry_type", lambda s: int((s == "harmful").sum())),
                                       median_chars=("text", lambda s: int(s.str.len().median())))
    print(table.to_string())
    print("total", len(out), "->", a.out)


if __name__ == "__main__":
    main()
