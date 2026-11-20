"""Does our copy of EuroSafeGuard's probe reproduce their released scores? Compares the sorted probe scores of
score_prompt_probe.py on XSTest (safe, unsafe) and OR-Bench-hard with the scores in their bundle
(EuroSafeGuard/paper/scores/overflag.csv.gz, same prompts, ids without text). Sorted, because the bundle has no
prompt text to join on; equal sets of prompts with the same scorer give equal sorted lists.

    uv run python scripts/compare_eurosafeguard_scores.py <dir with xs_test.0-of-1.jsonl, or_bench.0-of-1.jsonl> \
        <EuroSafeGuard/paper/scores>
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SETS = {  # bundle source -> (our file, filter on our rows)
    "xstest_safe": ("xs_test", lambda r: r["label"] == "safe"),
    "xstest_unsafe": ("xs_test", lambda r: r["label"] == "unsafe"),
    "orbench_hard": ("or_bench", lambda r: True),
}


def main(ours_dir: str, bundle_dir: str) -> int:
    theirs = pd.read_csv(Path(bundle_dir) / "overflag.csv.gz")
    ok = True
    for source, (name, keep) in SETS.items():
        rows = [json.loads(x) for x in (Path(ours_dir) / f"{name}.0-of-1.jsonl").read_text().splitlines()]
        a = np.sort([r["score"] for r in rows if keep(r)])
        b = np.sort(theirs.loc[theirs.source == source, "probe"].to_numpy())
        if len(a) != len(b):
            print(f"{source}: {len(a)} of our prompts vs {len(b)} in the bundle")
            ok = False
            continue
        diff = np.abs(a - b)
        print(f"{source:14s} n={len(a):5d}  mean ours {a.mean():7.3f} theirs {b.mean():7.3f}  "
              f"max |diff| of sorted scores {diff.max():.4f}, median {np.median(diff):.4f}")
        ok &= diff.max() < 0.05
    print("match" if ok else "NO MATCH")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:3]))
