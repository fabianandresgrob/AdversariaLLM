"""A folder of model answers keyed by row id, written by generate_answers.py.

    <store>/meta.json                      what produced the answers (model, template, generation settings)
    <store>/answers.<shard>of<n>.jsonl     one line per prompt: row_id, answer, token counts, refusal flag
    <store>/judge.<judge>.jsonl            one line per judged answer: row_id, p_harmful, judge_against

Files are only appended to. A row id that is already in the store is skipped, so a restarted job continues where it
stopped and a larger prompt file adds its new rows to the same store. meta.json is fixed by the first run: a later
run with other settings is refused, since its answers would not be comparable with the stored ones.
"""

import glob
import json
import os


def read_jsonl(pattern: str) -> list[dict]:
    """Every complete line of the files matching `pattern`. A last line cut off by a killed job is dropped."""
    rows = []
    for path in sorted(glob.glob(pattern)):
        with open(path) as fh:
            for line in fh:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return rows


def done_ids(store: str, prefix: str) -> set:
    """Row ids already in the store's `<prefix>*jsonl` files (prefix `answers.` or `judge.<name>.`, which also
    matches the unsharded judge.<name>.jsonl)."""
    return {r["row_id"] for r in read_jsonl(os.path.join(store, f"{prefix}*jsonl"))}


def append(path: str, rows: list[dict]) -> None:
    """Append rows and flush, starting on a fresh line if a killed job left the last one unfinished."""
    cut = False
    if os.path.exists(path) and os.path.getsize(path) > 0:
        with open(path, "rb") as fh:
            fh.seek(-1, os.SEEK_END)
            cut = fh.read(1) != b"\n"
    with open(path, "a") as fh:
        if cut:
            fh.write("\n")
        fh.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
        fh.flush()
        os.fsync(fh.fileno())


def check_meta(store: str, meta: dict, fixed: tuple) -> None:
    """Write meta.json on the store's first run. Later runs must agree on the `fixed` keys."""
    os.makedirs(store, exist_ok=True)
    path = os.path.join(store, "meta.json")
    if not os.path.exists(path):
        with open(path, "w") as fh:
            json.dump(meta, fh, indent=1)
        return
    with open(path) as fh:
        old = json.load(fh)
    differ = {k: (old.get(k), meta.get(k)) for k in fixed if old.get(k) != meta.get(k)}
    if differ:
        raise ValueError(f"{store} was written with other settings (stored, now): {differ}. Use another store name.")
