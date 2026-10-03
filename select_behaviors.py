"""Extend the coop training behaviors with HarmBench and AdvBench, keeping everything that resembles JBB out.

    pixi run --frozen python select_behaviors.py            # writes the review list, or the set once reviewed

JBB is the test set, and it takes 45 of its 100 behaviors from HarmBench and AdvBench, so every candidate is
string-matched (difflib ratio, as in train_response_head.py) against all 100 JBB behaviors:

  1. candidates   HarmBench standard text behaviors (contextual ones need a context, copyright is out of
                  scope) + AdvBench, each with its dataset's one target
  2. drop         ratio > --drop to any JBB goal or behavior, or to a current training behavior
  3. dedup        ratio > --drop to a candidate kept earlier (AdvBench repeats itself)
  4. review       ratio in (--review, --drop] to a JBB goal: a person decides keep/drop in the review csv; the
                  script stops until every such row has a decision, so the filter is reproducible from the csv
  5. write        the current behaviors + the kept candidates, in the training format

Writes (data/ paths):
  behavior_datasets/extra_behavior_datasets/adv_training_varied_behaviors.csv   old + new behaviors
  optimizer_targets/extra_targets/adv_training_varied_targets.json             old targets + one per new behavior
  behavior_datasets/extra_behavior_datasets/adv_training_varied_selection.json  counts per step, every drop + reason
"""

import argparse
import csv
import difflib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
EXTRA = "behavior_datasets/extra_behavior_datasets/"
REVIEW_CSV = EXTRA + "adv_training_varied_review.csv"
REVIEW_FIELDS = ["candidate_id", "candidate", "closest_jbb", "ratio", "decision", "note"]


def norm(text: str) -> str:
    return " ".join(str(text).lower().split())


def closest(text: str, pool: list[str]) -> tuple[float, str]:
    """(ratio, item) of the pool item most similar to text."""
    t = norm(text)
    return max(((difflib.SequenceMatcher(None, t, norm(p)).ratio(), p) for p in pool), default=(0.0, ""))


def select(candidates: list[dict], jbb: list[str], current: list[str], drop: float, review: float,
           decisions: dict[str, str]) -> tuple[list[dict], list[dict], list[dict]]:
    """candidates: dicts with id, behavior, target, source, category. Returns (kept, dropped, pending): dropped
    rows carry a reason; pending rows are in the review band without a decision yet."""
    kept, dropped, pending, seen = [], [], [], []
    for c in candidates:
        r_jbb, near_jbb = closest(c["behavior"], jbb)
        r_cur, near_cur = closest(c["behavior"], current)
        r_seen, near_seen = closest(c["behavior"], seen)
        if r_jbb > drop:
            dropped.append({**c, "reason": "near JBB", "ratio": round(r_jbb, 3), "match": near_jbb})
        elif r_cur > drop:
            dropped.append({**c, "reason": "near current training behavior", "ratio": round(r_cur, 3), "match": near_cur})
        elif r_seen > drop:
            dropped.append({**c, "reason": "near another candidate", "ratio": round(r_seen, 3), "match": near_seen})
        elif r_jbb > review and decisions.get(c["id"]) not in ("keep", "drop"):
            pending.append({**c, "ratio": round(r_jbb, 3), "match": near_jbb})
        elif r_jbb > review and decisions[c["id"]] == "drop":
            dropped.append({**c, "reason": "reviewed: same request as JBB", "ratio": round(r_jbb, 3), "match": near_jbb})
        else:
            kept.append(c)
            seen.append(c["behavior"])
    return kept, dropped, pending


def read_csv(path: Path) -> list[dict]:
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def load_candidates(data: Path) -> list[dict]:
    hb_targets = json.loads((data / "optimizer_targets/harmbench_targets_text.json").read_text())
    ab_targets = json.loads((data / "optimizer_targets/extra_targets/advbench_targets.json").read_text())
    out = [{"id": r["BehaviorID"], "behavior": r["Behavior"], "target": hb_targets[r["BehaviorID"]],
            "source": "HarmBench", "category": r["SemanticCategory"]}
           for r in read_csv(data / "behavior_datasets/harmbench_behaviors_text_all.csv")
           if r["FunctionalCategory"] == "standard" and r["BehaviorID"] in hb_targets]
    out += [{"id": r["BehaviorID"], "behavior": r["Behavior"], "target": ab_targets[r["BehaviorID"]],
             "source": "AdvBench", "category": ""}
            for r in read_csv(data / (EXTRA + "advbench_behaviors.csv")) if r["BehaviorID"] in ab_targets]
    return out


def main(argv=None, repo: Path = REPO) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--drop", type=float, default=0.8, help="ratio above which a candidate is dropped")
    parser.add_argument("--review", type=float, default=0.6, help="ratio above which a person decides")
    args = parser.parse_args(argv)
    data = repo / "data"

    import jailbreakbench as jbb

    jdf = jbb.read_dataset().as_dataframe()
    jbb_texts = list(jdf.Goal) + list(jdf.Behavior)
    current = read_csv(data / (EXTRA + "adv_training_behaviors.csv"))
    current_targets = json.loads((data / "optimizer_targets/extra_targets/adv_training_targets.json").read_text())
    candidates = load_candidates(data)

    review_path = data / REVIEW_CSV
    reviewed = read_csv(review_path) if review_path.exists() else []
    decisions = {r["candidate_id"]: r["decision"].strip().lower() for r in reviewed}
    kept, dropped, pending = select(candidates, jbb_texts, [r["Behavior"] for r in current], args.drop,
                                    args.review, decisions)

    if pending:  # add the undecided rows to the review csv and stop
        rows = reviewed + [{"candidate_id": p["id"], "candidate": p["behavior"], "closest_jbb": p["match"],
                            "ratio": p["ratio"], "decision": "", "note": ""}
                           for p in pending if p["id"] not in decisions]
        with open(review_path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=REVIEW_FIELDS)
            w.writeheader()
            w.writerows(rows)
        print(f"{len(pending)} candidates need a keep/drop decision in {review_path}; rerun once filled in")
        return 1

    new_rows = [{"Behavior": c["behavior"], "Category": f"{c['source']}:{c['category']}".rstrip(":"), "Tags": "",
                 "ContextString": "", "BehaviorID": c["id"]} for c in kept]
    with open(data / (EXTRA + "adv_training_varied_behaviors.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["Behavior", "Category", "Tags", "ContextString", "BehaviorID"])
        w.writeheader()
        w.writerows([{k: r.get(k, "") for k in w.fieldnames} for r in current] + new_rows)
    targets = {**current_targets, **{c["id"]: [c["target"]] for c in kept}}
    (data / "optimizer_targets/extra_targets/adv_training_varied_targets.json").write_text(json.dumps(targets, indent=1))
    summary = {"thresholds": {"drop": args.drop, "review": args.review},
               "counts": {"candidates": len(candidates), "kept": len(kept), "dropped": len(dropped),
                          "current": len(current), "total": len(current) + len(kept)},
               "dropped_by_reason": {r: sum(d["reason"] == r for d in dropped) for r in sorted({d["reason"] for d in dropped})},
               "kept_by_source": {s: sum(c["source"] == s for c in kept) for s in ("HarmBench", "AdvBench")},
               "dropped": dropped}
    (data / (EXTRA + "adv_training_varied_selection.json")).write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: summary[k] for k in ("counts", "dropped_by_reason", "kept_by_source")}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
