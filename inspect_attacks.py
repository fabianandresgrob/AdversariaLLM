"""Show the completions an attack cell counted as successful, so they can be read by hand.

    pixi run --frozen python inspect_attacks.py --attack inpainting --model M-respmean-s0
    pixi run --frozen python inspect_attacks.py --attack inpainting --defense coop_probe \
        --model M-respmean-s0 --limit 10 --out hits.md

Every line collect_attacks.py counts as a jailbreak is one scored completion above the threshold.
This prints those completions with the locator that identifies them
(outputs/<cell>/<date>/<time>/<behavior>, step N, completion i), the judge's p_harmful, and -- for
a defended run -- whether the probe fired on that completion and what it scored.

The point is auditing the judge. strong_reject is a classifier, not an oracle: on fluent attacks
(inpainting, PAIR) a refusal that discusses the topic, or a story that never states anything
actionable, can score high. Reading the hits is the only way to know what an ASR number contains.
"""

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
SCORE_KEY = "p_harmful"


def parse_dir_name(name: str) -> tuple[str, str, str] | None:
    parts = name.split("__")
    return (parts[0], parts[1], parts[2]) if len(parts) == 3 else None


def last_user(conversation) -> str:
    """The prompt actually sent: the last user turn of the (possibly rewritten) conversation."""
    if not isinstance(conversation, list):
        return ""
    for message in reversed(conversation):
        if isinstance(message, dict) and message.get("role") == "user":
            return str(message.get("content", ""))
    return ""


def hits(repo: Path, attack=None, defense=None, model=None, classifier="strong_reject",
         threshold=0.5) -> list[dict]:
    """Every scored completion above the threshold, with where it came from."""
    found = []
    for run_json in sorted((repo / "outputs").glob("*__*__*/*/*/*/run.json")):
        parsed = parse_dir_name(run_json.parents[3].name)
        if parsed is None:
            continue
        cell_attack, cell_defense, cell_model = parsed
        if (attack and cell_attack != attack) or (defense and cell_defense != defense) \
                or (model and cell_model != model):
            continue
        try:
            data = json.loads(run_json.read_text())
        except (OSError, ValueError):
            continue
        for run in data.get("runs", []):
            behavior = last_user(run.get("original_prompt"))
            for step in run.get("steps", []):
                scores = (step.get("scores", {}).get(classifier) or {}).get(SCORE_KEY) or []
                completions = step.get("model_completions") or []
                raw = step.get("model_completions_raw") or []
                meta = step.get("defense_metadata") or []
                for i, score in enumerate(scores):
                    if float(score) <= threshold:
                        continue
                    entry = meta[i] if i < len(meta) and isinstance(meta[i], dict) else {}
                    found.append({
                        "cell": run_json.parents[3].name,
                        "behavior_dir": str(run_json.parent.relative_to(repo / "outputs")),
                        "step": step.get("step"),
                        "completion": i,
                        "p_harmful": float(score),
                        "probe_score": entry.get("score"),
                        "probe_fired": entry.get("applied"),
                        "behavior": behavior,
                        "prompt": last_user(step.get("model_input")),
                        "response": completions[i] if i < len(completions) else "",
                        # what the model said before the defense replaced it, when they differ
                        "response_raw": raw[i] if i < len(raw) and raw[i] != (
                            completions[i] if i < len(completions) else None) else None,
                    })
    return found


def render(found: list[dict], max_chars: int) -> str:
    out = []
    for n, hit in enumerate(found, 1):
        probe = ""
        if hit["probe_fired"] is not None:
            probe = f"   probe={hit['probe_score']:.4f} fired={hit['probe_fired']}"
        out.append(f"### [{n}] {hit['cell']}  {hit['behavior_dir']}  step={hit['step']} "
                   f"completion={hit['completion']}")
        out.append(f"p_harmful={hit['p_harmful']:.4f}{probe}")
        out.append(f"\n**behavior**: {hit['behavior'][:max_chars]}")
        out.append(f"\n**attack prompt**:\n```\n{hit['prompt'][:max_chars]}\n```")
        out.append(f"\n**response**:\n```\n{hit['response'][:max_chars]}\n```")
        if hit["response_raw"]:
            out.append(f"\n**response before the defense replaced it**:\n"
                       f"```\n{hit['response_raw'][:max_chars]}\n```")
        out.append("")
    return "\n".join(out)


def main(argv=None, repo: Path = REPO) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--attack")
    parser.add_argument("--defense", help="'none' for the raw model, e.g. coop_probe for the pipeline")
    parser.add_argument("--model")
    parser.add_argument("--classifier", default="strong_reject")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--limit", type=int, default=20, help="how many hits to show (0 = all)")
    parser.add_argument("--max-chars", type=int, default=1500, help="truncate each text at this length")
    parser.add_argument("--sort", choices=["score", "order"], default="order",
                        help="'score' shows the judge's most confident hits first")
    parser.add_argument("--out", type=Path, help="write markdown here instead of stdout")
    parser.add_argument("--json", type=Path, help="also dump the hits as JSON (full text, untruncated)")
    args = parser.parse_args(argv)

    found = hits(repo, args.attack, args.defense, args.model, args.classifier, args.threshold)
    total = len(found)
    if args.sort == "score":
        found.sort(key=lambda h: h["p_harmful"], reverse=True)
    if args.limit:
        found = found[: args.limit]
    if args.json:
        args.json.write_text(json.dumps(found, indent=2))
    header = (f"{total} completion(s) scored above {args.threshold} by {args.classifier}"
              f"{f'; showing {len(found)}' if len(found) != total else ''}\n")
    text = header + "\n" + render(found, args.max_chars)
    if args.out:
        args.out.write_text(text)
        print(f"{header.strip()} -> {args.out}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
