"""Register finished coop checkpoints as conf/models/models.yaml entries, so run_attacks.py and
defense=coop_probe pick them up by name (model=<run name>).

Usage (from the repo root, after training):
    pixi run --frozen python gen_coop_models.py checkpoints_coop/*/*
    pixi run --frozen python gen_coop_models.py checkpoints_coop/A-eps-sweep/* --dry-run

A directory counts as a checkpoint when it holds final_adapter/, final_reader.pt and
run_config.json. Each becomes one entry named after the run, with dots replaced by "p"
(OmegaConf interpolations such as ${models.${model}.reader_path} split keys on dots). The entry
copies the exact base-model params stored in the run's run_config.json -- including the chat
template it trained with -- and adds adapter_path/reader_path. Entries live between the markers
below; re-running replaces only that block, so hand-written entries are never touched.
"""

import argparse
import json
import sys
from pathlib import Path

import yaml

BEGIN = "# >>> generated coop checkpoints (gen_coop_models.py) >>>"
END = "# <<< generated coop checkpoints <<<"
REPO = Path(__file__).resolve().parent


def entry_name(run_name: str) -> str:
    return run_name.replace(".", "p")


def is_checkpoint(path: Path) -> bool:
    return (path / "final_adapter").is_dir() and (path / "final_reader.pt").is_file() and (path / "run_config.json").is_file()


def build_entry(ckpt_dir: Path, repo: Path) -> tuple[str, dict]:
    cfg = json.loads((ckpt_dir / "run_config.json").read_text())
    entry = {k: v for k, v in cfg["models"][cfg["model"]].items() if k not in ("adapter_path", "reader_path")}
    rel = ckpt_dir.resolve().relative_to(repo.resolve()).as_posix()
    entry["adapter_path"] = "${root_dir}/" + rel + "/final_adapter"
    entry["reader_path"] = "${root_dir}/" + rel + "/final_reader.pt"
    return entry_name(cfg["name"]), entry


def strip_generated(text: str) -> str:
    if BEGIN not in text:
        return text
    head, rest = text.split(BEGIN, 1)
    if END not in rest:
        raise ValueError(f"found {BEGIN!r} without {END!r}")
    return head.rstrip("\n") + "\n" + rest.split(END, 1)[1].lstrip("\n")


def update_models_yaml(text: str, entries: dict[str, dict]) -> str:
    base = strip_generated(text)
    clashes = sorted(set(yaml.safe_load(base) or {}) & set(entries))
    if clashes:
        raise ValueError(f"generated names clash with hand-written models.yaml entries: {clashes}")
    if not entries:
        return base
    block = yaml.safe_dump(dict(sorted(entries.items())), sort_keys=False, default_flow_style=False)
    return base.rstrip("\n") + "\n" + BEGIN + "\n" + block + END + "\n"


def main(argv=None, repo: Path = REPO) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paths", nargs="+", type=Path, help="checkpoint directories (globs expanded by the shell)")
    parser.add_argument("--dry-run", action="store_true", help="print the generated block instead of writing")
    args = parser.parse_args(argv)

    entries, skipped = {}, []
    for path in args.paths:
        if not is_checkpoint(path):
            skipped.append(path)
            continue
        name, entry = build_entry(path, repo)
        if name in entries:
            raise ValueError(f"two checkpoints map to the entry name {name!r}")
        entries[name] = entry

    models_yaml = repo / "conf" / "models" / "models.yaml"
    new_text = update_models_yaml(models_yaml.read_text(), entries)
    for path in skipped:
        print(f"skipped (no final_adapter/final_reader.pt/run_config.json): {path}")
    if args.dry_run:
        print(new_text.split(BEGIN, 1)[1] if BEGIN in new_text else "(no checkpoints found)")
    else:
        models_yaml.write_text(new_text)
    print(f"{len(entries)} checkpoint entr{'y' if len(entries) == 1 else 'ies'} {'would be ' if args.dry_run else ''}written to {models_yaml}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
