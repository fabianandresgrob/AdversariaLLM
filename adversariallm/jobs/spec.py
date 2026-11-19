"""Experiment specs (experiments/<name>.yaml) -> concrete runs with stable names, plus the launch profiles of
conf/jobs/profiles.yaml. Taken from jsc-jobs (commit 3b6b9d5, spec.py and profiles.py) without the Slurm parts."""

import dataclasses
import hashlib
import itertools
import json
import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import yaml

_ALLOWED_KEYS = {"exp", "entrypoint", "name", "overrides", "sweep", "artifacts", "time_per_run"}
_NAME_OK = re.compile(r"^[A-Za-z0-9][\w.,+=-]*$", re.ASCII)
_PLACEHOLDER = re.compile(r"\{([^{}]+)\}")
_BARE = re.compile(r"^[\w./:@${}\-+*]+$", re.ASCII)


class SpecError(ValueError):
    pass


@dataclass
class ExperimentSpec:
    exp: str
    entrypoint: str
    name: str | None = None
    overrides: dict = field(default_factory=dict)
    sweep: dict = field(default_factory=dict)
    artifacts: list = field(default_factory=list)
    time_per_run: str | None = None

    def to_dict(self) -> dict:
        data = {"exp": self.exp, "entrypoint": self.entrypoint}
        for key in ("name", "overrides", "sweep", "artifacts", "time_per_run"):
            value = getattr(self, key)
            if value:
                data[key] = value
        return data


@dataclass(frozen=True)
class RunSpec:
    exp: str
    name: str
    entrypoint: str
    overrides: dict
    artifacts: tuple
    time_per_run: str | None
    config_hash: str


def check_name(name: str, what: str) -> None:
    if not _NAME_OK.match(name) or "__" in name:
        raise SpecError(f"invalid {what} name {name!r}: use letters, digits, . , + = - _ (no '__', no leading symbol)")


def spec_from_dict(data: dict, default_exp: str) -> ExperimentSpec:
    unknown = set(data) - _ALLOWED_KEYS
    if unknown:
        raise SpecError(f"unknown keys in experiment spec: {sorted(unknown)}")
    if not data.get("entrypoint"):
        raise SpecError("experiment spec needs 'entrypoint'")
    exp = str(data.get("exp") or default_exp)
    check_name(exp, "experiment")
    sweep = dict(data.get("sweep") or {})
    for key, values in sweep.items():
        if not isinstance(values, list) or not values:
            raise SpecError(f"sweep axis {key!r} must be a non-empty list")
    tpr = data.get("time_per_run")
    return ExperimentSpec(
        exp=exp,
        entrypoint=str(data["entrypoint"]),
        name=data.get("name"),
        overrides=dict(data.get("overrides") or {}),
        sweep=sweep,
        artifacts=list(data.get("artifacts") or []),
        time_per_run=str(tpr) if tpr is not None else None,
    )


def load_experiment(path) -> ExperimentSpec:
    path = Path(path)
    return spec_from_dict(yaml.safe_load(path.read_text()) or {}, default_exp=path.stem)


def _split_top_level(raw: str) -> list[str]:
    """Split on commas outside brackets/quotes: 'a,b' -> [a, b]; '[a,b]' -> ['[a,b]']."""
    parts, current, depth, quote = [], [], 0, None
    for ch in raw:
        if quote:
            current.append(ch)
            if ch == quote:
                quote = None
            continue
        if ch in "'\"":
            quote = ch
        elif ch in "[{(":
            depth += 1
        elif ch in "]})":
            depth -= 1
        elif ch == "," and depth == 0:
            parts.append("".join(current))
            current = []
            continue
        current.append(ch)
    parts.append("".join(current))
    return parts


def _parse_value(text: str):
    try:
        return yaml.safe_load(text)
    except yaml.YAMLError:
        return text


def parse_cli(entrypoint, tokens, exp=None, name=None, time_per_run=None, now=None) -> ExperimentSpec:
    overrides, sweep = {}, {}
    for token in tokens:
        if "=" not in token:
            raise SpecError(f"expected key=value, got {token!r}")
        key, raw = token.split("=", 1)
        parts = _split_top_level(raw)
        if len(parts) > 1:
            sweep[key] = [_parse_value(p) for p in parts]
        else:
            overrides[key] = _parse_value(raw)
    if exp is None:
        stamp = (now or datetime.now()).strftime("%Y%m%d-%H%M%S")
        exp = f"adhoc-{Path(entrypoint).stem}-{stamp}"
    data = {"exp": exp, "entrypoint": entrypoint, "name": name, "overrides": overrides, "sweep": sweep,
            "time_per_run": time_per_run}
    return spec_from_dict(data, default_exp=exp)


def format_value(value) -> str:
    """Python value -> Hydra command-line override value."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(format_value(v) for v in value) + "]"
    if isinstance(value, dict):
        return "{" + ",".join(f"{k}:{format_value(v)}" for k, v in value.items()) + "}"
    text = str(value)
    if _BARE.match(text):
        return text
    return "'" + text.replace("\\", "\\\\").replace("'", "\\'") + "'"


def format_overrides(overrides: dict) -> list[str]:
    return [f"{key}={format_value(value)}" for key, value in overrides.items()]


def config_hash(entrypoint: str, overrides: dict) -> str:
    payload = json.dumps({"entrypoint": entrypoint, "overrides": overrides}, sort_keys=True,
                         separators=(",", ":"), default=str)
    return "sha256:" + hashlib.sha256(payload.encode()).hexdigest()


def _render(template: str, values: dict) -> str:
    def substitute(match):
        key = match.group(1)
        if key not in values:
            raise SpecError(f"template {template!r}: no value for {{{key}}}")
        value = values[key]
        return format_value(value) if isinstance(value, (bool, list, tuple, dict)) or value is None else str(value)

    return _PLACEHOLDER.sub(substitute, template)


def expand(spec: ExperimentSpec, name_key: str | None = None) -> list[RunSpec]:
    keys = list(spec.sweep)
    points = list(itertools.product(*(spec.sweep[k] for k in keys))) if keys else [()]
    runs, seen = [], set()
    for index, point in enumerate(points):
        overrides = {**spec.overrides, **dict(zip(keys, point))}
        name = _render(spec.name, overrides) if spec.name else f"{spec.exp}-{index}"
        check_name(name, "run")
        if name in seen:
            raise SpecError(f"duplicate run name {name!r} in experiment {spec.exp!r}")
        seen.add(name)
        if name_key:
            overrides[name_key] = name
        artifacts = tuple(_render(a, {**overrides, "name": name}) for a in spec.artifacts)
        runs.append(RunSpec(spec.exp, name, spec.entrypoint, overrides, artifacts, spec.time_per_run,
                            config_hash(spec.entrypoint, overrides)))
    return runs


@dataclass(frozen=True)
class Profile:
    config_name: str | None = None          # Hydra config the entrypoint uses (checked by the project's tests)
    name_key: str | None = None             # override that receives the run name
    hydra_dir: bool = True                  # inject hydra.run.dir=<attempt>/hydra
    wandb_dir_key: str | None = None        # inject <key>=<attempt>/wandb
    uses_db: bool = False                   # pass ADVLLM_DB into the run environment
    extra_overrides: dict = field(default_factory=dict)   # always injected; str values get env-var expansion
    rerun_overrides: dict = field(default_factory=dict)   # injected for attempt >= 2


_FIELDS = {f.name for f in dataclasses.fields(Profile)}
PROFILES_FILE = "conf/jobs/profiles.yaml"


def load_profiles(project_root: Path) -> dict[str, Profile]:
    data = yaml.safe_load((Path(project_root) / PROFILES_FILE).read_text()) or {}
    profiles = {}
    for entrypoint, fields in data.items():
        fields = fields or {}
        unknown = set(fields) - _FIELDS
        if unknown:
            raise SpecError(f"profile {entrypoint!r}: unknown fields {sorted(unknown)}")
        profiles[str(entrypoint)] = Profile(**fields)
    return profiles


def get_profile(profiles: dict[str, Profile], entrypoint: str) -> Profile:
    try:
        return profiles[entrypoint]
    except KeyError:
        known = ", ".join(sorted(profiles))
        raise SpecError(f"no launch profile for {entrypoint!r} in {PROFILES_FILE}; known entrypoints: {known}") from None


def injected_overrides(profile: Profile, attempt_dir: Path, attempt: int) -> dict:
    out = {}
    if profile.hydra_dir:
        out["hydra.run.dir"] = str(attempt_dir / "hydra")
    if profile.wandb_dir_key:
        out[profile.wandb_dir_key] = str(attempt_dir / "wandb")
    for key, value in profile.extra_overrides.items():
        out[key] = os.path.expandvars(value) if isinstance(value, str) else value
    if attempt >= 2:
        out.update(profile.rerun_overrides)
    return out


def launch_args(entrypoint: str) -> list[str]:
    return [entrypoint] if entrypoint.endswith(".py") else ["-m", entrypoint]


_DURATION = re.compile(r"^(?:(\d+)-)?(?:(\d+):)?(\d+):(\d+)$")


def parse_duration(text: str | int) -> int:
    """Time string -> seconds. Accepts D-HH:MM:SS, HH:MM:SS, MM:SS, or plain seconds."""
    if isinstance(text, int):
        return text
    s = str(text).strip()
    if s.isdigit():
        return int(s)
    m = _DURATION.match(s)
    if not m:
        raise ValueError(f"unparsable duration: {text!r}")
    days, hours, minutes, seconds = (int(g) if g else 0 for g in m.groups())
    return ((days * 24 + hours) * 60 + minutes) * 60 + seconds
