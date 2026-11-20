"""HTCondor submit files, cost estimates and the run store for condor_jobs.py.

Pricing (IT page "Cluster banking system"): a job costs computing units x running price per hour, units are
max(cpus, memory_GB / 16) + 24 per GPU. The running price is the highest waiting bid of another user that could
run on the node, capped by the job's bid, and the cap rises 10% per hour after the first hour. The price limit
(+MaxRunningPrice) stops the job once the price would exceed it.
"""

import json
import os
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import yaml

from .launch import RECORD, write_record
from .spec import parse_duration

SITE_FILE = "conf/jobs/mpi.yaml"
CHECKPOINT_EXIT_CODE = 3  # launch.py: "checkpointed, continue"; the submit file re-queues the job


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load_site(repo: Path) -> dict:
    return yaml.safe_load((Path(repo) / SITE_FILE).read_text())


# ---- resources and cost ----

@dataclass(frozen=True)
class Resources:
    cpus: int
    memory_gb: int
    gpus: int
    disk_gb: int
    gpu: str | None  # key of site["gpus"]; None for CPU-only jobs

    @property
    def units(self) -> float:
        return max(self.cpus, self.memory_gb / 16) + 24 * self.gpus


@dataclass(frozen=True)
class Limits:
    bid: float
    max_price: float          # per computing unit and hour, as the bid
    price_action: str         # "kill" or "restart"
    max_hours: float          # runtime of one start before the job is held
    max_parallel: int | None  # None: no concurrency limit


def cost_lines(n_jobs: int, res: Resources, limits: Limits, time_per_run_h: float, min_bid: float) -> list[str]:
    u = res.units
    fee = n_jobs * min_bid / 10
    lines = [
        f"{n_jobs} job(s) x {u:g} units ({res.cpus} cores, {res.memory_gb} GB, {res.gpus} GPU)",
        (f"per job and hour:  {u * limits.bid:,.0f} at the bid ({limits.bid:g}), "
         f"at most {u * limits.max_price:,.0f} (price limit {limits.max_price:g})"),
        f"all jobs per hour: {n_jobs * u * limits.bid:,.0f} at the bid, at most {n_jobs * u * limits.max_price:,.0f}",
        (f"whole run ({time_per_run_h:g} h per job): {n_jobs * u * limits.bid * time_per_run_h:,.0f} if the price "
         f"stays at the bid, at most {n_jobs * u * limits.max_price * time_per_run_h:,.0f}"),
        f"submission fee: {fee:,.1f}",
    ]
    if limits.max_parallel:
        lines.append(f"at most {limits.max_parallel} job(s) at once")
    return lines


# ---- submit file ----

def concurrency_tag(exp: str) -> str:
    return "user." + re.sub(r"[^A-Za-z0-9_]", "_", exp)


def render_submit(site: dict, workdir: Path, submit_dir: Path, exp: str, res: Resources, limits: Limits) -> str:
    python = Path(site["repo"]) / ".venv" / "bin" / "python"
    lines = [
        f"# {exp}: one job per line of runs.txt (an attempt directory with its run.json)",
        "universe       = vanilla",
        f"executable     = {python}",
        'arguments      = "-u -m adversariallm.jobs.launch $(attempt_dir)"',
        f"initialdir     = {workdir}",
        f'environment    = "HOME={site["home"]} PYTHONUNBUFFERED=1"',
        "# one log pair per start: hold/release and restarts would overwrite them otherwise",
        "output         = $(attempt_dir)/condor.$$([NumJobStarts]).out",
        "error          = $(attempt_dir)/condor.$$([NumJobStarts]).err",
        f"log            = {submit_dir}/condor.log",
        f"request_cpus   = {res.cpus}",
        f"request_memory = {res.memory_gb * 1024}",
        f"request_disk   = {res.disk_gb}G",
    ]
    if res.gpus:
        lines += [f"request_gpus   = {res.gpus}",
                  f'requirements   = TARGET.CUDADeviceName == "{site["gpus"][res.gpu]["device"]}"']
    lines += [
        "",
        "# price limit per computing unit and hour (IT page 'Limit the running price of a job')",
        f"+MaxRunningPrice = {limits.max_price:g}",
        f'+RunningPriceExceededAction = "{limits.price_action}"',
        "",
        "# runtime limit per start: a hanging job is held instead of costing money (condor_release to continue)",
        f"MaxTime = {int(limits.max_hours * 3600)}",
        "periodic_hold = (JobStatus =?= 2) && ((CurrentTime - JobCurrentStartDate) >= $(MaxTime))",
        'periodic_hold_reason = "Job runtime exceeded"',
        "",
        (f"# exit code {CHECKPOINT_EXIT_CODE} = checkpointed: hold and release, the price starts fresh "
         "(IT page 'How to automatically restart jobs according to the exit code')"),
        f"on_exit_hold = (ExitCode =?= {CHECKPOINT_EXIT_CODE})",
        'on_exit_hold_reason = "Checkpointed, will resume"',
        "on_exit_hold_subcode = 1",
        "periodic_release = ( (JobStatus =?= 5) && (HoldReasonCode =?= 3) && (HoldReasonSubCode =?= 1) )",
    ]
    if limits.max_parallel:
        lines += ["", f"# at most {limits.max_parallel} at once (10000 tokens per tag)",
                  f"concurrency_limits = {concurrency_tag(exp)}:{10000 // limits.max_parallel}"]
    lines += ["", "queue attempt_dir from runs.txt", ""]
    return "\n".join(lines)


# ---- run store: <jobs_root>/<exp>/<run>/attempts/<n>/run.json ----

class RunStore:
    def __init__(self, root):
        self.root = Path(root)

    def run_dir(self, exp: str, run: str) -> Path:
        return self.root / exp / run

    def attempts(self, exp: str, run: str) -> list[int]:
        base = self.run_dir(exp, run) / "attempts"
        return sorted(int(p.name) for p in base.iterdir() if p.name.isdigit()) if base.is_dir() else []

    def latest(self, exp: str, run: str) -> tuple[int, dict] | None:
        attempts = self.attempts(exp, run)
        if not attempts:
            return None
        path = self.run_dir(exp, run) / "attempts" / str(attempts[-1]) / RECORD
        return attempts[-1], json.loads(path.read_text())

    def next_attempt(self, exp: str, run: str) -> tuple[int, Path]:
        attempt = (self.attempts(exp, run) or [0])[-1] + 1
        return attempt, self.run_dir(exp, run) / "attempts" / str(attempt)

    def create_attempt(self, exp: str, run: str, attempt: int, record: dict) -> Path:
        attempt_dir = self.run_dir(exp, run) / "attempts" / str(attempt)
        attempt_dir.mkdir(parents=True)  # fails if it exists: never reuse an attempt
        write_record(attempt_dir, {**record, "exp": exp, "run": run, "attempt": attempt})
        link = self.run_dir(exp, run) / "latest"
        if link.is_symlink():
            link.unlink()
        os.symlink(f"attempts/{attempt}", link)
        return attempt_dir

    def experiments(self) -> list[str]:
        if not self.root.is_dir():
            return []
        return sorted(p.name for p in self.root.iterdir() if p.is_dir() and not p.name.startswith("_"))

    def runs(self, exp: str) -> list[str]:
        d = self.root / exp
        return sorted(p.name for p in d.iterdir() if (p / "attempts").is_dir()) if d.is_dir() else []

    def records(self):
        """run.json of every attempt of every run (older attempts can still be queued after --rerun)."""
        for exp in self.experiments():
            for run in self.runs(exp):
                for attempt in self.attempts(exp, run):
                    yield json.loads((self.run_dir(exp, run) / "attempts" / str(attempt) / RECORD).read_text())


# ---- status ----

CONDOR_STATES = {1: "idle", 2: "running", 3: "removed", 4: "completed", 5: "held", 6: "transferring", 7: "suspended"}


def condor_queue() -> dict[str, tuple[str, str]] | None:
    """{"cluster.proc": (state, hold reason)} of the user's jobs; None when condor_q is not available."""
    try:
        out = subprocess.run(["condor_q", "-af:t", "ClusterId", "ProcId", "JobStatus", "HoldReason"],
                             capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return None
    jobs = {}
    for line in out.splitlines():
        parts = line.split("\t")
        if len(parts) >= 3 and parts[0].isdigit():
            reason = parts[3] if len(parts) > 3 and parts[3] != "undefined" else ""
            jobs[f"{parts[0]}.{parts[1]}"] = (CONDOR_STATES.get(int(parts[2]), parts[2]), reason)
    return jobs


def run_status(record: dict, queue: dict | None) -> tuple[str, str]:
    """(status, detail) of an attempt from its run.json and, if available, the condor queue."""
    in_queue = queue.get(record.get("condor_job")) if queue is not None else None
    if in_queue and in_queue[0] in ("held", "idle", "running"):
        state, reason = in_queue
        if state == "idle" and record.get("starts"):
            return "requeued", f"start {len(record['starts'])} ended with {record.get('exit_code')}"
        return state, reason
    if "exit_code" not in record:
        if not record.get("condor_job"):
            return "not submitted", ""
        if queue is None:
            return "submitted", "condor_q not available"
        return ("lost", "left the queue without an exit code") if record.get("starts") else \
            ("removed", "left the queue before starting")
    if record["exit_code"] == 0:
        missing = record.get("missing_artifacts") or []
        return ("failed", f"missing {', '.join(missing)}") if missing else ("done", "")
    if record["exit_code"] == CHECKPOINT_EXIT_CODE:
        return "checkpointed", "not in the queue anymore" if queue is not None else ""
    return "failed", f"exit code {record['exit_code']}"


# ---- frozen code ----

def git(repo: Path, *args) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=True).stdout.strip()


def frozen_checkout(site: dict, commit: str) -> Path:
    """A detached worktree of `commit` under <jobs_root>/_code, with the checkout's /fast symlinks. Jobs run from
    it, so a restarted job runs the code it was submitted with, not whatever the checkout has by then."""
    repo = Path(site["repo"])
    path = Path(site["jobs_root"]) / "_code" / commit[:12]
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        git(repo, "worktree", "add", "--detach", str(path), commit)
    for name in site.get("links", []):
        source, link = repo / name, path / name
        if source.exists() and not link.exists():  # the link's own target (/fast/...), not the /lustre real path
            os.symlink(os.readlink(source) if source.is_symlink() else source, link)
    return path


ACTIVE = {"idle", "running", "held", "requeued", "submitted"}  # run_status values of attempts that can still start


def prune_frozen_checkouts(site: dict, store: RunStore, queue, keep: str | None = None) -> list[tuple[Path, str]]:
    """Remove the frozen copies under <jobs_root>/_code that no queued or running attempt uses. Without the condor
    queue (queue is None) nothing is removed. `git worktree remove` refuses a copy with files git does not ignore,
    and that copy is kept. Returns [(path, "removed" or the reason it was kept)]."""
    base = Path(site["jobs_root"]) / "_code"
    if queue is None or not base.is_dir():
        return []
    in_use = {record.get("workdir") for record in store.records() if run_status(record, queue)[0] in ACTIVE}
    result = []
    for path in sorted(p for p in base.iterdir() if p.is_dir()):
        if str(path) in in_use or (keep and path.name == keep[:12]):
            continue
        try:
            git(Path(site["repo"]), "worktree", "remove", str(path))
            result.append((path, "removed"))
        except subprocess.CalledProcessError as e:
            result.append((path, e.stderr.strip() or "git worktree remove failed"))
    return result


def parse_cluster_id(output: str) -> int:
    m = re.search(r"submitted to cluster (\d+)", output)
    if not m:
        raise RuntimeError(f"no cluster id in condor_submit_bid output:\n{output}")
    return int(m.group(1))


def hours(duration) -> float:
    return parse_duration(duration) / 3600
