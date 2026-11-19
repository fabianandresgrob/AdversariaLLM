"""Runs one attempt inside an HTCondor job: `python -m adversariallm.jobs.launch <attempt_dir>`.

Reads the command from <attempt_dir>/run.json (written by condor_jobs.py), sets the environment, switches file
locks to SoftFileLock (/fast has no POSIX locks), runs the entrypoint in this process and records every start
and its exit code in run.json. Exit code 3 means "checkpointed, continue": the submit file then puts the job
back in the queue, and the next start appends a new entry to `starts`.
"""

import json
import os
import runpy
import socket
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

RECORD = "run.json"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def read_record(attempt_dir: Path) -> dict:
    return json.loads((attempt_dir / RECORD).read_text())


def write_record(attempt_dir: Path, record: dict) -> None:
    """Atomic: a reader (condor_jobs.py status) never sees a half-written file."""
    fd, tmp = tempfile.mkstemp(dir=attempt_dir, prefix=f".{RECORD}.", suffix=".tmp")
    with os.fdopen(fd, "w") as f:
        json.dump(record, f, indent=2, sort_keys=True, default=str)
        f.write("\n")
    os.replace(tmp, attempt_dir / RECORD)


def use_soft_file_locks() -> None:
    """Hugging Face and datasets lock their caches with filelock.FileLock, which fails on /fast. SoftFileLock only
    checks that the lock file exists. A job killed while holding one leaves a stale *.lock under HF_HOME."""
    import filelock

    filelock.FileLock = filelock.SoftFileLock


def run_entrypoint(entrypoint: str, args: list[str]) -> int:
    """Run a script path or module as __main__ with the given argv and return its exit code."""
    sys.argv = [entrypoint, *args]
    try:
        if entrypoint.endswith(".py"):
            runpy.run_path(entrypoint, run_name="__main__")
        else:
            runpy.run_module(entrypoint, run_name="__main__", alter_sys=True)
    except SystemExit as e:
        if e.code is None:
            return 0
        if isinstance(e.code, int):
            return e.code
        print(e.code, file=sys.stderr)
        return 1
    return 0


def missing_artifacts(workdir: Path, artifacts: list[str]) -> list[str]:
    return [a for a in artifacts if not (workdir / a).exists()]


def main(attempt_dir: str) -> int:
    attempt_dir = Path(attempt_dir)
    record = read_record(attempt_dir)
    workdir = Path(record["workdir"])
    os.chdir(workdir)
    os.environ["PWD"] = str(workdir)  # conf/paths.yaml: root_dir = ${oc.env:PWD}
    os.environ.update(record.get("env", {}))
    if os.environ.get("SOFTFILELOCK") == "1":
        use_soft_file_locks()

    start = {"started_at": now_iso(), "host": socket.gethostname(),
             "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}
    record.setdefault("starts", []).append(start)
    write_record(attempt_dir, record)
    print(f"[launch] {record['exp']}/{record['run']} attempt {record['attempt']} start {len(record['starts'])} "
          f"on {start['host']}: {record['entrypoint']} {' '.join(record['args'])}", flush=True)

    code = 1
    try:
        code = run_entrypoint(record["entrypoint"], record["args"])
    finally:
        record = read_record(attempt_dir)  # the run may not touch it, but never overwrite newer data
        start = record["starts"][-1]
        start["ended_at"], start["exit_code"] = now_iso(), code
        record["exit_code"] = code
        if code == 0:
            record["missing_artifacts"] = missing_artifacts(workdir, record.get("artifacts", []))
        write_record(attempt_dir, record)
        print(f"[launch] exit code {code}", flush=True)
    return code


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
