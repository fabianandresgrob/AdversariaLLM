"""Test job for condor_jobs.py: checks what a job sees on a compute node, without a model.

    uv run python condor_jobs.py submit smoke-condor --bid 15 --gpus 0 --cpus 1 --mem-gb 2 --disk-gb 1

Checks that the code comes from the frozen copy (<jobs_root>/_code/<commit>), that HOME, PWD and the site env are
set, that filelock is patched, that POSIX locks work under db_root (/home) and that the /fast links are writable.
With restart=1 the first start exits with 3 and the second with 0, which tests the hold and release in the
submit file. Every failed check is printed, and the exit code is 1 if any failed.
"""

import fcntl
import os
import sys
from pathlib import Path


def main(args: dict[str, str]) -> int:
    import filelock
    import torch

    import adversariallm

    name = args["name"]
    cwd = Path.cwd()
    checks = {
        "code from the frozen copy": "/_code/" in adversariallm.__file__ and Path(adversariallm.__file__).is_relative_to(cwd),
        "PWD is the working dir": os.environ.get("PWD") == str(cwd),
        "HOME is /home/fgrob": os.environ.get("HOME") == "/home/fgrob",
        "HF_HOME on /fast": os.environ.get("HF_HOME", "").startswith("/fast/"),
        "HF_HUB_OFFLINE=1": os.environ.get("HF_HUB_OFFLINE") == "1",
        "filelock patched": filelock.FileLock is filelock.SoftFileLock,
    }
    for key in ("sys.executable", "cwd", "adversariallm", "torch", "cuda", "nproc", "CUDA_VISIBLE_DEVICES",
                "ADVLLM_DB", "_CONDOR_SCRATCH_DIR"):
        value = {"sys.executable": sys.executable, "cwd": cwd, "adversariallm": adversariallm.__file__,
                 "torch": torch.__version__, "cuda": torch.cuda.is_available(), "nproc": os.cpu_count()}.get(
            key, os.environ.get(key))
        print(f"[smoke] {key}: {value}", flush=True)

    db = os.environ.get("ADVLLM_DB", "")
    if db.startswith("file:"):
        lock_dir = Path(db[len("file:"):]) / "_smoke"
        lock_dir.mkdir(parents=True, exist_ok=True)
        try:
            with open(lock_dir / "lock", "w") as f:
                fcntl.lockf(f, fcntl.LOCK_EX)
                fcntl.lockf(f, fcntl.LOCK_UN)
            checks["lockf under db_root"] = True
        except OSError as e:
            print(f"[smoke] lockf failed: {e}", flush=True)
            checks["lockf under db_root"] = False

    out = cwd / "outputs" / "smoke"
    out.mkdir(parents=True, exist_ok=True)
    marker = out / f"{name}.restarted"
    if args.get("restart") == "1" and not marker.exists():
        marker.write_text("first start\n")
        print("[smoke] first start: exit 3 (checkpointed, continue)", flush=True)
        return 3
    (out / f"{name}.txt").write_text(f"{os.uname().nodename}\n")
    checks["outputs/ writable (/fast)"] = (out / f"{name}.txt").exists() and str(out.resolve()).startswith(
        ("/fast/", "/lustre/fast/"))

    for check, ok in checks.items():
        print(f"[smoke] {'ok  ' if ok else 'FAIL'} {check}", flush=True)
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    sys.exit(main(dict(a.split("=", 1) for a in sys.argv[1:])))
