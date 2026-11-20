"""Submit experiments/<name>.yaml to the MPI-IS HTCondor cluster and show the status of its runs.

    uv run python condor_jobs.py submit coop-W-coop --bid 25                 # dry run: runs, submit file, cost
    uv run python condor_jobs.py submit coop-W-coop --bid 25 --submit        # on a login node: really submit
    uv run python condor_jobs.py submit coop-W-coop --bid 25 --only 'W-coop-s0' --rerun --submit
    uv run python condor_jobs.py status [coop-W-coop] [-v]
    uv run python condor_jobs.py prune                                       # remove unused frozen code copies

One HTCondor job per run, all runs of an experiment in one submission (`queue attempt_dir from runs.txt`). Every
submission is a new attempt: <jobs_root>/<exp>/<run>/attempts/<n>/ holds run.json (command, commit, condor job,
every start with its exit code), the condor logs of each start, and the Hydra and W&B dirs. Jobs run from a frozen
copy of the committed code (<jobs_root>/_code/<commit>), so commit before submitting. Paths, environment and
resources are in conf/jobs/mpi.yaml, the per-script arguments in conf/jobs/profiles.yaml.

Runs that are done, or still in the queue, are skipped unless --rerun is given. Every submission (and `prune`)
removes the frozen copies that no queued or running run uses anymore.
"""

import argparse
import fnmatch
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

from adversariallm.jobs.condor import (
    ACTIVE,
    SITE_FILE,
    Limits,
    Resources,
    RunStore,
    condor_queue,
    cost_lines,
    frozen_checkout,
    git,
    hours,
    load_site,
    now_iso,
    parse_cluster_id,
    prune_frozen_checkouts,
    render_submit,
    run_status,
)
from adversariallm.jobs.launch import write_record
from adversariallm.jobs.spec import (
    expand,
    format_overrides,
    get_profile,
    injected_overrides,
    load_experiment,
    load_profiles,
)

REPO = Path(__file__).resolve().parent


def experiment_path(name: str) -> Path:
    path = Path(name)
    return path if path.suffix == ".yaml" else REPO / "experiments" / f"{name}.yaml"


def select_runs(runs, only: list[str] | None, store: RunStore, queue, rerun: bool):
    """(runs to submit, [(name, reason)] skipped)."""
    chosen, skipped = [], []
    for run in runs:
        if only and not any(fnmatch.fnmatch(run.name, pattern) for pattern in only):
            continue
        latest = store.latest(run.exp, run.name)
        status = run_status(latest[1], queue)[0] if latest else "new"
        if not rerun and (status == "done" or status in ACTIVE):
            skipped.append((run.name, status))
        else:
            chosen.append(run)
    return chosen, skipped


def build_record(run, profile, attempt_dir: Path, site: dict, workdir: Path, commit: str, limits, res, attempt: int):
    injected = injected_overrides(profile, attempt_dir, attempt)
    env = dict(site.get("env", {}))
    if profile.uses_db:
        env["ADVLLM_DB"] = f"file:{site['db_root']}"
    return {
        "entrypoint": run.entrypoint,
        "args": [*format_overrides(run.overrides), *format_overrides(injected)],
        "overrides": run.overrides,
        "injected_overrides": injected,
        "config_hash": run.config_hash,
        "artifacts": list(run.artifacts),
        "time_per_run": run.time_per_run,
        "env": env,
        "workdir": str(workdir),
        "commit": commit,
        "submitted_at": now_iso(),
        "bid": limits.bid,
        "max_price": limits.max_price,
        "resources": {"cpus": res.cpus, "memory_gb": res.memory_gb, "gpus": res.gpus, "gpu": res.gpu},
    }


def cmd_submit(args) -> int:
    site = load_site(REPO)
    spec = load_experiment(experiment_path(args.experiment))
    profile = get_profile(load_profiles(REPO), spec.entrypoint)
    runs = expand(spec, profile.name_key)

    gpus = site["resources"]["gpus"] if args.gpus is None else args.gpus
    gpu = (args.gpu or site["default_gpu"]) if gpus else None
    if gpu is not None and gpu not in site["gpus"]:
        sys.exit(f"unknown --gpu {gpu!r}; known: {', '.join(site['gpus'])}")
    res = Resources(cpus=args.cpus or site["resources"]["cpus"],
                    memory_gb=args.mem_gb or site["resources"]["memory_gb"],
                    gpus=gpus, disk_gb=args.disk_gb or site["resources"]["disk_gb"], gpu=gpu)
    min_bid = site["gpus"][gpu]["min_bid"] if gpu else 1
    if args.bid < min_bid:
        sys.exit(f"--bid {args.bid:g} is below the minimum bid {min_bid} of {gpu}")
    time_per_run = spec.time_per_run or site["default_time_per_run"]
    limits = Limits(bid=args.bid, max_price=args.max_price or 2 * args.bid, price_action=args.price_action,
                    max_hours=args.max_hours or 2 * hours(time_per_run), max_parallel=args.max_parallel)
    if limits.max_price < limits.bid:
        sys.exit("--max-price below --bid: the job would be stopped as soon as the price reaches the bid")

    store = RunStore(site["jobs_root"])
    queue = condor_queue()
    chosen, skipped = select_runs(runs, args.only, store, queue, args.rerun)

    print(f"experiment {spec.exp}: {len(runs)} run(s), entrypoint {spec.entrypoint}, {time_per_run} per run")
    for name, status in skipped:
        print(f"  skip   {name} ({status}; --rerun to submit again)")
    for run in chosen:
        print(f"  submit {run.name}")
    if not chosen:
        print("nothing to submit")
        return 0
    print(f"resources: {gpu or 'CPU only'}, runtime limit {limits.max_hours:g} h per start, "
          f"price limit {limits.max_price:g} ({limits.price_action})")
    for line in cost_lines(len(chosen), res, limits, hours(time_per_run), min_bid):
        print("  " + line)

    if not args.submit:
        example = chosen[0]
        print("\ndry run: nothing written. Example command:\n  "
              + " ".join([example.entrypoint, *format_overrides(example.overrides)]))
        print("\nsubmit file:\n" + render_submit(site, Path(site["jobs_root"]) / "_code" / "<commit>",
                                                Path(site["jobs_root"]) / spec.exp / "_submit" / "<time>",
                                                spec.exp, res, limits))
        return 0

    if shutil.which("condor_submit_bid") is None:
        sys.exit("condor_submit_bid not found: --submit only works on a login node")
    if REPO != Path(site["repo"]).resolve():
        sys.exit(f"run this from the checkout named in {SITE_FILE}: {site['repo']}")
    dirty = git(REPO, "status", "--porcelain", "--untracked-files=no")
    if dirty and not args.allow_dirty:
        sys.exit(f"uncommitted changes (jobs run the committed code):\n{dirty}\ncommit, or pass --allow-dirty")
    commit = git(REPO, "rev-parse", "HEAD")
    print_pruned(prune_frozen_checkouts(site, store, queue, keep=commit))
    workdir = frozen_checkout(site, commit)

    stamp = now_iso().replace(":", "")
    submit_dir = Path(site["jobs_root"]) / spec.exp / "_submit" / stamp
    submit_dir.mkdir(parents=True)
    (submit_dir / "experiment.yaml").write_text(yaml.safe_dump(spec.to_dict(), sort_keys=False))
    attempt_dirs = []
    for run in chosen:
        attempt, attempt_dir = store.next_attempt(spec.exp, run.name)
        record = build_record(run, profile, attempt_dir, site, workdir, commit, limits, res, attempt)
        attempt_dirs.append(store.create_attempt(spec.exp, run.name, attempt, record))
    (submit_dir / "runs.txt").write_text("".join(f"{d}\n" for d in attempt_dirs))
    (submit_dir / "job.sub").write_text(render_submit(site, workdir, submit_dir, spec.exp, res, limits))

    out = subprocess.run(["condor_submit_bid", f"{args.bid:g}", "job.sub"], cwd=submit_dir,
                         capture_output=True, text=True, check=False)
    print(out.stdout.strip())
    if out.returncode != 0:
        print(out.stderr.strip(), file=sys.stderr)
        sys.exit(f"condor_submit_bid failed; attempts were created but not submitted ({submit_dir})")
    cluster = parse_cluster_id(out.stdout)
    for proc, attempt_dir in enumerate(attempt_dirs):
        record = store.latest(spec.exp, attempt_dir.parent.parent.name)[1]
        record["condor_job"] = f"{cluster}.{proc}"
        write_record(attempt_dir, record)
    print(f"cluster {cluster}: {len(attempt_dirs)} job(s), files in {submit_dir}")
    return 0


def print_pruned(result) -> None:
    for path, outcome in result:
        print(f"frozen copy {path.name}: {outcome}")


def cmd_prune(args) -> int:
    site = load_site(REPO)
    queue = condor_queue()
    if queue is None:
        sys.exit("condor_q not available: cannot tell which frozen copies are still in use")
    result = prune_frozen_checkouts(site, RunStore(site["jobs_root"]), queue)
    print_pruned(result)
    if not result:
        print("nothing to remove")
    return 0


def cmd_status(args) -> int:
    site = load_site(REPO)
    store = RunStore(site["jobs_root"])
    queue = condor_queue()
    if queue is None:
        print("(condor_q not available: status from run.json only)")
    for exp in [args.experiment] if args.experiment else store.experiments():
        counts, rows = {}, []
        for run in store.runs(exp):
            attempt, record = store.latest(exp, run)
            status, detail = run_status(record, queue)
            counts[status] = counts.get(status, 0) + 1
            rows.append((run, attempt, status, detail, record.get("condor_job", "")))
        print(f"{exp}: " + ", ".join(f"{n} {s}" for s, n in sorted(counts.items())))
        if args.verbose:
            for run, attempt, status, detail, job in rows:
                print(f"  {run:40s} attempt {attempt:<3d} {status:13s} {job:12s} {detail}")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("submit", help="submit the runs of one experiment (dry run without --submit)")
    p.add_argument("experiment", help="name in experiments/ or a path to a yaml file")
    p.add_argument("--bid", type=float, required=True, help="bid per computing unit and hour (queue position)")
    p.add_argument("--max-price", type=float, help="price limit per unit and hour (default: 2 x bid)")
    p.add_argument("--price-action", choices=["kill", "restart"], default="kill",
                   help="at the price limit: end the job (kill) or put it back in the queue (restart, only for "
                        "runs that resume from a checkpoint)")
    p.add_argument("--max-hours", type=float, help="runtime limit per start (default: 2 x time_per_run)")
    p.add_argument("--max-parallel", type=int, help="at most this many jobs of the experiment at once")
    p.add_argument("--gpu", help="GPU type from conf/jobs/mpi.yaml (default there: default_gpu)")
    p.add_argument("--gpus", type=int, help="GPUs per job (0: CPU only)")
    p.add_argument("--cpus", type=int)
    p.add_argument("--mem-gb", type=int)
    p.add_argument("--disk-gb", type=int)
    p.add_argument("--only", nargs="+", help="run names or globs")
    p.add_argument("--rerun", action="store_true", help="also submit runs that are done or still queued")
    p.add_argument("--allow-dirty", action="store_true", help="submit with uncommitted changes (they are not used)")
    p.add_argument("--submit", action="store_true", help="really submit (default: dry run)")
    p.set_defaults(func=cmd_submit)

    p = sub.add_parser("status", help="status of the runs, from run.json and condor_q")
    p.add_argument("experiment", nargs="?")
    p.add_argument("-v", "--verbose", action="store_true")
    p.set_defaults(func=cmd_status)

    p = sub.add_parser("prune", help="remove frozen code copies that no queued or running run uses")
    p.set_defaults(func=cmd_prune)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
