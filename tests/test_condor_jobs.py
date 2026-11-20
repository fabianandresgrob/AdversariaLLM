"""condor_jobs.py and adversariallm/jobs: run selection, submit file, cost, status, launcher, frozen code."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from adversariallm.jobs.condor import (
    Limits,
    Resources,
    RunStore,
    cost_lines,
    frozen_checkout,
    load_site,
    parse_cluster_id,
    prune_frozen_checkouts,
    render_submit,
    run_status,
)
from adversariallm.jobs.spec import (
    expand,
    get_profile,
    injected_overrides,
    load_experiment,
    load_profiles,
    parse_duration,
)

REPO = Path(__file__).resolve().parents[1]
RES = Resources(cpus=8, memory_gb=64, gpus=1, disk_gb=50, gpu="a100-80")
LIMITS = Limits(bid=25, max_price=50, price_action="kill", max_hours=24, max_parallel=None)


def test_units_follow_the_banking_rule():
    assert RES.units == 32                                    # max(8, 64/16) + 24
    assert Resources(2, 128, 0, 10, None).units == 8          # memory dominates: 128/16
    assert Resources(1, 16, 2, 10, "a100-80").units == 49


def test_cost_lines():
    lines = cost_lines(3, RES, LIMITS, time_per_run_h=12, min_bid=25)
    text = "\n".join(lines)
    assert "800 at the bid" in text and "at most 1,600" in text   # per job and hour
    assert "2,400 at the bid, at most 4,800" in text               # all jobs per hour
    assert "28,800" in text and "57,600" in text                   # whole run
    assert "submission fee: 7.5" in text


def test_submit_file_has_resources_and_all_limits(tmp_path):
    site = load_site(REPO)
    text = render_submit(site, tmp_path / "code", tmp_path / "sub", "coop-W-coop", RES,
                         Limits(bid=25, max_price=40, price_action="restart", max_hours=6, max_parallel=4))
    assert 'requirements   = TARGET.CUDADeviceName == "NVIDIA A100-SXM4-80GB"' in text
    assert "request_memory = 65536" in text and "request_gpus   = 1" in text
    assert "+MaxRunningPrice = 40" in text and '+RunningPriceExceededAction = "restart"' in text
    assert "MaxTime = 21600" in text
    assert "on_exit_hold = (ExitCode =?= 3)" in text
    assert "concurrency_limits = user.coop_W_coop:2500" in text
    assert "$$([NumJobStarts])" in text
    assert text.rstrip().endswith("queue attempt_dir from runs.txt")


def test_cpu_job_has_no_gpu_lines(tmp_path):
    text = render_submit(load_site(REPO), tmp_path, tmp_path, "x", Resources(4, 16, 0, 10, None), LIMITS)
    assert "request_gpus" not in text and "CUDADeviceName" not in text and "concurrency_limits" not in text


def test_parse_cluster_id():
    assert parse_cluster_id("Submitting job(s)...\n3 job(s) submitted to cluster 12345.\n") == 12345
    with pytest.raises(RuntimeError):
        parse_cluster_id("ERROR: something")


@pytest.mark.parametrize("record,queue,expected", [
    ({}, {}, "not submitted"),
    ({"condor_job": "7.0"}, {"7.0": ("idle", "")}, "idle"),
    ({"condor_job": "7.0"}, {"7.0": ("held", "Job runtime exceeded")}, "held"),
    ({"condor_job": "7.0", "starts": [{}], "exit_code": 3}, {"7.0": ("idle", "")}, "requeued"),
    ({"condor_job": "7.0", "starts": [{}]}, {"7.0": ("running", "")}, "running"),
    ({"condor_job": "7.0", "starts": [{}]}, {}, "lost"),
    ({"condor_job": "7.0"}, {}, "removed"),
    ({"condor_job": "7.0"}, None, "submitted"),
    ({"condor_job": "7.0", "exit_code": 0, "missing_artifacts": []}, {}, "done"),
    ({"condor_job": "7.0", "exit_code": 0, "missing_artifacts": ["a.pt"]}, {}, "failed"),
    ({"condor_job": "7.0", "exit_code": 1}, {}, "failed"),
    ({"condor_job": "7.0", "exit_code": 3}, {}, "checkpointed"),
])
def test_run_status(record, queue, expected):
    assert run_status(record, queue)[0] == expected


def test_experiment_expands_with_profile_name_key():
    spec = load_experiment(REPO / "experiments" / "coop-W-coop.yaml")
    profile = get_profile(load_profiles(REPO), spec.entrypoint)
    runs = expand(spec, profile.name_key)
    assert [r.name for r in runs] == ["W-coop-s0", "W-coop-s1", "W-coop-s2"]
    assert runs[1].overrides["seed"] == 1 and runs[1].overrides["name"] == "W-coop-s1"
    assert parse_duration(spec.time_per_run) == 12 * 3600
    injected = injected_overrides(profile, Path("/runs/a/1"), attempt=2)
    assert injected["hydra.run.dir"] == "/runs/a/1/hydra" and injected["wandb.dir"] == "/runs/a/1/wandb"
    assert injected["output.overwrite"] is True  # rerun_overrides from attempt 2 on


def test_select_runs_skips_done_and_queued(tmp_path):
    sys.path.insert(0, str(REPO))
    from condor_jobs import select_runs

    spec = load_experiment(REPO / "experiments" / "coop-W-coop.yaml")
    runs = expand(spec, "name")
    store = RunStore(tmp_path)
    store.create_attempt(spec.exp, "W-coop-s0", 1, {"condor_job": "5.0", "exit_code": 0, "missing_artifacts": []})
    store.create_attempt(spec.exp, "W-coop-s1", 1, {"condor_job": "5.1"})
    queue = {"5.1": ("running", "")}
    chosen, skipped = select_runs(runs, None, store, queue, rerun=False)
    assert [r.name for r in chosen] == ["W-coop-s2"]
    assert dict(skipped) == {"W-coop-s0": "done", "W-coop-s1": "running"}
    chosen, _ = select_runs(runs, ["W-coop-s0"], store, queue, rerun=True)
    assert [r.name for r in chosen] == ["W-coop-s0"]


def test_run_store_attempts_and_latest_link(tmp_path):
    store = RunStore(tmp_path)
    assert store.next_attempt("e", "r")[0] == 1
    store.create_attempt("e", "r", 1, {"a": 1})
    store.create_attempt("e", "r", 2, {"a": 2})
    assert store.latest("e", "r")[0] == 2 and store.latest("e", "r")[1]["a"] == 2
    assert os.readlink(tmp_path / "e" / "r" / "latest") == "attempts/2"
    assert store.experiments() == ["e"] and store.runs("e") == ["r"]
    with pytest.raises(FileExistsError):
        store.create_attempt("e", "r", 2, {})


def _attempt(tmp_path, script: str, artifacts=()):
    workdir = tmp_path / "work"
    workdir.mkdir(parents=True)
    (workdir / "entry.py").write_text(script)
    attempt_dir = tmp_path / "attempt"
    attempt_dir.mkdir(parents=True)
    (attempt_dir / "run.json").write_text(json.dumps({
        "exp": "e", "run": "r", "attempt": 1, "entrypoint": "entry.py", "args": ["x=1", "name=r"],
        "workdir": str(workdir), "env": {"SOFTFILELOCK": "1", "MY_VAR": "set"}, "artifacts": list(artifacts)}))
    return workdir, attempt_dir


def _launch(attempt_dir):
    return subprocess.run([sys.executable, "-m", "adversariallm.jobs.launch", str(attempt_dir)], cwd=REPO,
                          capture_output=True, text=True, check=False)


def test_launcher_runs_the_entrypoint_and_records_the_start(tmp_path):
    workdir, attempt_dir = _attempt(tmp_path, (
        "import os, sys, filelock\n"
        "assert filelock.FileLock is filelock.SoftFileLock\n"
        "open('out.txt', 'w').write(' '.join(sys.argv[1:]) + '|' + os.environ['PWD'] + '|' + os.environ['MY_VAR'])\n"
    ), artifacts=["out.txt"])
    out = _launch(attempt_dir)
    assert out.returncode == 0, out.stderr
    assert (workdir / "out.txt").read_text() == f"x=1 name=r|{workdir}|set"
    record = json.loads((attempt_dir / "run.json").read_text())
    assert record["exit_code"] == 0 and record["missing_artifacts"] == []
    assert len(record["starts"]) == 1 and record["starts"][0]["exit_code"] == 0


def test_launcher_passes_exit_code_3_through_and_counts_starts(tmp_path):
    _, attempt_dir = _attempt(tmp_path, "import sys\nsys.exit(3)\n")
    assert _launch(attempt_dir).returncode == 3
    assert _launch(attempt_dir).returncode == 3
    record = json.loads((attempt_dir / "run.json").read_text())
    assert [s["exit_code"] for s in record["starts"]] == [3, 3] and record["exit_code"] == 3


def test_launcher_records_a_crash_and_missing_artifacts(tmp_path):
    _, attempt_dir = _attempt(tmp_path, "raise RuntimeError('boom')\n")
    out = _launch(attempt_dir)
    assert out.returncode != 0 and "boom" in out.stderr
    assert json.loads((attempt_dir / "run.json").read_text())["exit_code"] == 1

    _, attempt_dir = _attempt(tmp_path / "b", "pass\n", artifacts=["never.pt"])
    assert _launch(attempt_dir).returncode == 0
    record = json.loads((attempt_dir / "run.json").read_text())
    assert record["missing_artifacts"] == ["never.pt"] and run_status(record, {})[0] == "failed"


def _repo_with_commits(tmp_path, n):
    repo = tmp_path / "repo"
    repo.mkdir()
    git = lambda *a: subprocess.run(["git", "-C", str(repo), *a], check=True, capture_output=True, text=True)
    git("init", "-q")
    commits = []
    for i in range(n):
        (repo / "code.py").write_text(f"v{i}\n")
        git("add", "code.py")
        git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", f"v{i}")
        commits.append(git("rev-parse", "HEAD").stdout.strip())
    return repo, commits


def test_prune_removes_only_unused_frozen_copies(tmp_path):
    repo, (old, used, current) = _repo_with_commits(tmp_path, 3)
    site = {"repo": str(repo), "jobs_root": str(tmp_path / "runs"), "links": []}
    paths = {c: frozen_checkout(site, c) for c in (old, used, current)}
    (paths[used] / "stray.txt").write_text("not ignored by git\n")
    store = RunStore(tmp_path / "runs")
    store.create_attempt("e", "done", 1, {"condor_job": "1.0", "workdir": str(paths[old]), "exit_code": 0})
    store.create_attempt("e", "queued", 1, {"condor_job": "2.0", "workdir": str(paths[used])})
    queue = {"2.0": ("idle", "")}

    assert prune_frozen_checkouts(site, store, None) == []            # no condor_q: never remove
    result = dict(prune_frozen_checkouts(site, store, queue, keep=current))
    assert result == {paths[old]: "removed"}                           # used and current are kept
    assert not paths[old].exists() and paths[used].exists() and paths[current].exists()

    queue = {}                                                         # the queued run is gone
    result = dict(prune_frozen_checkouts(site, store, queue))
    assert result[paths[current]] == "removed"
    assert result[paths[used]] != "removed" and paths[used].exists()   # untracked file: git refuses, copy kept


def test_frozen_checkout_is_a_worktree_with_the_fast_links(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    git = lambda *a: subprocess.run(["git", "-C", str(repo), *a], check=True, capture_output=True, text=True)
    git("init", "-q")
    (repo / "code.py").write_text("v1\n")
    git("add", "code.py")
    git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "v1")
    commit = git("rev-parse", "HEAD").stdout.strip()
    target = tmp_path / "fast" / "outputs"
    target.mkdir(parents=True)
    (repo / "outputs").symlink_to(target)
    (repo / "code.py").write_text("v2, not committed\n")

    site = {"repo": str(repo), "jobs_root": str(tmp_path / "runs"), "links": ["outputs", "checkpoints_cat"]}
    path = frozen_checkout(site, commit)
    assert path == tmp_path / "runs" / "_code" / commit[:12]
    assert (path / "code.py").read_text() == "v1\n"
    assert os.readlink(path / "outputs") == str(target)       # the link's target, not a resolved real path
    assert not (path / "checkpoints_cat").exists()             # absent in the checkout: no link
    assert frozen_checkout(site, commit) == path                # reused for the same commit


def test_dry_run_prints_cost_and_writes_nothing(tmp_path, capsys, monkeypatch):
    sys.path.insert(0, str(REPO))
    import condor_jobs

    monkeypatch.setattr(condor_jobs, "condor_queue", lambda: None)
    monkeypatch.setattr(condor_jobs, "load_site", lambda repo: {**load_site(REPO), "jobs_root": str(tmp_path)})
    assert condor_jobs.main(["submit", "coop-W-coop", "--bid", "25"]) == 0
    out = capsys.readouterr().out
    assert "submit W-coop-s2" in out and "800 at the bid" in out and "queue attempt_dir from runs.txt" in out
    assert list(tmp_path.iterdir()) == []


def test_bid_below_the_minimum_is_refused():
    sys.path.insert(0, str(REPO))
    import condor_jobs

    with pytest.raises(SystemExit, match="minimum bid"):
        condor_jobs.main(["submit", "coop-W-coop", "--bid", "10", "--gpu", "a100-80"])
