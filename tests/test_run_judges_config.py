"""Standalone run_judges.py invocation: the two things that only worked while run_attacks.py built
the judge config in-process (a missing filter_by key, and CLI suffixes arriving as ints)."""

from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]


def _fake_db(rows):
    class _Collection:
        def find(self):
            return rows

    return type("_DB", (), {"runs": _Collection()})()


def test_judge_config_defines_filter_by_so_the_entrypoint_runs_standalone():
    cfg = yaml.safe_load((REPO / "conf" / "judge.yaml").read_text())
    assert "filter_by" in cfg and cfg["filter_by"] is None


@pytest.mark.parametrize("suffixes", [[0, 1], ["0", "1"], 0])
def test_suffixes_work_whatever_type_the_cli_passes(monkeypatch, suffixes):
    """`suffixes=[0,1]` on the command line arrives as ints; endswith() rejects those."""
    run_judges = pytest.importorskip("run_judges")
    rows = [{"log_file": "/out/a/2026-09-19/10-00-00/3/run.json", "scored_by": []},
            {"log_file": "/out/a/2026-09-19/10-00-07/3/run.json", "scored_by": []}]
    monkeypatch.setattr(run_judges, "delete_orphaned_runs", lambda: None)
    monkeypatch.setattr(run_judges, "get_mongodb_connection", lambda: _fake_db(rows))
    # the suffix matches the timestamp dir two levels above run.json ("10-00-00" / "10-00-07")
    assert run_judges.collect_run_paths(suffixes, "strong_reject", None) == ["/out/a/2026-09-19/10-00-00/3/run.json"]


def test_runs_already_scored_by_this_classifier_are_skipped(monkeypatch):
    run_judges = pytest.importorskip("run_judges")
    rows = [{"log_file": "/out/a/2026-09-19/10-00-00/3/run.json", "scored_by": ["strong_reject"]},
            {"log_file": "/out/a/2026-09-19/20-00-00/3/run.json", "scored_by": ["harmbench"]}]
    monkeypatch.setattr(run_judges, "delete_orphaned_runs", lambda: None)
    monkeypatch.setattr(run_judges, "get_mongodb_connection", lambda: _fake_db(rows))
    assert run_judges.collect_run_paths([0], "strong_reject", None) == ["/out/a/2026-09-19/20-00-00/3/run.json"]
