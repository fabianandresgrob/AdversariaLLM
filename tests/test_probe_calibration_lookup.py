from __future__ import annotations

import json

import pytest

from adversariallm.defenses.monitor_defense import _calibrated_threshold


def _write_calibration(dirpath, fpr, threshold):
    # Mirrors run_calibrate_probe.py, which names its output for the FPR it calibrated at.
    path = dirpath / f"threshold_{int(float(fpr) * 100)}pct.json"
    path.write_text(json.dumps({"threshold": threshold, "fpr": fpr}))
    return path


def test_reads_the_calibrated_point(tmp_path):
    _write_calibration(tmp_path, 0.01, 0.0217)
    cfg = {"checkpoint_path": str(tmp_path / "final_reader.pt")}
    assert _calibrated_threshold(cfg, 0.5) == pytest.approx(0.0217)


def test_finds_a_non_default_fpr(tmp_path):
    # Calibrating at fpr=0.05 writes threshold_5pct.json; a reader hardcoded to
    # threshold_1pct.json silently fell back to 0.5, ~25x off the intended point.
    _write_calibration(tmp_path, 0.05, 0.0413)
    cfg = {"checkpoint_path": str(tmp_path / "final_reader.pt")}
    assert _calibrated_threshold(cfg, 0.5) == pytest.approx(0.0413)


def test_falls_back_when_nothing_is_calibrated(tmp_path):
    cfg = {"checkpoint_path": str(tmp_path / "final_reader.pt")}
    assert _calibrated_threshold(cfg, 0.5) == 0.5


def test_falls_back_without_a_checkpoint():
    assert _calibrated_threshold({}, 0.5) == 0.5
    assert _calibrated_threshold(None, 0.5) == 0.5


def test_ambiguous_calibration_raises(tmp_path):
    # Two operating points next to one checkpoint: picking either silently would hide
    # which one the run actually used.
    _write_calibration(tmp_path, 0.01, 0.0217)
    _write_calibration(tmp_path, 0.05, 0.0413)
    cfg = {"checkpoint_path": str(tmp_path / "final_reader.pt")}
    with pytest.raises(ValueError, match="calibration_path"):
        _calibrated_threshold(cfg, 0.5)


def test_explicit_path_wins_over_the_search(tmp_path):
    _write_calibration(tmp_path, 0.01, 0.0217)
    chosen = _write_calibration(tmp_path, 0.05, 0.0413)
    cfg = {"checkpoint_path": str(tmp_path / "final_reader.pt")}
    assert _calibrated_threshold(cfg, 0.5, str(chosen)) == pytest.approx(0.0413)


def test_missing_explicit_path_raises(tmp_path):
    cfg = {"checkpoint_path": str(tmp_path / "final_reader.pt")}
    with pytest.raises(FileNotFoundError):
        _calibrated_threshold(cfg, 0.5, str(tmp_path / "nope.json"))
