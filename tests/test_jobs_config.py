"""Job config: the MPI site file, launch profiles vs Hydra configs, DB switch."""

from pathlib import Path

import pytest

from adversariallm.jobs.condor import load_site
from adversariallm.jobs.spec import load_profiles, parse_duration

REPO = Path(__file__).resolve().parents[1]
EXPECTED_ENTRYPOINTS = {
    "run_coop_train.py", "run_train.py", "run_attacks.py", "run_judges.py", "run_sampling.py",
    "run_cross_attack_eval.py", "run_overrefusal_eval.py", "run_pretrain_probe.py", "run_calibrate_probe.py",
    "run_gate_diagnostics.py", "run_generate_benign_targets.py", "run_utility_eval.py",
    "probe_diagnose.py", "train_response_head.py", "inspect_embedding_attack.py", "build_answer_pool.py", "rescore_answer_probe.py", "rejudge_answers.py", "score_embedding_attack.py", "probe_prompt_conditions.py", "reread_monitor_eval.py", "offline_probe_study.py", "judge_flagged_raw.py", "score_prompt_probe.py",
}
_MISSING = object()


def _hydra_profiles():
    return sorted((e, p) for e, p in load_profiles(REPO).items() if p.config_name)


def test_mpi_site():
    site = load_site(REPO)
    assert site["repo"] == "/home/fgrob/Projects/coop-detect/AdversariaLLM"
    for key in ("jobs_root", "db_root"):  # run records and the file DB need POSIX locks: /home, never /fast
        assert site[key].startswith("/home/")
    assert site["env"]["HF_HOME"].startswith("/fast/") and site["env"]["HF_HUB_OFFLINE"] == "1"
    assert site["default_gpu"] in site["gpus"]
    assert parse_duration(site["default_time_per_run"]) == 4 * 3600


def test_all_entrypoints_have_profiles():
    assert {e for e, _ in _hydra_profiles()} == EXPECTED_ENTRYPOINTS


@pytest.mark.parametrize("entrypoint,profile", _hydra_profiles(), ids=lambda x: x if isinstance(x, str) else "")
def test_profile_matches_hydra_config(entrypoint, profile):
    hydra = pytest.importorskip("hydra")
    from omegaconf import OmegaConf

    assert f'config_name="{profile.config_name}"' in (REPO / entrypoint).read_text()
    with hydra.initialize_config_dir(config_dir=str(REPO / "conf"), version_base="1.3"):
        cfg = hydra.compose(config_name=profile.config_name)
    keys = [k for k in (profile.name_key, profile.wandb_dir_key) if k]
    keys += list(profile.extra_overrides) + list(profile.rerun_overrides)
    for key in keys:
        assert OmegaConf.select(cfg, key, default=_MISSING, throw_on_resolution_failure=False) is not _MISSING, key


def test_get_mongodb_connection_uses_file_db(monkeypatch, tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("pymongo")
    monkeypatch.setenv("ADVLLM_DB", f"file:{tmp_path}")
    from adversariallm.io_utils import get_mongodb_connection
    from adversariallm.io_utils.filedb import FileDatabase

    db = get_mongodb_connection()
    assert isinstance(db, FileDatabase) and db.root == tmp_path
