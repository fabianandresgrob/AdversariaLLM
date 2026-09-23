"""AdversariaLLM's jsc-jobs project config: site values, launch profiles vs Hydra configs, DB switch."""

from pathlib import Path

import pytest

pytest.importorskip("jsc_jobs")

from jsc_jobs.profiles import load_profiles  # noqa: E402
from jsc_jobs.site import load_site, parse_duration  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
EXPECTED_ENTRYPOINTS = {
    "run_coop_train.py", "run_train.py", "run_attacks.py", "run_judges.py", "run_sampling.py",
    "run_cross_attack_eval.py", "run_overrefusal_eval.py", "run_pretrain_probe.py", "run_calibrate_probe.py",
    "run_gate_diagnostics.py", "run_generate_benign_targets.py", "run_utility_eval.py",
    "probe_diagnose.py", "train_response_head.py", "inspect_embedding_attack.py", "build_answer_pool.py",
}
_MISSING = object()


def _hydra_profiles():
    return sorted((e, p) for e, p in load_profiles(REPO).items() if p.config_name)


def test_jureca_site():
    site = load_site(REPO, "jureca")
    assert site.account == "hai_1370" and site.partition == "dc-hwai"
    assert site.gpus_per_node == 4 and site.cpus_per_slot == 16 and site.mem_per_slot == "120G"
    assert parse_duration(site.node_time) == 24 * 3600 and parse_duration(site.default_time_per_run) == 12 * 3600
    assert site.max_nodes_per_submit == 4
    assert site.env == {"WANDB_MODE": "offline", "HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1",
                        "TRANSFORMERS_OFFLINE": "1"}


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
    from jsc_jobs.filedb import FileDatabase

    db = get_mongodb_connection()
    assert isinstance(db, FileDatabase) and db.root == tmp_path
