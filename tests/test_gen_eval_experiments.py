import json

import pytest

pytest.importorskip("jsc_jobs")
pytest.importorskip("hydra")

from hydra import compose, initialize_config_dir  # noqa: E402
from jsc_jobs.spec import expand, format_overrides, load_experiment  # noqa: E402

from gen_eval_experiments import main  # noqa: E402
from gen_coop_models import REPO  # noqa: E402


def _checkpoint(repo, block, run, complete=True):
    d = repo / "checkpoints_coop" / block / run
    d.mkdir(parents=True)
    (d / "run_config.json").write_text(json.dumps({"name": run}))
    if complete:
        (d / "final_adapter").mkdir()
        (d / "final_reader.pt").write_bytes(b"x")
    return d


@pytest.fixture
def repo(tmp_path):
    (tmp_path / "experiments").mkdir()
    return tmp_path


def test_writes_one_file_per_eval_and_block(repo):
    paths = [_checkpoint(repo, "A-eps-sweep", "A-eps0.05-s0"), _checkpoint(repo, "A-eps-sweep", "A-eps0.0-s1"),
             _checkpoint(repo, "G-delta", "G-delta1-s0"), _checkpoint(repo, "G-delta", "G-delta1-s1", complete=False)]
    main([str(p) for p in paths], repo=repo)
    names = sorted(p.name for p in (repo / "experiments").iterdir())
    assert names == ["eval-calib-A-eps-sweep.yaml", "eval-calib-G-delta.yaml", "eval-overrefusal-A-eps-sweep.yaml",
                     "eval-overrefusal-G-delta.yaml", "eval-overrefusal-reference.yaml", "eval-utility-A-eps-sweep.yaml",
                     "eval-utility-G-delta.yaml", "eval-utility-reference.yaml"]
    calib = load_experiment(repo / "experiments" / "eval-calib-A-eps-sweep.yaml")
    runs = expand(calib)
    assert [r.name for r in runs] == ["calib-A-eps0.0-s1", "calib-A-eps0.05-s0"]
    assert runs[0].artifacts == ("checkpoints_coop/A-eps-sweep/A-eps0.0-s1/threshold_1pct_calib.json",)
    assert [r.name for r in expand(load_experiment(repo / "experiments" / "eval-calib-G-delta.yaml"))] == [
        "calib-G-delta1-s0"]


@pytest.mark.parametrize("config_name,file", [("calibrate_probe", "eval-calib-A-eps-sweep.yaml"),
                                               ("overrefusal", "eval-overrefusal-A-eps-sweep.yaml"),
                                               ("overrefusal", "eval-overrefusal-reference.yaml"),
                                               ("utility_eval", "eval-utility-A-eps-sweep.yaml"),
                                               ("utility_eval", "eval-utility-reference.yaml")])
def test_overrides_compose_against_real_configs(repo, monkeypatch, config_name, file):
    monkeypatch.setenv("PWD", "/proj")
    main([str(_checkpoint(repo, "A-eps-sweep", "A-eps0.05-s0"))], repo=repo)
    run = expand(load_experiment(repo / "experiments" / file))[0]
    with initialize_config_dir(config_dir=str(REPO / "conf"), version_base="1.3"):
        cfg = compose(config_name=config_name, overrides=format_overrides(run.overrides))
    ckpt = "/proj/checkpoints_coop/A-eps-sweep/A-eps0.05-s0"
    if config_name == "calibrate_probe":
        assert (cfg.adapter_path, cfg.checkpoint_path) == (ckpt + "/final_adapter", ckpt + "/final_reader.pt")
    elif file == "eval-utility-A-eps-sweep.yaml":
        assert cfg.adapter_path == ckpt + "/final_adapter"
        assert cfg.out == "/proj/outputs/eval/utility/A-eps-sweep/A-eps0.05-s0"
    elif file == "eval-utility-reference.yaml":
        assert cfg.adapter_path is None and cfg.out == "/proj/outputs/eval/utility/reference/base"
    elif file == "eval-overrefusal-reference.yaml":
        assert dict(cfg.checkpoints) == {"base": "base"} and cfg.out == "/proj/outputs/eval/overrefusal/reference/base"
    else:
        assert dict(cfg.checkpoints) == {"model": ckpt + "/final_adapter"}
        assert cfg.out == cfg.gens_cache_dir == "/proj/outputs/eval/overrefusal/A-eps-sweep/A-eps0.05-s0"
        assert cfg.judge.enabled is True
