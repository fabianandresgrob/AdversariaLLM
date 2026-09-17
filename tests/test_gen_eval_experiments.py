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


def _cat_checkpoint(repo, block, run):
    d = repo / "checkpoints_cat" / block / run
    (d / "final_adapter").mkdir(parents=True)
    return d


def test_cat_checkpoints_get_overrefusal_and_utility_but_no_calibration(repo):
    main([str(_cat_checkpoint(repo, "J-cat", "J-ce-away1-s0"))], repo=repo)
    names = sorted(p.name for p in (repo / "experiments").iterdir())
    assert names == ["eval-overrefusal-J-cat.yaml", "eval-overrefusal-reference.yaml", "eval-utility-J-cat.yaml",
                     "eval-utility-reference.yaml"]
    run = expand(load_experiment(repo / "experiments" / "eval-utility-J-cat.yaml"))[0]
    assert run.overrides["adapter_path"] == "${root_dir}/checkpoints_cat/${block}/${run}/final_adapter"


def test_cross_attack_selects_runs_and_sweeps_attack_seeds(repo, monkeypatch):
    monkeypatch.setenv("PWD", "/proj")
    paths = [_checkpoint(repo, "A-eps-sweep", r) for r in ("A-eps0.05-s0", "A-eps0.05-s1", "A-eps1.0-s0")]
    main([str(p) for p in paths] + ["--cross-attack", "A-eps-sweep:A-eps0.05-*"], repo=repo)
    runs = expand(load_experiment(repo / "experiments" / "eval-crossattack-A-eps-sweep.yaml"))
    assert [r.name for r in runs] == [f"xattack-A-eps0.05-s{s}-a{a}" for s in (0, 1) for a in (0, 1, 2)]
    assert runs[1].artifacts == ("outputs/eval/cross_attack/A-eps-sweep/A-eps0.05-s0/seed1/cross_attack_eval.json",)
    with initialize_config_dir(config_dir=str(REPO / "conf"), version_base="1.3"):
        cfg = compose(config_name="cross_attack_eval", overrides=format_overrides(runs[1].overrides))
    assert cfg.adapter_path == "/proj/checkpoints_coop/A-eps-sweep/A-eps0.05-s0/final_adapter"
    assert list(cfg.eval_use_detector) == [False, True] and cfg.seed == 1
    assert cfg.out == "/proj/outputs/eval/cross_attack/A-eps-sweep/A-eps0.05-s0/seed1"


@pytest.mark.parametrize("spec,match", [("A-eps-sweep:nope-*", "matches no finished run"), ("Z-missing", "Z-missing")])
def test_cross_attack_errors(repo, spec, match):
    path = _checkpoint(repo, "A-eps-sweep", "A-eps0.05-s0")
    with pytest.raises(ValueError, match=match):
        main([str(path), "--cross-attack", spec], repo=repo)
