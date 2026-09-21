import yaml

from gen_attack_experiments import build, main, shard_bounds


def test_shards_partition_the_behavior_set_without_gaps_or_overlap():
    bounds = shard_bounds(100, 4)
    assert bounds == [(0, 25), (25, 50), (50, 75), (75, 100)]
    assert shard_bounds(10, 4) == [(0, 3), (3, 6), (6, 9), (9, 10)]  # last shard is short, never past the end


def test_one_file_per_shard_sweeping_models():
    files = build(["gcg"], ["base", "E-nd6-s0"], shards=2, n_behaviors=100, defense=None)
    assert sorted(files) == ["attack-gcg-shard0.yaml", "attack-gcg-shard1.yaml"]
    spec = files["attack-gcg-shard1.yaml"]
    assert spec["entrypoint"] == "run_attacks.py"
    assert spec["overrides"]["datasets.jbb_behaviors.idx"] == "list(range(50,100))"
    assert spec["sweep"] == {"model": ["base", "E-nd6-s0"]}
    assert spec["name"] == "gcg-{model}-b50"
    assert "defense" not in spec["overrides"]


def test_defense_goes_into_the_overrides_and_the_file_name():
    files = build(["gcg"], ["E-nd6-s0"], shards=1, n_behaviors=4, defense="coop_probe")
    assert list(files) == ["attack-gcg-coop_probe-shard0.yaml"]
    assert files["attack-gcg-coop_probe-shard0.yaml"]["overrides"]["defense"] == "coop_probe"


def test_main_writes_the_files(tmp_path):
    (tmp_path / "experiments").mkdir()
    assert main(["--attacks", "inpainting", "--models", "base", "--shards", "2"], repo=tmp_path) == 0
    written = sorted(p.name for p in (tmp_path / "experiments").iterdir())
    assert written == ["attack-inpainting-shard0.yaml", "attack-inpainting-shard1.yaml"]
    spec = yaml.safe_load((tmp_path / "experiments" / "attack-inpainting-shard0.yaml").read_text())
    assert spec["time_per_run"] == "03:00:00" and spec["overrides"]["attack"] == "inpainting"


def test_inpainting_gets_the_agreed_128_generation_budget():
    spec = build(["inpainting"], ["E-nd6-s0"], shards=1, n_behaviors=20, defense="coop_probe")[
        "attack-inpainting-coop_probe-shard0.yaml"]
    assert spec["overrides"]["attacks.inpainting.num_samples_per_behavior"] == 128


def test_detector_aware_gcg_adds_the_evasion_objective_and_its_own_file_name():
    files = build(["gcg"], ["M-respmean-s0"], shards=1, n_behaviors=20, defense="coop_probe",
                  detector_aware=True)
    assert list(files) == ["attack-gcg-coop_probe-adaptive-shard0.yaml"]
    overrides = files["attack-gcg-coop_probe-adaptive-shard0.yaml"]["overrides"]
    # resolved per swept model out of models.yaml, so one file covers the whole sweep
    assert overrides["attacks.gcg.detector_checkpoint"] == "${models.${model}.reader_path}"
    assert overrides["attacks.gcg.detector_loss_coeff"] == 0.5


def test_detector_aware_leaves_other_attacks_alone():
    # inpainting and pair have nothing to be aware of -- they never see the probe's gradients
    files = build(["inpainting"], ["E-nd6-s0"], shards=1, n_behaviors=20, defense="coop_probe",
                  detector_aware=True)
    assert list(files) == ["attack-inpainting-coop_probe-shard0.yaml"]
    assert "attacks.gcg.detector_checkpoint" not in files["attack-inpainting-coop_probe-shard0.yaml"]["overrides"]
