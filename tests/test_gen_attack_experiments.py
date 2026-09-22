import pytest

import yaml

from gen_attack_experiments import build, main, shard_bounds


def test_shards_partition_the_behavior_set_without_gaps_or_overlap():
    bounds = shard_bounds(100, 4)
    assert bounds == [(0, 25), (25, 50), (50, 75), (75, 100)]
    assert shard_bounds(10, 4) == [(0, 3), (3, 6), (6, 9), (9, 10)]  # last shard is short, never past the end


def test_one_file_per_shard_sweeping_models():
    files = build(["gcg"], ["base", "E-nd6-s0"], shards=2, n_behaviors=100, defense=None)
    assert sorted(files) == ["attack-gcg-b0-50.yaml", "attack-gcg-b50-100.yaml"]
    spec = files["attack-gcg-b50-100.yaml"]
    assert spec["entrypoint"] == "run_attacks.py"
    assert spec["overrides"]["datasets.jbb_behaviors.idx"] == "list(range(50,100))"
    assert spec["sweep"] == {"model": ["base", "E-nd6-s0"]}
    assert spec["name"] == "gcg-{model}-b50-100"
    assert "defense" not in spec["overrides"]


def test_defense_goes_into_the_overrides_and_the_file_name():
    # pair, not gcg: an optimisation attack cannot run against a runtime defense
    files = build(["pair"], ["E-nd6-s0"], shards=1, n_behaviors=4, defense="coop_probe")
    assert list(files) == ["attack-pair-coop_probe-b0-4.yaml"]
    assert files["attack-pair-coop_probe-b0-4.yaml"]["overrides"]["defense"] == "coop_probe"


def test_main_writes_the_files(tmp_path):
    (tmp_path / "experiments").mkdir()
    assert main(["--attacks", "inpainting", "--models", "base", "--shards", "2"], repo=tmp_path) == 0
    written = sorted(p.name for p in (tmp_path / "experiments").iterdir())
    assert written == ["attack-inpainting-b0-50.yaml", "attack-inpainting-b50-100.yaml"]
    spec = yaml.safe_load((tmp_path / "experiments" / "attack-inpainting-b0-50.yaml").read_text())
    assert spec["time_per_run"] == "03:00:00" and spec["overrides"]["attack"] == "inpainting"


def test_inpainting_gets_the_agreed_128_generation_budget():
    spec = build(["inpainting"], ["E-nd6-s0"], shards=1, n_behaviors=20, defense="coop_probe")[
        "attack-inpainting-coop_probe-b0-20.yaml"]
    assert spec["overrides"]["attacks.inpainting.num_samples_per_behavior"] == 128




def test_pair_judges_with_the_attacker_not_the_target():
    spec = build(["pair"], ["E-nd6-s0"], shards=1, n_behaviors=20, defense="coop_probe")[
        "attack-pair-coop_probe-b0-20.yaml"]
    # null would mean the target judges itself -- a different judge per arm
    assert spec["overrides"]["attacks.pair.judge_model.id"] == "lmsys/vicuna-13b-v1.5"
    assert spec["overrides"]["attacks.pair.judge_model.id"] == spec["overrides"].get(
        "attacks.pair.judge_model.tokenizer_id")


def test_run_names_carry_the_behavior_window_so_protocols_cannot_collide():
    """b0 alone meant 0-24 in the 100-behavior batch and 0-19 in the 20-behavior one: submitting the
    second was silently skipped because a run of that name already existed."""
    wide = build(["pair"], ["E-nd6-s0"], shards=4, n_behaviors=100, defense=None)["attack-pair-b0-25.yaml"]
    narrow = build(["pair"], ["E-nd6-s0"], shards=1, n_behaviors=20, defense=None)["attack-pair-b0-20.yaml"]
    assert wide["name"] != narrow["name"]
    assert (wide["name"], narrow["name"]) == ("pair-{model}-b0-25", "pair-{model}-b0-20")


def test_the_detector_checkpoint_override_is_parseable_by_hydra():
    """A bare nested interpolation fails at startup with "extraneous input '}' expecting <EOF>",
    which is how the first adaptive-GCG batch died."""
    parser = pytest.importorskip("hydra.core.override_parser.overrides_parser").OverridesParser.create()
    spec = build(["gcg_adaptive"], ["M-respmean-s0"], shards=1, n_behaviors=20,
                 defense=None)["attack-gcg_adaptive-b0-20.yaml"]
    key = "attacks.gcg_adaptive.detector_checkpoint"
    parsed = parser.parse_overrides([f"{key}={spec['overrides'][key]}"])[0]
    assert parsed.value() == "${models.${model}.reader_path}"  # quotes consumed, interpolation intact


def test_an_optimisation_attack_against_a_runtime_defense_is_refused_at_generation_time():
    """gcg+coop_probe submitted fine and then died in every run with 'Attack gcg is incompatible
    with runtime defenses'. The generator now refuses it before any job is created."""
    with pytest.raises(ValueError, match="replay"):
        build(["gcg"], ["M-respmean-s0"], shards=1, n_behaviors=20, defense="coop_probe")
    with pytest.raises(ValueError):
        build(["gcg_adaptive"], ["M-respmean-s0"], shards=1, n_behaviors=20, defense="coop_probe")
    # the black-box attacks are still allowed against the pipeline
    assert build(["pair", "inpainting", "direct"], ["M-respmean-s0"], shards=1, n_behaviors=20,
                 defense="coop_probe")



def test_adaptive_gcg_is_its_own_attack_with_its_own_results_dir():
    """Run outputs are filed under outputs/<attack>__<defense>__<model>/, so adaptive GCG must be a
    distinct attack NAME or its results merge into the vanilla gcg cell and the ASRs average."""
    files = build(["gcg", "gcg_adaptive"], ["M-respmean-s0"], shards=1, n_behaviors=20, defense=None)
    assert sorted(files) == ["attack-gcg-b0-20.yaml", "attack-gcg_adaptive-b0-20.yaml"]
    vanilla = files["attack-gcg-b0-20.yaml"]["overrides"]
    adaptive = files["attack-gcg_adaptive-b0-20.yaml"]["overrides"]
    assert vanilla["attack"] == "gcg" and adaptive["attack"] == "gcg_adaptive"
    assert "attacks.gcg.detector_checkpoint" not in vanilla  # vanilla never sees the probe
    assert adaptive["attacks.gcg_adaptive.detector_checkpoint"]


def test_adaptive_gcg_cannot_be_run_against_the_runtime_defense_either():
    with pytest.raises(ValueError, match="replay"):
        build(["gcg_adaptive"], ["M-respmean-s0"], shards=1, n_behaviors=20, defense="coop_probe")
