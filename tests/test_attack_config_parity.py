"""gcg_adaptive must expose exactly gcg's knobs, so the two attacks stay comparable.

The first version of the config was copied from gcg_refusal and interpolated keys gcg does not
have (max_new_tokens, use_mellowmax, ...), which blew up at compose time with
"Interpolation key 'attacks.gcg.max_new_tokens' not found".
"""

import pytest

yaml = pytest.importorskip("yaml")

from pathlib import Path  # noqa: E402

CONF = Path(__file__).resolve().parents[1] / "conf" / "attacks" / "attacks.yaml"
PROBE_ONLY = {"detector_checkpoint", "detector_loss_coeff", "detector_layer"}


def _attacks():
    return yaml.safe_load(CONF.read_text())


def test_gcg_adaptive_has_the_same_knobs_as_gcg():
    attacks = _attacks()
    gcg, adaptive = set(attacks["gcg"]), set(attacks["gcg_adaptive"])
    assert adaptive == gcg, f"only in gcg: {gcg - adaptive}; only in gcg_adaptive: {adaptive - gcg}"


def test_gcg_adaptive_inherits_every_search_knob_from_gcg():
    adaptive = _attacks()["gcg_adaptive"]
    for key, value in adaptive.items():
        if key in PROBE_ONLY | {"name", "version"}:
            continue
        assert value == "${attacks.gcg.%s}" % key, f"{key} is not inherited from gcg: {value!r}"


def test_gcg_adaptive_turns_the_probe_terms_on():
    adaptive = _attacks()["gcg_adaptive"]
    assert adaptive["name"] == "gcg_adaptive"
    assert adaptive["detector_loss_coeff"] == 0.5          # gcg defaults this to 0 = probe ignored
    # resolves the selected model's probe; a CLI override cannot express this (nested interpolation)
    assert adaptive["detector_checkpoint"] == "${models.${model}.reader_path}"
