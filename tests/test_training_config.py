from __future__ import annotations

from adversariallm.training.loop import build_objective


def test_ce_objective_uses_away_toward_kl():
    obj = build_objective({"model_objective": "ce", "lambda_away": 1.0, "lambda_toward": 1.0, "lambda_kl": 1.0})
    assert set(obj.active_terms) == {"away", "toward", "kl"}
    assert obj.away_variant == "ce"


def test_ul_objective_sets_variant():
    obj = build_objective({"model_objective": "ul", "lambda_kl": 1.0})
    assert set(obj.active_terms) == {"away", "toward", "kl"}
    assert obj.away_variant == "ul"


def test_ipo_objective_drops_away_toward():
    obj = build_objective({"model_objective": "ipo", "beta": 0.1, "lambda_kl": 1.0})
    assert set(obj.active_terms) == {"ipo", "kl"}


def test_kl_can_be_disabled():
    obj = build_objective({"model_objective": "ce", "lambda_kl": 0.0})
    assert "kl" not in obj.active_terms
