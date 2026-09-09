from __future__ import annotations

import contextlib

import pytest
from omegaconf import OmegaConf

from adversariallm.attacks.gcg import DetectorAware, SubstitutionSelectionStrategy, _cfg


def _gcg_cfg():
    cfg = OmegaConf.merge(OmegaConf.load("conf/paths.yaml"), OmegaConf.load("conf/attacks/attacks.yaml"))
    return cfg.gcg


def test_attacks_yaml_declares_the_detector_knobs():
    # Attacks get the raw DictConfig, never an instantiated GCGConfig, so the dataclass
    # defaults do not apply at runtime -- a knob absent here raises ConfigAttributeError
    # in run() before the attack starts.
    cfg = _gcg_cfg()
    assert cfg.detector_checkpoint is None
    assert cfg.detector_loss_coeff == 0.0
    assert cfg.detector_layer == -1


def test_detector_gate_is_off_for_the_stock_config():
    cfg = _gcg_cfg()
    assert not (_cfg(cfg, "detector_checkpoint") and _cfg(cfg, "detector_loss_coeff", 0.0) > 0)


def test_cfg_falls_back_when_a_knob_is_absent():
    # A config written before detector-aware GCG must still run.
    old = OmegaConf.create({"name": "gcg", "num_steps": 2})
    OmegaConf.set_struct(old, True)
    assert _cfg(old, "detector_checkpoint") is None
    assert _cfg(old, "detector_loss_coeff", 0.0) == 0.0
    assert _cfg(old, "detector_layer", -1) == -1


@pytest.mark.parametrize("cls", [DetectorAware, SubstitutionSelectionStrategy])
def test_detector_hooks_reachable_on_both_users(cls):
    # The token-selection strategy calls _capture/_evasion_term on itself; when those lived
    # only on GCGAttack every GCG run died on step 0 with AttributeError.
    assert callable(getattr(cls, "_capture"))
    assert callable(getattr(cls, "_evasion_term"))


def test_strategy_hooks_are_noops_without_a_detector():
    strategy = SubstitutionSelectionStrategy(
        config=_gcg_cfg(),
        prefix_cache=None,
        pre_prompt_embeds=None,
        post_embeds=None,
        target_embeds=None,
        target_ids=None,
        not_allowed_ids=None,
        tokenizer=None,
    )
    assert strategy.detector is None
    # model is never touched on the no-detector path, so None is a fine stand-in
    with strategy._capture(None) as capture:
        assert capture is None or isinstance(capture, contextlib.nullcontext)
    assert strategy._evasion_term(None, 1, 1, None) is None
