import pytest

torch = pytest.importorskip("torch")


def test_gcg_adaptive_is_registered_and_requires_a_probe():
    from types import SimpleNamespace
    from adversariallm.attacks.attack import Attack
    from adversariallm.attacks.gcg_adaptive import GCGAdaptiveAttack

    assert Attack.from_name("gcg_adaptive") is GCGAdaptiveAttack

    # a missing probe or a zero coefficient would silently degrade to plain gcg under this name
    for cfg in (SimpleNamespace(seed=0, detector_checkpoint=None, detector_loss_coeff=0.5),
                SimpleNamespace(seed=0, detector_checkpoint="/tmp/probe.pt", detector_loss_coeff=0.0)):
        with pytest.raises(ValueError, match="gcg"):
            GCGAdaptiveAttack(cfg)
