from __future__ import annotations

import pytest

from adversariallm.defenses.monitors.base import Monitor, build_monitor, register_monitor


def test_register_and_build_monitor():
    @register_monitor
    class _ToyMonitor(Monitor):
        NAME = "toy"

        def score(self, prompts, responses, *, target_model, target_tokenizer):
            return [1.0 for _ in prompts]

        @classmethod
        def from_config(cls, cfg):
            return cls()

    monitor = build_monitor({"name": "toy"})
    assert isinstance(monitor, _ToyMonitor)
    assert monitor.score(["p"], ["r"], target_model=None, target_tokenizer=None) == [1.0]


def test_build_unknown_monitor_raises():
    with pytest.raises(ValueError, match="Unknown monitor"):
        build_monitor({"name": "does_not_exist"})
