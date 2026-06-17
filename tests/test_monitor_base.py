from __future__ import annotations

from adversariallm.defenses.monitors._activation_detector_model import get_chat_template


def test_get_chat_template_llama31_strings():
    first_user_msg, response_template, response_key, first_part, second_part = get_chat_template(
        "meta-llama/Llama-3.1-8B-Instruct"
    )
    # Fixed system prompt must be reproduced for activation parity.
    assert "You are a helpful assistant" in first_user_msg
    assert first_user_msg.endswith("{instruction}")
    assert response_key == "<|start_header_id|>assistant<|end_header_id|>\n\n"
    assert response_template == response_key + "{target}<|eot_id|>"
    assert second_part == "<|eot_id|>"


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
