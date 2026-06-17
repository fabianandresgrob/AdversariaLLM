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
