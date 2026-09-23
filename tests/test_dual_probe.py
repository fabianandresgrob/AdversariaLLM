from __future__ import annotations

import math

import pytest
import torch

from adversariallm.training.readers import DualProbe, LinearProbe, build_reader, load_reader, window_max


def _batch(D=8, prompt=3, resp=5, pad=2, seed=0):
    torch.manual_seed(seed)
    T = prompt + resp + pad
    hidden = torch.randn(2, T, D)
    tgt = torch.zeros(2, T, dtype=torch.long)
    tgt[:, prompt:prompt + resp] = 7
    attn = torch.zeros(2, T, dtype=torch.long)
    attn[:, :prompt + resp] = 1
    return hidden, tgt, attn


def test_window_max_is_the_best_rolling_mean():
    values = torch.tensor([[0.0, 4.0, 0.0, 1.0, 1.0, 9.0]])
    mask = torch.tensor([[1, 1, 1, 1, 1, 0]])  # last position is padding: its 9 must not count
    assert window_max(values, mask, 2).item() == pytest.approx(2.0)   # (0+4)/2 and (4+0)/2
    assert window_max(values, mask, 1).item() == pytest.approx(4.0)


def test_window_max_on_a_short_row_is_its_mean_and_empty_row_is_minus_inf():
    values = torch.tensor([[1.0, 3.0, 0.0], [5.0, 5.0, 5.0]])
    mask = torch.tensor([[1, 1, 0], [0, 0, 0]])
    out = window_max(values, mask, 16)
    assert out[0].item() == pytest.approx(2.0)
    assert out[1].item() == -math.inf


def test_untrained_dual_scores_exactly_like_its_prompt_channel():
    # the response head starts at p ~ 0, so wrapping a probe changes nothing until it is trained
    hidden, tgt, attn = _batch()
    dual = DualProbe(8)
    assert torch.allclose(dual.p_harmful(hidden, tgt, attn), dual.prompt.p_harmful(hidden, tgt, attn), atol=1e-6)


def test_dual_is_the_max_of_its_channels_and_softmax_logits_agree():
    hidden, tgt, attn = _batch()
    dual = DualProbe(8, response_window=2)
    torch.nn.init.normal_(dual.response.weight)
    dual.response.bias.data.zero_()
    p_prompt = dual.prompt.p_harmful(hidden, tgt, attn)
    margin, has = dual.response_margin(hidden, tgt, attn)
    assert has.all()
    expected = torch.maximum(p_prompt, torch.sigmoid(margin))
    assert torch.allclose(dual.p_harmful(hidden, tgt, attn), expected, atol=1e-6)
    assert torch.allclose(torch.softmax(dual.logits(hidden, tgt, attn), -1)[:, 0], expected, atol=1e-5)


def test_a_benign_response_cannot_lower_the_prompt_score():
    hidden, tgt, attn = _batch()
    dual = DualProbe(8)
    dual.response.bias.data = torch.tensor([-50.0, 50.0])  # response channel: confidently benign
    assert (dual.p_harmful(hidden, tgt, attn) >= dual.prompt.p_harmful(hidden, tgt, attn) - 1e-6).all()


def test_prompt_only_rows_score_on_the_prompt_channel():
    hidden, _, attn = _batch()
    tgt = torch.zeros_like(attn)  # no response region
    dual = DualProbe(8)
    dual.response.bias.data = torch.tensor([50.0, -50.0])  # would fire if it read anything
    assert torch.allclose(dual.p_harmful(hidden, tgt, attn), dual.prompt.p_harmful(hidden, tgt, attn), atol=1e-6)


def test_dual_response_score_is_padding_invariant():
    hidden, tgt, attn = _batch(pad=0)
    dual = DualProbe(8, response_window=3)
    torch.nn.init.normal_(dual.response.weight)
    padded = torch.cat([hidden, torch.randn(2, 4, 8)], dim=1)
    tgt_p = torch.cat([tgt, torch.zeros(2, 4, dtype=torch.long)], dim=1)
    attn_p = torch.cat([attn, torch.zeros(2, 4, dtype=torch.long)], dim=1)
    assert torch.allclose(dual.p_harmful(hidden, tgt, attn), dual.p_harmful(padded, tgt_p, attn_p), atol=1e-6)


def test_load_reader_round_trips_both_types(tmp_path):
    hidden, tgt, attn = _batch()
    dual = DualProbe(8, response_window=4)
    torch.nn.init.normal_(dual.response.weight)
    path = tmp_path / "dual_reader.pt"
    torch.save({"reader": dual.state_dict(), "cfg": {"reader": {"type": "dual", "response_window": 4}}}, path)
    loaded = load_reader(str(path))
    assert isinstance(loaded, DualProbe) and loaded.response_window == 4
    assert torch.allclose(loaded.p_harmful(hidden, tgt, attn), dual.p_harmful(hidden, tgt, attn))

    linear = LinearProbe(8, readout="stream_last")
    path = tmp_path / "linear_reader.pt"
    torch.save({"reader": linear.state_dict(), "cfg": {"reader": {"type": "linear", "readout": "stream_last"}}}, path)
    loaded = load_reader(str(path))
    assert isinstance(loaded, LinearProbe) and loaded.readout_mode == "stream_last"
    bare = tmp_path / "bare.pt"
    torch.save(LinearProbe(8).state_dict(), bare)
    assert load_reader(str(bare)).readout_mode == "prompt_last"


def test_a_dual_probe_refuses_readout_overrides(tmp_path):
    path = tmp_path / "dual_reader.pt"
    torch.save({"reader": DualProbe(8).state_dict(), "cfg": {"reader": {"type": "dual"}}}, path)
    with pytest.raises(ValueError, match="dual"):
        load_reader(str(path), readout="stream_last")


def test_monitor_loads_a_dual_probe_and_generates_for_calibration(tmp_path):
    from adversariallm.defenses.monitors.linear_probe import LinearProbeMonitor

    path = tmp_path / "dual_reader.pt"
    torch.save({"reader": DualProbe(8).state_dict(), "cfg": {"reader": {"type": "dual"}}}, path)
    monitor = LinearProbeMonitor(checkpoint_path=str(path), target_model_id="x")
    assert monitor.reads_response(torch.nn.Linear(8, 8))
    assert isinstance(monitor._probe, DualProbe)


def test_gcg_detector_loader_accepts_a_dual_probe(tmp_path):
    from adversariallm.attacks.gcg import load_detector

    path = tmp_path / "dual_reader.pt"
    torch.save({"reader": DualProbe(8).state_dict(), "cfg": {"reader": {"type": "dual"}}}, path)
    assert isinstance(load_detector(str(path), torch.nn.Linear(8, 8)), DualProbe)


def test_build_reader_builds_a_dual():
    assert isinstance(build_reader({"type": "dual", "response_window": 5}, 8), DualProbe)
