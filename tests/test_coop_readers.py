from __future__ import annotations

import torch

from adversariallm.training.readers import LinearProbe, build_reader, readout_index


def test_readout_index_is_last_prompt_token():
    # prompt = 3 tokens (target_ids 0), response = 2 tokens (nonzero) -> last prompt = idx 2
    tgt = torch.tensor([[0, 0, 0, 5, 6]])
    am = torch.tensor([[1, 1, 1, 1, 1]])
    assert readout_index(tgt, am).tolist() == [2]


def test_readout_index_padding_invariant():
    # Sanity gate 2: the readout position (and the vector read) must not change under padding.
    tgt_unpad = torch.tensor([[0, 0, 0, 5, 6]])
    am_unpad = torch.tensor([[1, 1, 1, 1, 1]])
    tgt_pad = torch.tensor([[0, 0, 0, 5, 6, 0, 0]])   # two right-pad columns
    am_pad = torch.tensor([[1, 1, 1, 1, 1, 0, 0]])
    assert readout_index(tgt_pad, am_pad).tolist() == readout_index(tgt_unpad, am_unpad).tolist()

    torch.manual_seed(0)
    D = 8
    h_unpad = torch.randn(1, 5, D)
    h_pad = torch.cat([h_unpad, torch.zeros(1, 2, D)], dim=1)
    probe = LinearProbe(D)
    assert torch.allclose(
        probe.readout(h_unpad, tgt_unpad, am_unpad),
        probe.readout(h_pad, tgt_pad, am_pad),
    ), "readout changed under padding"


def test_prompt_only_falls_back_to_last_real_token():
    tgt = torch.tensor([[0, 0, 0, 0]])   # no response region
    am = torch.tensor([[1, 1, 1, 0]])    # last attended token = idx 2
    assert readout_index(tgt, am).tolist() == [2]


def test_readout_is_unit_norm_and_fp32_from_bf16():
    D = 8
    probe = LinearProbe(D)
    hidden = torch.randn(2, 5, D).to(torch.bfloat16)
    tgt = torch.tensor([[0, 0, 3, 4, 0], [0, 0, 0, 7, 8]])
    am = torch.tensor([[1, 1, 1, 1, 0], [1, 1, 1, 1, 1]])
    r = probe.readout(hidden, tgt, am)
    assert r.dtype == torch.float32
    assert torch.allclose(r.norm(dim=-1), torch.ones(2), atol=1e-5)


def test_logits_shape_and_label_convention():
    D = 8
    probe = build_reader({"type": "linear"}, D)
    hidden = torch.randn(1, 4, D)
    tgt = torch.tensor([[0, 0, 3, 4]])
    am = torch.tensor([[1, 1, 1, 1]])
    lg = probe.logits(hidden, tgt, am)
    assert lg.shape == (1, 2)
    # p_harmful is column 0 of the softmax
    assert torch.allclose(probe.p_harmful(hidden, tgt, am), torch.softmax(lg.float(), dim=-1)[:, 0])


def test_build_reader_rejects_unknown_type():
    import pytest
    with pytest.raises(ValueError):
        build_reader({"type": "mlp"}, 8)


# ---- readout modes: each one only changes WHICH position is read ---------------------------

def test_stream_last_index_is_the_last_real_token():
    from adversariallm.training.readers import stream_last_index

    tgt = torch.tensor([[0, 0, 0, 5, 6, 0], [0, 0, 7, 0, 0, 0]])
    am = torch.tensor([[1, 1, 1, 1, 1, 0], [1, 1, 1, 0, 0, 0]])
    assert stream_last_index(tgt, am).tolist() == [4, 2]


def test_response_mean_mask_covers_first_k_response_tokens():
    from adversariallm.training.readers import response_mean_mask

    # prompt = 2 tokens, response = 4 tokens, window k=2 -> response positions 2 and 3
    tgt = torch.tensor([[0, 0, 5, 6, 7, 8]])
    am = torch.ones(1, 6, dtype=torch.long)
    assert response_mean_mask(tgt, am, k=2).tolist() == [[False, False, True, True, False, False]]
    # k larger than the response stops at the last real token, never runs into padding
    tgt_pad = torch.tensor([[0, 0, 5, 6, 0, 0]])
    am_pad = torch.tensor([[1, 1, 1, 1, 0, 0]])
    assert response_mean_mask(tgt_pad, am_pad, k=99).tolist() == [[False, False, True, True, False, False]]


def test_response_mean_falls_back_to_last_real_token_without_a_response():
    from adversariallm.training.readers import response_mean_mask

    tgt = torch.tensor([[0, 0, 0, 0]])
    am = torch.tensor([[1, 1, 1, 0]])
    assert response_mean_mask(tgt, am, k=4).tolist() == [[False, False, True, False]]


def test_every_mode_is_padding_invariant():
    from adversariallm.training.readers import READOUT_MODES

    torch.manual_seed(0)
    D = 8
    tgt_unpad = torch.tensor([[0, 0, 4, 5, 6]])
    am_unpad = torch.tensor([[1, 1, 1, 1, 1]])
    tgt_pad = torch.tensor([[0, 0, 4, 5, 6, 0, 0]])
    am_pad = torch.tensor([[1, 1, 1, 1, 1, 0, 0]])
    h_unpad = torch.randn(1, 5, D)
    h_pad = torch.cat([h_unpad, torch.randn(1, 2, D)], dim=1)  # junk in the pad columns
    for mode in READOUT_MODES:
        probe = LinearProbe(D, readout=mode, readout_k=3)
        assert torch.allclose(
            probe.readout(h_unpad, tgt_unpad, am_unpad),
            probe.readout(h_pad, tgt_pad, am_pad),
            atol=1e-6,
        ), f"{mode} readout changed under padding"


def test_modes_read_different_positions_but_keep_the_parameter_shape():
    D = 8
    torch.manual_seed(0)
    hidden = torch.randn(1, 5, D)
    tgt = torch.tensor([[0, 0, 4, 5, 6]])
    am = torch.ones(1, 5, dtype=torch.long)
    vectors = {}
    for mode in ("prompt_last", "stream_last", "response_mean"):
        probe = LinearProbe(D, readout=mode, readout_k=2)
        vectors[mode] = probe.readout(hidden, tgt, am)
        assert probe.linear.weight.shape == (2, D)  # same state_dict in every mode
    unit = lambda v: v / v.norm()  # noqa: E731
    assert torch.allclose(vectors["prompt_last"], unit(hidden[0, 1]), atol=1e-6)
    assert torch.allclose(vectors["stream_last"], unit(hidden[0, 4]), atol=1e-6)
    assert torch.allclose(vectors["response_mean"], unit(hidden[0, 2:4].mean(0)), atol=1e-6)


def test_a_probe_trained_in_one_mode_loads_into_another():
    # the readout is not a parameter, so checkpoints stay compatible across modes
    D = 8
    trained = LinearProbe(D, readout="response_mean")
    loaded = LinearProbe(D, readout="prompt_last")
    loaded.load_state_dict(trained.state_dict())
    assert torch.equal(loaded.linear.weight, trained.linear.weight)


def test_build_reader_passes_the_readout_through_and_defaults_to_prompt_last():
    assert build_reader({"type": "linear"}, 8).readout_mode == "prompt_last"
    probe = build_reader({"type": "linear", "readout": "response_mean", "readout_k": 4}, 8)
    assert (probe.readout_mode, probe.readout_k) == ("response_mean", 4)


def test_unknown_readout_mode_is_rejected():
    import pytest
    with pytest.raises(ValueError):
        build_reader({"type": "linear", "readout": "middle_token"}, 8)


def test_monitor_restores_the_readout_the_probe_was_trained_with(tmp_path):
    # Scoring a response_mean probe at the last prompt token would be a different detector, so
    # the monitor must take the readout from the pair checkpoint, not from its own defaults.
    from adversariallm.defenses.monitors.linear_probe import LinearProbeMonitor

    D = 8
    probe = LinearProbe(D, readout="response_mean", readout_k=4)
    path = tmp_path / "final_reader.pt"
    torch.save({"reader": probe.state_dict(), "cfg": {"reader": {"type": "linear", "readout": "response_mean",
                                                                 "readout_k": 4}}, "step": 1}, path)
    target = torch.nn.Linear(D, D)  # stand-in for the target model: only .parameters() is used

    monitor = LinearProbeMonitor(checkpoint_path=str(path), target_model_id="x")
    monitor._ensure_head(target)
    assert (monitor._probe.readout_mode, monitor._probe.readout_k) == ("response_mean", 4)

    override = LinearProbeMonitor(checkpoint_path=str(path), target_model_id="x", readout="stream_last")
    override._ensure_head(target)
    assert override._probe.readout_mode == "stream_last"  # explicit ablation still wins


def test_monitor_defaults_to_prompt_last_for_a_checkpoint_without_a_saved_readout(tmp_path):
    from adversariallm.defenses.monitors.linear_probe import LinearProbeMonitor

    D = 8
    path = tmp_path / "bare_probe.pt"
    torch.save(LinearProbe(D).state_dict(), path)  # pre-readout checkpoint: a bare state_dict
    monitor = LinearProbeMonitor(checkpoint_path=str(path), target_model_id="x")
    monitor._ensure_head(torch.nn.Linear(D, D))
    assert monitor._probe.readout_mode == "prompt_last"
