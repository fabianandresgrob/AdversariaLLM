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
