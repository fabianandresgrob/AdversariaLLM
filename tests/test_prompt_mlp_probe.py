"""PromptMLPProbe against EuroSafeGuard's own code (the EuroSafeGuard checkout next to this repo, when present)."""
import sys
from pathlib import Path

import pytest
import torch

from adversariallm.defenses.monitors.eurosafeguard_probe import MLPProbe, ema_max
from adversariallm.training.readers import PromptMLPProbe, build_reader, load_reader, parse_layer, select_hidden

ESG = Path(__file__).resolve().parents[2] / "EuroSafeGuard"
B, T, D = 3, 40, 24


def _batch(prompt_lens=(30, 5, 22), total_lens=(40, 12, 22)):
    """Right-padded rows: prompt, then answer (target_ids > 0), then padding. Row 2 has no answer."""
    target_ids = torch.zeros(B, T, dtype=torch.long)
    attn = torch.zeros(B, T, dtype=torch.long)
    for i, (p, n) in enumerate(zip(prompt_lens, total_lens)):
        attn[i, :n] = 1
        target_ids[i, p:n] = 7
    return target_ids, attn


def _probe():
    torch.manual_seed(0)
    probe = PromptMLPProbe(D, hidden=8)
    torch.nn.init.normal_(probe.w2.weight)  # zero at init would make every score equal
    probe.set_scale(torch.rand(D) + 0.5)
    return probe


def test_layer_selection():
    hs = tuple(torch.full((1, 2, 3), float(i)) for i in range(5))
    assert parse_layer("all", 4) == [1, 2, 3, 4] and parse_layer(-1, 4) == -1 and parse_layer([2, 3], 4) == [2, 3]
    assert select_hidden(hs, -1).equal(hs[4])
    cat = select_hidden(hs, [1, 3])
    assert cat.shape == (1, 2, 6) and cat[0, 0].tolist() == [1, 1, 1, 3, 3, 3]


def test_prompt_mask_covers_the_prompt_only():
    target_ids, attn = _batch()
    mask = PromptMLPProbe.prompt_mask(target_ids, attn)
    assert mask.sum(dim=1).tolist() == [30, 5, 22]
    assert not (mask & (target_ids > 0)).any()


def test_score_matches_their_inference_code():
    probe, (target_ids, attn) = _probe(), _batch()
    theirs = MLPProbe(D, 8)
    theirs.load_state_dict(probe.state_dict())  # same parameter names: their weights load into ours and back
    hidden = torch.randn(B, T, D)
    want = ema_max(theirs(hidden), PromptMLPProbe.prompt_mask(target_ids, attn), 16)
    got = probe.score(hidden, target_ids, attn)
    assert torch.allclose(got, want, atol=1e-5)
    logits = probe.logits(hidden, target_ids, attn)
    assert torch.allclose(probe.p_harmful(hidden, target_ids, attn), torch.sigmoid(got), atol=1e-6)
    assert logits[:, 1].abs().max() == 0


def test_score_ignores_the_answer():
    probe, (target_ids, attn) = _probe(), _batch()
    hidden = torch.randn(B, T, D)
    other = hidden.clone()
    other[target_ids > 0] = torch.randn_like(other[target_ids > 0])
    assert torch.allclose(probe.score(hidden, target_ids, attn), probe.score(other, target_ids, attn))


@pytest.mark.skipif(not (ESG / "training" / "probe_lib.py").exists(), reason="no EuroSafeGuard checkout")
def test_loss_matches_their_training_code():
    sys.path[:0] = [str(ESG), str(ESG / "training")]
    try:
        import probe_lib
    finally:
        del sys.path[:2]
    probe, (target_ids, attn) = _probe(), _batch()
    hidden = torch.randn(B, T, D)
    labels = torch.tensor([0, 1, 0])  # ours: 0 = harmful
    mask = PromptMLPProbe.prompt_mask(target_ids, attn)
    want = probe_lib.probe_loss(probe.token_logits(hidden), mask, (labels == 0).float(), M=16)
    got = probe.example_loss(hidden, target_ids, attn, labels)
    assert got.shape == (B,) and torch.allclose(got.mean(), want, atol=1e-5)


def test_gradients_reach_the_features_and_the_probe():
    probe, (target_ids, attn) = _probe(), _batch()
    hidden = torch.randn(B, T, D, requires_grad=True)
    probe.evasion_loss(hidden, target_ids, attn).backward()  # the probe-aware attack's term
    assert hidden.grad.abs().sum() > 0 and hidden.grad[target_ids > 0].abs().sum() == 0
    probe.zero_grad()
    probe.example_loss(hidden.detach(), target_ids, attn, torch.tensor([0, 1, 0])).mean().backward()
    assert probe.W1.weight.grad.abs().sum() > 0


def test_build_and_reload(tmp_path):
    cfg = {"type": "prompt_mlp", "layer": "all", "hidden": 8}
    reader = build_reader(cfg, input_dim=6, num_layers=4)
    assert isinstance(reader, PromptMLPProbe) and reader.W1.in_features == 24
    with pytest.raises(ValueError):
        build_reader({"type": "linear", "layer": "all"}, input_dim=6, num_layers=4)
    path = tmp_path / "final_reader.pt"
    torch.save({"reader": reader.state_dict(), "cfg": {"reader": cfg}}, path)
    again = load_reader(str(path))
    assert isinstance(again, PromptMLPProbe) and again.W1.weight.equal(reader.W1.weight)
