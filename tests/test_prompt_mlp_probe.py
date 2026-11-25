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


def test_pretraining_split_and_twins():
    import importlib.util

    spec = importlib.util.spec_from_file_location("ppp", Path(__file__).resolve().parents[1] / "pretrain_prompt_probe.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    groups = [f"g{i // 3}" for i in range(300)]  # 100 groups of 3 rows
    train, val = mod.split_groups(groups, 0.05, seed=0)
    assert len(val) == 15 and len(train) == 285
    assert not {groups[i] for i in train} & {groups[i] for i in val}
    assert mod.split_groups(groups, 0.05, seed=0) == (train, val)
    twins = mod.plain_twins(["a", "a", "b", "c", "c"], ["plain", "redact_jb", "jf", "dan_h", "plain"])
    assert twins == {1: 0, 3: 4}
    assert mod.auroc([1, 1, 0, 0], [0.9, 0.8, 0.1, 0.2]) == 1.0
    assert mod.auroc([1, 0, 1, 0], [0.5, 0.5, 0.9, 0.1]) == 0.875  # one tie counts half: (0.5 + 1 + 1 + 1) / 4
    assert mod.auroc([1, 1], [0.1, 0.2]) != mod.auroc([1, 1], [0.1, 0.2])  # nan


def test_pretraining_steps_on_a_tiny_model():
    """features, rms_scale and a few probe updates on a random 2-layer model: the loss falls and scores separate."""
    import importlib.util

    from transformers import GPT2Config, GPT2LMHeadModel

    spec = importlib.util.spec_from_file_location("ppp", Path(__file__).resolve().parents[1] / "pretrain_prompt_probe.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    torch.manual_seed(0)
    model = GPT2LMHeadModel(GPT2Config(n_layer=2, n_embd=16, n_head=2, vocab_size=50)).eval()
    layer = parse_layer("all", 2)
    harmful = [[3, 4, 5, 6, 7] * k for k in (2, 5, 8, 3)]      # token 3..7 patterns
    benign = [[20, 21, 22, 23] * k for k in (3, 6, 2, 7)]
    items, labels = harmful + benign, torch.tensor([0] * 4 + [1] * 4)
    psi, tgt, att = mod.features(model, items, layer, pad_id=0)
    assert psi.shape == (8, 40, 32) and tgt.sum() == 0 and att.sum(dim=1).tolist() == [len(x) for x in items]
    reader = build_reader({"type": "prompt_mlp", "layer": "all", "hidden": 8}, 16, 2)
    scale = mod.rms_scale(model, items, layer, 0, 16, bs=4)
    assert scale.shape == (32,) and scale[:16].unique().numel() == 1
    reader.set_scale(scale)
    opt = torch.optim.AdamW(reader.parameters(), lr=1e-2)
    first = None
    for _ in range(60):
        loss = reader.example_loss(psi, tgt, att, labels).mean()
        first = first if first is not None else loss.item()
        opt.zero_grad()
        loss.backward()
        opt.step()
    scores = mod.score_all(model, reader.eval(), items, layer, 0, bs=3)
    assert loss.item() < 0.5 * first
    assert min(scores[:4]) > max(scores[4:])
