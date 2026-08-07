from __future__ import annotations

import torch

from adversariallm.training.gating import (
    avg_logprob, w_harm, w_miss, behavior_gate, rep_gate,
)


def test_w_miss_is_stop_gradient():
    # Sanity gate 6 (sg half): the behavior term = behavior_gate(w_miss) * CE must carry
    # NO gradient into whatever produced the detector score. Otherwise the model could
    # shrink the term by making activations detectable instead of by refusing (§14.9).
    det = torch.tensor([2.0], requires_grad=True)
    p_harm = torch.sigmoid(det)                 # stand-in detector P(harmful)
    ce = torch.tensor([3.0], requires_grad=True)  # stand-in behavior CE (the real route)
    term = (behavior_gate(w_miss(p_harm), epsilon=0.0) * ce).sum()
    term.backward()
    assert det.grad is None or torch.allclose(det.grad, torch.zeros_like(det))
    assert ce.grad is not None and ce.grad.abs().sum() > 0   # route 1 stays open


def test_w_harm_is_stop_gradient():
    lp_h = torch.tensor([1.0], requires_grad=True)
    lp_s = torch.tensor([0.0], requires_grad=True)
    rep = torch.tensor([2.0], requires_grad=True)
    term = (rep_gate(w_harm(lp_h, lp_s), delta=0.0) * rep).sum()
    term.backward()
    assert lp_h.grad is None or torch.allclose(lp_h.grad, torch.zeros_like(lp_h))
    assert lp_s.grad is None or torch.allclose(lp_s.grad, torch.zeros_like(lp_s))
    assert rep.grad.abs().sum() > 0


def test_w_harm_margin_and_w_miss_value():
    # comply example: logp(harm)=-0.8, logp(safe)=-3.3 -> margin 2.5 -> sigmoid(2.5)
    got = w_harm(torch.tensor([-0.8]), torch.tensor([-3.3]), tau=1.0)
    assert torch.allclose(got, torch.sigmoid(torch.tensor([2.5])), atol=1e-5)
    # detector P(harmful)=0.10 -> w_miss=0.90
    assert torch.allclose(w_miss(torch.tensor([0.10])), torch.tensor([0.90]), atol=1e-6)


def test_gate_floors():
    wm = torch.tensor([0.9])
    assert torch.allclose(behavior_gate(wm, 0.0), wm)                    # pure coupled
    assert torch.allclose(behavior_gate(wm, 1.0), torch.ones_like(wm))   # always on
    wh = torch.tensor([0.05])
    assert torch.allclose(rep_gate(wh, 0.0), wh)                         # off in case C
    assert torch.allclose(rep_gate(wh, 1.0), torch.ones_like(wh))        # always reshape


def test_avg_logprob_is_length_normalized_and_masks_ignore_index():
    logits = torch.zeros(1, 3, 4)                     # uniform -> logp = log(1/4) per token
    targets = torch.tensor([[0, 1, -100]])            # 2 valid tokens, 1 ignored
    out = avg_logprob(logits, targets)
    assert torch.allclose(out, torch.log(torch.tensor([0.25])), atol=1e-5)


def test_w_refuse_high_when_model_prefers_refusal():
    from adversariallm.training.gating import w_refuse
    # refusal far more likely than help -> weight ~1
    w = w_refuse(torch.tensor([-0.1]), torch.tensor([-5.0]), tau=1.0)
    assert w.item() > 0.98
    # help far more likely than refusal -> weight ~0
    w = w_refuse(torch.tensor([-5.0]), torch.tensor([-0.1]), tau=1.0)
    assert w.item() < 0.02


def test_w_refuse_is_stop_gradient():
    from adversariallm.training.gating import w_refuse
    lp_r = torch.tensor([-1.0], requires_grad=True)
    lp_h = torch.tensor([-2.0], requires_grad=True)
    assert not w_refuse(lp_r, lp_h).requires_grad
