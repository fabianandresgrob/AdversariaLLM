from __future__ import annotations

import torch
import torch.nn.functional as F

from adversariallm.training.losses import (
    away_from_harmful, toward_benign, utility_kl,
    sequence_logprob, ipo_preference,
)


def _logits_targets():
    torch.manual_seed(0)
    logits = torch.randn(2, 5, 32)          # (B, T, V)
    targets = torch.randint(0, 32, (2, 5))  # (B, T)
    return logits, targets


def test_away_ce_is_negative_cross_entropy():
    logits, targets = _logits_targets()
    ce = F.cross_entropy(logits.reshape(-1, 32), targets.reshape(-1))
    out = away_from_harmful(logits, targets, variant="ce")
    assert torch.allclose(out, -ce, atol=1e-5)


def test_away_ul_is_bounded_and_vanishes_when_target_unlikely():
    logits = torch.full((1, 1, 4), -10.0)
    logits[0, 0, 1] = -10.0            # target token id 1 stays tiny
    logits[0, 0, 0] = 10.0             # mass goes elsewhere
    targets = torch.tensor([[1]])
    ul = away_from_harmful(logits, targets, variant="ul")
    assert ul.item() >= 0.0
    assert ul.item() < 0.05           # bounded, near zero when target already unlikely


def test_toward_benign_is_cross_entropy():
    logits, targets = _logits_targets()
    ce = F.cross_entropy(logits.reshape(-1, 32), targets.reshape(-1))
    assert torch.allclose(toward_benign(logits, targets), ce, atol=1e-5)


def test_utility_kl_zero_for_identical_and_nonnegative():
    torch.manual_seed(1)
    logits = torch.randn(2, 4, 16)
    assert abs(utility_kl(logits, logits.clone()).item()) < 1e-6
    other = torch.randn(2, 4, 16)
    assert utility_kl(logits, other).item() >= 0.0


def test_sequence_logprob_sums_target_logprobs():
    logits = torch.zeros(1, 3, 4)          # uniform -> logp = log(1/4) per token
    targets = torch.tensor([[0, 1, 2]])
    out = sequence_logprob(logits, targets)
    assert torch.allclose(out, torch.tensor([3 * torch.log(torch.tensor(0.25))]), atol=1e-5)


def test_ipo_matches_formula():
    beta = 0.5
    pi_c, pi_r = torch.tensor([-1.0]), torch.tensor([-3.0])      # model: benign>harmful
    ref_c, ref_r = torch.tensor([-2.0]), torch.tensor([-2.0])    # ref: indifferent
    h = (pi_c - pi_r) - (ref_c - ref_r)                          # = 2.0
    expected = ((h - 1.0 / (2 * beta)) ** 2).mean()
    out = ipo_preference(pi_c, pi_r, ref_c, ref_r, beta=beta)
    assert torch.allclose(out, expected, atol=1e-6)
