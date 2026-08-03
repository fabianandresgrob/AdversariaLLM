from __future__ import annotations

import torch

from adversariallm.training.coop_losses import per_example_ce, detector_ce
from adversariallm.training.readers import LinearProbe


def test_per_example_ce_length_normalized_and_masks():
    logits = torch.zeros(1, 3, 4)              # uniform -> per-token CE = -log(1/4)
    targets = torch.tensor([[0, 1, -100]])     # 2 valid, 1 ignored
    out = per_example_ce(logits, targets)
    assert out.shape == (1,)
    assert torch.allclose(out, -torch.log(torch.tensor([0.25])), atol=1e-5)


def test_detector_ce_label_convention():
    # column 0 = harmful. Logits that strongly predict column 0:
    logits = torch.tensor([[10.0, -10.0]])
    assert detector_ce(logits, torch.tensor([0])).item() < 1e-3   # label harmful -> low loss
    assert detector_ce(logits, torch.tensor([1])).item() > 5.0    # label benign -> high loss


def test_detector_ce_per_example_reduction():
    out = detector_ce(torch.randn(3, 2), torch.tensor([0, 1, 0]), reduction="none")
    assert out.shape == (3,)


def test_rep_term_reshapes_activations_with_reader_frozen():
    # The representation term is detector_ce(reader(h), harmful) with the reader FROZEN
    # (detector frozen during the model step) and gradient flowing into the activations.
    D = 8
    probe = LinearProbe(D)
    for p in probe.parameters():
        p.requires_grad_(False)                     # frozen reader
    hidden = torch.randn(1, 4, D, requires_grad=True)
    tgt = torch.tensor([[0, 0, 3, 4]])
    am = torch.tensor([[1, 1, 1, 1]])
    loss = detector_ce(probe.logits(hidden, tgt, am), torch.tensor([0]))  # push toward harmful
    loss.backward()
    assert hidden.grad is not None and hidden.grad.abs().sum() > 0        # flows into h_theta
    assert all(p.grad is None for p in probe.parameters())               # frozen reader untouched
