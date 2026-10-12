"""Coop's benign-perturbation KL over chunks of the batch (data.utility_chunk) is the same loss and gradient."""
import torch

from adversariallm.training.coop_loop import _benign_perturb_kl
from adversariallm.training.reference import FrozenModelReference
from tests.test_training_step import _TinyLM


def test_chunked_benign_perturbation_kl_matches_the_whole_batch():
    torch.manual_seed(0)
    ref_model = _TinyLM()
    ids = torch.randint(0, 7, (3, 6))
    attn = torch.tensor([[1] * 6, [1] * 4 + [0] * 2, [1] * 5 + [0]])
    labels = torch.full_like(ids, -100)
    labels[:, 3:] = ids[:, 3:]  # answers start at position 3
    labels[attn == 0] = -100
    batch = {"input_ids": ids, "attn": attn, "labels": labels, "perturb_mask": torch.zeros_like(ids, dtype=torch.bool)}
    results = []
    for chunk in (None, 1, 2):
        torch.manual_seed(1)
        model = _TinyLM()
        hp = {"kl_head_tokens": 2, "benign_radius": 0.0, "utility_chunk": chunk}
        kl = _benign_perturb_kl(model, FrozenModelReference(ref_model), batch, hp, weight=0.5)
        results.append((kl, {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}))
    (kl0, g0), *rest = results
    assert kl0 > 0
    for kl, g in rest:
        assert abs(kl - kl0) < 1e-6
        for n in g0:
            assert torch.allclose(g[n], g0[n], atol=1e-6), n
