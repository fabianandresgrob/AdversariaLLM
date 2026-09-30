import torch

from adversariallm.training.coop_loop import _perturb_user


def test_noise_has_the_radius_on_user_tokens_and_is_zero_elsewhere():
    torch.manual_seed(0)
    emb = torch.randn(2, 5, 8)
    mask = torch.tensor([[0, 1, 1, 0, 0], [0, 0, 1, 1, 1]], dtype=torch.bool)
    delta = _perturb_user(emb, mask, 0.3) - emb
    norms = delta.norm(dim=-1)
    assert torch.allclose(norms[mask], torch.full((int(mask.sum()),), 0.3), atol=1e-5)
    assert torch.all(norms[~mask] == 0)
