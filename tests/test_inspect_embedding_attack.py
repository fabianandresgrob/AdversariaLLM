import torch

from inspect_embedding_attack import nearest_tokens


def test_nearest_tokens_recovers_unperturbed_and_slightly_perturbed_rows():
    torch.manual_seed(0)
    weight = torch.randn(50, 16)
    ids = torch.tensor([3, 17, 42, 0])
    assert nearest_tokens(weight[ids], weight, chunk=3).tolist() == ids.tolist()
    noisy = weight[ids] + 0.01 * torch.randn(4, 16)
    assert nearest_tokens(noisy, weight).tolist() == ids.tolist()


def test_nearest_tokens_follows_a_large_perturbation_to_another_token():
    weight = torch.eye(4) * 10
    moved = weight[1] + 0.9 * (weight[2] - weight[1])  # 90% of the way from token 1 to token 2
    assert nearest_tokens(moved.unsqueeze(0), weight).tolist() == [2]
