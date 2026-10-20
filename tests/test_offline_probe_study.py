import torch


def test_auroc_matches_pairwise_definition():
    from offline_probe_study import auroc

    assert auroc([0.9, 0.8], [0.1, 0.2]) == 1.0
    assert auroc([0.1], [0.9]) == 0.0
    assert auroc([0.5], [0.5]) == 0.5  # tie counts half
    assert abs(auroc([0.3, 0.7], [0.5, 0.1]) - 0.75) < 1e-9  # 3 of 4 pairs ordered right


def test_fit_probe_separates_a_linear_problem():
    from offline_probe_study import fit_probe

    torch.manual_seed(0)
    x = torch.nn.functional.normalize(torch.randn(200, 8), dim=-1)
    y = (x[:, 0] > 0).long()
    w, b = fit_probe(x, y, steps=500, lr=0.1)
    acc = (((x @ w + b) > 0).long() == y).float().mean()
    assert acc > 0.95
