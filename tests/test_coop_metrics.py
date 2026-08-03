from __future__ import annotations

import torch

from adversariallm.training.coop_metrics import (
    recall_at_fpr, four_case_frequencies, refusal_rate, fresh_refit_recall,
)


def test_recall_at_fpr_perfect_separation():
    benign = [0.0 + 0.001 * i for i in range(100)]   # 0.00 .. 0.099
    harmful = [0.9] * 100                              # all well above any benign quantile
    assert recall_at_fpr(benign, harmful, fpr=0.01) == 1.0


def test_recall_at_fpr_respects_the_budget():
    # benign and harmful identically distributed -> at 1% FPR recall should be ~1%.
    torch.manual_seed(0)
    benign = torch.rand(1000).tolist()
    harmful = torch.rand(1000).tolist()
    r = recall_at_fpr(benign, harmful, fpr=0.01)
    assert 0.0 <= r <= 0.05    # near the budget, not near 1.0


def test_recall_at_fpr_accepts_tensors():
    assert recall_at_fpr(torch.zeros(50), torch.ones(50), fpr=0.01) == 1.0


def test_four_case_frequencies_corners():
    # comply+miss (both high) = case D; refuse+fire (both low) = case A
    wh = torch.tensor([0.9, 0.1, 0.9, 0.1])
    wm = torch.tensor([0.9, 0.1, 0.1, 0.9])
    f = four_case_frequencies(wh, wm, thr=0.5)
    assert f["D"] == 0.25   # (0.9, 0.9): model failed + detector missed
    assert f["A"] == 0.25   # (0.1, 0.1): model refused + detector fired
    assert f["B"] == 0.25   # (0.9, 0.1): model failed + detector fired
    assert f["C"] == 0.25   # (0.1, 0.9): model refused + detector missed
    assert abs(sum(f.values()) - 1.0) < 1e-9


def test_refusal_rate():
    comps = ["Sorry, I can't do that.", "Here is the answer: 42", "I cannot help with that", "Sure!"]
    assert abs(refusal_rate(comps) - 0.5) < 1e-9


def test_fresh_refit_recall_separable_features():
    # harmful and benign live in clearly separated feature regions -> a fresh probe nails it
    torch.manual_seed(0)
    D = 16
    harmful_feat = torch.randn(50, D) + 3.0
    benign_feat = torch.randn(50, D) - 3.0
    train_features = torch.cat([harmful_feat, benign_feat], dim=0)
    train_labels = torch.cat([torch.zeros(50, dtype=torch.long), torch.ones(50, dtype=torch.long)])
    r = fresh_refit_recall(train_features, train_labels, benign_feat, harmful_feat, steps=200, lr=1e-2)
    assert r > 0.9


def test_fresh_refit_recall_works_under_no_grad():
    # It is called from @torch.no_grad() validation, so it must re-enable grad internally.
    torch.manual_seed(0)
    D = 16
    harmful_feat = torch.randn(40, D) + 3.0
    benign_feat = torch.randn(40, D) - 3.0
    train = torch.cat([harmful_feat, benign_feat], dim=0)
    labels = torch.cat([torch.zeros(40, dtype=torch.long), torch.ones(40, dtype=torch.long)])
    with torch.no_grad():
        r = fresh_refit_recall(train, labels, benign_feat, harmful_feat, steps=150, lr=1e-2)
    assert r > 0.9
