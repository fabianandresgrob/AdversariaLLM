from __future__ import annotations

import torch

from adversariallm.training.coop_metrics import (
    recall_at_fpr, threshold_at_fpr, fpr_at_threshold,
    four_case_frequencies, refusal_rate, fresh_refit_recall,
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


def test_threshold_at_fpr_is_the_recall_operating_point():
    # the threshold recall@fpr uses internally: the (1-fpr) quantile of benign.
    benign = [0.0 + 0.01 * i for i in range(101)]        # 0.00 .. 1.00, evenly spaced
    tau = threshold_at_fpr(benign, fpr=0.01)             # 99th percentile -> ~0.99
    assert abs(tau - 0.99) < 1e-6
    harmful = [tau + 0.001, tau - 0.001]                 # one above, one below
    assert recall_at_fpr(benign, harmful, fpr=0.01) == 0.5   # only the above-tau one counts
    assert torch.isnan(torch.tensor(threshold_at_fpr([])))


def test_fpr_at_threshold_exposes_over_firing():
    # plain FPR at a fixed 0.5 threshold: fraction of benign scored above it.
    assert fpr_at_threshold([0.1, 0.2, 0.3, 0.4], thr=0.5) == 0.0     # calibrated: none fire
    assert fpr_at_threshold([0.6, 0.7, 0.1, 0.2], thr=0.5) == 0.5     # over-firing: half fire
    assert torch.isnan(torch.tensor(fpr_at_threshold([])))


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


def test_gate_stats_reports_mean_open_and_saturation():
    from adversariallm.training.coop_metrics import gate_stats
    s = gate_stats([0.0, 0.4, 0.6, 0.995, 0.001])
    assert abs(s["mean"] - 0.3992) < 1e-3
    assert abs(s["frac_open"] - 0.4) < 1e-9      # 0.6 and 0.995 are > 0.5
    assert abs(s["frac_sat_hi"] - 0.2) < 1e-9    # only 0.995 > 0.99
    assert abs(s["frac_sat_lo"] - 0.4) < 1e-9    # 0.0 and 0.001 < 0.01


def test_gate_stats_empty_is_nan():
    import math
    from adversariallm.training.coop_metrics import gate_stats
    s = gate_stats([])
    assert all(math.isnan(v) for v in s.values())
