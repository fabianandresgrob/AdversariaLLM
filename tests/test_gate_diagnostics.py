import math

from adversariallm.training.gate_diagnostics import _gate, truncate_completion, variant_stats


class _FakeTok:
    def __call__(self, text, **kw):
        return {"input_ids": [ord(c) for c in text]}

    def decode(self, ids, **kw):
        return "".join(chr(i) for i in ids)


def test_truncate_completion_keeps_first_k_tokens():
    assert truncate_completion(_FakeTok(), "abcdef", 3) == "abc"
    assert truncate_completion(_FakeTok(), "ab", 5) == "ab"          # shorter than k


def test_gate_is_sigmoid_of_neg_minus_pos():
    # equal sides -> 0.5; refuse (neg) more probable -> >0.5
    assert _gate([-2.0], [-2.0], 1.0)[0] == 0.5
    assert _gate([-5.0], [-2.0], 1.0)[0] > 0.5
    assert math.isnan(_gate([float("nan")], [-2.0], 1.0)[0])


def test_variant_stats_drops_nan_and_counts():
    s = variant_stats([0.9, 0.1, float("nan"), 0.9])
    assert s["n"] == 3
    assert s["frac_gt0.5"] == 2 / 3
