from rejudge_answers import table


def _r(sr, hb, flagged):
    return {"p_harmful": sr, "p_harmful_harmbench": hb, "flagged": {"trained": flagged}}


def test_table_counts_harm_and_catches_under_both_judges():
    rows = [_r(0.9, 0.9, True), _r(0.9, 0.1, False), _r(0.1, 0.9, True), _r(0.1, 0.1, True)]
    t = table(rows, "harmbench")
    assert t["strong_reject"] == {"harmful": 2, "flagged": 1}
    assert t["harmbench"] == {"harmful": 2, "flagged": 2}
    assert t["both"] == {"harmful": 1, "flagged": 1}
    assert (t["flagged_total"], t["flagged_harmful_harmbench"]) == (3, 2)
