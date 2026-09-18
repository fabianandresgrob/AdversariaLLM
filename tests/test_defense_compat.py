from adversariallm.defenses import DEFENSE_COMPATIBLE_ATTACKS


def test_runtime_defense_support_matrix():
    assert "actor" in DEFENSE_COMPATIBLE_ATTACKS
    assert "crescendo" in DEFENSE_COMPATIBLE_ATTACKS
    assert "inpainting" in DEFENSE_COMPATIBLE_ATTACKS
    # IHO optimizes its attacker by DPO on a judge score over the TARGET SYSTEM's returned
    # text (preference.py: DEFAULT_SCORE_COL = "judge_score_training"), and it already talks
    # to the target through TargetSystem -- so a probe fire arrives as a refusal, scores low,
    # and lands in the rejected pool. That makes it adaptive against model+detector jointly.
    assert "iho" in DEFENSE_COMPATIBLE_ATTACKS
    assert "gcg" not in DEFENSE_COMPATIBLE_ATTACKS
