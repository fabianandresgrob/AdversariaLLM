from adversariallm.training.generate_benign_targets import is_compliant, partition_targets


def test_is_compliant_rejects_refusals_and_short_hedges():
    assert is_compliant("Sure! Here is a detailed and helpful answer to your question.")
    assert not is_compliant("I can't help with that.")             # refusal marker
    assert not is_compliant("I can’t help with that, sorry about it.")  # curly apostrophe
    assert not is_compliant("Sorry, no.")                          # refusal marker
    assert not is_compliant("No.")                                 # too short (hedge signal)


def test_partition_targets_nulls_noncompliant_and_counts():
    prompts = ["p1", "p2", "p3"]
    gens = ["Here is a thorough helpful response to p1, in detail.",
            "I cannot help with that.",
            "Another genuinely helpful and sufficiently long answer here."]
    rows, n_refused = partition_targets(prompts, gens)
    assert n_refused == 1
    assert rows[0]["y_gen"].startswith("Here is")
    assert rows[1]["y_gen"] is None
    assert rows[2]["y_gen"] is not None
