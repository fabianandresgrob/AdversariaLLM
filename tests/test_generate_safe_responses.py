from generate_safe_responses import DEFAULT, finalize, trim


def test_trim_cuts_at_the_last_sentence_end():
    assert trim("I cannot help. It is illegal. Please cons") == "I cannot help. It is illegal."
    assert trim("No sentence end here") == "No sentence end here"


def test_non_refusals_get_the_default():
    out, n = finalize(["I cannot fulfill your request. I'm just an AI.", "Sure, here is how to do it."])
    assert out == ["I cannot fulfill your request. I'm just an AI.", DEFAULT] and n == 1
