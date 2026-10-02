"""PAIR results: every stream's answer must sit next to the prompt that produced it."""

import pytest

torch = pytest.importorskip("torch")

from adversariallm.attacks.pair import stream_steps  # noqa: E402


def _conv(text):
    return [{"role": "user", "content": text}, {"role": "assistant", "content": ""}]


def test_one_step_per_stream_with_its_own_prompt():
    # 2 iterations x 3 streams; prompts and tokens are flat, completions per iteration
    attacks = [_conv(f"p{i}{s}") for i in range(2) for s in range(3)]
    tokens = [torch.tensor([10 * i + s]) for i in range(2) for s in range(3)]
    completions = [[f"a{i}{s}" for s in range(3)] for i in range(2)]
    raw = [[f"r{i}{s}" for s in range(3)] for i in range(2)]
    meta = [[{"score": i + s / 10} for s in range(3)] for i in range(2)]
    steps = stream_steps(attacks, tokens, completions, raw, meta, [3.0, 6.0], [30, 60], num_streams=3)
    assert [st.step for st in steps] == list(range(6))
    for st in steps:
        i, s = divmod(st.step, 3)
        assert st.model_input[0]["content"] == f"p{i}{s}"
        assert st.model_completions == [f"a{i}{s}"]
        assert st.model_completions_raw == [f"r{i}{s}"]
        assert st.defense_metadata == [{"score": i + s / 10}]
        assert st.model_input_tokens == [10 * i + s]
    assert steps[0].time_taken == 1.0 and steps[3].flops == 20


def test_extra_samples_stay_with_their_stream_and_order_is_kept():
    attacks = [_conv("x"), _conv("y")]
    tokens = [torch.tensor([1]), torch.tensor([2])]
    completions = [["x0", "x1", "y0", "y1"]]  # each stream's first answer, then its extra sample
    steps = stream_steps(attacks, tokens, completions, [None], [None], [1.0], [2], num_streams=2)
    assert [st.model_completions for st in steps] == [["x0", "x1"], ["y0", "y1"]]
    assert steps[0].model_completions_raw is None and steps[1].defense_metadata is None
    assert [c for st in steps for c in st.model_completions] == completions[0]
