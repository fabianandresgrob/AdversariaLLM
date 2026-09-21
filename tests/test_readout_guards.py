"""Guards against warm-starting or fitting a probe at a readout it does not belong to.

Parameter shapes match across readouts -- that is what makes checkpoints portable -- so a mismatch
produces plausible numbers rather than an error. These are the two places it could happen silently.
"""

import pytest

torch = pytest.importorskip("torch")

from adversariallm.training.coop_loop import load_probe_init  # noqa: E402
from adversariallm.training.pretrain_probe import check_pretrain_readout  # noqa: E402
from adversariallm.training.readers import LinearProbe  # noqa: E402


def _probe_pt(tmp_path, readout=None):
    """A run_pretrain_probe checkpoint. readout=None mimics one written before the readout modes."""
    reader_cfg = {"layer": -1} if readout is None else {"layer": -1, "readout": readout}
    path = tmp_path / "probe.pt"
    probe = LinearProbe(8)
    torch.nn.init.constant_(probe.linear.weight, 0.25)
    torch.save({"state": probe.state_dict(), "cfg": {"reader": reader_cfg}, "input_dim": 8, "layer": -1}, path)
    return path


def test_warm_starting_a_response_readout_from_a_prompt_probe_is_refused(tmp_path):
    with pytest.raises(ValueError, match="readout"):
        load_probe_init(LinearProbe(8, readout="response_mean"), str(_probe_pt(tmp_path)), "cpu")


def test_an_old_probe_without_a_recorded_readout_still_warm_starts(tmp_path):
    # backwards compatibility: every probe.pt on disk predates the readout modes
    reader = LinearProbe(8, readout="prompt_last")
    assert load_probe_init(reader, str(_probe_pt(tmp_path)), "cpu") == "prompt_last"
    assert torch.allclose(reader.linear.weight, torch.full((2, 8), 0.25))  # weights really loaded


def test_a_probe_recorded_as_prompt_last_warm_starts_a_prompt_last_run(tmp_path):
    reader = LinearProbe(8, readout="prompt_last")
    assert load_probe_init(reader, str(_probe_pt(tmp_path, "prompt_last")), "cpu") == "prompt_last"


def test_pretrain_refuses_a_readout_it_cannot_fit():
    with pytest.raises(NotImplementedError, match="prompt_last"):
        check_pretrain_readout({"reader": {"layer": -1, "readout": "stream_last"}})
    with pytest.raises(NotImplementedError):
        check_pretrain_readout({"reader": {"layer": -1, "readout": "response_mean"}})


def test_pretrain_accepts_the_prompt_readout_and_a_config_without_one():
    assert check_pretrain_readout({"reader": {"layer": -1}}) == "prompt_last"
    assert check_pretrain_readout({"reader": {"layer": -1, "readout": "prompt_last"}}) == "prompt_last"
    assert check_pretrain_readout({}) == "prompt_last"
