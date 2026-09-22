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


def test_detector_aware_gcg_loads_the_probe_at_its_trained_readout(tmp_path):
    """The attacker must evade the detector the defense actually runs.

    gcg.load_detector built LinearProbe(input_dim) with defaults, which would have optimised
    suffixes against a response probe read at the prompt position -- a detector nobody deploys.
    Needs no GPU: load_detector only reads .parameters() for the device."""
    from adversariallm.attacks.gcg import load_detector

    for mode, k in (("response_mean", 24), ("stream_last", 24), ("prompt_last", 8)):
        path = tmp_path / f"{mode}_reader.pt"
        torch.save({"reader": LinearProbe(8, readout=mode, readout_k=k).state_dict(),
                    "cfg": {"reader": {"type": "linear", "readout": mode, "readout_k": k}}}, path)
        probe = load_detector(str(path), torch.nn.Linear(8, 8))
        assert (probe.readout_mode, probe.readout_k) == (mode, k)


def test_detector_aware_gcg_defaults_a_bare_probe_to_prompt_last(tmp_path):
    from adversariallm.attacks.gcg import load_detector

    path = tmp_path / "bare.pt"
    torch.save(LinearProbe(8).state_dict(), path)  # pre-readout checkpoint: a bare state_dict
    assert load_detector(str(path), torch.nn.Linear(8, 8)).readout_mode == "prompt_last"
