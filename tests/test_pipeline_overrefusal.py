import json

import pytest

from adversariallm.training.probe_scoring import calibrated_threshold, pipeline_refusals
from collect_eval import overrefusal_columns
from run_overrefusal_eval import _judge_refused


def test_pipeline_refuses_when_the_model_refuses_or_the_probe_flags():
    assert pipeline_refusals([True, False, False, True], [0.1, 0.9, 0.2, 0.9], threshold=0.5) == [True, True, False, True]
    with pytest.raises(ValueError):
        pipeline_refusals([True], [0.1, 0.2], threshold=0.5)


def test_judge_refusal_counts_full_refusals_and_degenerate_answers_like_the_adjusted_rate():
    jd = {"judgments": ["2_full_refusal", "1_full_compliance", "3_partial_refusal", "1_full_compliance"],
          "degeneration_reasons": [[], [], [], ["repeated_chars"]]}
    assert _judge_refused(jd) == [True, False, False, True]


def test_calibrated_threshold_must_belong_to_the_reader(tmp_path):
    reader = tmp_path / "ema_reader.pt"
    (tmp_path / "threshold_1pct_calib.json").write_text(json.dumps({"threshold": 0.4, "checkpoint_path": str(reader)}))
    assert calibrated_threshold(str(reader)) == 0.4
    with pytest.raises(ValueError):
        calibrated_threshold(str(tmp_path / "ema_step750_reader.pt"))


def test_collect_eval_reads_the_pipeline_overrefusal(tmp_path):
    path = tmp_path / "overrefusal.json"
    path.write_text(json.dumps({"results": {"model": {"xs_test": 0.1}},
                                "pipeline": {"model": {"xs_test": {"threshold": 0.4, "probe_flagged": 0.05,
                                                                   "pipeline_refusal_string": 0.13,
                                                                   "pipeline_refusal_judge": 0.2}}}}))
    cols = overrefusal_columns(path)
    assert cols["xstest_refusal_string"] == 0.1
    assert (cols["xstest_probe_flagged"], cols["xstest_refusal_pipeline_string"], cols["xstest_refusal_pipeline_judge"]) \
        == (0.05, 0.13, 0.2)
    path.write_text(json.dumps({"results": {"model": {"xs_test": 0.1}}}))
    assert "xstest_probe_flagged" not in overrefusal_columns(path)
