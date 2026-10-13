from run_cross_attack_eval import judged_asr


def test_judged_asr_counts_judged_harm_and_probe_misses():
    samples = [{"probe_score": s} for s in (0.1, 0.9, 0.2, 0.3)]
    out = judged_asr(samples, [0.8, 0.9, 0.1, 0.5], thr=0.5, threshold=0.5)
    # judged harmful: samples 0 and 1 (0.5 is not above the threshold); the probe misses 0 (0.1 <= 0.5) only
    assert out == {"judged_asr_model": 0.5, "judged_asr_pipeline": 0.25}


def test_judged_pipeline_uses_the_score_on_the_generated_answer_when_there_is_one():
    samples = [{"probe_score": 0.9, "probe_score_gen": 0.1}, {"probe_score": 0.1, "probe_score_gen": 0.9}]
    out = judged_asr(samples, [0.8, 0.8], thr=0.5, threshold=0.5)
    assert out == {"judged_asr_model": 1.0, "judged_asr_pipeline": 0.5}  # only the first answer slipped past


def test_one_adapter_takes_an_explicit_probe_or_the_one_next_to_it():
    from omegaconf import OmegaConf

    from run_cross_attack_eval import checkpoints_to_eval

    cfg = OmegaConf.create({"adapter_path": "/a/ema_adapter", "reader_path": "/p/ema_reader.pt", "checkpoints": {"x": {}}})
    assert checkpoints_to_eval(cfg) == {"model": {"adapter": "/a/ema_adapter", "reader": "/p/ema_reader.pt"}}
    cfg.reader_path = None
    assert checkpoints_to_eval(cfg)["model"]["reader"] is None  # derived from the adapter's directory later
    assert checkpoints_to_eval(OmegaConf.create({"adapter_path": None, "checkpoints": {"x": {}}})) == {"x": {}}
