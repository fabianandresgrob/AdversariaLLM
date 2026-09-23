import json

from train_response_head import collect_examples, evaluate, split_examples, threshold_at


def _write(repo, idx, attack="inpainting", defense="none", model="E-nd6-s0", scores=(0.9, 0.3, 0.05), sub="0"):
    path = repo / "outputs" / f"{attack}__{defense}__{model}" / "2026-09-19" / "08-00-00" / sub / "run.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    step = {"model_completions": [f"c{i}" for i in range(len(scores))],
            "model_input": [{"role": "user", "content": f"attacked {idx}"}],
            "scores": {"strong_reject": {"p_harmful": list(scores)}}}
    path.write_text(json.dumps({"config": {"dataset_params": {"idx": [idx]}}, "runs": [
        {"original_prompt": [{"role": "user", "content": f"behavior {idx}"}], "steps": [step]}]}))


def test_labels_hits_and_clear_misses_and_drops_the_unsure_band(tmp_path):
    _write(tmp_path, 3)
    ex = collect_examples(tmp_path, "E-nd6-s0", ["inpainting"])
    assert [(e["response"], e["label"]) for e in ex] == [("c0", 1), ("c2", 0)]
    assert ex[0]["idx"] == 3 and ex[0]["prompt"] == "attacked 3"


def test_only_the_requested_model_defense_and_attacks_are_read(tmp_path):
    _write(tmp_path, 1)
    _write(tmp_path, 2, defense="coop_probe")
    _write(tmp_path, 3, model="M-respmean-s0")
    _write(tmp_path, 4, attack="direct")
    assert {e["idx"] for e in collect_examples(tmp_path, "E-nd6-s0", ["inpainting", "pair"])} == {1}


def test_a_run_without_a_single_dataset_index_is_skipped(tmp_path):
    _write(tmp_path, 1)
    path = next((tmp_path / "outputs").rglob("run.json"))
    data = json.loads(path.read_text())
    data["config"]["dataset_params"]["idx"] = None
    path.write_text(json.dumps(data))
    assert collect_examples(tmp_path, "E-nd6-s0", ["inpainting"]) == []


def _ex(idx, attack, label, n):
    return [{"idx": idx, "attack": attack, "label": label, "k": k} for k in range(n)]


def test_split_is_behavior_disjoint_capped_and_keeps_every_test_hit():
    examples = _ex(5, "inpainting", 1, 30) + _ex(5, "inpainting", 0, 30) + _ex(25, "inpainting", 1, 30) \
        + _ex(25, "pair", 1, 30) + _ex(25, "inpainting", 0, 30)
    train, test = split_examples(examples, test_below=20, train_attacks=["inpainting"], cap=4)
    assert {e["idx"] for e in train} == {25} and {e["idx"] for e in test} == {5}
    assert {e["attack"] for e in train} == {"inpainting"}           # pair left out of training
    assert sum(e["label"] for e in train) == 4 and len(train) == 8   # capped per label
    assert sum(e["label"] for e in test) == 30                       # test hits never capped
    assert len(test) == 34


def test_threshold_leaves_the_requested_fraction_above():
    scores = [i / 100 for i in range(100)]
    tau = threshold_at(scores, 0.01)
    assert sum(s > tau for s in scores) == 0 and tau == 0.99
    assert sum(s > threshold_at(scores, 0.1) for s in scores) == 9


def test_asr_after_counts_a_behavior_only_while_one_of_its_hits_is_unflagged():
    test = [{"idx": 0, "attack": "pair", "label": 1}, {"idx": 0, "attack": "pair", "label": 1},
            {"idx": 1, "attack": "pair", "label": 1}, {"idx": 2, "attack": "pair", "label": 0}]
    p_prompt = [0.0, 0.0, 0.0, 0.0]
    p_resp = [0.9, 0.1, 0.9, 0.9]     # behavior 0 keeps one unflagged hit; behavior 1 is fully caught
    m = evaluate(test, p_prompt, p_resp, tau_prompt=0.5, tau_dual=0.5)["pair"]
    assert m["asr_before"] == 2 / 3 and m["asr_after_prompt"] == 2 / 3 and m["asr_after_dual"] == 1 / 3
    assert m["tpr_dual"] == round(2 / 3, 4) and m["tpr_prompt"] == 0.0
    assert m["flag_rate_misses_dual"] == 1.0
