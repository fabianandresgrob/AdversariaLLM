import pytest

pytest.importorskip("hydra")

from hydra import compose, initialize_config_dir  # noqa: E402

from gen_coop_models import REPO  # noqa: E402
from run_utility_eval import build_command, lm_eval_env  # noqa: E402


def _cfg(monkeypatch, overrides=()):
    monkeypatch.setenv("MYPROJECT", "/p/me")
    monkeypatch.setenv("PWD", "/p/me/AdversariaLLM")
    with initialize_config_dir(config_dir=str(REPO / "conf"), version_base="1.3"):
        return compose(config_name="utility_eval", overrides=list(overrides))


def test_base_model_follows_meta_recipe(monkeypatch):
    cmd = build_command(_cfg(monkeypatch))
    assert cmd[0] == "/p/me/venvs/lm_eval/bin/lm_eval"
    assert cmd[cmd.index("--model") + 1] == "vllm"
    assert cmd[cmd.index("--tasks") + 1] == "mmlu_llama,arc_challenge_llama,gsm8k_llama"
    assert {"--apply_chat_template", "--fewshot_as_multiturn", "--log_samples"} <= set(cmd)
    assert "--limit" not in cmd
    args = cmd[cmd.index("--model_args") + 1]
    assert "pretrained=meta-llama/Meta-Llama-3.1-8B-Instruct" in args and "lora" not in args
    assert cmd[cmd.index("--output_path") + 1] == "/p/me/AdversariaLLM/outputs/eval/utility/reference/base"


def test_adapter_is_loaded_as_lora(monkeypatch):
    cmd = build_command(_cfg(monkeypatch, ["adapter_path=/ckpt/final_adapter", "limit=5"]))
    args = cmd[cmd.index("--model_args") + 1].split(",")
    assert {"enable_lora=True", "max_lora_rank=16", "lora_local_path=/ckpt/final_adapter"} <= set(args)
    assert cmd[cmd.index("--limit") + 1] == "5"


def test_env_drops_pythonpath_keeps_offline_flags(monkeypatch):
    monkeypatch.setenv("PYTHONPATH", "/p/me/jsc-jobs")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("MYPROJECT", "/p/me")
    monkeypatch.delenv("VLLM_USE_FLASHINFER_SAMPLER", raising=False)
    monkeypatch.delenv("VLLM_CACHE_ROOT", raising=False)
    env = lm_eval_env()
    assert "PYTHONPATH" not in env and env["HF_HUB_OFFLINE"] == "1"
    assert env["VLLM_USE_FLASHINFER_SAMPLER"] == "0" and env["VLLM_CACHE_ROOT"] == "/p/me/.cache/vllm"


def test_tasks_with_shot_counts_run_one_call_each(monkeypatch):
    from run_utility_eval import build_commands

    cfg = _cfg(monkeypatch, ["model=google/gemma-3-12b-it", "tasks=[gsm8k:8,mmlu:5,arc_challenge:25]", "out=/o"])
    cmds = build_commands(cfg)
    assert [c[c.index("--tasks") + 1] for c in cmds] == ["gsm8k", "mmlu", "arc_challenge"]
    assert [c[c.index("--num_fewshot") + 1] for c in cmds] == ["8", "5", "25"]
    assert [c[c.index("--output_path") + 1] for c in cmds] == ["/o/gsm8k", "/o/mmlu", "/o/arc_challenge"]
    assert all({"--apply_chat_template", "--fewshot_as_multiturn"} <= set(c) for c in cmds)


def test_plain_tasks_keep_the_single_llama_call(monkeypatch):
    from run_utility_eval import build_commands

    cmds = build_commands(_cfg(monkeypatch))
    assert len(cmds) == 1 and "--num_fewshot" not in cmds[0]


def test_summary_merges_per_task_results(tmp_path, monkeypatch):
    import json
    import time

    from run_utility_eval import write_summary

    for task, metric, value in (("gsm8k", "exact_match,strict-match", 0.8), ("mmlu", "acc,none", 0.7),
                                ("arc_challenge", "acc_norm,none", 0.6)):
        d = tmp_path / task / "model"
        d.mkdir(parents=True)
        (d / "results_1.json").write_text(json.dumps({"results": {task: {metric: value}}, "n-shot": {task: 5}}))
        time.sleep(0.01)
    s = write_summary(str(tmp_path), _cfg(monkeypatch, ["tasks=[gsm8k:8,mmlu:5,arc_challenge:25]"]))
    assert s["summary"] == {"gsm8k_strict_pct": 80.0, "mmlu_acc_pct": 70.0, "arc_c_acc_norm_pct": 60.0}
    assert json.loads((tmp_path / "utility.json").read_text())["n_shot"]["mmlu"] == 5
