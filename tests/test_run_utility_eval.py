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
    env = lm_eval_env()
    assert "PYTHONPATH" not in env and env["HF_HUB_OFFLINE"] == "1"
