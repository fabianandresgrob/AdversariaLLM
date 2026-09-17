"""Utility eval (MMLU / ARC-C / GSM8K, Meta's Llama-3.1 recipe) for the base model or a LoRA checkpoint.

Thin wrapper around lm-evaluation-harness in its own venv (conf/utility_eval.yaml: lm_eval_venv),
using the vLLM backend with the adapter loaded as a LoRA. Adds no scoring code: the llama3 tasks
define prompts, few-shot, generation limits and metrics. Results and per-sample generations
(--log_samples) land in `out`.

    pixi run --frozen python run_utility_eval.py                                   # base model
    pixi run --frozen python run_utility_eval.py adapter_path=<ckpt>/final_adapter out=<dir>
    pixi run --frozen python run_utility_eval.py limit=20 out=/tmp/utility_smoke   # smoke test
"""

import logging
import os
import subprocess
import sys

import hydra
from omegaconf import DictConfig

log = logging.getLogger(__name__)


def build_command(cfg: DictConfig) -> list[str]:
    model_params = cfg.models[cfg.model]
    model_args = [
        f"pretrained={model_params.id}",
        f"tokenizer={model_params.tokenizer_id}",
        f"dtype={cfg.vllm.dtype}",
        f"gpu_memory_utilization={cfg.vllm.gpu_memory_utilization}",
        f"max_model_len={cfg.vllm.max_model_len}",
    ]
    if cfg.adapter_path:
        model_args += ["enable_lora=True", f"max_lora_rank={cfg.vllm.max_lora_rank}", f"lora_local_path={cfg.adapter_path}"]
    cmd = [
        os.path.join(str(cfg.lm_eval_venv), "bin", "lm_eval"),
        "--model", "vllm",
        "--model_args", ",".join(model_args),
        "--tasks", ",".join(cfg.tasks),
        "--batch_size", str(cfg.batch_size),
        "--output_path", str(cfg.out),
    ]
    if cfg.apply_chat_template:
        cmd.append("--apply_chat_template")
    if cfg.fewshot_as_multiturn:
        cmd.append("--fewshot_as_multiturn")
    if cfg.log_samples:
        cmd.append("--log_samples")
    if cfg.limit is not None:
        cmd += ["--limit", str(cfg.limit)]
    return cmd


def lm_eval_env() -> dict[str, str]:
    """Environment for the venv's lm_eval: keep HF/W&B offline flags and CUDA binding, drop the
    pixi env's Python path so the venv's own packages are used.

    VLLM_USE_FLASHINFER_SAMPLER=0: FlashInfer's sampler JIT-compiles a kernel at warmup, which needs
    nvcc (absent on compute nodes); the llama3 tasks decode greedily, so the torch sampler is equivalent.
    VLLM_CACHE_ROOT: torch.compile artifacts go to $MYPROJECT instead of the small $HOME quota."""
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONHOME", None)
    env.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    if "MYPROJECT" in env:
        env.setdefault("VLLM_CACHE_ROOT", os.path.join(env["MYPROJECT"], ".cache", "vllm"))
    return env


@hydra.main(config_path="./conf", config_name="utility_eval", version_base="1.3")
def main(cfg: DictConfig) -> None:
    if cfg.adapter_path and not os.path.isdir(str(cfg.adapter_path)):
        raise FileNotFoundError(f"adapter_path={cfg.adapter_path} is not a directory")
    os.makedirs(str(cfg.out), exist_ok=True)
    cmd = build_command(cfg)
    log.info("running: " + " ".join(cmd))
    result = subprocess.run(cmd, env=lm_eval_env())
    if result.returncode != 0:
        sys.exit(result.returncode)
    summary = write_summary(str(cfg.out), cfg)
    log.info(f"wrote lm_eval results to {cfg.out}: {summary['results']}")


def write_summary(out: str, cfg: DictConfig) -> dict:
    """lm_eval nests its output as <out>/<model>/results_<timestamp>.json; copy the newest run's
    per-task metrics to <out>/utility.json (fixed name, so jsc-jobs can check it as an artifact)."""
    import glob
    import json

    found = sorted(glob.glob(os.path.join(out, "**", "results_*.json"), recursive=True), key=os.path.getmtime)
    if not found:
        raise FileNotFoundError(f"lm_eval finished but wrote no results_*.json under {out}")
    with open(found[-1]) as fh:
        results = json.load(fh)
    summary = {
        "results": results["results"],
        "results_file": found[-1],
        "adapter_path": cfg.adapter_path,
        "tasks": list(cfg.tasks),
        "limit": cfg.limit,
    }
    with open(os.path.join(out, "utility.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    return summary


if __name__ == "__main__":
    main()
