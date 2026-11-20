"""Download the models and datasets the jobs load into HF_HOME, then check that they load offline.

Run on a login node of the MPI cluster, in tmux (compute nodes run with HF_HUB_OFFLINE=1 and cannot download):

    cd ~/Projects/coop-detect/AdversariaLLM
    uv run python scripts/fetch_hf.py all 2>&1 | tee -a /fast/fgrob/huggingface/fetch.log
    uv run python scripts/fetch_hf.py check          # offline check only, downloads nothing

Downloads one repo at a time (login nodes: half a core, 4 GB RAM; no parallel transfers). Models are downloaded
at `main`, the revision the code asks for, so offline loading finds them; the resolved commit is printed and
should go into the experiment notes. Datasets are loaded once with the same call the dataset classes use, which
fills the datasets cache that offline jobs read. A rerun skips what is complete.
"""

import argparse
import os
import subprocess
import sys
import time

HF_HOME = "/fast/fgrob/huggingface"  # conf/jobs/mpi.yaml env.HF_HOME
os.environ["HF_HOME"] = HF_HOME      # before huggingface_hub is imported

# repo id -> (why, files to skip). The weights in other formats (original/*.pth, GGUF) are not used.
MODELS = {
    "meta-llama/Meta-Llama-3.1-8B-Instruct": ("base model (conf/models/models.yaml)", ["original/*"]),
    "meta-llama/Llama-3.1-8B-Instruct": ("chat_template_id: tokenizer only", ["*.safetensors", "original/*"]),
    "qylu4156/strongreject-15k-v1": ("StrongREJECT judge, LoRA adapter (judgezoo)", []),
    "google/gemma-2b": ("base model of the StrongREJECT adapter", ["*.gguf", "*.bin"]),
    "lmsys/vicuna-13b-v1.5": ("PAIR attacker (conf/attacks/attacks.yaml), only .bin weights", []),
    "GraySwanAI/Llama-3-8B-Instruct-RR": ("Circuit Breakers baseline (models.yaml cb-llama3-8b-rr)", []),
    "cais/HarmBench-Llama-2-13b-cls": ("HarmBench judge (judgezoo, conf/judge.yaml)", ["training_args.bin"]),
}

# Repos without a tokenizer or model of their own: repo id -> why. Checked only for being complete in the cache.
FILES = {
    "EuroSafeAI/EuroSafeGuard-v1": "EuroSafeGuard probes on Llama-3.1-8B-Instruct (next steps R1.2), with their code",
}

# Adapters the configs load from a directory instead of the hub cache: repo id -> (why, path in the repo). The path
# is under a symlink into /fast (conf/jobs/mpi.yaml links), created here when it is missing.
FAST = "/fast/fgrob/coop-detect/AdversariaLLM"
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOCAL_MODELS = {
    "INSAIT-Institute/Llama3.1-8B-MixAT": ("MixAT baseline adapter (models.yaml mixat-llama31-8b)",
                                           "checkpoints_baselines/mixat-llama31-8b"),
}

# name -> python code that loads it exactly like adversariallm/dataset/*.py does. Not mmlu: the experiments only use
# it through lm_eval (run_utility_eval.py), which loads its own task data.
DATASETS = {
    "alpaca": 'load_dataset("tatsu-lab/alpaca")',
    "magpie": 'load_dataset("Magpie-Align/Magpie-Llama-3.1-Pro-300K-Filtered", split="train")',
    "or_bench": 'load_dataset("bench-llm/or-bench", "or-bench-hard-1k")',
    "xs_test": 'load_dataset("walledai/XSTest")',
    "strong_reject": ('load_dataset("csv", data_files="https://raw.githubusercontent.com/alexandrasouly/strongreject/'
                      'main/strongreject_dataset/strongreject_dataset.csv")'),
    "jbb_behaviors": "__import__('jailbreakbench').read_dataset().as_dataframe()",
}

OFFLINE = {"HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}


def log(msg: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def run_python(code: str, offline: bool) -> subprocess.CompletedProcess:
    """In a fresh process, so the offline flags are read at import time as in a job."""
    env = {**os.environ, "HF_HOME": HF_HOME, **(OFFLINE if offline else {})}
    if not offline:
        for key in OFFLINE:
            env.pop(key, None)
    return subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, check=False)


def fetch_models() -> None:
    from huggingface_hub import snapshot_download

    for repo, (why, skip) in MODELS.items():
        log(f"model {repo} ({why})")
        start = time.time()
        path = snapshot_download(repo, revision="main", ignore_patterns=skip)
        size = sum(os.path.getsize(os.path.join(d, f)) for d, _, fs in os.walk(path) for f in fs) / 1e9
        log(f"  done: commit {os.path.basename(path)}, {size:.1f} GB, {(time.time() - start) / 60:.0f} min")
    for repo, why in FILES.items():
        log(f"files {repo} ({why})")
        path = snapshot_download(repo, revision="main")
        log(f"  done: commit {os.path.basename(path)}")
    for repo, (why, rel) in LOCAL_MODELS.items():
        log(f"model {repo} ({why}) -> {rel}")
        top = rel.split("/")[0]
        link = os.path.join(REPO, top)
        if not os.path.islink(link):
            if os.path.exists(link):
                raise SystemExit(f"{link} exists and is not a symlink into {FAST}")
            os.makedirs(os.path.join(FAST, top), exist_ok=True)
            os.symlink(os.path.join(FAST, top), link)
            log(f"  created the symlink {link} -> {FAST}/{top}")
        info = snapshot_download(repo, revision="main", local_dir=os.path.join(REPO, rel), ignore_patterns=["assets/*"])
        log(f"  done: {info}")


def fetch_datasets() -> None:
    for name, call in DATASETS.items():
        log(f"dataset {name}")
        out = run_python(f"from datasets import load_dataset\nprint({call})", offline=False)
        log("  done" if out.returncode == 0 else f"  FAILED\n{out.stderr[-2000:]}")


def check() -> bool:
    """Offline, as in a job: tokenizer and config of every model (no weights: 4 GB RAM), every dataset."""
    ok = True
    for repo in MODELS:
        code = ("from transformers import AutoTokenizer\n"
                f"AutoTokenizer.from_pretrained({repo!r})\n")
        if "*.safetensors" not in MODELS[repo][1]:  # weights expected
            code += ("from huggingface_hub import snapshot_download\nimport glob\n"
                     f"p = snapshot_download({repo!r}, local_files_only=True)\n"
                     "assert glob.glob(p + '/*.safetensors') or glob.glob(p + '/*.bin'), 'no weights'\n")
        out = run_python(code, offline=True)
        ok &= out.returncode == 0
        log(f"check model   {repo}: " + ("ok" if out.returncode == 0 else "FAIL\n" + out.stderr[-1500:]))
    for repo in FILES:
        out = run_python(f"from huggingface_hub import snapshot_download\nsnapshot_download({repo!r})", offline=True)
        ok &= out.returncode == 0
        log(f"check files   {repo}: " + ("ok" if out.returncode == 0 else "FAIL\n" + out.stderr[-1500:]))
    for repo, (_, rel) in LOCAL_MODELS.items():
        found = os.path.exists(os.path.join(REPO, rel, "adapter_model.safetensors"))
        ok &= found
        log(f"check model   {repo}: " + ("ok" if found else f"FAIL: no adapter_model.safetensors in {rel}"))
    for name, call in DATASETS.items():
        out = run_python(f"from datasets import load_dataset\n{call}", offline=True)
        ok &= out.returncode == 0
        log(f"check dataset {name}: " + ("ok" if out.returncode == 0 else "FAIL\n" + out.stderr[-1500:]))
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("what", choices=["all", "models", "datasets", "check"])
    args = parser.parse_args()
    log(f"HF_HOME={HF_HOME}")
    if args.what in ("all", "models"):
        fetch_models()
    if args.what in ("all", "datasets"):
        fetch_datasets()
    ok = check()
    log("all checks ok" if ok else "some checks FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
