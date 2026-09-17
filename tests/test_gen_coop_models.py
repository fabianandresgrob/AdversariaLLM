import json
from pathlib import Path

import pytest
import yaml
from omegaconf import OmegaConf

from gen_coop_models import BEGIN, END, main, strip_generated, update_models_yaml

BASE = {
    "id": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "tokenizer_id": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "short_name": "Llama",
    "developer_name": "Meta",
    "compile": False,
    "dtype": "bfloat16",
    "chat_template": "llama-3-instruct",
    "trust_remote_code": True,
}
HAND_WRITTEN = """\
meta-llama/Meta-Llama-3.1-8B-Instruct:
  id: meta-llama/Meta-Llama-3.1-8B-Instruct
  chat_template: llama-3-instruct
cat_ul_450:
  id: meta-llama/Meta-Llama-3.1-8B-Instruct
  adapter_path: ${root_dir}/checkpoints_cat/cat_ul_450/best_adapter
"""


def _repo(tmp_path):
    (tmp_path / "conf" / "models").mkdir(parents=True)
    (tmp_path / "conf" / "models" / "models.yaml").write_text(HAND_WRITTEN)
    return tmp_path


def _checkpoint(repo, exp, name, complete=True, chat_template="llama-3-instruct"):
    d = repo / "checkpoints_coop" / exp / name
    d.mkdir(parents=True)
    run_cfg = {"name": name, "model": BASE["id"], "models": {BASE["id"]: {**BASE, "chat_template": chat_template}}}
    (d / "run_config.json").write_text(json.dumps(run_cfg))
    if complete:
        (d / "final_adapter").mkdir()
        (d / "final_reader.pt").write_bytes(b"x")
    return d


def _models(repo):
    return yaml.safe_load((repo / "conf" / "models" / "models.yaml").read_text())


def test_generates_entry_from_run_config(tmp_path):
    repo = _repo(tmp_path)
    ckpt = _checkpoint(repo, "A-eps-sweep", "A-eps0.05-s0", chat_template="llama-3-instruct-cooptrain")
    assert main([str(ckpt)], repo=repo) == 0
    entry = _models(repo)["A-eps0p05-s0"]
    assert entry == {
        **BASE,
        "chat_template": "llama-3-instruct-cooptrain",
        "adapter_path": "${root_dir}/checkpoints_coop/A-eps-sweep/A-eps0.05-s0/final_adapter",
        "reader_path": "${root_dir}/checkpoints_coop/A-eps-sweep/A-eps0.05-s0/final_reader.pt",
    }
    assert "cat_ul_450" in _models(repo)


def test_incomplete_checkpoint_is_skipped(tmp_path, capsys):
    repo = _repo(tmp_path)
    done = _checkpoint(repo, "A", "A-s0")
    running = _checkpoint(repo, "A", "A-s1", complete=False)
    main([str(done), str(running)], repo=repo)
    assert set(_models(repo)) == {"meta-llama/Meta-Llama-3.1-8B-Instruct", "cat_ul_450", "A-s0"}
    assert "skipped" in capsys.readouterr().out


def test_rerun_replaces_block_and_is_idempotent(tmp_path):
    repo = _repo(tmp_path)
    a = _checkpoint(repo, "A", "A-s0")
    main([str(a)], repo=repo)
    first = (repo / "conf" / "models" / "models.yaml").read_text()
    main([str(a)], repo=repo)
    assert (repo / "conf" / "models" / "models.yaml").read_text() == first
    b = _checkpoint(repo, "B", "B-s0")
    main([str(a), str(b)], repo=repo)
    text = (repo / "conf" / "models" / "models.yaml").read_text()
    assert text.count(BEGIN) == 1 and text.count(END) == 1
    assert {"A-s0", "B-s0"} <= set(_models(repo))


def test_dry_run_writes_nothing(tmp_path):
    repo = _repo(tmp_path)
    main([str(_checkpoint(repo, "A", "A-s0")), "--dry-run"], repo=repo)
    assert (repo / "conf" / "models" / "models.yaml").read_text() == HAND_WRITTEN


def test_clash_with_hand_written_entry_raises(tmp_path):
    repo = _repo(tmp_path)
    ckpt = _checkpoint(repo, "X", "cat_ul_450")
    with pytest.raises(ValueError, match="cat_ul_450"):
        main([str(ckpt)], repo=repo)
    assert (repo / "conf" / "models" / "models.yaml").read_text() == HAND_WRITTEN


def test_defense_interpolation_resolves_generated_name(tmp_path):
    repo = _repo(tmp_path)
    main([str(_checkpoint(repo, "A-eps-sweep", "A-eps0.25-s2"))], repo=repo)
    cfg = OmegaConf.create({
        "root_dir": "/r",
        "model": "A-eps0p25-s2",
        "models": _models(repo),
        "checkpoint_path": "${models.${model}.reader_path}",
    })
    assert cfg.checkpoint_path == "/r/checkpoints_coop/A-eps-sweep/A-eps0.25-s2/final_reader.pt"


def test_real_models_yaml_regenerates_identically():
    text = (Path(__file__).resolve().parents[1] / "conf" / "models" / "models.yaml").read_text()
    hand_written = yaml.safe_load(strip_generated(text)) or {}
    generated = {k: v for k, v in yaml.safe_load(text).items() if k not in hand_written}
    assert update_models_yaml(text, generated) == text
