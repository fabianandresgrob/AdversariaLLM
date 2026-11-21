"""score_prompt_probe.py and the copied EuroSafeGuard probe: prompt collection, shards, batches, resume, scoring."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from adversariallm.defenses.monitors.eurosafeguard_probe import MLPProbe, ema_max, score_ids
from score_prompt_probe import attack_prompts, batches, done_hashes, sha1, shard_of


def test_ema_max_matches_a_plain_loop():
    z = torch.tensor([[1.0, -2.0, 4.0, 0.5, 9.0], [3.0, 3.0, -1.0, 0.0, 0.0]])
    mask = torch.tensor([[1, 1, 1, 1, 0], [1, 1, 1, 0, 0]], dtype=torch.bool)  # padding is never scored
    a = 2.0 / 17
    expected = []
    for row, m in zip(z.tolist(), mask.tolist()):
        state, best = None, -1e9
        for x, keep in zip(row, m):
            if keep:
                state = x if state is None else (1 - a) * state + a * x
                best = max(best, state)
        expected.append(best)
    assert torch.allclose(ema_max(z, mask, M=16), torch.tensor(expected))


def test_score_ids_reads_the_configured_layers_and_ignores_padding():
    dim, layers = 4, [1, 2]
    probe = MLPProbe(dim * len(layers), hidden=3)
    torch.nn.init.normal_(probe.W1.weight)
    torch.nn.init.normal_(probe.w2.weight)

    class Fake(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = torch.nn.Embedding(50, dim)

        def forward(self, input_ids, attention_mask, output_hidden_states):
            h = self.emb(input_ids)
            return SimpleNamespace(hidden_states=(h * 0, h, 2 * h))

    model = Fake()
    one = score_ids(model, probe, [[5, 6, 7]], layers, M=16, pad_id=0)
    padded = score_ids(model, probe, [[5, 6, 7], [1, 2, 3, 4, 9, 9]], layers, M=16, pad_id=0)
    assert torch.allclose(one[0], padded[0], atol=1e-6)  # the same row, alone or next to a longer one


def test_attack_prompts_dedupes_the_user_message(tmp_path):
    def write(folder, prompts):
        path = tmp_path / folder / "2026-10-01" / "12-00-00" / "0" / "run.json"
        path.parent.mkdir(parents=True)
        steps = [{"model_input": [{"role": "user", "content": p}, {"role": "assistant", "content": ""}]} for p in prompts]
        path.write_text(json.dumps({"runs": [{"steps": steps}]}))

    write("pair__none__base-llama31-8b", ["a", "b"])
    write("pair__coop_probe__W-coop-s0-ema", ["b", "c"])
    write("gcg__none__base-llama31-8b", ["x"])
    rows = attack_prompts(str(tmp_path), "pair")
    assert sorted(r["prompt"] for r in rows.values()) == ["a", "b", "c"]
    assert set(rows) == {sha1(p) for p in "abc"}
    assert sorted(r["prompt"] for r in attack_prompts(str(tmp_path), "pair", "base-llama31-8b").values()) == ["a", "b"]


def test_shards_partition_and_batches_respect_the_budget():
    rows = {sha1(str(i)): {} for i in range(100)}
    shards = [shard_of(rows, k, 4) for k in range(4)]
    assert sorted(h for s in shards for h in s) == sorted(rows) and all(shards)
    lengths = {h: 10 + i for i, h in enumerate(rows)}
    out = list(batches(list(rows), lengths, token_budget=300, max_rows=8))
    assert sorted(h for b in out for h in b) == sorted(rows)
    assert all(len(b) <= 8 and len(b) * max(lengths[h] for h in b) <= 300 for b in out)


def test_done_hashes_skips_a_line_cut_by_a_kill(tmp_path):
    path = tmp_path / "x.jsonl"
    path.write_text('{"sha1": "aa", "score": 1.0}\n{"sha1": "bb", "sco')
    assert done_hashes(path) == {"aa"}
    assert done_hashes(tmp_path / "missing.jsonl") == set()


def test_unknown_source_is_rejected():
    from score_prompt_probe import benign_prompts

    with pytest.raises(ValueError, match="unknown benign set"):
        benign_prompts("wildchat", 10, 0)
