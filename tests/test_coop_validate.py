import torch

from adversariallm.training.coop_loop import _generate_from_embeds


class _Tok:
    pad_token_id, eos_token_id = 0, 0

    def batch_decode(self, ids, **kw):
        return [f"n={row.numel()}" for row in ids]


class _RecordingModel:
    def __init__(self):
        self.calls = []

    def generate(self, inputs_embeds, attention_mask, max_new_tokens, **kw):
        self.calls.append((inputs_embeds.clone(), attention_mask.clone()))
        return torch.zeros(inputs_embeds.size(0), max_new_tokens, dtype=torch.long)


def test_generate_from_embeds_left_pads_so_every_prompt_ends_at_the_onset():
    a, b = torch.full((2, 3), 1.0), torch.full((4, 3), 2.0)
    model = _RecordingModel()
    out = _generate_from_embeds(model, _Tok(), [a, b], max_new_tokens=5, batch_size=8)
    embeds, attn = model.calls[0]
    assert attn.tolist() == [[0, 0, 1, 1], [1, 1, 1, 1]]
    assert torch.equal(embeds[0, 2:], a) and torch.all(embeds[0, :2] == 0)
    assert torch.equal(embeds[1], b)
    assert out == ["n=5", "n=5"]


def test_generate_from_embeds_batches():
    model = _RecordingModel()
    prompts = [torch.ones(2, 3) for _ in range(5)]
    assert len(_generate_from_embeds(model, _Tok(), prompts, max_new_tokens=1, batch_size=2)) == 5
    assert [c[0].size(0) for c in model.calls] == [2, 2, 1]
