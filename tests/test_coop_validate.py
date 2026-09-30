import pytest
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


def test_answer_head_mask_selects_the_first_answer_predictions():
    from adversariallm.training.coop_loop import _answer_head_mask

    # prompt 3 tokens (-100), answer 4 tokens, then 1 pad
    labels = torch.tensor([[-100, -100, -100, 5, 6, 7, 8, -100]])
    attn = torch.tensor([[1, 1, 1, 1, 1, 1, 1, 0]])
    head = _answer_head_mask(labels, attn, n_tokens=2)
    # position t predicts token t+1: positions 2 and 3 predict the first two answer tokens
    assert head.tolist() == [[False, False, True, True, False, False, False]]
    assert _answer_head_mask(labels, attn, n_tokens=10).sum() == 4


def test_head_kl_equals_the_masked_full_kl():
    from adversariallm.training.coop_loop import _head_kl
    from adversariallm.training.losses import utility_kl

    torch.manual_seed(0)
    m, r = torch.randn(2, 5, 7), torch.randn(2, 5, 7)
    head = torch.tensor([[False, True, True, False, False], [False, False, True, False, False]])
    assert torch.allclose(_head_kl(m, r, head), utility_kl(m, r, attention_mask=head), atol=1e-6)
    assert _head_kl(m, r, torch.zeros_like(head)).item() == 0.0


def test_score_answers_marks_only_the_generated_tokens_and_caps_them():
    from types import SimpleNamespace

    from adversariallm.training.coop_loop import _score_answers

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = torch.nn.Embedding(50, 3)

        def get_input_embeddings(self):
            return self.emb

        def forward(self, inputs_embeds=None, input_ids=None, attention_mask=None, output_hidden_states=False):
            return SimpleNamespace(hidden_states=[inputs_embeds], logits=None)

    class Tok:
        def __call__(self, text, add_special_tokens=False, return_tensors=None):
            return {"input_ids": torch.tensor([[10 + i for i in range(len(text))]])}

    seen = []

    class Reader:
        def p_harmful(self, hidden, target_ids, attn):
            seen.append((hidden.shape[1], target_ids[0].tolist()))
            return torch.tensor([0.7])

    prefix = torch.zeros(3, 3)
    scores = _score_answers(Model(), Reader(), 0, [prefix, prefix], ["ab", "abcdef"], Tok(), max_tokens=4)
    assert scores == pytest.approx([0.7, 0.7])
    assert seen[0] == (5, [0, 0, 0, 10, 11])          # prompt positions 0, answer tokens marked
    assert seen[1] == (7, [0, 0, 0, 10, 11, 12, 13])  # capped at 4 answer tokens
