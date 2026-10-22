from types import SimpleNamespace

import torch

from adversariallm.training.coop_loop import _detector_step
from adversariallm.training.data import drop_longer_than, split_by_label
from adversariallm.training.readers import LinearProbe


class _FakeTok:
    """user -> "U<content>", assistant -> "R<content>E": one id per character."""

    def __call__(self, text, **kw):
        return {"input_ids": [ord(c) for c in text]}

    def apply_chat_template(self, conv, tokenize=False, add_generation_prompt=False, **kw):
        s = "".join(("U" if m["role"] == "user" else "R") + m["content"] + ("E" if m["role"] == "assistant" else "")
                    for m in conv)
        return s + ("R" if add_generation_prompt else "")


def test_split_by_label_keeps_only_labelled_rows_with_prompt_and_completion():
    records = [{"p": "a", "c": "x", "l": "harmful"}, {"p": "b", "c": "y", "l": "unharmful"},
               {"p": "c", "c": "z", "l": None}, {"p": "d", "c": None, "l": "harmful"}, {"p": "", "c": "w", "l": "harmful"}]
    harmful, harmless = split_by_label(records, "p", "c", "l")
    assert harmful == [("a", "x")] and harmless == [("b", "y")]


def test_drop_longer_than_drops_rather_than_cuts():
    rows = [("ab", "cd"), ("a" * 20, "b")]
    assert drop_longer_than(rows, _FakeTok(), 8) == [("ab", "cd")]  # "UabRcdE" = 7 tokens


class _Embed(torch.nn.Module):
    """Model stub: hidden state = an embedding of the token id, so labels are learnable from the ids."""

    def __init__(self, d=8):
        super().__init__()
        self.emb = torch.nn.Embedding(16, d)

    def forward(self, input_ids=None, inputs_embeds=None, attention_mask=None, output_hidden_states=True):
        h = self.emb(input_ids) if inputs_embeds is None else inputs_embeds
        return SimpleNamespace(hidden_states=[h], logits=None)


def _batch(token, n=2):
    ids = torch.full((n, 4), token)
    return {"d_ids": ids, "d_attn": torch.ones_like(ids), "d_targetids": torch.tensor([[0, 0, 1, 1]] * n)}


def test_extra_batches_train_the_probe_with_their_labels_and_no_easy_batch():
    torch.manual_seed(0)
    model = _Embed()
    reader = LinearProbe(8)
    opt = torch.optim.Adam(reader.parameters(), lr=0.05)
    adv = {"h_attn": torch.ones(2, 4, dtype=torch.long), "h_targetids": torch.tensor([[0, 0, 1, 1]] * 2)}
    adv_embeds = model.emb(torch.full((2, 4), 1)).detach()  # attacked harmful (label 0)
    extra = [(_batch(5), 0), (_batch(9), 1)]                 # extra harmful = token 5, extra harmless = token 9
    for _ in range(100):
        _detector_step(model, reader, opt, 0, adv_embeds, adv, None, "cpu", extra_batches=extra)
    with torch.no_grad():
        p = {t: reader.p_harmful(model.emb(_batch(t)["d_ids"]), _batch(t)["d_targetids"], _batch(t)["d_attn"]).mean()
             for t in (5, 9)}
    assert p[5] > 0.9 and p[9] < 0.1
