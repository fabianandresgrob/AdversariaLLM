import pytest
import torch


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="continuous attack needs a GPU + real model; validated in the smoke run",
)
def test_attack_smoke_placeholder():
    pass


def test_reader_evasion_loss_targets_benign():
    """Stage C: the reader's evasion_loss (detector-side attack term) drives it toward benign
    (BENIGN_COL). Label convention col0=harmful / col1=benign."""
    from adversariallm.training.readers import BENIGN_COL, HARMFUL_COL, Reader

    assert (HARMFUL_COL, BENIGN_COL) == (0, 1)

    class _FakeReader(Reader):
        def __init__(self, logits):
            self._logits = logits

        def logits(self, hidden, target_ids, attention_mask):
            return self._logits

    benign = _FakeReader(torch.tensor([[-5.0, 5.0]]))    # already scores benign (col 1 high)
    harmful = _FakeReader(torch.tensor([[5.0, -5.0]]))   # scores harmful (col 0 high)
    # the attacker minimizes evasion_loss -> lower when the reader already says benign
    assert benign.evasion_loss(None, None, None) < harmful.evasion_loss(None, None, None)


class _Tok:
    eos_token_id, unk_token_id = 9, None

    def convert_tokens_to_ids(self, token):
        return 8 if token == "<|eot_id|>" else None

    def decode(self, ids, **kwargs):
        return " ".join(map(str, ids.tolist()))


class _TinyLM(torch.nn.Module):
    """inputs_embeds -> logits over a 10-token vocab, with the HF output fields the attack reads."""

    def __init__(self, dim=6, vocab=10):
        super().__init__()
        self.head = torch.nn.Linear(dim, vocab)

    def forward(self, inputs_embeds, attention_mask=None, output_hidden_states=False):
        from types import SimpleNamespace
        return SimpleNamespace(logits=self.head(inputs_embeds), hidden_states=(inputs_embeds,))


def _attack(**kwargs):
    from adversariallm.training.attacks import ContinuousEmbeddingAttack

    torch.manual_seed(0)
    return ContinuousEmbeddingAttack(torch.randn(10, 6), None, _Tok(), iters=3, eps=1.0, lr=0.1, **kwargs)


def test_target_eot_false_drops_end_of_turn_from_the_attack_loss_only():
    # prompt 3 tokens, target [5, 6], then end-of-turn (8)
    target_ids = torch.tensor([[0, 0, 0, 5, 6, 8]])
    with_eot = _attack()._attack.get_loss_mask(target_ids)
    without = _attack(target_eot=False)._attack.get_loss_mask(target_ids)
    assert with_eot.tolist() == [[False, False, True, True, True]]
    assert without.tolist() == [[False, False, True, True, False]]


def test_perturb_mask_confines_the_perturbation():
    ids = torch.tensor([[1, 2, 3, 4, 5, 6]])
    target_ids = torch.tensor([[0, 0, 0, 0, 5, 6]])
    attn = torch.ones_like(ids)
    only = torch.tensor([[False, False, True, False, False, False]])
    attack = _attack(target_eot=False)
    model = _TinyLM()
    model.requires_grad_(False)
    out = attack.attack(model, {"h_ids": ids, "h_targetids": target_ids, "h_attn": attn, "h_perturb_mask": only})
    clean = attack._attack.get_embeddings(ids)
    moved = (out - clean).norm(dim=-1)[0]
    assert moved[2] > 0
    assert torch.all(moved[[0, 1, 3, 4, 5]] == 0)


def test_without_a_mask_every_prompt_position_may_move():
    ids = torch.tensor([[1, 2, 3, 4, 5, 6]])
    target_ids = torch.tensor([[0, 0, 0, 0, 5, 6]])
    attack = _attack()
    model = _TinyLM()
    model.requires_grad_(False)
    out = attack.attack(model, {"h_ids": ids, "h_targetids": target_ids, "h_attn": torch.ones_like(ids)})
    moved = (out - attack._attack.get_embeddings(ids)).norm(dim=-1)[0]
    assert torch.all(moved[:4] > 0) and torch.all(moved[4:] == 0)
