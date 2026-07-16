from __future__ import annotations

from types import SimpleNamespace

import torch

from adversariallm.training.losses import away_from_harmful
from adversariallm.training.loop import Objective, train_step


class _TinyLM(torch.nn.Module):
    """Minimal stand-in for a causal LM: embeddings + a linear head to vocab."""

    def __init__(self, vocab=7, dim=4):
        super().__init__()
        self.emb = torch.nn.Embedding(vocab, dim)
        self.head = torch.nn.Linear(dim, vocab)

    def get_input_embeddings(self):
        return self.emb

    def forward(self, inputs_embeds=None, input_ids=None, attention_mask=None):
        if inputs_embeds is None:
            inputs_embeds = self.emb(input_ids)
        return SimpleNamespace(logits=self.head(inputs_embeds))


class _PollutingAttack:
    """Attack stand-in that mimics the real one's gradient behaviour.

    ContinuousEmbeddingAttack optimizes its perturbation with an optimizer built over
    the perturbation alone, so its internal backward() accumulates into the model's
    parameters as a side effect, then returns detached embeddings.
    """

    def __init__(self, adv_embeds):
        self.adv_embeds = adv_embeds

    def attack(self, model, batch, detector=None, use_detector=False):
        pert = torch.zeros_like(self.adv_embeds, requires_grad=True)
        out = model(inputs_embeds=self.adv_embeds + pert, attention_mask=batch["h_attn"])
        out.logits.square().sum().backward()
        return self.adv_embeds.detach()


def _grads(model):
    return {n: (torch.zeros_like(p) if p.grad is None else p.grad.clone())
            for n, p in model.named_parameters()}


def test_train_step_excludes_attack_gradients_from_the_update():
    """The optimizer must step on the objective's gradients alone.

    The attack leaves its own gradients on the model's parameters; if they survive
    into the update, the model trains on objective + attack instead of the objective.
    """
    torch.manual_seed(0)
    model = _TinyLM()
    B, T = 2, 3
    adv_embeds = torch.randn(B, T, 4)
    adv_batch = {
        "h_attn": torch.ones(B, T, dtype=torch.long),
        "h_labels": torch.tensor([[-100, 1, 2], [-100, 3, 4]]),
    }
    objective = Objective(active_terms={"away"}, away_variant="ce", lambda_away=1.0)

    loss, _ = train_step(
        model,
        ref=None,
        attack=_PollutingAttack(adv_embeds),
        objective=objective,
        adv_batch=adv_batch,
        util_batch=None,
        device=torch.device("cpu"),
    )
    loss.backward()
    actual = _grads(model)

    # Reference: the away term alone, on the same adversarial embeddings.
    model.zero_grad(set_to_none=True)
    logits = model(inputs_embeds=adv_embeds, attention_mask=adv_batch["h_attn"]).logits
    away_from_harmful(logits[:, :-1], adv_batch["h_labels"][:, 1:], variant="ce").backward()
    expected = _grads(model)

    for name in expected:
        assert torch.allclose(actual[name], expected[name], atol=1e-6), (
            f"{name}: attack gradients leaked into the update"
        )


def test_polluting_attack_stub_actually_pollutes():
    """Guards the test above: if the stub stopped leaving grads, it would pass vacuously."""
    torch.manual_seed(0)
    model = _TinyLM()
    adv_batch = {"h_attn": torch.ones(2, 3, dtype=torch.long)}
    _PollutingAttack(torch.randn(2, 3, 4)).attack(model, adv_batch)
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
