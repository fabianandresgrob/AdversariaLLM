from __future__ import annotations

import pytest
import torch

from adversariallm.training._embedding_attack_core import EmbeddingSpaceAttack


def _attack(relative_lr, lr=0.01, eps=0.05):
    # rows of norm 2.0 -> embedding_norm == 2.0, so eps scales to 0.1
    embed_weights = torch.ones(4, 4)
    return EmbeddingSpaceAttack(
        embed_weights=embed_weights,
        response_key="### Response:",
        tokenizer=None,
        hidden_state_detector_index=-1,
        opt_config={"type": "adam", "lr": lr},
        eps=eps,
        relative_lr=relative_lr,
    )


def test_eps_scales_with_the_embedding_norm():
    attack = _attack(relative_lr=False)
    assert float(attack.embedding_norm) == pytest.approx(2.0)
    assert float(attack.eps) == pytest.approx(0.1)


def test_absolute_lr_is_left_alone():
    assert float(_attack(relative_lr=False).opt_config["lr"]) == pytest.approx(0.01)


def test_relative_lr_scales_by_the_ball_radius():
    # Constructing with relative_lr=True used to read self.eps one line before it was
    # assigned, so this raised AttributeError instead of scaling anything.
    attack = _attack(relative_lr=True)
    assert float(attack.eps) == pytest.approx(0.1)
    assert float(attack.opt_config["lr"]) == pytest.approx(0.001)  # 0.01 * eps
