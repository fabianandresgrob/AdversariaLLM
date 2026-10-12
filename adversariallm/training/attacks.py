from __future__ import annotations

from abc import ABC, abstractmethod

import torch

from ._embedding_attack_core import EmbeddingSpaceAttack


def input_embedding_matrix(model):
    """The token embedding matrix as the model sees it: Gemma scales its embeddings by sqrt(hidden size) inside the
    embedding layer (embed_scale), so an attack that builds inputs_embeds from the raw weights must apply the same
    scale, and its radius (eps x mean embedding norm) is measured on that scale. Unchanged for models without it."""
    layer = model.get_input_embeddings()
    scale = getattr(layer, "embed_scale", None)
    return layer.weight if scale is None else layer.weight * scale.to(layer.weight)


class TrainingAttack(ABC):
    @abstractmethod
    def attack(self, model, batch, detector=None, use_detector: bool = False):
        """Return adversarial input embeddings (B,T,D) that try to elicit y_harmful.
        detector/use_detector are ignored by model-CAT; used by detector-AT/joint later."""
        ...


class ContinuousEmbeddingAttack(TrainingAttack):
    def __init__(
        self,
        embed_weights,
        response_key,
        tokenizer,
        *,
        iters,
        eps,
        lr,
        detector_loss_coeff=0.5,
        detector_layer=-1,
        target_eot=True,
        optimizer="adam",
        relative_lr=False,
        perturb="all",
    ):
        """target_eot: whether the attack's loss covers the end-of-turn token that closes the target.
        True (the original objective) optimizes "say the target, then stop", which elicits the stub
        and nothing after it; False optimizes the target alone, so the answer can continue.
        optimizer: "adam" or "sign" (signed-gradient steps, as pgd); relative_lr expresses lr as a
        fraction of the eps ball. perturb: "all" (every prompt position, chat template included --
        the original attack, which pushes the model off-manifold into loops and noise) or "user"
        (only the user message, batch["h_perturb_mask"]; the answers it elicits stay coherent)."""
        if perturb not in ("all", "user"):
            raise ValueError(f"perturb must be 'all' or 'user', got {perturb!r}")
        self.perturb = perturb
        # EmbeddingSpaceAttack.__init__ signature (from source):
        #   (embed_weights, response_key, tokenizer, hidden_state_detector_index,
        #    iters=8, opt_config=None, eps=1.0, init_type="instruction",
        #    suffix_tokens=10, relative_lr=False, detector_loss_coeff=0.5,
        #    wandb_run=None, *args, **kwargs)
        # Note: `detector` is NOT in __init__; it is passed per-call to .attack().
        # hidden_state_detector_index=-1 uses the last hidden layer (fine for model-CAT
        # where detector=None and the hidden state is never used downstream).
        self._attack = EmbeddingSpaceAttack(
            embed_weights,
            response_key,
            tokenizer,
            hidden_state_detector_index=detector_layer,
            iters=iters,
            opt_config={"type": optimizer, "lr": lr},
            eps=eps,
            init_type="instruction",
            detector_loss_coeff=detector_loss_coeff,
            relative_lr=relative_lr,
            wandb_run=None,
        )
        if not target_eot:
            from .data import end_of_turn_ids

            self._attack.loss_exclude_ids = torch.tensor(end_of_turn_ids(tokenizer), dtype=torch.long)

    @property
    def detector_loss_coeff(self):
        """Attacker's evade-detector vs elicit-harm split: (1-c)*harm + c*evade. Settable so a
        post-hoc eval can sweep the split on one attack object."""
        return self._attack.detector_loss_coeff

    @detector_loss_coeff.setter
    def detector_loss_coeff(self, value):
        self._attack.detector_loss_coeff = float(value)

    def attack(self, model, batch, detector=None, use_detector: bool = False):
        # EmbeddingSpaceAttack.attack returns an 8-tuple:
        #   (input_embeds, adv_perturbation, adv_perturbation_mask,
        #    perturbed_embeds,   # <-- index 3, shape (B, T, D)
        #    hidden_states, all_total_losses, all_losses, all_detector_losses)
        result = self._attack.attack(
            model=model,
            input_ids=batch["h_ids"],
            target_ids=batch["h_targetids"],
            attention_mask=batch["h_attn"],
            detector=detector,
            use_detector=use_detector,
            perturb_mask=batch["h_perturb_mask"] if self.perturb == "user" else None,
            elicit_mask=batch.get("h_elicit_mask"),
        )
        # the final iteration's losses (batch means), for logging: does the attack still win?
        self.last_target_loss = float(result[6][-1]) if result[6] else float("nan")
        self.last_detector_loss = float(result[7][-1]) if result[7] else float("nan")
        # Return only the perturbed embeddings (index 3).
        perturbed_embeds = result[3]
        return perturbed_embeds
