from __future__ import annotations

from abc import ABC, abstractmethod

from ._embedding_attack_core import EmbeddingSpaceAttack


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
    ):
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
            hidden_state_detector_index=-1,
            iters=iters,
            opt_config={"type": "adam", "lr": lr},
            eps=eps,
            init_type="instruction",
            detector_loss_coeff=detector_loss_coeff,
            wandb_run=None,
        )

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
        )
        # Return only the perturbed embeddings (index 3).
        perturbed_embeds = result[3]
        return perturbed_embeds
