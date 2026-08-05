from __future__ import annotations

from abc import ABC, abstractmethod

import torch


class ReferenceProvider(ABC):
    @abstractmethod
    def logits(self, *, inputs_embeds: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor: ...


class LoRADisableReference(ReferenceProvider):
    """Reference = the LoRA model with its adapter disabled (zero extra memory)."""

    def __init__(self, peft_model):
        self.model = peft_model

    @torch.no_grad()
    def logits(self, *, inputs_embeds, attention_mask):
        with self.model.disable_adapter():
            return self.model(inputs_embeds=inputs_embeds, attention_mask=attention_mask).logits


class FrozenModelReference(ReferenceProvider):
    """Reference = a separate frozen copy of the base model (for full fine-tuning)."""

    def __init__(self, frozen_model):
        self.model = frozen_model

    @torch.no_grad()
    def logits(self, *, inputs_embeds, attention_mask):
        return self.model(inputs_embeds=inputs_embeds, attention_mask=attention_mask).logits
