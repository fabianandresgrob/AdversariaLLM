from __future__ import annotations

from abc import abstractmethod

import torch

from .base import Monitor
from .tokenization import build_detector_batch


class ActivationMonitor(Monitor):
    """Base for monitors that read the target model's hidden states.

    Owns the shared path: build the detector batch, run the target model once, pull the
    chosen layer's hidden states, softmax the head's yes/no logits (column 0 == harmful).
    Subclasses supply the head via _ensure_head + _head_logits (a gemma projection detector,
    a linear probe, ...). Requires attrs target_model_id, index_hidden_layer_detector, batch_size.
    """

    target_model_id: str
    index_hidden_layer_detector: int
    batch_size: int

    @abstractmethod
    def _ensure_head(self, target_model) -> None:
        """Lazily build the scoring head once the target hidden dim/device is known."""
        raise NotImplementedError

    @abstractmethod
    def _head_logits(self, hidden, target_ids, attention_mask) -> torch.Tensor:
        """Map (B,T,D) hidden states to (B,2) yes/no logits; column 0 == harmful."""
        raise NotImplementedError

    @torch.no_grad()
    def score(self, prompts, responses, *, target_model, target_tokenizer):
        self._ensure_head(target_model)
        device = next(target_model.parameters()).device
        scores: list[float] = []
        for start in range(0, len(prompts), self.batch_size):
            p_chunk = prompts[start : start + self.batch_size]
            r_chunk = responses[start : start + self.batch_size]
            input_ids, target_ids, attention_mask = build_detector_batch(
                p_chunk, r_chunk, target_tokenizer, self.target_model_id
            )
            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)
            attention_mask = attention_mask.to(device)
            embeds = target_model.get_input_embeddings()(input_ids)
            out = target_model(
                inputs_embeds=embeds,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
            hidden = out.hidden_states[self.index_hidden_layer_detector]
            logits = self._head_logits(hidden, target_ids, attention_mask)
            scores.extend(torch.softmax(logits.float(), dim=1)[:, 0].tolist())
        return scores
