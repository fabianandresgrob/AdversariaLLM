from __future__ import annotations

from typing import Any

import torch

from ...training.readers import load_reader
from .activation_monitor import ActivationMonitor
from .base import register_monitor


@register_monitor
class LinearProbeMonitor(ActivationMonitor):
    """Cooperative-pipeline detector: a linear probe on the target model's activation
    (readers.LinearProbe). Same readout position + template as coop training -- the position is
    taken from the pair checkpoint -- so the co-trained probe scores identically at eval. Reads
    the target model directly — no second model.
    Loads a coop pair checkpoint (`{tag}_reader.pt`, key "reader") or a bare probe state_dict;
    a checkpoint whose cfg.reader.type is "dual" loads a DualProbe (prompt + response channel)."""

    NAME = "linear_probe"

    def __init__(self, *, checkpoint_path, target_model_id, index_hidden_layer_detector=-1, batch_size=16,
                 readout=None, readout_k=None):
        self.checkpoint_path = checkpoint_path
        self.target_model_id = target_model_id
        self.index_hidden_layer_detector = index_hidden_layer_detector
        self.batch_size = batch_size
        # None = read at the position the probe was trained with, as recorded in the pair
        # checkpoint. Scoring a probe at a position it never trained on is a different detector,
        # so these overrides are an ablation, never a default.
        self.readout = readout
        self.readout_k = readout_k
        self._probe = None  # lazily built once the target device is known

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "LinearProbeMonitor":
        return cls(
            checkpoint_path=cfg["checkpoint_path"],
            target_model_id=cfg["target_model_id"],
            index_hidden_layer_detector=cfg.get("index_hidden_layer_detector", -1),
            batch_size=cfg.get("batch_size", 16),
            readout=cfg.get("readout"),
            readout_k=cfg.get("readout_k"),
        )

    def _ensure_head(self, target_model) -> None:
        if self._probe is not None:
            return
        # fp32 params; the readout casts hidden to fp32. Type (linear | dual) and position come
        # from the checkpoint; readout/readout_k are the LinearProbe ablation overrides.
        probe = load_reader(self.checkpoint_path, readout=self.readout, readout_k=self.readout_k)
        probe.to(next(target_model.parameters()).device)
        self._probe = probe

    def reads_response(self, target_model) -> bool:
        """Does this probe's score depend on the response text?

        False for a prompt-only readout, where score(prompt, "") is the real operating point.
        True for the response readouts, where anything that scores with an empty response (e.g.
        threshold calibration) must generate one first or it measures a position the probe never
        trained on."""
        self._ensure_head(target_model)
        return self._probe.readout_mode != "prompt_last"

    def _head_logits(self, hidden, target_ids, attention_mask) -> torch.Tensor:
        return self._probe.logits(hidden, target_ids, attention_mask)
