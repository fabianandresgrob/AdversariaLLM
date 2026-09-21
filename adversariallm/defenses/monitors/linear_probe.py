from __future__ import annotations

from typing import Any

import torch

from ...training.readers import DEFAULT_READOUT, DEFAULT_READOUT_K, LinearProbe
from .activation_monitor import ActivationMonitor
from .base import register_monitor


@register_monitor
class LinearProbeMonitor(ActivationMonitor):
    """Cooperative-pipeline detector: a linear probe on the target model's activation
    (readers.LinearProbe). Same readout position + template as coop training -- the position is
    taken from the pair checkpoint -- so the co-trained probe scores identically at eval. Reads
    the target model directly — no second model.
    Loads a coop pair checkpoint (`{tag}_reader.pt`, key "reader") or a bare probe state_dict."""

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
        ckpt = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
        is_pair = isinstance(ckpt, dict) and "reader" in ckpt
        state = ckpt["reader"] if is_pair else ckpt
        trained = ((ckpt.get("cfg") or {}).get("reader") or {}) if is_pair else {}
        input_dim = state["linear.weight"].shape[1]  # (2, input_dim)
        probe = LinearProbe(  # fp32 params; readout casts hidden to fp32
            input_dim,
            readout=self.readout or trained.get("readout") or DEFAULT_READOUT,
            readout_k=self.readout_k or trained.get("readout_k") or DEFAULT_READOUT_K,
        )
        probe.load_state_dict(state)
        probe.to(next(target_model.parameters()).device).eval()
        self._probe = probe

    def _head_logits(self, hidden, target_ids, attention_mask) -> torch.Tensor:
        return self._probe.logits(hidden, target_ids, attention_mask)
