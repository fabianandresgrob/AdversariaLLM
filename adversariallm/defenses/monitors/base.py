from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from transformers import PreTrainedModel, PreTrainedTokenizerBase


class Monitor(ABC):
    """A harmfulness monitor. Activation monitors use target_model/target_tokenizer;
    text-only monitors ignore them."""

    NAME: str

    @abstractmethod
    def score(
        self,
        prompts: list[str],
        responses: list[str],
        *,
        target_model: PreTrainedModel | None,
        target_tokenizer: PreTrainedTokenizerBase | None,
    ) -> list[float]:
        """Return a per-item harmfulness score (higher = more harmful)."""
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def from_config(cls, cfg: dict[str, Any]) -> "Monitor":
        raise NotImplementedError


_MONITOR_REGISTRY: dict[str, type[Monitor]] = {}


def register_monitor(cls: type[Monitor]) -> type[Monitor]:
    _MONITOR_REGISTRY[cls.NAME] = cls
    return cls


def build_monitor(cfg: dict[str, Any]) -> Monitor:
    name = cfg.get("name")
    monitor_cls = _MONITOR_REGISTRY.get(name)
    if monitor_cls is None:
        raise ValueError(f"Unknown monitor: {name}")
    return monitor_cls.from_config(cfg)
