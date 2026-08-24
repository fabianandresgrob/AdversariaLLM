from __future__ import annotations

from typing import Any

import torch
from peft import LoraConfig, TaskType, get_peft_model

from ...io_utils import load_model_and_tokenizer
from ._activation_detector_model import Detector, get_embed_weights
from .activation_monitor import ActivationMonitor
from .base import register_monitor


@register_monitor
class ActivationLLMMonitor(ActivationMonitor):
    """Mathieu's activation detector: reads the target model's hidden states."""

    NAME = "activation_llm"

    def __init__(
        self,
        detector_model,
        detector_tokenizer,
        *,
        checkpoint_path: str,
        prompt: str,
        target_model_id: str,
        detector_model_id: str,
        index_hidden_layer_detector: int = -1,
        batch_size: int = 16,
    ):
        self.detector_model = detector_model
        self.detector_tokenizer = detector_tokenizer
        self.checkpoint_path = checkpoint_path
        self.prompt = prompt
        self.target_model_id = target_model_id
        self.detector_model_id = detector_model_id
        self.index_hidden_layer_detector = index_hidden_layer_detector
        self.batch_size = batch_size
        self._detector = None  # lazily built once target hidden dim is known

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "ActivationLLMMonitor":
        model, tokenizer = load_model_and_tokenizer(
            {
                "id": cfg["detector_model_id"],
                "tokenizer_id": cfg.get("tokenizer_id", cfg["detector_model_id"]),
                "short_name": "activation_detector",
                "developer_name": "thesis",
                "compile": False,
                "dtype": cfg.get("dtype", "bfloat16"),
                "chat_template": None,
                "trust_remote_code": True,
            }
        )
        if cfg.get("lora", True):
            lora_config = LoraConfig(
                r=8,
                lora_alpha=32,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                lora_dropout=0.1,
                bias="none",
                task_type=TaskType.CAUSAL_LM,
            )
            model = get_peft_model(model, lora_config)
        return cls(
            model,
            tokenizer,
            checkpoint_path=cfg["checkpoint_path"],
            prompt=cfg["prompt"],
            target_model_id=cfg["target_model_id"],
            detector_model_id=cfg["detector_model_id"],
            index_hidden_layer_detector=cfg.get("index_hidden_layer_detector", -1),
            batch_size=cfg.get("batch_size", 16),
        )

    def _ensure_head(self, target_model):
        if self._detector is not None:
            return
        input_shape = get_embed_weights(target_model).shape[-1]
        projected_shape = get_embed_weights(self.detector_model).shape[-1]
        detector_config = {
            "prompt": self.prompt,
            "input_shape": input_shape,
            "projected_shape": projected_shape,
            "debug": False,
        }
        # NB: the Detector's own "Is this harmful?" scaffold is tokenized/embedded with
        # the DETECTOR (gemma) tokenizer, so it must use the detector model's chat
        # template — not the target model's. The input hidden states fed in are still
        # built with the target model's template (see build_detector_batch in score()).
        detector = Detector(self.detector_model, self.detector_tokenizer, detector_config, self.detector_model_id)
        device = next(target_model.parameters()).device
        detector.to(device)
        checkpoint = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
        detector.load_state_dict(checkpoint["detector_state_dict"])
        detector.eval()
        self._detector = detector

    def _head_logits(self, hidden, target_ids, attention_mask):
        # column 0 == "yes" == harmful; the projection detector returns (aux, logits_yes_no)
        _, logits_yes_no = self._detector(hidden, target_ids, attention_mask)
        return logits_yes_no
