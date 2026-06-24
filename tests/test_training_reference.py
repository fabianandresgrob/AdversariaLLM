from __future__ import annotations

from contextlib import contextmanager
import torch

from adversariallm.training.reference import LoRADisableReference, FrozenModelReference


class _FakePeftModel:
    def __init__(self):
        self.disabled_during_call = None
        self._disabled = False

    @contextmanager
    def disable_adapter(self):
        self._disabled = True
        try:
            yield
        finally:
            self._disabled = False

    def __call__(self, inputs_embeds=None, attention_mask=None):
        self.disabled_during_call = self._disabled
        class _Out: ...
        o = _Out(); o.logits = torch.zeros(1, 2, 3) + (1.0 if self._disabled else 9.0)
        return o


def test_lora_reference_calls_with_adapter_disabled():
    m = _FakePeftModel()
    ref = LoRADisableReference(m)
    out = ref.logits(inputs_embeds=torch.zeros(1, 2, 4), attention_mask=torch.ones(1, 2))
    assert m.disabled_during_call is True
    assert torch.allclose(out, torch.ones(1, 2, 3))


def test_frozen_reference_uses_separate_model():
    class _Frozen:
        def __call__(self, inputs_embeds=None, attention_mask=None):
            class _Out: ...
            o = _Out(); o.logits = torch.full((1, 2, 3), 5.0)
            return o
    ref = FrozenModelReference(_Frozen())
    out = ref.logits(inputs_embeds=torch.zeros(1, 2, 4), attention_mask=torch.ones(1, 2))
    assert torch.allclose(out, torch.full((1, 2, 3), 5.0))
