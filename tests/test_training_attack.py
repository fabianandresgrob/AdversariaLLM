import pytest
import torch


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="continuous attack needs a GPU + real model; validated in the smoke run",
)
def test_attack_smoke_placeholder():
    pass
