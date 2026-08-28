import pytest
import torch


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="continuous attack needs a GPU + real model; validated in the smoke run",
)
def test_attack_smoke_placeholder():
    pass


def test_reader_evasion_loss_targets_benign():
    """Stage C: the reader's evasion_loss (detector-side attack term) drives it toward benign
    (BENIGN_COL). Label convention col0=harmful / col1=benign."""
    from adversariallm.training.readers import BENIGN_COL, HARMFUL_COL, Reader

    assert (HARMFUL_COL, BENIGN_COL) == (0, 1)

    class _FakeReader(Reader):
        def __init__(self, logits):
            self._logits = logits

        def logits(self, hidden, target_ids, attention_mask):
            return self._logits

    benign = _FakeReader(torch.tensor([[-5.0, 5.0]]))    # already scores benign (col 1 high)
    harmful = _FakeReader(torch.tensor([[5.0, -5.0]]))   # scores harmful (col 0 high)
    # the attacker minimizes evasion_loss -> lower when the reader already says benign
    assert benign.evasion_loss(None, None, None) < harmful.evasion_loss(None, None, None)
