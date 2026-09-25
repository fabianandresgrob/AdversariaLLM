import torch

from adversariallm.training.coop_loop import _replay_loss
from adversariallm.training.readers import LinearProbe


def test_replay_loss_trains_the_head_to_separate_buffered_features():
    torch.manual_seed(0)
    reader = LinearProbe(8)
    harmful = torch.randn(40, 8, dtype=torch.bfloat16) + 2.0  # buffered features come back bf16 from the cpu
    benign = torch.randn(40, 8, dtype=torch.bfloat16) - 2.0
    opt = torch.optim.Adam(reader.parameters(), lr=0.05)
    first = _replay_loss(reader, (harmful, benign), "cpu")
    for _ in range(50):
        opt.zero_grad()
        _replay_loss(reader, (harmful, benign), "cpu").backward()
        opt.step()
    p_harmful = torch.softmax(reader.linear(torch.cat([harmful, benign]).float()), dim=-1)[:, 0]
    assert _replay_loss(reader, (harmful, benign), "cpu") < first
    assert (p_harmful[:40] > 0.5).all() and (p_harmful[40:] < 0.5).all()  # harmful = column 0
