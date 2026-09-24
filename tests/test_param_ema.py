import torch

from adversariallm.training.coop_loop import ParamEMA


def test_ema_tracks_the_average_and_swap_restores_the_live_weights():
    p = torch.nn.Parameter(torch.zeros(3, dtype=torch.bfloat16))
    ema = ParamEMA([p], decay=0.5)
    with torch.no_grad():
        p.fill_(2.0)
    ema.update()  # 0.5 * 0 + 0.5 * 2
    with torch.no_grad():
        p.fill_(4.0)
    ema.update()  # 0.5 * 1 + 0.5 * 4
    assert torch.allclose(ema.shadow[0], torch.full((3,), 2.5))
    with ema.swapped_in():
        assert torch.allclose(p.float(), torch.full((3,), 2.5))
    assert torch.allclose(p.float(), torch.full((3,), 4.0)) and p.dtype == torch.bfloat16
