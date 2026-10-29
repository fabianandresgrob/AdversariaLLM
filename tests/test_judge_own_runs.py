from types import SimpleNamespace

from adversariallm.io_utils.database import check_match
from run_attacks import own_runs_filter


def test_own_runs_filter_matches_this_run_and_not_a_neighbour_started_in_the_same_second():
    mine = [SimpleNamespace(model="W-coop-s0-ema", attack="jailbreak_r1", defense=None)]
    f = own_runs_filter(mine)
    assert check_match({"model": "W-coop-s0-ema", "attack": "jailbreak_r1", "defense": None}, f)
    assert not check_match({"model": "W-coop-s0-ema", "attack": "jailbreak_r1", "defense": "coop_probe"}, f)
    assert not check_match({"model": "X-coop-qwen25-7b-s0-ema", "attack": "jailbreak_r1", "defense": None}, f)


def test_own_runs_filter_covers_every_cell_of_a_multi_cell_run():
    runs = [SimpleNamespace(model=m, attack="pair", defense=d) for m in ("a", "b") for d in (None, "coop_probe")]
    f = own_runs_filter(runs)
    assert all(check_match({"model": r.model, "attack": r.attack, "defense": r.defense}, f) for r in runs)
