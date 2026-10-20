import torch


def test_answer_is_cut_to_max_tokens_and_prompt_kept(monkeypatch):
    import adversariallm.training.data as data
    from probe_prompt_conditions import example

    # prompt of 3 tokens, answer of 6 (target_ids 0 on the prompt)
    ids = torch.tensor([1, 2, 3, 10, 11, 12, 13, 14, 15])
    tgt = torch.tensor([0, 0, 0, 10, 11, 12, 13, 14, 15])
    monkeypatch.setattr(data, "build_example_full", lambda p, a, t: (ids, None, tgt, torch.ones_like(ids)))
    monkeypatch.setattr(data, "user_token_mask", lambda t, p, n: torch.zeros(n, dtype=torch.bool))
    ex = example("p", "a", None, max_answer_tokens=4)
    assert ex["h_ids"].tolist() == [1, 2, 3, 10, 11, 12, 13]
    assert ex["h_targetids"].tolist() == [0, 0, 0, 10, 11, 12, 13]
    assert ex["h_perturb_mask"].numel() == 7


def test_prompt_is_stripped_before_the_template(monkeypatch):
    import adversariallm.training.data as data
    from probe_prompt_conditions import example

    seen = []
    ids = torch.tensor([1, 2, 10])
    monkeypatch.setattr(data, "build_example_full", lambda p, a, t: (seen.append(p), (ids, None, torch.tensor([0, 0, 10]), torch.ones(3)))[1])
    monkeypatch.setattr(data, "user_token_mask", lambda t, p, n: (seen.append(p), torch.zeros(n, dtype=torch.bool))[1])
    example("  You are a hacker. ", "a", None, max_answer_tokens=8)
    assert seen == ["You are a hacker.", "You are a hacker."]
