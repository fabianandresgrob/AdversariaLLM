"""Batched greedy generation shared by probe calibration, the answer pool and coop training."""


def generate_responses(model, tokenizer, prompts, max_new_tokens: int, batch_size: int,
                       prefills: list[str] | None = None) -> list[str]:
    """Greedy generations for the calibration prompts — what the monitor scores at eval.

    Batched with left padding (generation reads the rightmost tokens), restoring the tokenizer's
    padding settings afterwards so nothing else in the process inherits them. `prefills` (one per
    prompt) starts the assistant turn with that text; the returned string is the continuation only."""
    from adversariallm.training.data import render_prompt

    device = next(model.parameters()).device
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    side, pad_token = tokenizer.padding_side, tokenizer.pad_token
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    generations: list[str] = []
    try:
        for start in range(0, len(prompts), batch_size):
            chunk = prompts[start : start + batch_size]
            heads = prefills[start : start + batch_size] if prefills else [""] * len(chunk)
            enc = tokenizer([render_prompt(tokenizer, p) + h for p, h in zip(chunk, heads)], return_tensors="pt",
                            padding=True, add_special_tokens=False).to(device)
            out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=pad_id)
            generations += tokenizer.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
    finally:
        tokenizer.padding_side, tokenizer.pad_token = side, pad_token
    return generations
