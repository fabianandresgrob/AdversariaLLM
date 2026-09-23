import torch

from build_answer_pool import pool_rows, render


def test_one_plain_answer_then_one_per_target_with_the_coop_split():
    rows = [("steal", "Sure, a", "no"), ("steal", "Here is b", "no"), ("hack", "Sure, c", "no")]
    jobs = pool_rows(rows, val_behaviors={"hack"})
    assert [(j["behavior"], j["prefill"], j["target_index"], j["target"]) for j in jobs] == [
        ("steal", "none", None, ""), ("steal", "target", 0, "Sure, a"), ("steal", "target", 1, "Here is b"),
        ("hack", "none", None, ""), ("hack", "target", 0, "Sure, c")]
    assert [j["split"] for j in jobs] == ["train", "train", "train", "val", "val"]


def test_render_summarizes_hits_per_prefill_mode():
    recs = [{"behavior": "b", "split": "train", "prefill": "none", "target_index": None, "completion": "x", "p_harmful": 0.9},
            {"behavior": "b", "split": "train", "prefill": "target", "target_index": 0, "completion": "y", "p_harmful": 0.2}]
    md = render(recs, 0.5, 100, "# h")
    assert "prefill=none: 1/1 judged harmful" in md and "prefill=target: 0/1 judged harmful" in md
    assert "**prefill target 0** | p_harmful=0.200" in md


class _Tok:
    padding_side, pad_token, pad_token_id, eos_token_id, eos_token = "right", None, None, 0, "</s>"

    def __init__(self):
        self.seen = []

    def apply_chat_template(self, conv, tokenize=False, add_generation_prompt=False, **kw):
        return "U" + conv[0]["content"] + ("A" if add_generation_prompt else "")

    def __call__(self, texts, **kw):
        self.seen += texts
        ids = torch.ones(len(texts), 3, dtype=torch.long)

        class _Enc(dict):
            def to(self, device):
                return self
        return _Enc(input_ids=ids, attention_mask=ids)

    def batch_decode(self, ids, **kw):
        return ["gen"] * ids.size(0)


class _Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.p = torch.nn.Parameter(torch.zeros(1))

    def generate(self, input_ids, **kw):
        return torch.cat([input_ids, torch.full((input_ids.size(0), 2), 7)], dim=1)


def test_generate_responses_prefills_the_assistant_turn_and_returns_the_continuation():
    from run_calibrate_probe import generate_responses

    tok = _Tok()
    out = generate_responses(_Model(), tok, ["p1", "p2"], max_new_tokens=2, batch_size=8, prefills=["Sure,", ""])
    assert tok.seen == ["Up1ASure,", "Up2A"]
    assert out == ["gen", "gen"]
