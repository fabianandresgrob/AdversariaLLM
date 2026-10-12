"""Coop training on other models than Llama 3.1: Gemma's scaled embeddings, each template's end of turn, and LoRA on
the language model only of a vision-language class."""
import pytest
import torch

from adversariallm.training.attacks import input_embedding_matrix
from adversariallm.training.coop_loop import LORA_TARGETS, lora_target_modules


class _Embedding(torch.nn.Embedding):
    pass


class _Model(torch.nn.Module):
    def __init__(self, scale=None, vision=False):
        super().__init__()
        self.embed = _Embedding(5, 3)
        if scale is not None:
            self.embed.register_buffer("embed_scale", torch.tensor(scale))
        if vision:
            self.vision_tower = torch.nn.Module()
            self.vision_tower.q_proj = torch.nn.Linear(3, 3)

    def get_input_embeddings(self):
        return self.embed


def test_embedding_matrix_applies_gemmas_scale_and_leaves_other_models_alone():
    plain = _Model()
    assert input_embedding_matrix(plain) is plain.embed.weight
    gemma = _Model(scale=8.0)
    assert torch.allclose(input_embedding_matrix(gemma), gemma.embed.weight * 8.0)


def test_lora_targets_the_language_model_only_when_there_is_a_vision_tower():
    assert lora_target_modules(_Model()) == list(LORA_TARGETS)
    pattern = lora_target_modules(_Model(vision=True))
    import re

    assert re.fullmatch(pattern, "model.language_model.layers.0.self_attn.q_proj")
    assert re.fullmatch(pattern, "language_model.model.layers.3.mlp.down_proj")
    assert not re.fullmatch(pattern, "model.vision_tower.vision_model.encoder.layers.0.self_attn.q_proj")


@pytest.mark.parametrize("model_id,token", [("meta-llama/Meta-Llama-3.1-8B-Instruct", "<|eot_id|>"),
                                            ("Qwen/Qwen2.5-7B-Instruct", "<|im_end|>"),
                                            ("google/gemma-3-12b-it", "<end_of_turn>")])
def test_end_of_turn_comes_from_each_models_template(model_id, token):
    transformers = pytest.importorskip("transformers")
    from adversariallm.training.data import end_of_turn_ids

    try:
        tok = transformers.AutoTokenizer.from_pretrained(model_id, local_files_only=True)
    except Exception:
        pytest.skip(f"{model_id} tokenizer not cached")
    ids = end_of_turn_ids(tok)
    assert tok.convert_tokens_to_ids(token) in ids
    assert tok.eos_token_id in ids
