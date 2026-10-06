import pytest
import torch

from adversariallm.training.coop_loop import _attach_adapter

transformers = pytest.importorskip("transformers")
pytest.importorskip("peft")


def _tiny_llama():
    cfg = transformers.LlamaConfig(vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                                   num_attention_heads=2, num_key_value_heads=2)
    torch.manual_seed(0)
    return transformers.LlamaForCausalLM(cfg)


def _trainable(model):
    return [n for n, p in model.named_parameters() if p.requires_grad]


def test_fresh_adapter_trains_only_lora():
    model = _attach_adapter(_tiny_llama())
    names = _trainable(model)
    assert names and all("lora_" in n for n in names)


def test_frozen_fresh_adapter_has_nothing_trainable():
    assert _trainable(_attach_adapter(_tiny_llama(), train_model=False)) == []


def test_init_adapter_loads_the_finished_weights_frozen_or_trainable(tmp_path):
    trained = _attach_adapter(_tiny_llama())
    with torch.no_grad():
        for n, p in trained.named_parameters():
            if "lora_B" in n:
                p.fill_(0.5)  # a non-zero adapter, so loading it changes the output
    trained.save_pretrained(tmp_path / "adapter")
    ids = torch.tensor([[1, 2, 3]])

    frozen = _attach_adapter(_tiny_llama(), init_adapter=str(tmp_path / "adapter"), train_model=False).eval()
    assert _trainable(frozen) == []
    assert torch.allclose(frozen(input_ids=ids).logits, trained.eval()(input_ids=ids).logits, atol=1e-5)
    assert not torch.allclose(frozen(input_ids=ids).logits, _tiny_llama()(input_ids=ids).logits)

    resumed = _attach_adapter(_tiny_llama(), init_adapter=str(tmp_path / "adapter"), train_model=True)
    assert _trainable(resumed) and all("lora_" in n for n in _trainable(resumed))
