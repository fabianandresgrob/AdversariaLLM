"""
Magpie-Align/Magpie-Llama-3.1-Pro-300K-Filtered — instruction-following benign data.

@article{xu2024magpie,
  title={Magpie: Alignment Data Synthesis from Scratch by Prompting Aligned LLMs with Nothing},
  author={Xu, Zhangchen and Jiang, Fengqing and Niu, Luyao and Deng, Yuntian and Poovendran, Radha and Choi, Yejin and Lin, Bill Yuchen},
  journal={arXiv preprint arXiv:2406.08464},
  year={2024}
}
"""

from dataclasses import dataclass

from datasets import load_dataset

from ..types import Conversation

from .prompt_dataset import PromptDataset


@dataclass
class MagpieConfig:
    name: str = "magpie"
    seed: int = 0
    idx: list[int] | int | str | None = None
    shuffle: bool = True


@PromptDataset.register("magpie")
class MagpieDataset(PromptDataset):
    def __init__(self, config: MagpieConfig):
        super().__init__(config)
        # arrow-backed (memory-mapped); only the selected window is materialized.
        dataset = load_dataset("Magpie-Align/Magpie-Llama-3.1-Pro-300K-Filtered", split="train")
        self.idx, self.config_idx = self._select_idx(config, len(dataset))
        self.data = dataset.select(self.idx.tolist())

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> Conversation:
        row = self.data[idx]
        return [
            {"role": "user", "content": row["instruction"]},
            {"role": "assistant", "content": row["response"]},
        ]
