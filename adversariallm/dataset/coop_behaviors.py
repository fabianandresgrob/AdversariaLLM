"""Coop's adversarial-training behaviors (the HarmBench adv-training file run_coop_train.py trains on), split into
train and val behaviors exactly as coop training splits them (training.data.split_adv_stream on the same files and
val_size / val_seed). Each behavior is one conversation: the behavior and its first training target.

Used to attack the training behaviors outside training, for example to collect a model's own answers under
Jailbreak-R1 for an offline probe (vault note "Content probe on a prompt-by-answer grid").
"""

import os
from dataclasses import dataclass, field

from ..types import Conversation
from .prompt_dataset import PromptDataset


@dataclass
class CoopBehaviorsConfig:
    name: str = "coop_behaviors"
    data_dir: str = "./data"
    behaviors: str = "behavior_datasets/extra_behavior_datasets/adv_training_behaviors.csv"
    targets: str = "optimizer_targets/extra_targets/adv_training_targets.json"
    safe: str = "safe_responses/adv_training_behaviors_safe_responses_longer.csv"
    split: str = "train"  # train, val or all: the behaviors coop trains on, the ones it validates on, or both
    val_size: int = 24  # as data.val_size and data.val_seed of the W runs
    val_seed: int = 0
    exclude_categories: list[str] = field(default_factory=list)  # Category column, e.g. ["copyright"]
    seed: int = 0
    idx: list[int] | int | str | None = None
    shuffle: bool = False  # keep the training file's order; idx then selects from it


@PromptDataset.register("coop_behaviors")
class CoopBehaviorsDataset(PromptDataset):
    def __init__(self, config: CoopBehaviorsConfig):
        super().__init__(config)
        from ..training.data import AdvTupleStream, split_adv_stream  # training imports stay out of module load

        stream = AdvTupleStream(config.data_dir, config.behaviors, config.targets, config.safe, None, None)
        first_target = {}
        for behavior, target, _ in stream.rows:
            first_target.setdefault(behavior, target)
        split = getattr(config, "split", "train")
        if split == "all":
            behaviors = list(first_target)
        elif split in ("train", "val"):
            train, val = split_adv_stream(stream, val_size=int(config.val_size), seed=int(config.val_seed))
            subset = train if split == "train" else val
            behaviors = list(dict.fromkeys(stream.rows[i][0] for i in subset.indices))
        else:
            raise ValueError(f"split must be train, val or all, got {split!r}")
        excluded = set(getattr(config, "exclude_categories", None) or [])
        behaviors = [b for b in behaviors if stream.categories.get(b, "") not in excluded]

        self.idx, self.config_idx = self._select_idx(config, len(behaviors))
        self.messages = [behaviors[i] for i in self.idx]
        self.targets = [first_target[b] for b in self.messages]
        self.source = os.path.join(config.data_dir, config.behaviors)

    def __len__(self):
        return len(self.messages)

    def __getitem__(self, idx: int) -> Conversation:
        return [
            {"role": "user", "content": self.messages[idx]},
            {"role": "assistant", "content": self.targets[idx]},
        ]
