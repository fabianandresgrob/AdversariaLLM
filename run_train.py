import logging

import hydra
from omegaconf import DictConfig

from adversariallm.training.loop import run_training

log = logging.getLogger(__name__)


@hydra.main(config_path="./conf", config_name="train", version_base="1.3")
def main(cfg: DictConfig) -> None:
    try:
        run_training(cfg)
    except Exception:
        # Otherwise the traceback only reaches stderr and never run_train.log.
        log.exception("training run crashed")
        raise


if __name__ == "__main__":
    main()
