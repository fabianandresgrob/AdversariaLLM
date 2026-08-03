import logging

import hydra
from omegaconf import DictConfig

from adversariallm.training.coop_loop import run_coop_training

log = logging.getLogger(__name__)


@hydra.main(config_path="./conf", config_name="coop_train", version_base="1.3")
def main(cfg: DictConfig) -> None:
    try:
        run_coop_training(cfg)
    except Exception:
        log.exception("cooperative training run crashed")
        raise


if __name__ == "__main__":
    main()
