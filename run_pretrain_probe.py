import logging

import hydra
from omegaconf import DictConfig

from adversariallm.training.pretrain_probe import run_pretrain_probe

log = logging.getLogger(__name__)


@hydra.main(config_path="./conf", config_name="pretrain_probe", version_base="1.3")
def main(cfg: DictConfig) -> None:
    try:
        run_pretrain_probe(cfg)
    except Exception:
        log.exception("probe pretraining crashed")
        raise


if __name__ == "__main__":
    main()
