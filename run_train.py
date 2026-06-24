import hydra
from omegaconf import DictConfig

from adversariallm.training.loop import run_training


@hydra.main(config_path="./conf", config_name="train", version_base="1.3")
def main(cfg: DictConfig) -> None:
    run_training(cfg)


if __name__ == "__main__":
    main()
