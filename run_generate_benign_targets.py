import hydra

from adversariallm.training.generate_benign_targets import run_generate_benign_targets


@hydra.main(version_base=None, config_path="conf", config_name="generate_benign_targets")
def main(cfg):
    run_generate_benign_targets(cfg)


if __name__ == "__main__":
    main()
