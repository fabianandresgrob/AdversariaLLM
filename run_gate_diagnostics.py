import hydra

from adversariallm.training.gate_diagnostics import run_gate_diagnostics


@hydra.main(version_base=None, config_path="conf", config_name="gate_diagnostics")
def main(cfg):
    run_gate_diagnostics(cfg)


if __name__ == "__main__":
    main()
