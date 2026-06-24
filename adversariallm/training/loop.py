from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Objective:
    active_terms: set
    away_variant: str = "ce"
    lambda_away: float = 1.0
    lambda_toward: float = 1.0
    lambda_kl: float = 1.0
    beta: float = 0.1


def build_objective(cfg: dict) -> Objective:
    mode = cfg.get("model_objective", "ce")
    kl = {"kl"} if cfg.get("lambda_kl", 1.0) else set()
    if mode in ("ce", "ul"):
        return Objective(
            active_terms={"away", "toward"} | kl,
            away_variant=mode,
            lambda_away=cfg.get("lambda_away", 1.0),
            lambda_toward=cfg.get("lambda_toward", 1.0),
            lambda_kl=cfg.get("lambda_kl", 1.0),
        )
    if mode == "ipo":
        return Objective(active_terms={"ipo"} | kl, beta=cfg.get("beta", 0.1),
                         lambda_kl=cfg.get("lambda_kl", 1.0))
    raise ValueError(f"unknown model_objective: {mode}")
