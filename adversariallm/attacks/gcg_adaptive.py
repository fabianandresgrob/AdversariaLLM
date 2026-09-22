"""Adaptive (detector-aware) GCG: one attack that knows the cooperative pipeline's probe.

Standard GCG optimises a suffix to elicit an affirmative target from the model. Adaptive GCG
optimises the same suffix against the model AND the probe that guards it:

    loss = (1 - detector_loss_coeff) * elicit_target + detector_loss_coeff * evade_detector

in both the gradient step and the candidate ranking, exactly as the co-training loop's
detector-aware attacker does (coop attack.use_detector). The mechanics live in GCGAttack, which
already implements the detector hooks; this subclass exists so the adaptive attack is a NAME --
`attack=gcg_adaptive` -- rather than a pair of overrides on `gcg`.

That naming is what keeps the results honest: run outputs are filed under
outputs/<attack>__<defense>__<model>/, so running adaptive GCG as overrides on `gcg` writes into
the same directory as the vanilla run and the two silently merge into one ASR cell.

It cannot run against a runtime defense (TargetSystem filters generated text and exposes no
gradients; see defenses.registry.DEFENSE_COMPATIBLE_ATTACKS). The full pipeline evaluation is two
stages, which is a property of optimisation attacks, not a limitation of this one:

    1. attack=gcg_adaptive defense=none          # attacker optimises against model + probe
    2. attack=replay defense=coop_probe source=<stage-1 results dir>   # probe filters at runtime

Stage 1 is where the attacker's knowledge of the probe is used; stage 2 is where the probe acts.
"""

from dataclasses import dataclass

from .gcg import GCGAttack, GCGConfig


@dataclass
class GCGAdaptiveConfig(GCGConfig):
    """GCGConfig with the detector terms promoted from optional to required.

    detector_loss_coeff defaults to 0.5 (an even split between eliciting the target and evading
    the probe) rather than GCGConfig's 0.0, which would make this attack identical to `gcg`.
    """

    name: str = "gcg_adaptive"
    detector_loss_coeff: float = 0.5


class GCGAdaptiveAttack(GCGAttack):
    """GCG whose objective includes the coop probe's evasion loss.

    Validates up front that a probe is actually configured: GCGAttack treats a missing checkpoint
    or a zero coefficient as "not detector-aware" and silently runs the standard attack, which
    under this name would produce a vanilla-GCG result labelled adaptive.
    """

    def __init__(self, config):
        checkpoint = getattr(config, "detector_checkpoint", None)
        coeff = float(getattr(config, "detector_loss_coeff", 0.0) or 0.0)
        if not checkpoint:
            raise ValueError(
                "gcg_adaptive needs attacks.gcg_adaptive.detector_checkpoint (a coop <tag>_reader.pt). "
                "Without it the attack is plain gcg; use attack=gcg if that is what you want."
            )
        if coeff <= 0.0:
            raise ValueError(
                f"gcg_adaptive needs attacks.gcg_adaptive.detector_loss_coeff > 0, got {coeff}. "
                "At 0 the probe term vanishes and the attack is plain gcg."
            )
        super().__init__(config)
