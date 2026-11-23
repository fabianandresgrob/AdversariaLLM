"""R1.6: EuroSafeGuard's prompt probe on prompts under coop's training attack (context/next_steps_2026-11-10.md).

    uv run python prompt_probe_embedding_attack.py variant=plain
    uv run python prompt_probe_embedding_attack.py variant=aware

Base Llama-3.1-8B-Instruct under Meta's chat template (the probe's training setup, and track E's), coop's training
behaviors split as in the W runs (76 train behaviors with all 12 targets, 24 val behaviors with 6). Every
(behavior, target) gets coop's embedding attack (ContinuousEmbeddingAttack with W's iters/eps/lr, user message only,
the target sentence as the objective). Two variants:

  plain   the attack elicits the target only, as against a model without a probe
  aware   the attack also pushes the probe's score down: loss (1 - c) * target CE + c * softplus(probe logit), the
          same split as coop's detector-aware attack against our answer probe (softplus(logit) = CE toward benign)

The probe reads the attacked prompt as in their release: all hidden_states in probe_config["layers"], a logit per
prompt token, EMA over the tokens, max. The attack hands a detector one layer only, so PromptProbeDetector keeps all
layers of the model's forward through a hook. Per row it records the probe logit of the clean and of the attacked
prompt, the attack's final losses, and the greedy answer to the attacked prompt.

Writes <out>/<name>/<split>.jsonl (appended per behavior, a restart skips finished behaviors) and, at the end,
outputs/eval/embedding_attack/<name>.json for score_embedding_attack.py (StrongREJECT on gen_attacked). The probe's
benign reference (WildChat-clean, Alpaca) comes from R1.2.
"""

import json
import logging
import time
from pathlib import Path

import hydra
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf

from adversariallm.defenses.monitors.eurosafeguard_probe import ema_max, load_probe, prompt_ids, score_ids, snapshot

log = logging.getLogger(__name__)


def probe_logit(model, probe, layers, M, inputs_embeds, mask) -> torch.Tensor:
    """EMA-max probe logit of the positions in `mask`, from embeddings (with gradients if enabled)."""
    hs = model(inputs_embeds=inputs_embeds, attention_mask=mask.long(), output_hidden_states=True).hidden_states
    return ema_max(probe(torch.cat([hs[layer] for layer in layers], dim=-1)), mask.bool(), M)


class PromptProbeDetector:
    """The prompt probe as the attack's detector. The attack passes evasion_loss one layer's hidden states; the probe
    reads all of `layers`, so a forward hook keeps the hidden_states of the model's last forward (the attack's)."""

    def __init__(self, model, probe, layers, M):
        self.probe, self.layers, self.M, self.hidden = probe, list(layers), int(M), None
        model.register_forward_hook(self._keep)

    def _keep(self, module, args, output):
        self.hidden = getattr(output, "hidden_states", None)

    def evasion_loss(self, hidden, target_ids, attention_mask):
        if self.hidden is None:
            raise RuntimeError("no hidden_states from the attack's forward")
        prompt = (target_ids == 0) & attention_mask.bool()  # 0 = prompt or padding in target_ids
        psi = torch.cat([self.hidden[layer] for layer in self.layers], dim=-1)
        self.hidden = None
        return F.softplus(ema_max(self.probe(psi), prompt, self.M)).mean()


def done_behaviors(path: Path) -> set[str]:
    if not path.exists():
        return set()
    out = set()
    for line in path.read_text().splitlines():
        try:
            out.add(json.loads(line)["behavior"])
        except (json.JSONDecodeError, KeyError):
            pass  # a line cut by a kill: that behavior runs again
    return out


@hydra.main(version_base=None, config_path="conf", config_name="prompt_probe_embedding_attack")
def main(cfg: DictConfig) -> None:
    from transformers import AutoTokenizer

    from adversariallm.io_utils import load_model_and_tokenizer
    from adversariallm.training.attacks import ContinuousEmbeddingAttack, embedding_scale
    from adversariallm.training.data import AdvTupleStream, collate_adv, split_adv_stream
    from inspect_embedding_attack import generate

    log.info("config:\n%s", OmegaConf.to_yaml(cfg))
    if cfg.variant not in ("plain", "aware"):
        raise ValueError(f"variant must be plain or aware, got {cfg.variant!r}")
    if not torch.cuda.is_available():
        raise RuntimeError("no GPU visible")
    torch.manual_seed(int(cfg.seed))
    name = cfg.name or f"pp16-{cfg.variant}"

    model, tok = load_model_and_tokenizer(OmegaConf.to_container(cfg.models[cfg.model], resolve=True))
    if cfg.template == "theirs":
        tok.chat_template = AutoTokenizer.from_pretrained(cfg.base_model).chat_template
    elif cfg.template != "ours":
        raise ValueError(f"template must be theirs or ours, got {cfg.template!r}")
    model.eval()
    model.requires_grad_(False)
    device = next(model.parameters()).device
    log.info(
        "template %s, first prompt: %r",
        cfg.template,
        tok.apply_chat_template([{"role": "user", "content": "X"}], tokenize=False, add_generation_prompt=True),
    )

    probe, pcfg = load_probe(snapshot(cfg.probe_repo), device)
    probe.requires_grad_(False)
    layers, M = pcfg["layers"], pcfg["M"]
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    aware = cfg.variant == "aware"
    detector = PromptProbeDetector(model, probe, layers, M) if aware else None
    attack = ContinuousEmbeddingAttack(
        model.get_input_embeddings().weight,
        None,
        tok,
        iters=int(cfg.attack.iters),
        eps=float(cfg.attack.eps),
        lr=float(cfg.attack.lr),
        detector_loss_coeff=float(cfg.attack.detector_loss_coeff),
        target_eot=bool(cfg.attack.target_eot),
        optimizer=str(cfg.attack.optimizer),
        relative_lr=bool(cfg.attack.relative_lr),
        perturb=str(cfg.attack.perturb),
        embed_scale=embedding_scale(model),
    )

    ds = AdvTupleStream(cfg.data.dir, cfg.data.behaviors, cfg.data.targets, cfg.data.safe, tok, cfg.base_model)
    subsets = dict(
        zip(
            ("train", "val"),
            split_adv_stream(
                ds, val_size=int(cfg.data.val_size), seed=int(cfg.data.val_seed), val_targets=int(cfg.data.val_targets)
            ),
        )
    )
    out_dir = Path(cfg.out) / name
    out_dir.mkdir(parents=True, exist_ok=True)
    for split in cfg.splits:
        groups: dict[str, list[int]] = {}
        for i in subsets[split].indices:
            groups.setdefault(ds.rows[i][0], []).append(i)
        path = out_dir / f"{split}.jsonl"
        done = done_behaviors(path)
        log.info(
            "%s: %d behaviors, %d attacks, %d behaviors already in %s",
            split,
            len(groups),
            sum(map(len, groups.values())),
            len(done),
            path,
        )
        for b_i, (behavior, idx) in enumerate(groups.items()):
            if behavior in done:
                continue
            t0 = time.time()
            batch = collate_adv([ds[i] for i in idx])
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            prompt_len = int((batch["h_targetids"][0] != 0).float().argmax())  # same prompt for every target
            clean = prompt_ids(tok, behavior, 10**9)
            if clean != batch["h_ids"][0, :prompt_len].tolist():
                raise ValueError(f"attack prompt and probe prompt differ for {behavior[:60]!r}")
            score_clean = float(score_ids(model, probe, [clean[-pcfg["max_len"] :]], layers, M, pad)[0])

            perturbed = attack.attack(model, batch, detector=detector, use_detector=aware)
            adv = perturbed[:, :prompt_len].detach().to(model.dtype)
            ones = torch.ones(adv.shape[:2], dtype=torch.bool, device=device)
            with torch.no_grad():
                scores = probe_logit(model, probe, layers, M, adv, ones).float().tolist()
                if b_i == 0:  # the embedding path gives the same score as the token path on the clean prompt
                    emb = model.get_input_embeddings()(batch["h_ids"][:1, :prompt_len]) * embedding_scale(model)
                    check = float(probe_logit(model, probe, layers, M, emb, ones[:1])[0])
                    log.info("clean score via tokens %.4f, via embeddings %.4f", score_clean, check)
            answers = generate(model, tok, int(cfg.max_new_tokens), inputs_embeds=adv)

            with path.open("a") as f:
                for r, i in enumerate(idx):
                    f.write(
                        json.dumps(
                            {
                                "split": split,
                                "variant": cfg.variant,
                                "behavior": behavior,
                                "target_index": r,
                                "target": ds.rows[i][1],
                                "n_prompt_tokens": prompt_len,
                                "score_clean": score_clean,
                                "score_attacked": scores[r],
                                "target_loss_end": attack.last_target_loss,
                                "detector_loss_end": attack.last_detector_loss,
                                "gen_attacked": answers[r],
                            }
                        )
                        + "\n"
                    )
            log.info(
                "%s %d/%d: probe %.2f -> mean %.2f, target loss %.3f, detector loss %.3f, %.0f s",
                split,
                b_i + 1,
                len(groups),
                score_clean,
                sum(scores) / len(scores),
                attack.last_target_loss,
                attack.last_detector_loss,
                time.time() - t0,
            )

    rows = [json.loads(line) for split in cfg.splits for line in (out_dir / f"{split}.jsonl").read_text().splitlines()]
    judge_dir = Path(cfg.root_dir) / "outputs" / "eval" / "embedding_attack"
    judge_dir.mkdir(parents=True, exist_ok=True)
    (judge_dir / f"{name}.json").write_text(json.dumps(rows, indent=1))
    log.info("%d rows -> %s and %s", len(rows), out_dir, judge_dir / f"{name}.json")


if __name__ == "__main__":
    main()
