"""Run coop training's embedding attack on a few training behaviors and write what it produces.

    pixi run --frozen python inspect_embedding_attack.py model=meta-llama/Meta-Llama-3.1-8B-Instruct
    pixi run --frozen python inspect_embedding_attack.py model=E-nd6-s0 n_behaviors=5

For each behavior (from the coop training split) and each of its targets (adv_training_targets.json,
12 per behavior) this runs the exact attack coop trains against -- ContinuousEmbeddingAttack with
coop_train.yaml's iters/eps/lr, detector-aware when the model has a probe -- and records:

  gen_clean      the model's answer to the unperturbed prompt (once per behavior)
  gen_attacked   the answer generated from the perturbed prompt embeddings (what coop's attack elicits)
  projected      the perturbed prompt snapped back to the nearest vocabulary token at every position,
                 how many positions changed, and the answer generated from that text

The coop target ends in the end-of-turn token (render_full terminates the assistant turn); with
attack.target_eot=true the attack optimizes "say the stub, then stop" and the answer is just the
target. false (default here and in coop_train.yaml) leaves that token out of the loss, so the
answer continues past the target. perturb picks what the attack may move: every prompt position
(all, coop's original), the user message only (user), or a pgd-style placeholder suffix appended
to it (suffix). attack.optimizer=sign with relative_lr=true and lr=0.001 is pgd's update rule.

eps bounds each token's perturbation by eps x the mean embedding norm (per token, L2) -- as far as
a typical embedding is long -- so the projection shows whether the attacked prompt is still text.
With perturb=all every non-target position is perturbed, chat-template tokens included, as in coop.

Writes outputs/eval/embedding_attack/<name>.md (to read) and .json (everything, untruncated).
"""

import json
import logging
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

log = logging.getLogger(__name__)


def nearest_tokens(embeds, weight, chunk: int = 256):
    """Nearest vocabulary row (L2) for each embedding: (P, D) -> (P,) token ids."""
    import torch

    w = weight.float()
    w_sq = (w * w).sum(dim=1)
    out = []
    for start in range(0, embeds.size(0), chunk):
        x = embeds[start:start + chunk].float()
        # argmin ||x - w||^2 = argmax 2 x.w - ||w||^2
        out.append((2 * x @ w.T - w_sq).argmax(dim=1))
    return torch.cat(out)


def generate(model, tokenizer, max_new_tokens, *, inputs_embeds=None, input_ids=None) -> list[str]:
    import torch

    x = inputs_embeds if inputs_embeds is not None else input_ids
    attn = torch.ones(x.shape[:2], dtype=torch.long, device=x.device)
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    with torch.no_grad():
        out = model.generate(inputs_embeds=inputs_embeds, input_ids=input_ids, attention_mask=attn,
                             max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=pad)
    if input_ids is not None:  # decoder-only generate returns prompt + new tokens for ids, new only for embeds
        out = out[:, input_ids.size(1):]
    return tokenizer.batch_decode(out, skip_special_tokens=True)


def render(records: list[dict], header: str, max_chars: int) -> str:
    lines = [header, ""]
    current = None
    for r in records:
        if r["behavior"] != current:
            current = r["behavior"]
            lines += [f"## {current}", "", f"**clean answer (no attack)**:\n```\n{r['gen_clean'][:max_chars]}\n```", ""]
        lines += [
            f"### target {r['target_index']}: {r['target']}",
            f"loss {r['loss_start']:.3f} -> {r['loss_end']:.3f} | perturbation norm / eps: mean "
            f"{r['delta_over_eps_mean']:.2f} | projection changed {r['n_changed']}/{r['n_prompt_tokens']} "
            f"prompt tokens ({r['n_changed_user']}/{r['n_user_tokens']} in the user message)",
            "",
            f"**answer under the embedding attack**:\n```\n{r['gen_attacked'][:max_chars]}\n```",
            f"**attacked prompt snapped back to tokens**:\n```\n{r['projected_prompt'][:max_chars]}\n```",
            f"**answer to that snapped-back text**:\n```\n{r['gen_projected'][:max_chars]}\n```",
            "",
        ]
    return "\n".join(lines)


@hydra.main(version_base=None, config_path="conf", config_name="inspect_embedding_attack")
def main(cfg: DictConfig) -> None:
    import torch

    from adversariallm.io_utils import load_model_and_tokenizer
    from adversariallm.training.attacks import ContinuousEmbeddingAttack
    from adversariallm.training.data import (AdvTupleStream, build_example_full, pad_collate, render_prompt,
                                             split_adv_stream, user_token_mask)
    from adversariallm.training.readers import load_reader

    torch.manual_seed(int(cfg.seed))
    entry = OmegaConf.to_container(cfg.models[cfg.model], resolve=True)
    model, tokenizer = load_model_and_tokenizer(entry)
    model.eval()
    model.requires_grad_(False)
    device = next(model.parameters()).device
    weight = model.get_input_embeddings().weight

    use_detector = cfg.use_detector if cfg.use_detector is not None else bool(entry.get("reader_path"))
    detector = load_reader(entry["reader_path"]).to(device) if use_detector else None
    attack = ContinuousEmbeddingAttack(weight, None, tokenizer, iters=int(cfg.attack.iters), eps=float(cfg.attack.eps),
                                       lr=float(cfg.attack.lr), detector_loss_coeff=float(cfg.attack.detector_loss_coeff),
                                       detector_layer=int(cfg.layer), target_eot=bool(cfg.attack.target_eot),
                                       optimizer=str(cfg.attack.optimizer), relative_lr=bool(cfg.attack.relative_lr))
    core = attack._attack
    if cfg.perturb not in ("all", "user", "suffix"):
        raise ValueError(f"perturb must be all | user | suffix, got {cfg.perturb!r}")
    suffix = str(cfg.suffix) if cfg.perturb == "suffix" else ""

    ds = AdvTupleStream(cfg.data.dir, cfg.data.behaviors, cfg.data.targets, cfg.data.safe, tokenizer, cfg.chat_template_id)
    train_ds, _ = split_adv_stream(ds, val_size=int(cfg.data.val_size), seed=int(cfg.data.val_seed))
    targets_by_behavior: dict[str, list[str]] = {}
    for i in train_ds.indices:
        x, y_h, _ = ds.rows[i]
        targets_by_behavior.setdefault(x, []).append(y_h)
    behaviors = list(targets_by_behavior)[: int(cfg.n_behaviors)]
    keys = ("h_ids", "h_labels", "h_targetids", "h_attn")

    records = []
    for b_i, behavior in enumerate(behaviors):
        prompt_text = behavior + suffix
        items = [dict(zip(keys, build_example_full(prompt_text, y, tokenizer))) for y in targets_by_behavior[behavior]]
        batch = {k: v.to(device) for k, v in pad_collate(items, list(keys), pad_id=0).items()}
        prompt_len = int((batch["h_targetids"][0] != 0).float().argmax())  # same prompt for every target

        # the user message in the rendered prompt; perturb=suffix narrows to the appended suffix
        length = batch["h_ids"].size(1)
        user = user_token_mask(tokenizer, prompt_text, length).to(device)
        span = (len(behavior), len(prompt_text)) if suffix else None
        perturb_mask = None if cfg.perturb == "all" else \
            user_token_mask(tokenizer, prompt_text, length, span).to(device).expand_as(batch["h_attn"])

        clean_ids = torch.tensor([tokenizer(render_prompt(tokenizer, behavior), add_special_tokens=False)["input_ids"]],
                                 device=device)
        gen_clean = generate(model, tokenizer, int(cfg.max_new_tokens), input_ids=clean_ids)[0]

        result = core.attack(model, batch["h_ids"], batch["h_targetids"], batch["h_attn"],
                             detector=detector, use_detector=use_detector, perturb_mask=perturb_mask)
        delta, perturbed, losses = result[1], result[3], result[6]
        adv_prompt = perturbed[:, :prompt_len].to(weight.dtype)
        gen_attacked = generate(model, tokenizer, int(cfg.max_new_tokens), inputs_embeds=adv_prompt)

        delta_norm = delta[:, :prompt_len].float().norm(dim=-1)  # (B, P)
        if perturb_mask is not None:
            delta_norm = delta_norm[:, perturb_mask[0, :prompt_len]]
        proj_ids = torch.stack([nearest_tokens(adv_prompt[r], weight) for r in range(adv_prompt.size(0))])
        gen_projected = generate(model, tokenizer, int(cfg.max_new_tokens), input_ids=proj_ids)
        changed = proj_ids != batch["h_ids"][:1, :prompt_len]
        for r in range(len(items)):
            records.append({
                "behavior": behavior, "target_index": r, "target": targets_by_behavior[behavior][r],
                "loss_start": float(losses[0]), "loss_end": float(losses[-1]),
                "delta_over_eps_mean": float(delta_norm[r].mean() / core.eps),
                "n_prompt_tokens": prompt_len, "n_changed": int(changed[r].sum()),
                "n_user_tokens": int(user.sum()), "n_changed_user": int(changed[r][user[:prompt_len]].sum()),
                "gen_clean": gen_clean, "gen_attacked": gen_attacked[r],
                "projected_prompt": tokenizer.decode(proj_ids[r], skip_special_tokens=False),
                "gen_projected": gen_projected[r],
            })
        log.info(f"behavior {b_i + 1}/{len(behaviors)} done (loss {losses[0]:.3f} -> {losses[-1]:.3f}; "
                 f"batch mean over targets)")

    out = Path(cfg.root_dir) / "outputs" / "eval" / "embedding_attack"
    out.mkdir(parents=True, exist_ok=True)
    name = cfg.name or str(cfg.model).replace("/", "_")
    (out / f"{name}.json").write_text(json.dumps(records, indent=2))
    header = (f"# Embedding attack on {cfg.model}\n\niters={cfg.attack.iters} eps={cfg.attack.eps} "
              f"lr={cfg.attack.lr} optimizer={cfg.attack.optimizer} detector-aware={use_detector} "
              f"target_eot={cfg.attack.target_eot} perturb={cfg.perturb} "
              f"| {len(behaviors)} behaviors x "
              f"{len(records) // max(len(behaviors), 1)} targets | greedy, {cfg.max_new_tokens} new tokens")
    (out / f"{name}.md").write_text(render(records, header, int(cfg.max_chars)))
    print(f"{len(records)} attacks -> {out / name}.md")


if __name__ == "__main__":
    main()
