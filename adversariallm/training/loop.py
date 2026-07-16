from __future__ import annotations

import os
from dataclasses import dataclass

import torch


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


# ---------------------------------------------------------------------------
# Training step
# ---------------------------------------------------------------------------

from .losses import (  # noqa: E402
    away_from_harmful,
    toward_benign,
    utility_kl,
    sequence_logprob,
    ipo_preference,
)


def _benign_under_adv_prompt(model, adv_embeds, adv_batch):
    """Build inputs_embeds + attn + labels for the benign continuation under the
    SAME (adversarial) prompt prefix as the harmful sequence.

    The harmful and benign examples share an identical prompt prefix (same x, same
    chat template, same response_key); they only differ in the response region.
    `adv_embeds` (B, Th, D) are the perturbed embeddings of the harmful sequence,
    so its prompt-region rows already carry the adversarial perturbation. We take
    those prompt rows and concatenate the (clean) benign response embeddings.

    Returns (embeds, attn, labels) where labels has the prompt region set to -100
    and the benign response region set to the benign target ids.
    """
    embed = model.get_input_embeddings()
    b_embeds = embed(adv_batch["b_ids"])  # (B, Tb, D)
    h_tgt = adv_batch["h_targetids"]      # (B, Th) 0 for prompt/pad
    h_attn = adv_batch["h_attn"]          # (B, Th)
    b_tgt = adv_batch["b_targetids"]      # (B, Tb)
    b_attn = adv_batch["b_attn"]          # (B, Tb)

    B = adv_embeds.size(0)
    D = adv_embeds.size(-1)
    device = adv_embeds.device

    rows_embeds, rows_attn, rows_labels = [], [], []
    for i in range(B):
        # prompt length = number of leading 0s in h_targetids that are attended to.
        # (response tokens are > 0; trailing pad is 0 but not attended.)
        h_resp_mask = (h_tgt[i] > 0)
        if h_resp_mask.any():
            p_len = int(h_resp_mask.nonzero()[0].item())
        else:
            p_len = int(h_attn[i].sum().item())
        prompt_rows = adv_embeds[i, :p_len]  # (p_len, D)

        # benign response rows: positions where b_targetids > 0
        b_resp_idx = (b_tgt[i] > 0).nonzero().squeeze(-1)
        resp_rows = b_embeds[i, b_resp_idx]               # (r_len, D)
        resp_ids = adv_batch["b_ids"][i, b_resp_idx]      # (r_len,)

        seq = torch.cat([prompt_rows, resp_rows], dim=0)  # (p_len+r_len, D)
        attn = torch.ones(seq.size(0), device=device, dtype=h_attn.dtype)
        lab = torch.full((seq.size(0),), -100, device=device, dtype=torch.long)
        lab[p_len:] = resp_ids
        rows_embeds.append(seq)
        rows_attn.append(attn)
        rows_labels.append(lab)

    max_len = max(s.size(0) for s in rows_embeds)
    embeds = torch.zeros(B, max_len, D, device=device, dtype=adv_embeds.dtype)
    attn = torch.zeros(B, max_len, device=device, dtype=h_attn.dtype)
    labels = torch.full((B, max_len), -100, device=device, dtype=torch.long)
    for i in range(B):
        L = rows_embeds[i].size(0)
        embeds[i, :L] = rows_embeds[i]
        attn[i, :L] = rows_attn[i]
        labels[i, :L] = rows_labels[i]
    return embeds, attn, labels


def train_step(model, ref, attack, objective, adv_batch, util_batch, device):
    """One model-CAT step. Returns scalar loss tensor + a dict of components for logging."""
    logs = {}
    total = torch.zeros((), device=device)

    # The attack optimizes its perturbation by backward()ing through the model, which
    # accumulates into the model's parameters; those grads must not reach opt.step().
    adv_embeds = attack.attack(model, adv_batch, detector=None, use_detector=False)
    model.zero_grad(set_to_none=True)

    if {"away", "toward"} & objective.active_terms:
        if "away" in objective.active_terms:
            logits_h = model(inputs_embeds=adv_embeds, attention_mask=adv_batch["h_attn"]).logits
            a = away_from_harmful(
                logits_h[:, :-1], adv_batch["h_labels"][:, 1:], variant=objective.away_variant
            )
            total = total + objective.lambda_away * a
            logs["away"] = a.item()
        if "toward" in objective.active_terms:
            # benign continuation under the SAME adversarial prompt
            be, b_attn, b_labels = _benign_under_adv_prompt(model, adv_embeds, adv_batch)
            logits_b = model(inputs_embeds=be, attention_mask=b_attn).logits
            t = toward_benign(logits_b[:, :-1], b_labels[:, 1:])
            total = total + objective.lambda_toward * t
            logs["toward"] = t.item()

    if "ipo" in objective.active_terms:
        # sequence log-probs of benign(chosen)/harmful(rejected) under model and
        # reference, both under the adversarial prompt prefix.
        be, b_attn, b_labels = _benign_under_adv_prompt(model, adv_embeds, adv_batch)
        logits_b = model(inputs_embeds=be, attention_mask=b_attn).logits
        logits_h = model(inputs_embeds=adv_embeds, attention_mask=adv_batch["h_attn"]).logits
        pi_b = sequence_logprob(logits_b[:, :-1], b_labels[:, 1:])
        pi_h = sequence_logprob(logits_h[:, :-1], adv_batch["h_labels"][:, 1:])

        ref_b_logits = ref.logits(inputs_embeds=be, attention_mask=b_attn)
        ref_h_logits = ref.logits(inputs_embeds=adv_embeds, attention_mask=adv_batch["h_attn"])
        ref_b = sequence_logprob(ref_b_logits[:, :-1], b_labels[:, 1:])
        ref_h = sequence_logprob(ref_h_logits[:, :-1], adv_batch["h_labels"][:, 1:])

        ipo = ipo_preference(pi_b, pi_h, ref_b, ref_h, beta=objective.beta)
        total = total + ipo
        logs["ipo"] = ipo.item()

    if "kl" in objective.active_terms:
        u_ids = util_batch["input_ids"]
        u_logits = model(input_ids=u_ids, attention_mask=util_batch["attn"]).logits
        r_logits = ref.logits(
            inputs_embeds=model.get_input_embeddings()(u_ids), attention_mask=util_batch["attn"]
        )
        k = utility_kl(u_logits, r_logits, attention_mask=util_batch["attn"])
        total = total + objective.lambda_kl * k
        logs["kl"] = k.item()

    logs["total"] = total.item()
    return total, logs


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


def _to_device(batch, device):
    out = {}
    for k, v in batch.items():
        out[k] = v.to(device) if torch.is_tensor(v) else v
    return out


def _cycle(loader):
    while True:
        for b in loader:
            yield b


def run_training(cfg):
    from omegaconf import OmegaConf
    from torch.utils.data import DataLoader

    from ..io_utils import load_model_and_tokenizer
    from ..defenses.monitors._activation_detector_model import get_chat_template
    from .attacks import ContinuousEmbeddingAttack
    from .reference import LoRADisableReference, FrozenModelReference
    from .data import AdvTupleStream, UtilityStream, collate_adv, collate_util, split_adv_stream

    container = OmegaConf.to_container(cfg, resolve=True)

    model_params = {
        "id": cfg.model.id,
        "tokenizer_id": cfg.model.tokenizer_id,
        "dtype": cfg.model.dtype,
        "trust_remote_code": cfg.model.trust_remote_code,
    }
    model, tokenizer = load_model_and_tokenizer(model_params)
    device = next(model.parameters()).device

    update_mode = cfg.model.update_mode
    if update_mode == "lora":
        import peft
        from peft import LoraConfig

        model = peft.get_peft_model(
            model,
            LoraConfig(
                r=8,
                lora_alpha=32,
                target_modules=[
                    "q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj",
                ],
                lora_dropout=0.05,
                task_type="CAUSAL_LM",
            ),
        )
        ref = LoRADisableReference(model)
    elif update_mode == "full":
        model.requires_grad_(True)
        model.train()
        frozen, _ = load_model_and_tokenizer(model_params)
        frozen.eval()
        frozen.requires_grad_(False)
        ref = FrozenModelReference(frozen)
    else:
        raise ValueError(f"unknown update_mode: {update_mode}")

    # data
    adv_ds = AdvTupleStream(
        data_dir=cfg.data.dir,
        behaviors_csv=cfg.data.behaviors,
        targets_json=cfg.data.targets,
        safe_csv=cfg.data.safe,
        tokenizer=tokenizer,
        model_name=cfg.model.id,
    )
    util_ds = UtilityStream(tokenizer, cfg.model.id, fraction=cfg.data.utility_fraction)

    adv_train_ds, adv_val_ds = split_adv_stream(
        adv_ds, val_size=cfg.data.val_size, seed=cfg.data.val_seed
    )

    adv_loader = DataLoader(
        adv_train_ds, batch_size=cfg.data.harmful_batch_size, shuffle=True, collate_fn=collate_adv
    )
    # Materialized once so every validation step scores the same held-out batches.
    val_batches = [
        _to_device(b, device)
        for b in DataLoader(
            adv_val_ds, batch_size=cfg.data.harmful_batch_size, shuffle=False, collate_fn=collate_adv
        )
    ]
    util_loader = DataLoader(
        util_ds, batch_size=cfg.data.utility_batch_size, shuffle=True, collate_fn=collate_util
    )

    # attack
    _, _, response_key, _, _ = get_chat_template(cfg.model.id)
    attack = ContinuousEmbeddingAttack(
        model.get_input_embeddings().weight,
        response_key,
        tokenizer,
        iters=cfg.attack.iters,
        eps=cfg.attack.eps,
        lr=cfg.attack.lr,
    )

    # objective: flatten the keys build_objective expects
    objective = build_objective(
        {
            "model_objective": container.get("model_objective", "ce"),
            "lambda_away": container.get("lambda_away", 1.0),
            "lambda_toward": container.get("lambda_toward", 1.0),
            "lambda_kl": container.get("lambda_kl", 1.0),
            "beta": container.get("beta", 0.1),
        }
    )

    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(trainable, lr=cfg.training.lr)

    adv_iter = _cycle(adv_loader)
    util_iter = _cycle(util_loader)

    run_name = container.get("name", None) or "run"
    out_dir = os.path.join(cfg.output.checkpoint_path, run_name)
    os.makedirs(out_dir, exist_ok=True)

    best_val = float("inf")
    model.train()
    for step in range(cfg.training.n_steps):
        adv_batch = _to_device(next(adv_iter), device)
        util_batch = _to_device(next(util_iter), device)

        loss, logs = train_step(model, ref, attack, objective, adv_batch, util_batch, device)
        loss.backward()
        opt.step()
        opt.zero_grad()

        print(f"[step {step}] " + " ".join(f"{k}={v:.4f}" for k, v in logs.items()))

        if (step + 1) % cfg.training.val_every == 0:
            val = _validate(model, attack, val_batches, device)
            print(f"[step {step}] val_toward={val:.4f}")
            if val < best_val:
                best_val = val
                _save_checkpoint(model, container, step, out_dir, update_mode, best=True)

    _save_checkpoint(model, container, cfg.training.n_steps, out_dir, update_mode, best=False)
    return out_dir


def _validate(model, attack, val_batches, device):
    """Mean toward-benign loss on the held-out behaviors under attack.

    Drives best-checkpoint selection. Grad must stay enabled around the attack, which
    optimizes its perturbation internally; only the scoring forward runs under no_grad.
    """
    was_training = model.training
    model.eval()
    losses = []
    for adv_batch in val_batches:
        adv_embeds = attack.attack(model, adv_batch, detector=None, use_detector=False)
        with torch.no_grad():
            be, b_attn, b_labels = _benign_under_adv_prompt(model, adv_embeds, adv_batch)
            logits_b = model(inputs_embeds=be, attention_mask=b_attn).logits
            losses.append(toward_benign(logits_b[:, :-1], b_labels[:, 1:]).item())
    model.zero_grad(set_to_none=True)
    if was_training:
        model.train()
    return sum(losses) / len(losses)


def _save_checkpoint(model, container, step, out_dir, update_mode, best):
    if update_mode == "lora":
        from peft.utils import get_peft_model_state_dict

        state = get_peft_model_state_dict(model)
        # also persist a reloadable adapter directory
        adapter_dir = os.path.join(out_dir, "best_adapter" if best else "final_adapter")
        model.save_pretrained(adapter_dir)
    else:
        state = model.state_dict()
    ckpt = {"state": state, "cfg": container, "step": step}
    fname = "best.pt" if best else "final.pt"
    torch.save(ckpt, os.path.join(out_dir, fname))
