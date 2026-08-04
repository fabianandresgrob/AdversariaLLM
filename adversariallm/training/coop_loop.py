"""Cooperative model+detector co-training loop (design doc §4, §5, §8).

Two-timescale schedule per step: N detector steps (model frozen, reader trainable),
then one model step (reader frozen, model trainable) whose loss couples the two via the
gated representation term. See the spec for the full architecture.

FIRST CUT — validated at the Phase C cluster smoke (plan Task 6). Watch-items there:
the frozen-set discipline (asserted below), the rep-term gradient path through the reader
into the model, and peak memory (the model step holds several forward graphs; if it OOMs,
apply the incremental-backward pattern from loop.py:train_step).
"""
from __future__ import annotations

import logging
import os

import torch

from .loop import _to_device, _cycle, _benign_under_adv_prompt, _init_wandb
from .losses import utility_kl
from .gating import avg_logprob, w_harm, w_miss, behavior_gate, rep_gate
from .coop_losses import per_example_ce, detector_ce
from .coop_metrics import recall_at_fpr, four_case_frequencies, refusal_rate, fresh_refit_recall
from .readers import build_reader

log = logging.getLogger(__name__)


def _labels_to_target_ids(labels):
    """Reader wants 0-masked target_ids (0 = prompt/pad); the utility stream carries the
    CE convention (-100 = prompt/pad). Convert."""
    t = labels.clone()
    t[t == -100] = 0
    return t


def _set_requires_grad(params, flag):
    for p in params:
        p.requires_grad_(flag)


def _assert_grad(params, flag, who):
    assert all(p.requires_grad == flag for p in params), f"frozen-set violation: {who} requires_grad != {flag}"


def _hidden_and_logits(model, layer, *, inputs_embeds=None, input_ids=None, attention_mask=None):
    out = model(inputs_embeds=inputs_embeds, input_ids=input_ids,
                attention_mask=attention_mask, output_hidden_states=True)
    return out.hidden_states[layer], out.logits


def _detector_step(model, reader, opt_det, layer, adv_embeds, adv_batch, util_batch, device):
    """One detector update: model frozen, reader trainable. CE over attacked (->harmful)
    and benign (->benign) examples. Model forward under no_grad so only the reader trains."""
    opt_det.zero_grad(set_to_none=True)
    with torch.no_grad():
        h_hidden, _ = _hidden_and_logits(model, layer, inputs_embeds=adv_embeds, attention_mask=adv_batch["h_attn"])
        b_hidden, _ = _hidden_and_logits(model, layer, input_ids=util_batch["input_ids"], attention_mask=util_batch["attn"])
    logits_h = reader.logits(h_hidden, adv_batch["h_targetids"], adv_batch["h_attn"])
    b_tgt = _labels_to_target_ids(util_batch["labels"])
    logits_b = reader.logits(b_hidden, b_tgt, util_batch["attn"])
    logits = torch.cat([logits_h, logits_b], dim=0)
    labels = torch.cat([
        torch.zeros(logits_h.size(0), dtype=torch.long, device=device),   # harmful = 0
        torch.ones(logits_b.size(0), dtype=torch.long, device=device),    # benign = 1
    ])
    loss = detector_ce(logits, labels)
    loss.backward()
    opt_det.step()
    return loss.item()


def _model_step(model, reader, ref, opt_model, layer, adv_embeds, adv_batch, util_batch, hp, use_rep, device):
    """One model update: reader frozen, model trainable.
    loss = lambda_kl*KL(UltraChat) + lambda_beh*behavior_gate*CE(y_safe) + lambda_rep*rep_gate*detector_ce(reader(h)).
    No away term. Gates are stop-gradient (computed in gating.py)."""
    opt_model.zero_grad(set_to_none=True)
    logs = {}

    # attacked prompt (harmful) and benign continuation under the SAME adversarial prompt
    h_hidden, logits_h = _hidden_and_logits(model, layer, inputs_embeds=adv_embeds, attention_mask=adv_batch["h_attn"])
    be, b_attn, b_labels = _benign_under_adv_prompt(model, adv_embeds, adv_batch)
    logits_b = model(inputs_embeds=be, attention_mask=b_attn).logits

    # gates (both detached inside gating.py)
    lp_h = avg_logprob(logits_h[:, :-1], adv_batch["h_labels"][:, 1:])
    lp_s = avg_logprob(logits_b[:, :-1], b_labels[:, 1:])
    wh = w_harm(lp_h, lp_s, tau=hp["tau"])
    wm = w_miss(reader.p_harmful(h_hidden, adv_batch["h_targetids"], adv_batch["h_attn"]))

    # behavior term: teach the safe response, scaled by detector failure
    beh_ce = per_example_ce(logits_b[:, :-1], b_labels[:, 1:])
    beh = (behavior_gate(wm, hp["epsilon"]) * beh_ce).mean()
    total = hp["lambda_beh"] * beh
    logs["beh"] = beh.item()

    # representation term: make h detectable, scaled by model failure (off during warmup)
    if use_rep:
        rep_logits = reader.logits(h_hidden, adv_batch["h_targetids"], adv_batch["h_attn"])
        harmful = torch.zeros(rep_logits.size(0), dtype=torch.long, device=device)
        rep_ce = detector_ce(rep_logits, harmful, reduction="none")
        rep = (rep_gate(wh, hp["delta"]) * rep_ce).mean()
        total = total + hp["lambda_rep"] * rep
        logs["rep"] = rep.item()

    # utility KL on UltraChat only
    u_ids = util_batch["input_ids"]
    u_logits = model(input_ids=u_ids, attention_mask=util_batch["attn"]).logits
    r_logits = ref.logits(inputs_embeds=model.get_input_embeddings()(u_ids), attention_mask=util_batch["attn"])
    kl = utility_kl(u_logits, r_logits, attention_mask=util_batch["attn"])
    total = total + hp["lambda_kl"] * kl
    logs["kl"] = kl.item()

    total.backward()
    opt_model.step()
    logs.update(w_harm=wh.mean().item(), w_miss=wm.mean().item(), total=total.item())
    return logs, wh.detach(), wm.detach()


@torch.no_grad()
def _coop_validate(model, reader, layer, harmful_batches, benign_batches, benign_prompts,
                   tokenizer, template_id, max_new_tokens):
    """Held-out metrics on CLEAN harmful vs OOD benign (never UltraChat):
      - val_recall@1fpr: the co-trained reader's recall at 1% FPR (detector quality).
      - val_fresh_refit_recall: a probe fit FROM SCRATCH on half the current model's
        features, evaluated on the other half (§11 collusion probe). Compare the two —
        co-trained >> fresh hints the reader leans on something a fresh probe can't
        recover. (Fullest form fits/evals on a held-out ATTACK distribution; this clean
        split is the from-run-1 version.)
      - val_ood_refusal: over-refusal from free generation on OOD benign.
    Attacked-recall is a fuller eval-side metric; this is the cheap training-time monitor."""
    from ..defenses.monitors._activation_detector_model import get_chat_template

    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    has_feats = hasattr(reader, "readout") and hasattr(reader, "linear")

    def _collect(batches, ids_key, tgt_key, attn_key):
        feats, scores = [], []
        for b in batches:
            hidden, _ = _hidden_and_logits(model, layer, input_ids=b[ids_key], attention_mask=b[attn_key])
            if has_feats:
                f = reader.readout(hidden, b[tgt_key], b[attn_key])
                feats.append(f)
                scores += torch.softmax(reader.linear(f).float(), dim=-1)[:, 0].tolist()
            else:
                scores += reader.p_harmful(hidden, b[tgt_key], b[attn_key]).tolist()
        return (torch.cat(feats, dim=0) if feats else None), scores

    h_feat, harmful_scores = _collect(harmful_batches, "h_ids", "h_targetids", "h_attn")
    b_feat, benign_scores = _collect(benign_batches, "d_ids", "d_targetids", "d_attn")
    metrics = {"val_recall@1fpr": recall_at_fpr(benign_scores, harmful_scores, fpr=0.01)}

    # collusion probe: fresh linear probe fit on the first half, evaluated on the second
    if has_feats and h_feat is not None and h_feat.size(0) >= 2 and b_feat.size(0) >= 2:
        nh, nb = h_feat.size(0) // 2, b_feat.size(0) // 2
        tr = torch.cat([h_feat[:nh], b_feat[:nb]], dim=0)
        lab = torch.cat([torch.zeros(nh, dtype=torch.long, device=device),
                         torch.ones(nb, dtype=torch.long, device=device)])
        metrics["val_fresh_refit_recall"] = fresh_refit_recall(tr, lab, b_feat[nb:], h_feat[nh:])

    first_user_msg, _, response_key, _, _ = get_chat_template(template_id)
    gens = []
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    for p in benign_prompts:
        enc = tokenizer(first_user_msg.format(instruction=p) + response_key, return_tensors="pt").to(device)
        out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=pad_id)
        gens.append(tokenizer.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True))
    metrics["val_ood_refusal"] = refusal_rate(gens)

    if was_training:
        model.train()
    return metrics


def _adv_embeds(attack, model, adv_batch):
    """Stage B: perturbed embeddings from the continuous attack. Stage A (attack is None):
    clean prompt embeddings, so the whole machine runs with no attack cost."""
    if attack is None:
        return model.get_input_embeddings()(adv_batch["h_ids"]).detach()
    return attack.attack(model, adv_batch, detector=None, use_detector=False)


def _save_pair(model, reader, container, step, out_dir, tag):
    """Checkpoint model adapter + reader together — a mismatched pair is nonsense (§14.9)."""
    model.save_pretrained(os.path.join(out_dir, f"{tag}_adapter"))
    torch.save({"reader": reader.state_dict(), "cfg": container, "step": step},
               os.path.join(out_dir, f"{tag}_reader.pt"))


def run_coop_training(cfg):
    from omegaconf import OmegaConf
    from torch.utils.data import DataLoader
    import peft
    from peft import LoraConfig

    from ..io_utils import load_model_and_tokenizer
    from ..defenses.monitors._activation_detector_model import get_chat_template
    from .attacks import ContinuousEmbeddingAttack
    from .reference import LoRADisableReference
    from .data import (
        AdvTupleStream, UtilityStream, OODBenignStream, load_dataset_prompts,
        collate_adv, collate_util, collate_ood, split_adv_stream,
    )

    container = OmegaConf.to_container(cfg, resolve=True)
    device_hp = {k: container[k] for k in ("tau", "epsilon", "delta", "lambda_beh", "lambda_rep", "lambda_kl")}

    model_params = cfg.models[cfg.model]
    template_id = cfg.chat_template_id
    model, tokenizer = load_model_and_tokenizer(model_params)
    device = next(model.parameters()).device

    model = peft.get_peft_model(
        model,
        LoraConfig(
            r=8, lora_alpha=32,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.05, task_type="CAUSAL_LM",
        ),
    )
    ref = LoRADisableReference(model)
    model_trainable = [p for p in model.parameters() if p.requires_grad]

    # reader (random init); input_dim = target hidden size
    hidden_dim = model.get_input_embeddings().weight.shape[-1]
    reader = build_reader(container.get("reader"), hidden_dim).to(device)
    reader_params = list(reader.parameters())

    # data
    adv_ds = AdvTupleStream(
        data_dir=cfg.data.dir, behaviors_csv=cfg.data.behaviors, targets_json=cfg.data.targets,
        safe_csv=cfg.data.safe, tokenizer=tokenizer, model_name=template_id,
    )
    util_ds = UtilityStream(tokenizer, template_id, window=cfg.splits.ultrachat.train)

    adv_train_ds, adv_val_ds = split_adv_stream(adv_ds, val_size=cfg.data.val_size, seed=cfg.data.val_seed)
    adv_loader = DataLoader(adv_train_ds, batch_size=cfg.data.harmful_batch_size, shuffle=True, collate_fn=collate_adv)
    util_loader = DataLoader(util_ds, batch_size=cfg.data.utility_batch_size, shuffle=True, collate_fn=collate_util)

    # held-out validation sets: clean harmful (from the split) + OOD benign (Alpaca, NOT UltraChat)
    harmful_val_batches = [
        _to_device(b, device) for b in
        DataLoader(adv_val_ds, batch_size=cfg.data.harmful_batch_size, shuffle=False, collate_fn=collate_adv)
    ]
    # OOD benign for validation = alpaca VAL window (canonical split; disjoint from train + test)
    ood_prompts, ood_resp = load_dataset_prompts(cfg.datasets, cfg.data.ood_benign, window=cfg.splits.alpaca.val,
                                                 seed=cfg.data.val_seed)
    ood_ds = OODBenignStream(list(zip(ood_prompts, ood_resp)), tokenizer, template_id)
    benign_val_batches = [
        _to_device(b, device) for b in
        DataLoader(ood_ds, batch_size=cfg.data.harmful_batch_size, shuffle=False, collate_fn=collate_ood)
    ]
    benign_prompts = [x for x, _ in ood_ds.rows[:int(cfg.training.benign_gen_n)]]

    # attack (Stage B) or None (Stage A)
    attack = None
    if cfg.attack.enabled:
        _, _, response_key, _, _ = get_chat_template(template_id)
        attack = ContinuousEmbeddingAttack(
            model.get_input_embeddings().weight, response_key, tokenizer,
            iters=cfg.attack.iters, eps=cfg.attack.eps, lr=cfg.attack.lr,
        )

    opt_model = torch.optim.Adam(model_trainable, lr=cfg.training.model_lr)
    opt_det = torch.optim.Adam(reader_params, lr=cfg.training.detector_lr)

    layer = int(container["reader"].get("layer", -1)) if container.get("reader") else -1
    n_det = int(cfg.training.n_detector_steps)
    warmup = int(cfg.training.rep_warmup_steps)

    adv_iter, util_iter = _cycle(adv_loader), _cycle(util_loader)
    run_name = container.get("name") or "coop_run"
    out_dir = os.path.join(cfg.output.checkpoint_path, run_name)
    os.makedirs(out_dir, exist_ok=True)
    wandb_run = _init_wandb(cfg, container)

    val_every = int(cfg.training.val_every)
    wh_hist, wm_hist = [], []   # accumulate per-example gates for four-case frequencies

    model.train()
    for step in range(cfg.training.n_steps):
        adv_batch = _to_device(next(adv_iter), device)
        util_batch = _to_device(next(util_iter), device)

        # ---- detector phase: model frozen, reader trainable ----
        _set_requires_grad(model_trainable, False)
        _set_requires_grad(reader_params, True)
        _assert_grad(model_trainable, False, "model(det phase)")
        _assert_grad(reader_params, True, "reader(det phase)")
        adv_embeds = _adv_embeds(attack, model, adv_batch)
        det_losses = [_detector_step(model, reader, opt_det, layer, adv_embeds, adv_batch, util_batch, device)
                      for _ in range(n_det)]

        # ---- model phase: reader frozen, model trainable ----
        _set_requires_grad(reader_params, False)
        _set_requires_grad(model_trainable, True)
        _assert_grad(reader_params, False, "reader(model phase)")
        _assert_grad(model_trainable, True, "model(model phase)")
        use_rep = step >= warmup
        logs, wh, wm = _model_step(model, reader, ref, opt_model, layer, adv_embeds, adv_batch, util_batch,
                                   device_hp, use_rep, device)
        logs["det"] = sum(det_losses) / len(det_losses)
        wh_hist.append(wh); wm_hist.append(wm)

        log.info(f"[step {step}] " + " ".join(f"{k}={v:.4f}" for k, v in logs.items()))
        if wandb_run is not None:
            wandb_run.log(logs, step=step)

        if val_every and (step + 1) % val_every == 0:
            metrics = _coop_validate(model, reader, layer, harmful_val_batches, benign_val_batches,
                                     benign_prompts, tokenizer, template_id,
                                     int(cfg.training.benign_val_max_new_tokens))
            cases = four_case_frequencies(torch.cat(wh_hist), torch.cat(wm_hist))
            metrics.update({f"case_{k}": v for k, v in cases.items()})
            wh_hist.clear(); wm_hist.clear()
            log.info(f"[step {step}] " + " ".join(f"{k}={v:.4f}" for k, v in metrics.items()))
            if wandb_run is not None:
                wandb_run.log(metrics, step=step)

        ckpt_every = int(cfg.training.checkpoint_every)
        if ckpt_every and (step + 1) % ckpt_every == 0:
            _save_pair(model, reader, container, step, out_dir, tag=f"step{step + 1}")

    _save_pair(model, reader, container, cfg.training.n_steps, out_dir, tag="final")
    if wandb_run is not None:
        wandb_run.finish()
    return out_dir
