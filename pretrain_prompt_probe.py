"""Train EuroSafeGuard's prompt probe design (PromptMLPProbe) on their prompt data, on a frozen model.

    uv run python pretrain_prompt_probe.py name=esg-probe-llama31-s0

Follows their trainer (EuroSafeGuard training/train_probe.py, configs/probe.json): features of all layers computed
on the fly, AdamW lr 1e-3 and weight decay 0.01, batches of 8, one epoch, every rewritten prompt trains next to its
plain twin, the per-feature scale is 1/RMS per layer from 32 unlabelled training rows, prompts longer than max_len
are cut from the left. The released data has no validation split, so val_frac of the input_id groups is held out
(seeded), as their data/prepare.py does.

Writes <out>/<name>/final_reader.pt (loads with load_reader, and as probe_init of coop training), metrics.json, and
probe/probe.safetensors with probe_config.json in the layout of their release, so score_prompt_probe.py reads it
with probe_repo=<out>/<name>. A checkpoint every save_every steps lets a restarted job continue. The loss goes to W&B
(project coop-detect, group prompt-probe); a job that continues from a checkpoint starts a new W&B run of the same
name at the checkpoint's step.
"""

import json
import logging
import os
import random

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

log = logging.getLogger(__name__)


def split_groups(input_ids: list, val_frac: float, seed: int) -> tuple[list[int], list[int]]:
    """Row indices of train and val, with val_frac of the input_id groups (a prompt and its rewrites) held out."""
    groups = sorted(set(input_ids))
    random.Random(seed).shuffle(groups)
    held = set(groups[: round(val_frac * len(groups))])
    train = [i for i, g in enumerate(input_ids) if g not in held]
    val = [i for i, g in enumerate(input_ids) if g in held]
    return train, val


def plain_twins(input_ids: list, sources: list) -> dict[int, int]:
    """Position of each rewritten row -> position of its plain twin (same input_id, source "plain")."""
    plain = {g: i for i, (g, s) in enumerate(zip(input_ids, sources)) if s == "plain"}
    return {i: plain[g] for i, (g, s) in enumerate(zip(input_ids, sources)) if s != "plain" and g in plain}


def auroc(harmful: list[int], scores: list[float]) -> float:
    """Probability that a harmful row scores above a benign one, ties counted half. nan with one class only."""
    pos = torch.tensor([s for y, s in zip(harmful, scores) if y], dtype=torch.float64)
    neg = torch.tensor([s for y, s in zip(harmful, scores) if not y], dtype=torch.float64)
    if not len(pos) or not len(neg):
        return float("nan")
    neg, _ = neg.sort()
    below = torch.searchsorted(neg, pos, right=False)
    upto = torch.searchsorted(neg, pos, right=True)
    return float((below + 0.5 * (upto - below)).sum() / (len(pos) * len(neg)))


def encode(tok, prompt: str, max_len: int) -> list[int]:
    text = tok.apply_chat_template([{"role": "user", "content": str(prompt)}], tokenize=False,
                                   add_generation_prompt=True)
    return tok(text, add_special_tokens=False)["input_ids"][-max_len:]


@torch.no_grad()
def features(model, ids: list[list[int]], layer, pad_id: int):
    """(B, T, features) hidden states of the chosen layers, target_ids (all 0: prompt only) and attention mask."""
    from adversariallm.training.readers import select_hidden

    device = next(model.parameters()).device
    T = max(map(len, ids))
    inp = torch.full((len(ids), T), pad_id, dtype=torch.long)
    att = torch.zeros((len(ids), T), dtype=torch.long)
    for k, x in enumerate(ids):
        inp[k, : len(x)] = torch.tensor(x)
        att[k, : len(x)] = 1
    inp, att = inp.to(device), att.to(device)
    hs = model(input_ids=inp, attention_mask=att, output_hidden_states=True).hidden_states
    return select_hidden(hs, layer), torch.zeros_like(inp), att


@torch.no_grad()
def rms_scale(model, items, layer, pad_id, hidden_size: int, bs: int) -> torch.Tensor:
    """1/RMS of every layer's activations on the prompt tokens of a few rows, repeated over that layer's features."""
    ss, cnt = None, 0
    for i in range(0, len(items), bs):
        psi, _, att = features(model, items[i:i + bs], layer, pad_id)
        x = psi[att.bool()]
        part = x.view(x.shape[0], -1, hidden_size).float().pow(2).sum((0, 2)).double()
        ss = part if ss is None else ss + part
        cnt += x.shape[0] * hidden_size
    return (1.0 / (ss / cnt).sqrt()).float().repeat_interleave(hidden_size)


@torch.no_grad()
def score_all(model, reader, items, layer, pad_id, bs: int) -> list[float]:
    out = [0.0] * len(items)
    order = sorted(range(len(items)), key=lambda i: len(items[i]))
    for b in range(0, len(order), bs):
        idx = order[b:b + bs]
        psi, tgt, att = features(model, [items[j] for j in idx], layer, pad_id)
        for j, s in zip(idx, reader.score(psi, tgt, att).float().tolist()):
            out[j] = s
    return out


def save_release_layout(reader, out_dir: str, cfg: dict) -> None:
    from safetensors.torch import save_file

    os.makedirs(os.path.join(out_dir, "probe"), exist_ok=True)
    save_file({k: v.float().contiguous().cpu() for k, v in reader.state_dict().items()},
              os.path.join(out_dir, "probe", "probe.safetensors"))
    with open(os.path.join(out_dir, "probe", "probe_config.json"), "w") as fh:
        json.dump(cfg, fh, indent=1)


@hydra.main(version_base=None, config_path="conf", config_name="pretrain_prompt_probe")
def main(cfg: DictConfig) -> None:
    import pandas as pd

    from adversariallm.training.loop import _init_wandb
    from adversariallm.training.readers import build_reader, parse_layer
    from generate_answers import load

    out_dir = os.path.join(cfg.out, cfg.name)
    os.makedirs(out_dir, exist_ok=True)
    if os.path.exists(os.path.join(out_dir, "final_reader.pt")) and not cfg.overwrite:
        raise FileExistsError(f"{out_dir} already holds a trained probe (overwrite=true to replace it)")
    torch.manual_seed(int(cfg.seed))
    rng = random.Random(int(cfg.seed))
    torch.zeros(1, device="cuda")  # a GPU that cannot be used would send the model to the CPU

    df = pd.read_parquet(cfg.prompts, columns=["row_id", "text", "label", "source", "input_id"])
    df = df[df.text.fillna("").str.strip() != ""].sort_values("row_id").reset_index(drop=True)
    if cfg.limit:
        df = df.sample(n=int(cfg.limit), random_state=int(cfg.seed)).sort_values("row_id").reset_index(drop=True)
    train_idx, val_idx = split_groups(df.input_id.tolist(), float(cfg.val_frac), int(cfg.split_seed))
    harmful = df.label.astype(int).tolist()  # their label: 1 = harmful

    model, tok = load(cfg)
    pad_id = tok.pad_token_id
    num_layers, hidden_size = model.config.num_hidden_layers, model.config.hidden_size
    reader_cfg = OmegaConf.to_container(cfg.reader, resolve=True)
    layer = parse_layer(reader_cfg["layer"], num_layers)
    reader = build_reader(reader_cfg, hidden_size, num_layers).to(next(model.parameters()).device)

    items = [encode(tok, t, int(cfg.max_len)) for t in df.text]
    tr = [items[i] for i in train_idx]
    y_tr = [harmful[i] for i in train_idx]
    twin = plain_twins([df.input_id[i] for i in train_idx], [df.source[i] for i in train_idx]) if cfg.twins else {}
    log.info("%d train rows, %d val rows, %d rewritten rows with a plain twin, %d harmful in train", len(tr),
             len(val_idx), len(twin), sum(y_tr))

    bs = int(cfg.batch_size)
    reader.set_scale(rms_scale(model, [tr[j] for j in rng.sample(range(len(tr)), min(32, len(tr)))], layer, pad_id,
                               hidden_size, bs))
    opt = torch.optim.AdamW(reader.parameters(), lr=float(cfg.lr), weight_decay=float(cfg.weight_decay))
    order = [j for _ in range(int(cfg.epochs)) for j in rng.sample(range(len(tr)), len(tr))]
    starts = list(range(0, len(order), bs))
    step, ckpt = 0, os.path.join(out_dir, "checkpoint.pt")
    if os.path.exists(ckpt):
        state = torch.load(ckpt, map_location="cpu", weights_only=False)
        reader.load_state_dict(state["reader"])
        opt.load_state_dict(state["opt"])
        step = int(state["step"])
        log.info("continuing from step %d of %d", step, len(starts))

    wandb_run = _init_wandb(cfg, {**OmegaConf.to_container(cfg, resolve=True), "models": None, "n_train": len(tr),
                                  "n_val": len(val_idx), "steps_total": len(starts), "start_step": step})
    reader.train()
    for b in starts[step:]:
        bi = order[b:b + bs]
        for i in list(bi):  # a rewritten prompt trains next to its plain twin
            t = twin.get(i)
            if t is not None and t not in bi:
                bi.append(t)
        psi, tgt, att = features(model, [tr[j] for j in bi], layer, pad_id)
        labels = torch.tensor([0 if y_tr[j] else 1 for j in bi], device=psi.device)  # ours: 0 = harmful
        loss = reader.example_loss(psi, tgt, att, labels).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        step += 1
        if wandb_run is not None and step % int(cfg.wandb.every) == 0:
            wandb_run.log({"probe/loss": loss.item(), "probe/batch_rows": len(bi), "probe/batch_tokens": int(att.sum())},
                          step=step)
        if step % int(cfg.log_every) == 0:
            log.info("step %d of %d, loss %.4f", step, len(starts), loss.item())
        if step % int(cfg.save_every) == 0:
            torch.save({"reader": reader.state_dict(), "opt": opt.state_dict(), "step": step}, ckpt + ".tmp")
            os.replace(ckpt + ".tmp", ckpt)

    reader.eval()
    scores = score_all(model, reader, [items[i] for i in val_idx], layer, pad_id, bs)
    y_va = [harmful[i] for i in val_idx]
    val_auroc = auroc(y_va, scores)
    log.info("val AUROC %.4f on %d rows", val_auroc, len(val_idx))
    if wandb_run is not None:
        wandb_run.log({"probe/val_auroc": val_auroc}, step=step)
        wandb_run.finish()

    meta = {"reader": reader_cfg, "name": cfg.name, "model": cfg.model, "template": cfg.template,
            "prompts": str(cfg.prompts), "seed": int(cfg.seed), "split_seed": int(cfg.split_seed)}
    torch.save({"reader": reader.state_dict(), "cfg": meta, "step": step}, os.path.join(out_dir, "final_reader.pt"))
    save_release_layout(reader, out_dir, {
        "kind": "mlp", "hidden": int(reader_cfg.get("hidden") or 256), "activation": "gelu",
        "base_model": cfg.models[cfg.model].id, "layers": layer, "hidden_size": hidden_size,
        "feature_dim": hidden_size * len(layer), "max_len": int(cfg.max_len), "M": int(reader_cfg.get("window") or 16),
        "val_auroc": val_auroc})
    with open(os.path.join(out_dir, "metrics.json"), "w") as fh:
        json.dump({"val_auroc": val_auroc, "n_train": len(tr), "n_val": len(val_idx), "steps": step,
                   "val_row_ids": df.row_id[val_idx].tolist()}, fh)
    if os.path.exists(ckpt):
        os.remove(ckpt)


if __name__ == "__main__":
    main()
