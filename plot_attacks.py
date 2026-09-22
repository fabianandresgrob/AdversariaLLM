"""Figures for collect_attacks.py.

asr_by_attack.png  one panel per attack family: behavior-level ASR over the 100 JBB behaviors by
                   model, undefended vs defended bars, so the probe's contribution is a model's gap.
                   GCG's defended bar comes from its replay run (the suffix is optimised against the
                   raw model, then transferred); inpainting and PAIR hit the defended pipelinedirectly.
budget_curve.png   ASR against query budget (1 / 10 / 100 / any) per model, one panel per attack -- how
                   many tries an attacker needs, which a single ASR number hides.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402
import pandas as pd  # noqa: E402

UNDEFENDED = "#2a78d6"    # categorical slot 1
DEFENDED = "#e08a1e"      # categorical slot 2
SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e6e5e1"
BUDGET_COLUMNS = [("asr_at_1", "1"), ("asr_at_10", "10"), ("asr_at_100", "100"),
                  ("asr_at_128", "128"), ("asr_behavior", "all")]


def _style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=8)
    ax.yaxis.set_major_locator(MaxNLocator(5))


def _legend(ax, labels):
    legend = ax.legend(labels, frameon=False, fontsize=8)
    for text in legend.get_texts():
        text.set_color(TEXT_SECONDARY)


def plot_by_attack(df: pd.DataFrame, path: Path) -> bool:
    attacks = sorted(df["family"].unique())
    if not attacks:
        return False
    models = sorted(df["model"].unique())
    fig, axes = plt.subplots(len(attacks), 1, figsize=(1.1 * len(models) + 3, 2.9 * len(attacks)),
                             facecolor=SURFACE, squeeze=False)
    for ax, attack in zip(axes[:, 0], attacks):
        _style(ax)
        rows = df[df["family"] == attack].set_index(["model", "defense"])
        width, drawn = 0.38, []
        for offset, defense, color, label in ((-width / 2, "none", UNDEFENDED, "model only"),
                                              (width / 2, "coop_probe", DEFENDED, "model + probe")):
            values = [rows.loc[(m, defense), "asr_behavior"] if (m, defense) in rows.index else None
                      for m in models]
            if all(v is None for v in values):
                continue
            positions = [i + offset for i, v in enumerate(values) if v is not None]
            heights = [v for v in values if v is not None]
            ax.bar(positions, heights, width=width, color=color, edgecolor=SURFACE, linewidth=1)
            for x, height in zip(positions, heights):  # value labels: the amber bar sits under 3:1 on
                ax.annotate(f"{height:.2f}", (x, height), xytext=(0, 2), textcoords="offset points",
                            fontsize=6, color=TEXT, ha="center")  # this surface, so it needs the relief
            drawn.append(label)
        ax.set_xticks(range(len(models)), models, rotation=30, ha="right", fontsize=7)
        ax.set_ylabel("ASR over 100 behaviors", fontsize=8, color=TEXT_SECONDARY)
        ax.set_title(attack, fontsize=9, color=TEXT, loc="left")
        if len(drawn) > 1:
            _legend(ax, drawn)
    fig.suptitle("Attack success rate over the 100 JBB behaviors (lower is better)", fontsize=10, color=TEXT, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    return True


def plot_budget_curve(df: pd.DataFrame, path: Path) -> bool:
    undefended = df[df["defense"] == "none"]
    attacks = sorted(undefended["family"].unique())
    if not attacks:
        return False
    fig, axes = plt.subplots(1, len(attacks), figsize=(3.4 * len(attacks), 3.2), facecolor=SURFACE, squeeze=False)
    for ax, attack in zip(axes[0], attacks):
        _style(ax)
        ax.grid(True, axis="both", color=GRID, linewidth=0.8)
        rows = undefended[undefended["family"] == attack]
        for _, row in rows.iterrows():
            values = [row[col] for col, _ in BUDGET_COLUMNS if col in row]
            ax.plot(range(len(values)), values, marker="o", ms=5, linewidth=1.6, color=UNDEFENDED, alpha=0.75)
            ax.annotate(row["model"], (len(values) - 1, values[-1]), xytext=(4, 0),
                        textcoords="offset points", fontsize=6, color=TEXT_SECONDARY, va="center")
        ax.set_xticks(range(len(BUDGET_COLUMNS)), [label for _, label in BUDGET_COLUMNS])
        ax.set_xlabel("completions tried per behavior", fontsize=8, color=TEXT_SECONDARY)
        ax.set_title(attack, fontsize=9, color=TEXT, loc="left")
    axes[0][0].set_ylabel("behaviors jailbroken", fontsize=8, color=TEXT_SECONDARY)
    fig.suptitle("Attack success against query budget (undefended)", fontsize=10, color=TEXT, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    return True


def plot_all(df: pd.DataFrame, out: Path) -> list[Path]:
    out.mkdir(parents=True, exist_ok=True)
    written = []
    for name, fn in (("asr_by_attack.png", plot_by_attack), ("budget_curve.png", plot_budget_curve)):
        if fn(df, out / name):
            written.append(out / name)
    return written
