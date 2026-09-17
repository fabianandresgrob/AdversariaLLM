"""Static figures for collect_eval.py (small multiples, one hue per panel, base model as gray reference).

tradeoff.png   one panel per block: mean utility (MMLU/ARC-C/GSM8K) vs xs_test over-refusal, one point per
               config (mean over seeds, std error bars), direct-labeled with the swept knob value.
sweep_<b>.png  one row per block with a swept knob: MMLU, ARC-C, GSM8K, xs_test refusal, train recall@1%FPR
               vs the knob value (separate panels -- different scales never share an axis).
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402
import pandas as pd  # noqa: E402

from collect_eval import swept_knobs  # noqa: E402

SERIES = "#2a78d6"        # categorical slot 1 (validated: lightness, chroma, contrast on #fcfcfb)
SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e6e5e1"
REFUSAL = "xstest_refusal_judge"
SWEEP_METRICS = [
    ("mmlu_pct", "MMLU (%)"),
    ("arc_c_pct", "ARC-C (%)"),
    ("gsm8k_pct", "GSM8K (%)"),
    (REFUSAL, "xs_test refusal (judge)"),
    ("train_recall_1fpr", "recall@1%FPR (train val)"),
]


def _style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=8)
    ax.xaxis.set_major_locator(MaxNLocator(5))
    ax.yaxis.set_major_locator(MaxNLocator(6))


def _refusal_col(df):
    return REFUSAL if REFUSAL in df and df[REFUSAL].notna().any() else "xstest_refusal_string"


def _config_means(block_df: pd.DataFrame, knob: str | None, cols: list[str]) -> pd.DataFrame:
    key = block_df[knob].astype(str) if knob else pd.Series("all", index=block_df.index)
    present = [c for c in cols if c in block_df]
    stats = block_df.groupby(key)[present].agg(["mean", "std"])
    stats.index.name = "knob"
    return stats


def _fmt(value) -> str:
    try:
        f = float(value)
        return str(int(f)) if f.is_integer() else f"{f:g}"
    except ValueError:
        return str(value)


def _knob_order(values):
    try:
        return sorted(values, key=float)
    except ValueError:
        return sorted(values)


def plot_tradeoff(df: pd.DataFrame, path: Path) -> None:
    coop = df[df["kind"] == "coop"]
    blocks = sorted(coop["block"].unique())
    if not blocks:
        return
    refusal = _refusal_col(df)
    base = df[(df["kind"] == "reference") & (df["run"] == "base")]
    ncols = 4
    nrows = -(-len(blocks) // ncols)
    # independent axes: a broken block (e.g. utility ~0) must not flatten every other panel
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.4 * ncols, 3.0 * nrows), facecolor=SURFACE, squeeze=False)
    for ax, block in zip(axes.flat, blocks):
        _style(ax)
        block_df = coop[coop["block"] == block]
        knobs = swept_knobs(block_df)
        knob = knobs[0] if knobs else None
        stats = _config_means(block_df, knob, ["utility_mean_pct", refusal])
        if not base.empty and base["utility_mean_pct"].notna().any():
            bx, by = base[refusal].iloc[0], base["utility_mean_pct"].iloc[0]
            if pd.notna(bx):
                ax.axvline(bx, color=TEXT_SECONDARY, linestyle="--", linewidth=1)
            ax.axhline(by, color=TEXT_SECONDARY, linestyle="--", linewidth=1)
            ax.annotate("base", (1, by), xycoords=("axes fraction", "data"), xytext=(-2, 3),
                        textcoords="offset points", fontsize=7, color=TEXT_SECONDARY, ha="right")
        for i, label in enumerate(_knob_order(list(stats.index))):
            row = stats.loc[label]
            x, y = row[(refusal, "mean")], row[("utility_mean_pct", "mean")]
            if pd.isna(x) or pd.isna(y):
                continue
            ax.errorbar(x, y, xerr=row[(refusal, "std")], yerr=row[("utility_mean_pct", "std")], fmt="o",
                        ms=6, color=SERIES, ecolor=SERIES, elinewidth=1, capsize=0,
                        markeredgecolor=SURFACE, markeredgewidth=1.5)
            if knob:  # alternate above/below so neighbouring configs don't overprint
                ax.annotate(_fmt(label), (x, y), xytext=(6, 7 if i % 2 == 0 else -12), textcoords="offset points",
                            fontsize=7, color=TEXT)
        ax.set_title(f"{block}" + (f"  ({knob})" if knob else ""), fontsize=9, color=TEXT, loc="left")
    for ax in axes.flat[len(blocks):]:
        ax.set_visible(False)
    for ax in axes.flat[: len(blocks)]:
        ax.set_xlabel("xs_test over-refusal" + (" (judge)" if refusal == REFUSAL else ""), fontsize=8,
                      color=TEXT_SECONDARY)
    for ax in axes[:, 0]:
        ax.set_ylabel("mean utility: MMLU/ARC-C/GSM8K (%)", fontsize=8, color=TEXT_SECONDARY)
    fig.suptitle("Utility vs over-refusal per ablation block (mean ± std over seeds)", fontsize=10, color=TEXT,
                 x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def plot_sweep(df: pd.DataFrame, block: str, path: Path) -> bool:
    coop = df[(df["kind"] == "coop") & (df["block"] == block)]
    knobs = swept_knobs(coop)
    if not knobs:
        return False
    knob = knobs[0]
    metrics = [(c, label) for c, label in SWEEP_METRICS if c in coop and coop[c].notna().any()]
    if not metrics:
        return False
    stats = _config_means(coop, knob, [c for c, _ in metrics])
    order = _knob_order(list(stats.index))
    base = df[(df["kind"] == "reference") & (df["run"] == "base")]
    fig, axes = plt.subplots(1, len(metrics), figsize=(2.7 * len(metrics), 2.6), facecolor=SURFACE, squeeze=False)
    for ax, (col, label) in zip(axes[0], metrics):
        _style(ax)
        xs = range(len(order))
        means = [stats.loc[k, (col, "mean")] for k in order]
        stds = [stats.loc[k, (col, "std")] for k in order]
        ax.errorbar(list(xs), means, yerr=stds, color=SERIES, linewidth=2, marker="o", ms=6, capsize=0,
                    elinewidth=1, markeredgecolor=SURFACE, markeredgewidth=1.5)
        if col in base and not base.empty and pd.notna(base[col].iloc[0]):
            ax.axhline(base[col].iloc[0], color=TEXT_SECONDARY, linestyle="--", linewidth=1)
            ax.annotate("base", (len(order) - 1, base[col].iloc[0]), xytext=(0, 3), textcoords="offset points",
                        fontsize=7, color=TEXT_SECONDARY, ha="right")
        ax.set_xticks(list(xs), [_fmt(k) for k in order])
        ax.set_xlabel(knob, fontsize=8, color=TEXT_SECONDARY)
        ax.set_title(label, fontsize=9, color=TEXT, loc="left")
    fig.suptitle(f"{block}: metrics vs {knob} (mean ± std over seeds)", fontsize=10, color=TEXT, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    return True


def plot_all(df: pd.DataFrame, out: Path) -> list[Path]:
    out.mkdir(parents=True, exist_ok=True)
    written = []
    if len(df):
        plot_tradeoff(df, out / "tradeoff.png")
        written.append(out / "tradeoff.png")
        for block in sorted(df.loc[df["kind"] == "coop", "block"].unique()):
            if plot_sweep(df, block, out / f"sweep_{block}.png"):
                written.append(out / f"sweep_{block}.png")
    return written
