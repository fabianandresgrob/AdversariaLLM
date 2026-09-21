"""Static figures for collect_eval.py (small multiples, one hue per panel, base model as gray reference).

utility.png    utility (MMLU/ARC-C/GSM8K mean) per config, coop and CAT, against the base model
leash.png      the CAT leash isolation: same objective, steps and batch, only the KL leash differs
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

from collect_eval import CHECKPOINT_ROOTS, swept_knobs  # noqa: E402

SERIES = "#2a78d6"        # categorical slot 1 (validated: lightness, chroma, contrast on #fcfcfb)
SERIES_2 = "#e08a1e"      # categorical slot 2 (same validator run; under 3:1 -> always value-labelled)
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
    coop = df[df["kind"].isin(CHECKPOINT_ROOTS)]
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
    coop = df[df["kind"].isin(CHECKPOINT_ROOTS) & (df["block"] == block)]
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
            ax.annotate("base", (0, base[col].iloc[0]), xycoords=("axes fraction", "data"), xytext=(2, 3),
                        textcoords="offset points", fontsize=7, color=TEXT_SECONDARY, ha="left")
        ax.set_xticks(list(xs), [_fmt(k) for k in order])
        ax.set_xlabel(knob, fontsize=8, color=TEXT_SECONDARY)
        ax.set_title(label, fontsize=9, color=TEXT, loc="left")
    fig.suptitle(f"{block}: metrics vs {knob} (mean ± std over seeds)", fontsize=10, color=TEXT, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    return True


def plot_robustness(df: pd.DataFrame, path: Path) -> bool:
    """The headline trade-off: pipeline ASR under the worst attack condition vs xs_test over-refusal.
    One point per config (mean over seeds). Coop and CAT are told apart by marker shape and a direct
    label, never by colour alone; the base model's refusal rate is a dashed reference."""
    if "xattack_asr_worst" not in df:
        return False
    refusal = _refusal_col(df)
    attacked = df[df["xattack_asr_worst"].notna() & df[refusal].notna() & df["kind"].isin(CHECKPOINT_ROOTS)]
    if attacked.empty:
        return False
    fig, ax = plt.subplots(figsize=(6.4, 4.4), facecolor=SURFACE)
    _style(ax)
    base = df[(df["kind"] == "reference") & (df["run"] == "base")]
    if not base.empty and pd.notna(base[refusal].iloc[0]):
        ax.axvline(base[refusal].iloc[0], color=TEXT_SECONDARY, linestyle="--", linewidth=1)
        ax.annotate("base refusal", (base[refusal].iloc[0], 1), xycoords=("data", "axes fraction"),
                    xytext=(4, -10), textcoords="offset points", fontsize=7, color=TEXT_SECONDARY)
    for kind, marker, label in (("coop", "o", "coop (model + probe)"), ("cat", "^", "CAT (model only)")):
        rows = attacked[attacked["kind"] == kind]
        if rows.empty:
            continue
        key = rows["run"].str.replace(r"-s\d+$", "", regex=True)
        stats = rows.groupby(key)[[refusal, "xattack_asr_worst"]].mean()
        ax.scatter(stats[refusal], stats["xattack_asr_worst"], marker=marker, s=48, color=SERIES,
                   edgecolor=SURFACE, linewidth=1.5, label=label)
        for name, row in stats.iterrows():
            ax.annotate(name, (row[refusal], row["xattack_asr_worst"]), xytext=(6, 4),
                        textcoords="offset points", fontsize=7, color=TEXT)
    ax.set_xlabel("xs_test over-refusal" + (" (judge)" if refusal == REFUSAL else ""), fontsize=8,
                  color=TEXT_SECONDARY)
    ax.set_ylabel("pipeline ASR, worst attack condition", fontsize=8, color=TEXT_SECONDARY)
    ax.set_title("Robustness vs over-refusal (lower-left is better)", fontsize=10, color=TEXT, loc="left")
    legend = ax.legend(frameon=False, fontsize=8, loc="upper right")
    for text in legend.get_texts():
        text.set_color(TEXT_SECONDARY)
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    return True


def _config_table(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """One row per config (seeds averaged), labelled '<block>/<run without -sN>'."""
    trained = df[df["kind"].isin(CHECKPOINT_ROOTS)]
    key = trained["block"] + "/" + trained["run"].str.replace(r"-s\d+$", "", regex=True)
    present = [c for c in columns if c in trained]
    stats = trained.groupby([key, trained["kind"]])[present].agg(["mean", "std"])
    stats.index = stats.index.set_names(["config", "kind"])
    return stats.reset_index()


def plot_utility(df: pd.DataFrame, path: Path) -> bool:
    """Utility per config against the base model, sorted, so a collapsed run (utility ~0) is obvious
    instead of hidden in a table."""
    stats = _config_table(df, ["utility_mean_pct"])
    stats = stats[stats[("utility_mean_pct", "mean")].notna()].sort_values(("utility_mean_pct", "mean"))
    if stats.empty:
        return False
    base = df[(df["kind"] == "reference") & (df["run"] == "base")]
    fig, ax = plt.subplots(figsize=(7.2, 0.26 * len(stats) + 1.8), facecolor=SURFACE)
    _style(ax)
    ax.grid(True, axis="x", color=GRID, linewidth=0.8)
    colors = [SERIES if kind == "coop" else SERIES_2 for kind in stats["kind"]]
    ax.barh(range(len(stats)), stats[("utility_mean_pct", "mean")], color=colors,
            xerr=stats[("utility_mean_pct", "std")].fillna(0),
            error_kw={"elinewidth": 1, "ecolor": TEXT_SECONDARY}, edgecolor=SURFACE, linewidth=1)
    for i, value in enumerate(stats[("utility_mean_pct", "mean")]):
        ax.annotate(f"{value:.1f}", (value, i), xytext=(3, 0), textcoords="offset points",
                    fontsize=6, color=TEXT, va="center")
    ax.set_yticks(range(len(stats)), stats["config"], fontsize=6)
    if not base.empty and pd.notna(base["utility_mean_pct"].iloc[0]):
        ax.axvline(base["utility_mean_pct"].iloc[0], color=TEXT_SECONDARY, linestyle="--", linewidth=1)
        ax.annotate("base", (base["utility_mean_pct"].iloc[0], 1), xycoords=("data", "axes fraction"),
                    xytext=(3, -9), textcoords="offset points", fontsize=7, color=TEXT_SECONDARY)
    ax.set_xlabel("mean utility: MMLU / ARC-C / GSM8K (%)", fontsize=8, color=TEXT_SECONDARY)
    handles = [plt.Rectangle((0, 0), 1, 1, color=SERIES), plt.Rectangle((0, 0), 1, 1, color=SERIES_2)]
    legend = ax.legend(handles, ["coop", "CAT"], frameon=False, fontsize=8, loc="lower right")
    for text in legend.get_texts():
        text.set_color(TEXT_SECONDARY)
    ax.set_title("Utility per config (mean ± std over seeds)", fontsize=10, color=TEXT, loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    return True


def plot_leash(df: pd.DataFrame, path: Path, blocks=("J-ultra450", "J-leash450", "J-legacy", "J-cat")) -> bool:
    """CAT leash isolation: J-ultra450 (ultrachat) and J-leash450b4 (magpie/1024) share objective, steps
    and batch size, so the gap between them is the leash alone; J-legacy and J-cat bracket the pair."""
    rows = df[df["block"].isin(blocks)]
    if rows.empty:
        return False
    refusal = _refusal_col(df)
    stats = _config_table(rows, ["utility_mean_pct", refusal]).sort_values(("utility_mean_pct", "mean"))
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 0.4 * len(stats) + 2.0), facecolor=SURFACE, sharey=True)
    for ax, col, label, color in ((axes[0], "utility_mean_pct", "mean utility (%)", SERIES),
                                  (axes[1], refusal, "xs_test over-refusal", SERIES_2)):
        _style(ax)
        ax.grid(True, axis="x", color=GRID, linewidth=0.8)
        values = stats[(col, "mean")]
        ax.barh(range(len(stats)), values.fillna(0), xerr=stats[(col, "std")].fillna(0),
                error_kw={"elinewidth": 1, "ecolor": TEXT_SECONDARY}, color=color,
                edgecolor=SURFACE, linewidth=1)
        for i, value in enumerate(values):
            if pd.notna(value):
                text = f"{value:.2f}" if col == refusal else f"{value:.1f}"
                ax.annotate(text, (value, i), xytext=(3, 0), textcoords="offset points",
                            fontsize=6, color=TEXT, va="center")
        ax.set_xlabel(label, fontsize=8, color=TEXT_SECONDARY)
    axes[0].set_yticks(range(len(stats)), stats["config"], fontsize=7)
    fig.suptitle("CAT: the KL leash decides whether the model survives training", fontsize=10, color=TEXT,
                 x=0.01, ha="left")
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
        for name, fn in (("utility.png", plot_utility), ("leash.png", plot_leash),
                         ("robustness.png", plot_robustness)):
            if fn(df, out / name):
                written.append(out / name)
        for block in sorted(df.loc[df["kind"].isin(CHECKPOINT_ROOTS), "block"].unique()):
            if plot_sweep(df, block, out / f"sweep_{block}.png"):
                written.append(out / f"sweep_{block}.png")
    return written
