"""Visualize steering generalizability across disagreement types.

Answers: Does a probe trained on one disagreement type reduce sycophancy
when evaluated against a different disagreement type?

Produces:
1. Heatmap of absolute sycophancy rate reduction (probe_type × eval_disagreement)
2. Grouped bar chart comparing baseline vs steered sycophancy rates
"""

from pathlib import Path

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

RESULT_PATH = Path(__file__).parent.parent / "result.csv"
OUT_DIR = Path(__file__).parent
FONT_DIR = OUT_DIR / "fonts"

DISAGREEMENT_ORDER = ["epistemic", "persuasion", "authority_pressure", "emotional_pressure"]
DISAGREEMENT_LABELS = {
    "epistemic": "Simple",
    "persuasion": "Opinion",
    "authority_pressure": "Authority",
    "emotional_pressure": "Emotional",
    "combined": "Combined",
}
SHORT_LABELS = {
    "epistemic": "Simple",
    "persuasion": "Opinion",
    "authority_pressure": "Authority",
    "emotional_pressure": "Emotional",
    "combined": "Combined",
}

COLOR_1 = "#1b7f5a"
COLOR_2 = "#D57758"
TEXT_COLOR = "#4a4844"


def setup_font():
    FONT_DIR.mkdir(exist_ok=True)
    font_path = FONT_DIR / "Nunito-VariableFont_wght.ttf"
    if not font_path.exists():
        import urllib.request
        url = "https://github.com/google/fonts/raw/main/ofl/nunito/Nunito%5Bwght%5D.ttf"
        urllib.request.urlretrieve(url, font_path)
    fm.fontManager.addfont(str(font_path))
    font_name = fm.FontProperties(fname=str(font_path)).get_name()
    plt.rcParams.update({
        "font.family": font_name,
        "text.color": TEXT_COLOR,
        "axes.labelcolor": TEXT_COLOR,
        "xtick.color": TEXT_COLOR,
        "ytick.color": TEXT_COLOR,
        "axes.edgecolor": TEXT_COLOR,
        "legend.edgecolor": TEXT_COLOR,
    })


def load_data():
    return pd.read_csv(RESULT_PATH)


def get_baseline_rates(df):
    """Return {(model, eval_disagreement): sycophancy_rate} for baseline rows."""
    baselines = df[df["method"] == "baseline"]
    return {
        (row["model"], row["eval_disagreement"]): row["sycophancy_rate"]
        for _, row in baselines.iterrows()
    }


# ---------------------------------------------------------------------------
# Plot 1: Generalizability heatmap — probe_type × eval_disagreement
# ---------------------------------------------------------------------------

def plot_generalization_heatmap(df):
    """Heatmap of sycophancy rate reduction when steering with each probe type.

    Rows = probe type used for steering, Columns = eval disagreement type.
    Cell value = absolute reduction in sycophancy rate vs baseline.
    One heatmap per model.
    """
    steered = df[df["method"] == "steer"]
    baselines = get_baseline_rates(df)
    models = sorted(steered["model"].unique())

    probe_types = [p for p in DISAGREEMENT_ORDER + ["combined"]
                   if p in steered["probe_type"].unique()]

    # Green (good reduction) to white (no change) to orange (worse)
    cmap = LinearSegmentedColormap.from_list(
        "reduction", [COLOR_2, "#f5f0eb", COLOR_1],
    )

    n_models = len(models)
    width_ratios = [1] * n_models + [0.05]
    fig, all_axes = plt.subplots(
        1, n_models + 1, figsize=(7 * n_models + 1, 0.8 * len(probe_types) + 2),
        gridspec_kw={"width_ratios": width_ratios}, squeeze=False,
    )
    axes = all_axes[0, :n_models]
    cbar_ax = all_axes[0, -1]

    im = None
    for col, model in enumerate(models):
        ax = axes[col]
        model_steered = steered[steered["model"] == model]

        grid = np.full((len(probe_types), len(DISAGREEMENT_ORDER)), np.nan)
        for i, pt in enumerate(probe_types):
            for j, ed in enumerate(DISAGREEMENT_ORDER):
                row = model_steered[
                    (model_steered["probe_type"] == pt)
                    & (model_steered["eval_disagreement"] == ed)
                ]
                if row.empty:
                    continue
                baseline = baselines.get((model, ed))
                if baseline is None:
                    continue
                grid[i, j] = baseline - row.iloc[0]["sycophancy_rate"]

        vmax = max(0.15, np.nanmax(np.abs(grid))) if not np.all(np.isnan(grid)) else 0.15
        im = ax.imshow(grid, cmap=cmap, vmin=-vmax, vmax=vmax,
                        aspect="auto", interpolation="nearest")

        # Annotate cells
        for i in range(len(probe_types)):
            for j in range(len(DISAGREEMENT_ORDER)):
                val = grid[i, j]
                if np.isnan(val):
                    continue
                sign = "+" if val > 0 else ""
                color = "white" if abs(val) > vmax * 0.6 else TEXT_COLOR
                fontweight = "bold" if i == j or probe_types[i] == "combined" else "normal"
                ax.text(j, i, f"{sign}{val:.1%}", ha="center", va="center",
                        fontsize=10, color=color, fontweight=fontweight)

        ax.set_xticks(range(len(DISAGREEMENT_ORDER)))
        ax.set_xticklabels([SHORT_LABELS[d] for d in DISAGREEMENT_ORDER], fontsize=10)
        ax.set_yticks(range(len(probe_types)))
        ax.set_yticklabels([SHORT_LABELS[p] for p in probe_types], fontsize=10)
        ax.set_xlabel("Eval Disagreement", fontsize=11)
        if col == 0:
            ax.set_ylabel("Probe Type", fontsize=11)
        ax.set_title(model, fontsize=13, fontweight="bold")

        # Highlight diagonal (same probe = same eval)
        for k, pt in enumerate(probe_types):
            if pt in DISAGREEMENT_ORDER:
                j = DISAGREEMENT_ORDER.index(pt)
                ax.add_patch(plt.Rectangle((j - 0.5, k - 0.5), 1, 1,
                             fill=False, edgecolor=TEXT_COLOR, linewidth=2))

    if im is not None:
        cbar = fig.colorbar(im, cax=cbar_ax)
        cbar.set_label("Sycophancy Rate Reduction", color=TEXT_COLOR, fontsize=10)
        cbar.ax.yaxis.set_tick_params(color=TEXT_COLOR)
        plt.setp(cbar.ax.yaxis.get_ticklabels(), color=TEXT_COLOR)

    fig.suptitle("Steering Generalizability: Probe Type vs Eval Disagreement",
                 fontsize=15, fontweight="bold", color=TEXT_COLOR, y=1.02)
    fig.tight_layout()
    out = OUT_DIR / "steering_generalization_heatmap.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


# ---------------------------------------------------------------------------
# Plot 2: Effectiveness bar chart — baseline vs steered
# ---------------------------------------------------------------------------

def plot_steering_effectiveness(df):
    """Grouped bar chart showing baseline sycophancy rate vs best steered rate
    for each eval disagreement type. One subplot per model.

    'Best steered' = probe type that gives the lowest sycophancy rate.
    """
    steered = df[df["method"] == "steer"]
    baselines_df = df[df["method"] == "baseline"]
    models = sorted(steered["model"].unique())

    n_models = len(models)
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5), sharey=True,
                              squeeze=False)

    x = np.arange(len(DISAGREEMENT_ORDER))
    width = 0.25

    for col, model in enumerate(models):
        ax = axes[0, col]
        model_baselines = baselines_df[baselines_df["model"] == model]
        model_steered = steered[steered["model"] == model]

        baseline_rates = []
        best_matched_rates = []
        best_cross_rates = []
        best_cross_labels = []

        for ed in DISAGREEMENT_ORDER:
            # Baseline
            bl = model_baselines[model_baselines["eval_disagreement"] == ed]
            baseline_rates.append(bl.iloc[0]["sycophancy_rate"] if not bl.empty else np.nan)

            ed_steered = model_steered[model_steered["eval_disagreement"] == ed]

            # Matched (same probe type as eval)
            matched = ed_steered[ed_steered["probe_type"] == ed]
            best_matched_rates.append(matched.iloc[0]["sycophancy_rate"] if not matched.empty else np.nan)

            # Best cross-type (lowest sycophancy rate from any probe)
            if not ed_steered.empty:
                best_idx = ed_steered["sycophancy_rate"].idxmin()
                best_cross_rates.append(ed_steered.loc[best_idx, "sycophancy_rate"])
                best_cross_labels.append(SHORT_LABELS.get(
                    ed_steered.loc[best_idx, "probe_type"],
                    ed_steered.loc[best_idx, "probe_type"],
                ))
            else:
                best_cross_rates.append(np.nan)
                best_cross_labels.append("")

        bars1 = ax.bar(x - width, baseline_rates, width, label="Baseline", color="#cccccc")
        bars2 = ax.bar(x, best_matched_rates, width, label="Matched Probe", color=COLOR_2)
        bars3 = ax.bar(x + width, best_cross_rates, width, label="Best Probe", color=COLOR_1)

        # Value labels
        for bars in [bars1, bars2, bars3]:
            for bar in bars:
                h = bar.get_height()
                if not np.isnan(h):
                    ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01, f"{h:.0%}",
                            ha="center", va="bottom", fontsize=8, color=TEXT_COLOR)

        # Best probe name — rotated vertically inside the green bar
        for bar, label in zip(bars3, best_cross_labels):
            h = bar.get_height()
            if not np.isnan(h) and label:
                ax.text(bar.get_x() + bar.get_width() / 2, h / 2, label,
                        ha="center", va="center", fontsize=7, color="white",
                        fontweight="bold", rotation=90)

        ax.set_xticks(x)
        ax.set_xticklabels([DISAGREEMENT_LABELS[d] for d in DISAGREEMENT_ORDER], fontsize=9)
        ax.set_title(model, fontsize=13, fontweight="bold")
        ax.set_ylim(0, 0.85)
        if col == 0:
            ax.set_ylabel("Sycophancy Rate")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(fontsize=8, framealpha=0.7, loc="upper right")

    fig.suptitle("Steering Effectiveness: Baseline vs Matched vs Best Probe",
                 fontsize=15, fontweight="bold", color=TEXT_COLOR, y=1.02)
    fig.tight_layout()
    out = OUT_DIR / "steering_effectiveness.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def main():
    setup_font()
    df = load_data()

    if df[df["method"] == "steer"].empty:
        print("No steering results found in result.csv — nothing to plot.")
        return

    plot_generalization_heatmap(df)
    plot_steering_effectiveness(df)


if __name__ == "__main__":
    main()
