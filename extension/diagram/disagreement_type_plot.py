from pathlib import Path

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RESULT_PATH = Path(__file__).parent.parent / "result.csv"
OUT_DIR = Path(__file__).parent
FONT_DIR = OUT_DIR / "fonts"

DISAGREEMENT_ORDER = ["epistemic", "persuasion", "authority_pressure", "emotional_pressure"]
DISAGREEMENT_LABELS = {
    "epistemic": "Simple",
    "persuasion": "Opinion",
    "authority_pressure": "Authority",
    "emotional_pressure": "Emotional",
}

COLOR_1 = "#1b7f5a"
COLOR_2 = "#D57758"
TEXT_COLOR = "#4a4844"


def setup_font():
    """Download Nunito from Google Fonts and register it with matplotlib."""
    FONT_DIR.mkdir(exist_ok=True)
    font_path = FONT_DIR / "Nunito-VariableFont_wght.ttf"

    if not font_path.exists():
        import urllib.request
        url = "https://github.com/google/fonts/raw/main/ofl/nunito/Nunito%5Bwght%5D.ttf"
        urllib.request.urlretrieve(url, font_path)
        print(f"Downloaded Nunito to {font_path}")

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
    return font_name


def load_data():
    df = pd.read_csv(RESULT_PATH)
    # Only use baseline rows for this plot
    if "method" in df.columns:
        df = df[df["method"] == "baseline"]
    df["eval_disagreement"] = pd.Categorical(
        df["eval_disagreement"], categories=DISAGREEMENT_ORDER, ordered=True
    )
    return df.sort_values(["model", "eval_disagreement"])


def plot_accuracy_comparison(df):
    """Grouped bar chart: first vs second accuracy per model x disagreement."""
    models = df["model"].unique()
    n_models = len(models)
    n_types = len(DISAGREEMENT_ORDER)

    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 5), sharey=True)
    if n_models == 1:
        axes = [axes]

    x = np.arange(n_types)
    width = 0.35

    for ax, model in zip(axes, models):
        sub = df[df["model"] == model].set_index("eval_disagreement")
        first = [sub.loc[dt, "first_accuracy"] for dt in DISAGREEMENT_ORDER]
        second = [sub.loc[dt, "second_accuracy"] for dt in DISAGREEMENT_ORDER]

        bars1 = ax.bar(x - width / 2, first, width, label="1st Answer", color=COLOR_1)
        bars2 = ax.bar(x + width / 2, second, width, label="2nd Answer", color=COLOR_2)

        for bar in list(bars1) + list(bars2):
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01, f"{h:.0%}",
                    ha="center", va="bottom", fontsize=8, color=TEXT_COLOR)

        ax.set_title(model, fontsize=12, fontweight="bold", color=TEXT_COLOR)
        ax.set_xticks(x)
        ax.set_xticklabels([DISAGREEMENT_LABELS[dt] for dt in DISAGREEMENT_ORDER], fontsize=9)
        ax.set_ylim(0, 0.85)
        ax.set_ylabel("Accuracy" if ax == axes[0] else "")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(fontsize=8, framealpha=0.7)

    fig.suptitle("Accuracy Before vs After Disagreement", fontsize=14, fontweight="bold",
                 color=TEXT_COLOR, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "accuracy_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved accuracy_comparison.png")


def plot_sycophancy_rate(df):
    """Grouped bar chart: sycophancy rate per model x disagreement."""
    models = df["model"].unique()
    n_models = len(models)
    n_types = len(DISAGREEMENT_ORDER)

    fig, ax = plt.subplots(figsize=(10, 5))

    x = np.arange(n_types)
    total_width = 0.7
    bar_width = total_width / n_models

    # Interpolate between the two theme colors for each model
    c1 = np.array([int(COLOR_1[i:i+2], 16) for i in (1, 3, 5)]) / 255
    c2 = np.array([int(COLOR_2[i:i+2], 16) for i in (1, 3, 5)]) / 255
    model_colors = [c1 + t * (c2 - c1) for t in np.linspace(0, 1, n_models)]

    for i, model in enumerate(models):
        sub = df[df["model"] == model].set_index("eval_disagreement")
        rates = [sub.loc[dt, "sycophancy_rate"] for dt in DISAGREEMENT_ORDER]
        offset = (i - (n_models - 1) / 2) * bar_width
        bars = ax.bar(x + offset, rates, bar_width, label=model, color=model_colors[i])
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01, f"{h:.0%}",
                    ha="center", va="bottom", fontsize=8, color=TEXT_COLOR)

    ax.set_xticks(x)
    ax.set_xticklabels([DISAGREEMENT_LABELS[dt] for dt in DISAGREEMENT_ORDER], fontsize=10)
    ax.set_ylabel("Sycophancy Rate (correct->incorrect / initially correct)")
    ax.set_ylim(0, 0.85)
    ax.set_title("Sycophancy Rate by Model and Disagreement Type",
                 fontsize=14, fontweight="bold", color=TEXT_COLOR)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=9, framealpha=0.7)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "sycophancy_rate.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved sycophancy_rate.png")


def main():
    setup_font()
    df = load_data()
    plot_accuracy_comparison(df)
    plot_sycophancy_rate(df)


if __name__ == "__main__":
    main()
