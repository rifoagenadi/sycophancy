"""Visualize correlation between uncertainty scores and sycophancy label.

Generates one box plot per (model, score) pair.

Usage:
    python visualize.py <file1.csv> [file2.csv ...]
"""

import argparse
from pathlib import Path

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats

OUTPUT_DIR = Path("uncertainty_result")
FONT_DIR = Path(__file__).parent.parent / "extension" / "diagram" / "fonts"

COLOR_1 = "#1b7f5a"
COLOR_2 = "#D57758"
TEXT_COLOR = "#4a4844"

SCORE_COLS = ["semantic_negentropy", "mean_token_negentropy"]


def setup_font():
    font_path = FONT_DIR / "Nunito-VariableFont_wght.ttf"
    if not font_path.exists():
        import urllib.request
        FONT_DIR.mkdir(parents=True, exist_ok=True)
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


def _model_name(filepath: Path) -> str:
    return filepath.stem.replace("_opinion_responses_labeled", "")


def plot_boxplot(df: pd.DataFrame, col: str, model_name: str):
    """Single box plot for one score vs sycophancy label."""
    fig, ax = plt.subplots(figsize=(5, 5))

    syco = df[df["label"] == "sycophancy"][col].dropna()
    non_syco = df[df["label"] == "non-sycophancy"][col].dropna()

    bp = ax.boxplot(
        [non_syco, syco],
        tick_labels=["non-sycophancy", "sycophancy"],
        patch_artist=True,
        boxprops=dict(linewidth=1.2),
        medianprops=dict(color=TEXT_COLOR, linewidth=1.5),
        whiskerprops=dict(color=TEXT_COLOR),
        capprops=dict(color=TEXT_COLOR),
        flierprops=dict(marker="o", markerfacecolor=TEXT_COLOR,
                        markersize=3, alpha=0.5, linestyle="none"),
    )
    bp["boxes"][0].set_facecolor(COLOR_1)
    bp["boxes"][0].set_alpha(0.8)
    bp["boxes"][1].set_facecolor(COLOR_2)
    bp["boxes"][1].set_alpha(0.8)

    _, mw_p = stats.mannwhitneyu(syco, non_syco, alternative="two-sided")
    rpb, rpb_p = stats.pointbiserialr(
        (df["label"] == "sycophancy").astype(int), df[col],
    )

    ax.set_title(
        f"{model_name}: {col}\n"
        f"Mann-Whitney p={mw_p:.4f} | r_pb={rpb:.3f} (p={rpb_p:.4f})",
        fontsize=10, fontweight="bold", color=TEXT_COLOR,
    )
    ax.set_ylabel("Score", fontsize=10)

    for i, data in enumerate([non_syco, syco]):
        ax.text(
            i + 1.2, data.median(), f"med={data.median():.3f}",
            ha="left", va="center", fontsize=9, fontweight="bold",
            color=TEXT_COLOR, rotation=90,
        )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    out = OUTPUT_DIR / f"{model_name}_{col}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def print_summary(df: pd.DataFrame, model_name: str):
    print(f"\n{'='*65}")
    print(f"Summary: {model_name}")
    print(f"{'='*65}")
    binary = (df["label"] == "sycophancy").astype(int)
    for col in SCORE_COLS:
        grouped = df.groupby("label")[col].agg(["mean", "median", "std", "count"])
        rpb, rpb_p = stats.pointbiserialr(binary, df[col])
        _, mw_p = stats.mannwhitneyu(
            df[df["label"] == "sycophancy"][col].dropna(),
            df[df["label"] == "non-sycophancy"][col].dropna(),
            alternative="two-sided",
        )
        print(f"\n{col}:")
        print(grouped.to_string())
        print(f"  Point-biserial r={rpb:.4f} (p={rpb_p:.4f})")
        print(f"  Mann-Whitney U p={mw_p:.4f}")
    print()


def main():
    parser = argparse.ArgumentParser(description="Visualize uncertainty vs sycophancy")
    parser.add_argument("files", nargs="+", type=Path, help="Labeled CSV files to visualize")
    args = parser.parse_args()

    setup_font()

    for filepath in args.files:
        df = pd.read_csv(filepath)
        model_name = _model_name(filepath)

        missing = [c for c in SCORE_COLS if c not in df.columns]
        if missing:
            print(f"SKIP {filepath.name}: missing columns {missing}")
            continue

        for col in SCORE_COLS:
            plot_boxplot(df, col, model_name)

        print_summary(df, model_name)


if __name__ == "__main__":
    main()
