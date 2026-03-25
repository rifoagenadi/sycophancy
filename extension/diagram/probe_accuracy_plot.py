"""Visualize linear probe accuracies to answer:
1) Which part of the model has the clearest sycophancy signal?
2) Do different disagreement types share the same subspace?
"""

import json
from pathlib import Path

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr

RESULTS_DIR = Path(__file__).parent.parent / "probe_results"
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

ACT_TYPES = ["mha", "mlp", "residual"]
ACT_LABELS = {"mha": "Attention Heads", "mlp": "MLP", "residual": "Residual Stream"}

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


def load_accuracies(model, dt, act_type):
    path = RESULTS_DIR / model / dt / f"{act_type}_accuracies.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def per_layer_accuracy(accs, act_type, num_layers):
    """Aggregate per-layer accuracy. For MHA, take max over heads per layer."""
    layer_accs = []
    if act_type == "mha":
        for layer in range(num_layers):
            head_accs = [v for k, v in accs.items() if k.startswith(f"{layer}_")]
            layer_accs.append(max(head_accs) if head_accs else 50.0)
    else:
        for layer in range(num_layers):
            layer_accs.append(accs.get(str(layer), 50.0))
    return np.array(layer_accs)


def get_num_layers(model):
    # Infer from residual accuracy keys (one key per layer)
    accs = load_accuracies(model, DISAGREEMENT_ORDER[0], "residual")
    if accs is None:
        return 0
    return len(accs)


# ---------------------------------------------------------------------------
# Plot 1: Per-layer probe accuracy (Q1 — where is the signal?)
# ---------------------------------------------------------------------------

def plot_layer_accuracy(models):
    """Line plot of probe accuracy across layers for each activation type.
    One row per model, one column per activation type.
    Shows combined + individual disagreement types.
    """
    n_models = len(models)
    fig, axes = plt.subplots(n_models, 3, figsize=(18, 5 * n_models), squeeze=False)

    c1 = np.array([int(COLOR_1[i:i+2], 16) for i in (1, 3, 5)]) / 255
    c2 = np.array([int(COLOR_2[i:i+2], 16) for i in (1, 3, 5)]) / 255
    dt_colors = {dt: c1 + t * (c2 - c1)
                 for dt, t in zip(DISAGREEMENT_ORDER, np.linspace(0, 1, len(DISAGREEMENT_ORDER)))}

    for row, model in enumerate(models):
        num_layers = get_num_layers(model)
        x = np.arange(num_layers)

        for col, act_type in enumerate(ACT_TYPES):
            ax = axes[row, col]

            # Individual disagreement types (thin, semi-transparent)
            for dt in DISAGREEMENT_ORDER:
                accs = load_accuracies(model, dt, act_type)
                if accs is None:
                    continue
                y = per_layer_accuracy(accs, act_type, num_layers)
                ax.plot(x, y, color=dt_colors[dt], alpha=0.4, linewidth=1.0,
                        label=DISAGREEMENT_LABELS[dt])

            # Combined (thick, prominent)
            accs_combined = load_accuracies(model, "combined", act_type)
            if accs_combined is not None:
                y_combined = per_layer_accuracy(accs_combined, act_type, num_layers)
                ax.plot(x, y_combined, color=TEXT_COLOR, linewidth=2.5,
                        label="Combined", zorder=5)

            ax.axhline(50, color="#cccccc", linestyle="--", linewidth=0.8, zorder=0)
            ax.set_xlabel("Layer")
            ax.set_ylabel("Probe Accuracy (%)" if col == 0 else "")
            ax.set_title(f"{ACT_LABELS[act_type]}", fontsize=12, fontweight="bold")
            ax.set_ylim(45, 102)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            if col == 2:
                ax.legend(fontsize=8, framealpha=0.7, loc="lower right")

        # Row label
        axes[row, 0].annotate(
            model, xy=(-0.3, 0.5), xycoords="axes fraction",
            fontsize=14, fontweight="bold", color=TEXT_COLOR,
            ha="center", va="center", rotation=90,
        )

    fig.suptitle("Sycophancy Probe Accuracy Across Layers",
                 fontsize=16, fontweight="bold", color=TEXT_COLOR, y=1.01)
    fig.tight_layout()
    out = OUT_DIR / "probe_layer_accuracy.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


# ---------------------------------------------------------------------------
# Plot 2: MHA heatmap — best heads (Q1 detail)
# ---------------------------------------------------------------------------

def plot_mha_heatmap(models):
    """Heatmap of MHA probe accuracy per (layer, head) for combined disagreement."""
    from matplotlib.colors import LinearSegmentedColormap

    # Broken white (#f5f0eb) -> orange (#D57758)
    cmap_orange = LinearSegmentedColormap.from_list("white_orange", ["#f5f0eb", "#D57758"])

    n_models = len(models)
    # Extra width ratio for the colorbar on the right
    width_ratios = [1] * n_models + [0.05]
    fig, all_axes = plt.subplots(
        1, n_models + 1, figsize=(8 * n_models + 1, 6),
        gridspec_kw={"width_ratios": width_ratios}, squeeze=False,
    )
    axes = all_axes[0, :n_models]
    cbar_ax = all_axes[0, -1]

    im = None
    for i, model in enumerate(models):
        ax = axes[i]
        accs = load_accuracies(model, "combined", "mha")
        if accs is None:
            continue

        num_layers = get_num_layers(model)
        num_heads = max(int(k.split("_")[1]) for k in accs) + 1
        grid = np.full((num_layers, num_heads), 50.0)
        for k, v in accs.items():
            layer, head = [int(x) for x in k.split("_")]
            grid[layer, head] = v

        im = ax.imshow(grid, aspect="auto", cmap=cmap_orange, vmin=45, vmax=95,
                        origin="lower", interpolation="nearest")
        ax.set_xlabel("Head")
        ax.set_ylabel("Layer" if i == 0 else "")
        ax.set_title(model, fontsize=12, fontweight="bold")

    if im is not None:
        cbar = fig.colorbar(im, cax=cbar_ax)
        cbar.set_label("Probe Accuracy (%)", color=TEXT_COLOR)
        cbar.ax.yaxis.set_tick_params(color=TEXT_COLOR)
        plt.setp(cbar.ax.yaxis.get_ticklabels(), color=TEXT_COLOR)

    fig.suptitle("Attention Head Probe Accuracy (Combined Disagreement Types)",
                 fontsize=14, fontweight="bold", color=TEXT_COLOR, y=1.02)
    fig.tight_layout()
    out = OUT_DIR / "probe_mha_heatmap.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


# ---------------------------------------------------------------------------
# Plot 3: Cross-disagreement correlation (Q2 — shared subspace?)
# ---------------------------------------------------------------------------

def plot_cross_disagreement_correlation(models):
    """Pairwise correlation of MHA probe accuracies between disagreement types.
    One heatmap per model, laid out horizontally.
    """
    from matplotlib.colors import LinearSegmentedColormap
    cmap_orange = LinearSegmentedColormap.from_list("white_orange", ["#f5f0eb", "#D57758"])

    n_models = len(models)
    n_dt = len(DISAGREEMENT_ORDER)

    width_ratios = [1] * n_models + [0.05]
    fig, all_axes = plt.subplots(
        1, n_models + 1, figsize=(6 * n_models + 1, 5),
        gridspec_kw={"width_ratios": width_ratios}, squeeze=False,
    )
    axes = all_axes[0, :n_models]
    cbar_ax = all_axes[0, -1]

    im = None
    for col, model in enumerate(models):
        ax = axes[col]

        all_accs = {}
        for dt in DISAGREEMENT_ORDER:
            accs = load_accuracies(model, dt, "mha")
            if accs is not None:
                all_accs[dt] = accs

        if len(all_accs) < 2:
            ax.set_visible(False)
            continue

        corr_matrix = np.ones((n_dt, n_dt))
        for i, dt_i in enumerate(DISAGREEMENT_ORDER):
            for j, dt_j in enumerate(DISAGREEMENT_ORDER):
                if i >= j:
                    continue
                if dt_i not in all_accs or dt_j not in all_accs:
                    corr_matrix[i, j] = corr_matrix[j, i] = np.nan
                    continue
                common = sorted(set(all_accs[dt_i]) & set(all_accs[dt_j]))
                v1 = [all_accs[dt_i][k] for k in common]
                v2 = [all_accs[dt_j][k] for k in common]
                r, _ = pearsonr(v1, v2)
                corr_matrix[i, j] = corr_matrix[j, i] = r

        im = ax.imshow(corr_matrix, cmap=cmap_orange, vmin=0.5, vmax=1.0,
                       interpolation="nearest")
        ax.set_xticks(range(n_dt))
        ax.set_yticks(range(n_dt))
        short_labels = ["Simple", "Opinion", "Authority", "Emotional"]
        ax.set_xticklabels(short_labels, fontsize=11, rotation=45, ha="right")
        ax.set_yticklabels(short_labels, fontsize=11)
        ax.set_title(model, fontsize=14, fontweight="bold")

        for i in range(n_dt):
            for j in range(n_dt):
                val = corr_matrix[i, j]
                if not np.isnan(val):
                    cell_color = "white" if val > 0.7 else TEXT_COLOR
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                            fontsize=12, fontweight="bold", color=cell_color)

    if im is not None:
        cbar = fig.colorbar(im, cax=cbar_ax)
        cbar.set_label("Pearson Correlation", color=TEXT_COLOR, fontsize=12)
        cbar.ax.yaxis.set_tick_params(color=TEXT_COLOR)
        plt.setp(cbar.ax.yaxis.get_ticklabels(), color=TEXT_COLOR)

    fig.suptitle("Cross-Disagreement Attention Head Probe Correlation",
                 fontsize=16, fontweight="bold", color=TEXT_COLOR, y=1.02)
    fig.tight_layout()
    out = OUT_DIR / "probe_cross_disagreement_corr.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


# ---------------------------------------------------------------------------
# Plot 4: Scatter — pairwise disagreement type accuracy (Q2 detail)
# ---------------------------------------------------------------------------

def plot_pairwise_scatter(models):
    """Scatter plot of per-component probe accuracies for all disagreement type pairs.
    Uses residual stream (cleanest signal). One row per model.
    """
    pairs = []
    for i, dt_i in enumerate(DISAGREEMENT_ORDER):
        for j, dt_j in enumerate(DISAGREEMENT_ORDER):
            if i < j:
                pairs.append((dt_i, dt_j))

    n_pairs = len(pairs)
    n_models = len(models)
    fig, axes = plt.subplots(n_models, n_pairs, figsize=(4 * n_pairs, 4 * n_models), squeeze=False)

    c1 = np.array([int(COLOR_1[i:i+2], 16) for i in (1, 3, 5)]) / 255
    c2 = np.array([int(COLOR_2[i:i+2], 16) for i in (1, 3, 5)]) / 255

    for row, model in enumerate(models):
        for col, (dt_a, dt_b) in enumerate(pairs):
            ax = axes[row, col]
            accs_a = load_accuracies(model, dt_a, "residual")
            accs_b = load_accuracies(model, dt_b, "residual")
            if accs_a is None or accs_b is None:
                ax.set_visible(False)
                continue

            common = sorted(set(accs_a) & set(accs_b), key=int)
            va = np.array([accs_a[k] for k in common])
            vb = np.array([accs_b[k] for k in common])

            color = c1 * 0.5 + c2 * 0.5
            ax.scatter(va, vb, s=20, alpha=0.7, color=color, edgecolors="white", linewidths=0.3)

            # Diagonal
            lims = [min(va.min(), vb.min()) - 2, max(va.max(), vb.max()) + 2]
            ax.plot(lims, lims, "--", color="#cccccc", linewidth=0.8, zorder=0)
            ax.set_xlim(lims)
            ax.set_ylim(lims)

            r, _ = pearsonr(va, vb)
            ax.set_title(f"r = {r:.3f}", fontsize=10, color=TEXT_COLOR)
            ax.set_xlabel(DISAGREEMENT_LABELS[dt_a].replace("\n", " "), fontsize=8)
            ax.set_ylabel(DISAGREEMENT_LABELS[dt_b].replace("\n", " "), fontsize=8)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        axes[row, 0].annotate(
            model, xy=(-0.5, 0.5), xycoords="axes fraction",
            fontsize=12, fontweight="bold", color=TEXT_COLOR,
            ha="center", va="center", rotation=90,
        )

    fig.suptitle("Residual Stream Probe Accuracy: Pairwise Disagreement Comparison",
                 fontsize=14, fontweight="bold", color=TEXT_COLOR, y=1.02)
    fig.tight_layout()
    out = OUT_DIR / "probe_pairwise_scatter.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def main():
    setup_font()
    models = sorted([d.name for d in RESULTS_DIR.iterdir() if d.is_dir()])
    print(f"Models: {models}")

    plot_layer_accuracy(models)
    plot_mha_heatmap(models)
    plot_cross_disagreement_correlation(models)
    plot_pairwise_scatter(models)


if __name__ == "__main__":
    main()
