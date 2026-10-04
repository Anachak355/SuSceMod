"""Make the README figures from the saved example outputs.

Run from the repository root after running the example notebooks:
    python docs/make_figures.py
"""
import os
import numpy as np
import pandas as pd
import rasterio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EXAMPLES = os.path.join("SuSceMod", "examples")
OUT = "docs"

# One colour per scenario in every figure
SCENARIOS = [
    ("growth_based", "Growth-based", "#2a78d6"),
    ("density_based", "Density-based", "#eb6834"),
    ("densification_only", "Densification-only", "#1baf7a"),
]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    "text.color": INK, "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.edgecolor": GRID, "font.size": 10,
})


def read(path):
    with rasterio.open(path) as src:
        return src.read(1)


def figure_changes():
    start = read(os.path.join(EXAMPLES, "input_rasters", "BU_2020.tif"))
    fig, axes = plt.subplots(len(SCENARIOS), 1, figsize=(8, 3.2 * len(SCENARIOS)))
    rows_b, cols_b = np.nonzero(start > 0)
    pad = 20
    for ax, (folder, label, color) in zip(axes, SCENARIOS):
        end = read(os.path.join(EXAMPLES, "output", folder, "sim_2030.tif"))
        changed = (end != start) & ~np.isnan(end)
        ax.imshow(np.where(start > 0, 1.0, np.nan), cmap="Greys", vmin=0, vmax=3, interpolation="nearest")
        rows, cols = np.nonzero(changed)
        ax.scatter(cols, rows, s=2.5, color=color, linewidths=0)
        ax.set_title(f"{label}: {int(changed.sum()):,} cells changed", loc="left", fontsize=11)
        ax.set_xlim(cols_b.min() - pad, cols_b.max() + pad)
        ax.set_ylim(rows_b.max() + pad, rows_b.min() - pad)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.suptitle("Cells that change between 2020 and 2030 (grey = built-up in 2020)", x=0.01, ha="left", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(os.path.join(OUT, "scenario_changes.png"), dpi=100, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


def figure_trends():
    start = read(os.path.join(EXAMPLES, "input_rasters", "BU_2020.tif"))
    start_counts = {c: int(np.sum(start == c)) for c in range(4)}
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.4), sharex=True)
    for folder, label, color in SCENARIOS:
        df = pd.read_csv(os.path.join(EXAMPLES, "output", folder, "class_counts_2020_2030.csv"), index_col=0)
        df = df[~df.index.astype(str).isin(["nan"])]
        years = [2020] + [int(y) for y in df.columns]
        for ax, c in zip(axes, range(4)):
            values = [0] + list(df.loc[float(c)].values - start_counts[c])
            ax.plot(years, values, color=color, linewidth=2, label=label)
    for ax, c in zip(axes, range(4)):
        ax.set_title(f"Class {c}", loc="left", fontsize=11)
        ax.axhline(0, color=GRID, linewidth=1)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.set_xticks([2020, 2025, 2030])
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel("Change in cells since 2020")
    axes[0].legend(frameon=False, fontsize=9, loc="best")
    fig.suptitle("Change in cell count per density class", x=0.01, ha="left", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(os.path.join(OUT, "class_trends.png"), dpi=100)
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    figure_changes()
    figure_trends()
    print("Figures saved in", OUT)
