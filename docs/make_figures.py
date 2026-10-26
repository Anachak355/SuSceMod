"""Make the README figures from the saved example outputs.

Run from the repository root after running the example notebooks:
    python docs/make_figures.py
"""
import contextlib
import io
import os
import numpy as np
import pandas as pd
import rasterio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from SuSceMod.simulation import data_utilities, input_handling, scenarios

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


# Growth-As-Usual keeps the blue it has in the other figures. Centralities is a new scenario with its own colour
GAU_COLOR, CENTRALITIES_COLOR = "#2a78d6", "#4a3aa7"


def _quiet(f, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return f(*args, **kwargs)


def centralities_runs():
    """
    Growth-As-Usual with the historical share inside the centralities kept, and the centralities scenario,
    both from the same map of 2030, with the same demand and the same seed. Only the share inside differs.
    """
    inputs = os.path.join(EXAMPLES, "input_rasters")
    saved = os.path.join(EXAMPLES, "output", "centralities")
    needed = [os.path.join(inputs, "BU_centralities.tif"), os.path.join(inputs, "ex_Zone_eau_binary.tif"),
              os.path.join(saved, "sim_2050.tif")]
    if not all(os.path.exists(f) for f in needed):
        return None

    load = lambda name: data_utilities.load_raster_data(os.path.join(inputs, name))
    start, before = load("BU_2020.tif"), load("BU_2010.tif")
    zones, water = load("BU_centralities.tif"), load("ex_Zone_eau_binary.tif")
    probs, _ = input_handling.load_prob_maps(os.path.join(EXAMPLES, "suitability_maps", "10_20"))
    change_maps, _ = input_handling.calculate_change_maps(os.path.join(inputs, "BU_2010.tif"), os.path.join(inputs, "BU_2020.tif"))
    rates = _quiet(input_handling.calculate_change_rates, change_maps, [2010, 2020])[0]
    historical = input_handling.calculate_central_shares(before, start, zones)

    forced = dict(water_zone=water, central_zones=zones, central_shares=historical, randomness="gumbel", seed=2118)
    to_2030 = _quiet(scenarios.simulate_growth_based_scenario, 2020, 2030, start, probs, rates, "o", **forced)[0]
    gau = _quiet(scenarios.simulate_growth_based_scenario, 2030, 2050, to_2030[2030], probs, rates, "o", **forced)[0]
    centralities = {year: data_utilities.load_raster_data(os.path.join(saved, f"sim_{year}.tif")) for year in range(2031, 2051)}
    return to_2030[2030], gau, centralities, zones


def figure_centralities_maps(runs):
    start_2030, gau, centralities, zones = runs
    built_up = read(os.path.join(EXAMPLES, "input_rasters", "BU_2020.tif"))
    rows_b, cols_b = np.nonzero(built_up > 0)
    pad = 20
    panels = [("Growth-As-Usual", GAU_COLOR, gau[2050]), ("Centralities", CENTRALITIES_COLOR, centralities[2050])]

    fig, axes = plt.subplots(len(panels), 1, figsize=(8, 3.4 * len(panels)))
    for ax, (label, color, end) in zip(axes, panels):
        changed = (end != start_2030) & ~np.isnan(end)
        exp_share, dens_share = input_handling.calculate_central_shares(start_2030, end, zones, decimals=2)
        ax.imshow(np.where(start_2030 > 0, 1.0, np.nan), cmap="Greys", vmin=0, vmax=3, interpolation="nearest")
        ax.contour((zones == 1).astype(float), levels=[0.5], colors=INK, linewidths=0.5)
        rows, cols = np.nonzero(changed)
        ax.scatter(cols, rows, s=2.5, color=color, linewidths=0)
        ax.set_title(f"{label}: {int(changed.sum()):,} cells changed, {exp_share:.0%} of the expansion and "
                     f"{dens_share:.0%} of the densification inside the centralities", loc="left", fontsize=10)
        ax.set_xlim(cols_b.min() - pad, cols_b.max() + pad)
        ax.set_ylim(rows_b.max() + pad, rows_b.min() - pad)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.suptitle("Cells that change between 2030 and 2050 (grey = built-up in 2030, black line = centralities)",
                 x=0.01, ha="left", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(os.path.join(OUT, "centralities_maps.png"), dpi=100, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


def figure_centralities_shares(runs):
    start_2030, gau, centralities, zones = runs
    fig, ax = plt.subplots(figsize=(7.5, 4))
    for label, color, maps in (("Growth-As-Usual", GAU_COLOR, gau), ("Centralities", CENTRALITIES_COLOR, centralities)):
        years, previous, shares = [], start_2030, []
        for year in range(2031, 2051):
            shares.append(input_handling.calculate_central_shares(previous, maps[year], zones, decimals=4))
            years.append(year)
            previous = maps[year]
        shares = np.array(shares)
        ax.plot(years, shares[:, 0], color=color, linewidth=2, label=f"{label}, expansion")
        ax.plot(years, shares[:, 1], color=color, linewidth=2, linestyle="--", label=f"{label}, densification")
    ax.axhline(0.75, color=MUTED, linewidth=1, linestyle=":")
    ax.text(2031, 0.765, "target 75%", color=MUTED, fontsize=9)
    ax.set_ylim(0, 1)
    ax.set_xticks([2031, 2035, 2040, 2045, 2050])
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_ylabel("Share of the change inside the centralities")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, fontsize=9, loc="center left", bbox_to_anchor=(0.0, 0.55))
    ax.set_title("Share of each year's change inside the centralities", loc="left", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "centralities_shares.png"), dpi=100, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


def figure_centralities():
    runs = centralities_runs()
    if runs is None:
        print("Centralities figures skipped: the centralities and water rasters or the saved centralities results are missing")
        return
    figure_centralities_maps(runs)
    figure_centralities_shares(runs)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    figure_changes()
    figure_trends()
    figure_centralities()
    print("Figures saved in", OUT)
