"""
Peak-current probability analysis for lightning strokes (positive and negative separately).

For each polarity this script:
  1. Builds the empirical cumulative probability P(I > i) of the stroke peak current.
  2. Fits a log-normal distribution (the standard model used in IEC 62305 / CIGRE).
  3. Reports the midpoint (median, 50 % probability) and other exceedance levels.
  4. Builds a probability matrix: P(I > threshold) for a set of kA thresholds.
  5. Plots the cumulative probability curve and the probability density (distribution).

Usage:
    python lightning_probability.py                       # defaults to the Covenant CSV, ground strokes
    python lightning_probability.py data.csv --type all   # all strokes (ground + cloud)
    python lightning_probability.py big.csv --chunksize 1000000   # for very large files (e.g. 9M rows)

Input columns used: current(KA) (sign = polarity) and gc (G = cloud-to-ground, C = intra-cloud).
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter
from scipy import stats

CURRENT_COL = "current(KA)"
TYPE_COL = "gc"

# Exceedance probabilities reported in the summary (P(I > i) = p)
LEVELS = [0.95, 0.90, 0.80, 0.50, 0.20, 0.10, 0.05, 0.01]
# Current thresholds (kA) used for the probability matrix
THRESHOLDS = [2, 3, 5, 10, 15, 20, 30, 40, 50, 75, 100, 150, 200]

COLORS = {"Negative": "#2a78d6", "Positive": "#eb6834"}
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def load(path, chunksize=None):
    cols = [CURRENT_COL, TYPE_COL]
    if chunksize:
        parts = pd.read_csv(path, usecols=cols, chunksize=chunksize)
        return pd.concat(parts, ignore_index=True)
    return pd.read_csv(path, usecols=cols)


def split_polarity(df, stroke_type):
    if stroke_type == "ground":
        df = df[df[TYPE_COL] == "G"]
    elif stroke_type == "cloud":
        df = df[df[TYPE_COL] == "C"]
    current = pd.to_numeric(df[CURRENT_COL], errors="coerce").dropna()
    current = current[current != 0]
    return {
        "Negative": np.sort(-current[current < 0].to_numpy()),
        "Positive": np.sort(current[current > 0].to_numpy()),
    }


def fit_lognormal(x):
    """Return (median kA, sigma of ln I) for a log-normal fit."""
    ln = np.log(x)
    return float(np.exp(ln.mean())), float(ln.std(ddof=1))


def exceedance_current(median, sigma, p):
    """Current exceeded with probability p under the log-normal fit."""
    return median * np.exp(sigma * stats.norm.ppf(1 - p))


def summarise(data):
    rows, matrix = [], []
    for pol, x in data.items():
        median, sigma = fit_lognormal(x)
        row = {
            "polarity": pol,
            "n_strokes": len(x),
            "empirical_median_kA": round(float(np.median(x)), 2),
            "lognormal_median_kA": round(median, 2),
            "sigma_ln": round(sigma, 3),
            "sigma_log10": round(sigma / np.log(10), 3),
            "mean_kA": round(float(x.mean()), 2),
            "max_kA": round(float(x.max()), 2),
        }
        for p in LEVELS:
            row[f"I_exceeded_{int(p * 100)}pct_kA"] = round(exceedance_current(median, sigma, p), 2)
        rows.append(row)

        for t in THRESHOLDS:
            matrix.append({
                "polarity": pol,
                "threshold_kA": t,
                "empirical_P(I>t)": round(float((x > t).mean()), 4),
                "lognormal_P(I>t)": round(float(stats.norm.sf((np.log(t) - np.log(median)) / sigma)), 4),
            })
    return pd.DataFrame(rows), pd.DataFrame(matrix)


def style_axes(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_2)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.grid(True, which="major", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def plot_cumulative(data, title, out):
    fig, ax = plt.subplots(figsize=(9, 5.8), facecolor=SURFACE)
    style_axes(ax)
    xmax = max(x.max() for x in data.values())
    grid_i = np.logspace(np.log10(0.5), np.log10(xmax * 1.2), 400)

    for pol, x in data.items():
        c = COLORS[pol]
        median, sigma = fit_lognormal(x)
        p_emp = 1 - np.arange(len(x)) / len(x)  # P(I >= x_k)
        ax.step(x, p_emp * 100, where="post", color=c, linewidth=2, alpha=0.45,
                label=f"{pol} strokes, measured (n = {len(x):,})")
        ax.plot(grid_i, stats.norm.sf((np.log(grid_i) - np.log(median)) / sigma) * 100,
                color=c, linewidth=2, linestyle="--",
                label=f"{pol} log-normal fit (median {median:.1f} kA, σ ln = {sigma:.2f})")
        ax.plot([median], [50], marker="o", markersize=9, color=c,
                markeredgecolor=SURFACE, markeredgewidth=2, zorder=5)
        ax.vlines(median, 0, 50, color=c, linewidth=1, linestyle=":")
        # negative label to the right of its marker, positive to the left, so they never collide
        right = pol == "Negative"
        ax.annotate(f"{pol}: {median:.1f} kA", (median, 50), xytext=(10 if right else -10, -16),
                    textcoords="offset points", ha="left" if right else "right",
                    color=INK, fontsize=9, fontweight="bold")

    ax.axhline(50, color=INK_2, linewidth=1, linestyle=":")
    ax.text(0.55, 51.5, "Midpoint: 50 % probability", color=INK_2, fontsize=9)
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_xlim(0.5, xmax * 1.2)
    ax.set_ylim(0, 102)
    ax.set_xlabel("Peak current magnitude, I (kA)", color=INK, fontsize=10)
    ax.set_ylabel("Probability that peak current exceeds I (%)", color=INK, fontsize=10)
    ax.set_title(title, color=INK, fontsize=12, loc="left", pad=12)
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK, loc="upper right")
    fig.tight_layout()
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def plot_density(data, title, out):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), facecolor=SURFACE)
    for ax, (pol, x) in zip(axes, data.items()):
        style_axes(ax)
        c = COLORS[pol]
        median, sigma = fit_lognormal(x)
        upper = np.percentile(x, 99.5)
        bins = np.linspace(0, upper, 40)
        ax.hist(x[x <= upper], bins=bins, density=True, color=c, alpha=0.55,
                edgecolor=SURFACE, linewidth=1.5, label="Measured (histogram)")
        grid_i = np.linspace(0.01, upper, 400)
        ax.plot(grid_i, stats.lognorm.pdf(grid_i, s=sigma, scale=median), color=c,
                linewidth=2, label="Log-normal fit")
        ax.axvline(median, color=INK_2, linestyle=":", linewidth=1)
        ax.text(median, ax.get_ylim()[1] * 0.95, f"  median {median:.1f} kA", color=INK, fontsize=9, va="top")
        ax.set_title(f"{pol} strokes (n = {len(x):,})", color=INK, fontsize=11, loc="left")
        ax.set_xlabel("Peak current magnitude, I (kA)", color=INK, fontsize=10)
        ax.set_ylabel("Probability density (1/kA)", color=INK, fontsize=10)
        ax.legend(frameon=False, fontsize=8.5, labelcolor=INK)
    fig.suptitle(title, color=INK, fontsize=12, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csv", nargs="?", default="Covenant_Nigeria_2015-2023.csv")
    parser.add_argument("--type", choices=["ground", "cloud", "all"], default="ground",
                        help="stroke type: ground = cloud-to-ground (gc == G, default), cloud = intra-cloud, all = both")
    parser.add_argument("--out", default="results")
    parser.add_argument("--chunksize", type=int, default=None, help="read the CSV in chunks (large files)")
    parser.add_argument("--label", default="Covenant University, Ota, 2015–2023")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    data = split_polarity(load(args.csv, args.chunksize), args.type)
    type_name = {"ground": "cloud-to-ground", "cloud": "intra-cloud", "all": "all"}[args.type]

    summary, matrix = summarise(data)
    summary.to_csv(os.path.join(args.out, f"summary_{args.type}.csv"), index=False)
    matrix.to_csv(os.path.join(args.out, f"probability_matrix_{args.type}.csv"), index=False)

    plot_cumulative(data, f"Cumulative probability of peak current, {type_name} strokes\n{args.label}",
                    os.path.join(args.out, f"cumulative_probability_{args.type}.png"))
    plot_density(data, f"Probability distribution of peak current, {type_name} strokes, {args.label}",
                 os.path.join(args.out, f"probability_distribution_{args.type}.png"))

    pd.set_option("display.width", 200)
    print(summary.T.to_string(header=False))
    print()
    print(matrix.pivot(index="threshold_kA", columns="polarity", values=["empirical_P(I>t)", "lognormal_P(I>t)"]))


if __name__ == "__main__":
    main()
