"""
Plot m1-m2 contours from posterior samples.

Usage
-----
python ligo_mass1_mass2_plot.py \
    --posterior-dir /path/to/results/ \
    --outfig        /path/to/mass1_mass2_plot.png \
    --highlight     inj001 inj002 inj003

Example: 
python ligo_mass1_mass2_plot.py --posterior-dir /home/xavier.garcia-sabat/Lensing/pesummaries_merge_all/ --outfig pesummaries_merge_all/mass1_mass2_plot.png --highlight inj001 inj002 inj003 --legend
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("agg")
matplotlib.rcParams["text.usetex"] = True
matplotlib.rcParams["font.size"] = 9
matplotlib.rcParams["savefig.dpi"] = 300
matplotlib.rcParams["text.latex.preamble"] = r"\usepackage{amsmath}"
matplotlib.rcParams["legend.fontsize"] = 9
matplotlib.rcParams["font.family"] = "serif"

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.legend_handler import HandlerLine2D
from matplotlib.lines import Line2D
from matplotlib.ticker import ScalarFormatter
from scipy.stats import gaussian_kde

PRIMARY_MASS_FIELDS   = ["mass_1_source", "m1_source", "mass_1", "m1", "mass1"]
SECONDARY_MASS_FIELDS = ["mass_2_source", "m2_source", "mass_2", "m2", "mass2"]

DEFAULT_EVENT_COLORS = [
    "#4F3A3A", "#4F774A", "#8A49D9",
    "#7A2F5B", "#A99853", "#D43D2D",
]

DEFAULT_TICKS       = [1, 2, 4, 7]
DEFAULT_MINOR_TICKS = [3, 5, 6, 8, 9]

# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def discover_posterior_files(base_dir: Path, filename: str) -> list[Path]:
    found = sorted(base_dir.glob(f"*/{filename}"))
    if not found:
        raise FileNotFoundError(f"No files named '{filename}' found under '{base_dir}'.")
    return found


def load_posterior_samples(path: Path) -> np.ndarray:
    with h5py.File(path, "r") as f:
        group = list(f.keys())[0]
        return f[group]["posterior_samples"][()]

# ---------------------------------------------------------------------------
# Mass extraction
# ---------------------------------------------------------------------------

def extract_mass_samples(samples):
    data = {n: np.asarray(samples[n], dtype=float).reshape(-1) for n in samples.dtype.names}

    def get(*names):
        for n in names:
            if n in data: return data[n]
        raise KeyError(f"None of these fields found: {', '.join(names)}")

    m1 = get(*PRIMARY_MASS_FIELDS)
    m2 = get(*SECONDARY_MASS_FIELDS)
    n  = min(len(m1), len(m2))
    m1, m2 = m1[:n], m2[:n]

    mask = np.isfinite(m1) & np.isfinite(m2) & (m1 > 0) & (m2 > 0)
    m1, m2 = m1[mask], m2[mask]

    # ensure m1 >= m2
    m1, m2 = np.maximum(m1, m2), np.minimum(m1, m2)
    mask = (m2 / m1) >= 1.0 / 50.0
    m1, m2 = m1[mask], m2[mask]

    if len(m1) < 2:
        raise ValueError("Not enough valid mass samples after filtering.")
    return m1, m2

# ---------------------------------------------------------------------------
# Density estimation
# ---------------------------------------------------------------------------

def compute_kde_grid(m1, m2, xgrid, ygrid):
    samples = np.vstack([np.log10(m1), np.log10(m2)])
    kde = gaussian_kde(samples, bw_method="scott")
    xx_log, yy_log = np.meshgrid(np.log10(xgrid), np.log10(ygrid))
    density = kde(np.vstack([xx_log.ravel(), yy_log.ravel()])).reshape(xx_log.shape)
    return density


def density_thresholds(density, credible_levels):
    flat = np.asarray(density, dtype=float).ravel()
    flat = flat[np.isfinite(flat)]
    order = np.argsort(flat)[::-1]
    cumulative = np.cumsum(flat[order])
    cumulative /= cumulative[-1]
    thresholds = []
    for level in credible_levels:
        idx = min(np.searchsorted(cumulative, level, side="left"), len(flat) - 1)
        thresholds.append(flat[order][idx])
    return sorted(set(thresholds))

# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def custom_log_ticks(limits, bases=DEFAULT_TICKS):
    lo, hi = float(limits[0]), float(limits[1])
    e_min = int(np.floor(np.log10(lo)))
    e_max = int(np.ceil(np.log10(hi)))
    return sorted({b * 10**e for e in range(e_min, e_max + 1)
                   for b in bases if lo <= b * 10**e <= hi})


def custom_log_minor_ticks(limits):
    return custom_log_ticks(limits, bases=DEFAULT_MINOR_TICKS)


def draw_m1m2_boundaries(ax):
    m = np.logspace(0, np.log10(200), 500)
    ax.fill_between(m, m, 200, color="lightgrey")
    ax.fill_between(m, m / 5, color="lightgrey", zorder=500)
    ax.plot(m, m / 5, ls="-", color="k", zorder=501)
    ax.text(72, 11, r"$q=1/5$", zorder=502)
    ax.text(40, 50, r"$q=1$")
    ax.plot(m, m, color="k", ls="-")

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Plot m1-m2 contours from posterior samples.")
    parser.add_argument("--posterior-dir",      required=True, metavar="DIR",
                        help="Base directory. Searches every subdirectory for --posterior-filename.")
    parser.add_argument("--posterior-filename", default="posterior_samples_joint_thin.h5",
                        help="H5 filename inside each subdirectory. (default: posterior_samples_joint_thin.h5)")
    parser.add_argument("--outfig",    default="ligo_mass1_mass2_plot.png", help="Output figure path.")
    parser.add_argument("--highlight", nargs="+", default=[],
                        help="Events to highlight with colour and include in legend.")
    parser.add_argument("--legend",    action="store_true", help="Show legend.")
    parser.add_argument("--levels",    nargs="+", type=float, default=[0.9],
                        help="Credible levels to contour. (default: 0.9)")
    parser.add_argument("--grid-size", type=int, default=180,
                        help="KDE grid size per axis. (default: 180)")
    parser.add_argument("--xlim",      nargs=2, type=float, default=[2.0, 200.0])
    parser.add_argument("--ylim",      nargs=2, type=float, default=[0.5, 200.0])
    return parser.parse_args()


def main():
    args = parse_args()

    posterior_paths = discover_posterior_files(Path(args.posterior_dir), args.posterior_filename)
    event_labels    = [p.parent.name for p in posterior_paths]

    print(f"Found {len(posterior_paths)} event(s).")
    loaded = []
    for label, path in zip(event_labels, posterior_paths):
        try:
            loaded.append((label, load_posterior_samples(path)))
        except Exception as exc:
            print(f"[skip] {label}: could not load posterior ({exc})")
    print(f"Loaded {len(loaded)} event(s) successfully.")

    xgrid = np.geomspace(args.xlim[0], args.xlim[1], args.grid_size)
    ygrid = np.geomspace(max(args.ylim[0], 1e-3), args.ylim[1], args.grid_size)
    xx, yy = np.meshgrid(xgrid, ygrid)
    physical_mask = (yy <= xx) & (yy >= xx / 50.0)

    fig, ax = plt.subplots(figsize=(6.75, 3.75))
    legend_elements = []

    for idx, (event, samples) in enumerate(loaded):
        print(f"Plotting {event}...")
        try:
            m1, m2  = extract_mass_samples(samples)
            density = compute_kde_grid(m1, m2, xgrid, ygrid)
            levels  = density_thresholds(density, args.levels)
            density = np.where(physical_mask, density, np.nan)

            if event in args.highlight:
                color = DEFAULT_EVENT_COLORS[idx % len(DEFAULT_EVENT_COLORS)]
                ax.contour(xx, yy, density, levels=levels,
                           colors=color, linewidths=1.5, zorder=100)
                label = event.replace("_", r"\_")
                legend_elements.append(
                    Line2D([], [], color=color, linewidth=1.5,
                           label=rf"$\mathrm{{{label}}}$")
                )
            else:
                ax.contour(xx, yy, density, levels=levels,
                           colors="k", linewidths=0.5, alpha=0.2)
        except Exception as exc:
            print(f"[skip] {event}: {exc}")

    draw_m1m2_boundaries(ax)

    ax.set_xlabel(r"Primary mass $m_1$ [$M_{\odot}$]", fontsize=12)
    ax.set_ylabel(r"Secondary mass $m_2$ [$M_{\odot}$]", fontsize=12)
    ax.set_xscale("log")
    ax.set_yscale("log")

    for axis, lim in [(ax.xaxis, args.xlim), (ax.yaxis, args.ylim)]:
        axis.set_ticks(custom_log_ticks(lim))
        axis.set_ticks(custom_log_minor_ticks(lim), minor=True)

    fmt = ScalarFormatter()
    fmt.set_scientific(False)
    ax.xaxis.set_major_formatter(fmt)
    ax.yaxis.set_major_formatter(fmt)
    ax.tick_params(axis="both", size=6, labelsize=12)
    ax.set_xlim(10,150)
    ax.set_ylim(7,150)

    if args.legend and legend_elements:
        ax.legend(
            handles=legend_elements,
            handlelength=3,
            bbox_to_anchor=(0, 1.02, 1, 0.2),
            loc="lower left",
            ncols=3,
            mode="expand",
            frameon=False,
            handler_map={line: HandlerLine2D(numpoints=3) for line in legend_elements},
        )

    plt.grid(False)
    fig.tight_layout()
    outpath = Path(args.outfig)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath)
    print(f"Saved plot to: {outpath.resolve()}")


if __name__ == "__main__":
    main()