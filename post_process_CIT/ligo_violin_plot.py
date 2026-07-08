"""
Generate LIGO-style violin/KDE summary plots from posterior samples and injection values.

This version overlays three KDE outlines:
- green solid line: joint posterior
  * For paired parameters (e.g. luminosity_distance): image 1 fills the UPPER half,
    image 2 fills the LOWER half of each violin row (asymmetric split-violin).
- red dash-dot line: image 1 single posterior
- blue dotted line: image 2 single posterior
- green vertical ticks: joint 90% credible interval
- black dotted vertical tick: joint median
- black solid vertical tick: injected value

Usage
-----
python ligo_violin_plot.py \
    --posterior-dir /path/to/results/ \
    --injections    /path/to/injections.txt \
    --parameters    chirp_mass mass_ratio chi_eff chi_p luminosity_distance \
    --output        /path/to/output.png

Example:
python ligo_violin_plot.py --posterior-dir /home/xavier.garcia-sabat/Lensing/pesummaries_merge_all/ --injections /home/xavier.garcia-sabat/Lensing/ler_data/n_lensed_params_bbh_filtered.txt --parameters chirp_mass mass_ratio a_1 a_2 chi_eff chi_p luminosity_distance --output /home/xavier.garcia-sabat/Lensing/pesummaries_merge_all/ligo_violin_plot.png
"""

from __future__ import annotations

import argparse
import ast
import json
import pickle
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde

plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "serif"


# Constants
# ---------------------------------------------------------------------------

DEFAULT_PARAMETERS = [
    "chirp_mass", "mass_ratio", "chi_eff", "chi_p",
    "luminosity_distance", "theta_jn", "ra", "dec",
    "psi", "phase", "a_1", "a_2", "tilt_1", "tilt_2",
]

LABEL_MAP = {
    "chirp_mass":           r"$\mathcal{M}\,[M_\odot]$",
    "chirp_mass_source":    r"$\mathcal{M}_{\rm source}$",
    "mass_ratio":           r"$q$",
    "chi_eff":              r"$\chi_\mathrm{eff}$",
    "chi_p":                r"$\chi_p$",
    "luminosity_distance":  r"$D_{\rm eff}\,[\mathrm{Mpc}]$",
    "theta_jn":             r"$\theta_{JN}$",
    "psi":                  r"$\psi$",
    "phase":                r"$\phi_c$",
    "a_1":                  r"$a_1$",
    "a_2":                  r"$a_2$",
    "tilt_1":               r"$\mathrm{tilt}_1$",
    "tilt_2":               r"$\mathrm{tilt}_2$",
    "ra":                   r"$\alpha$",
    "dec":                  r"$\delta$",
}

INJECTION_ALIASES = {"luminosity_distance": "effective_luminosity_distance"}

JOINT_COLOR = "#5DBB4A"
SINGLE1_COLOR = "#B44235"
SINGLE2_COLOR = "#4B55A1"


# I/O helpers
# ---------------------------------------------------------------------------

def load_posterior_samples(path: Path) -> np.ndarray:
    with h5py.File(path, "r") as f:
        group = list(f.keys())[0]
        return f[group]["posterior_samples"][()]


def load_injections(path: Path):
    suffix = path.suffix.lower()
    if suffix in {".h5", ".hdf5"}:
        with h5py.File(path, "r") as f:
            first = list(f.keys())[0]
            obj = f[first]
            if isinstance(obj, h5py.Dataset):
                return obj[()]
            if "injections" in obj:
                return obj["injections"][()]
            return {k: obj[k][()] for k in obj.keys()}
    if suffix in {".pkl", ".pickle"}:
        with open(path, "rb") as f:
            return pickle.load(f)
    if suffix == ".json":
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    if suffix == ".txt":
        with open(path, encoding="utf-8") as f:
            raw = f.read().strip()
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return ast.literal_eval(raw)
    raise ValueError(f"Unsupported injection file format: {path}")


def discover_posterior_files(base_dir: Path, filename: str) -> list[Path]:
    found = sorted(base_dir.glob(f"*/{filename}"))
    if not found:
        raise FileNotFoundError(
            f"No files named '{filename}' found in any subdirectory of '{base_dir}'."
        )
    return found


def matching_single_paths(joint_paths: list[Path], filename: str) -> list[Path | None]:
    paths = []
    for joint_path in joint_paths:
        candidate = joint_path.parent / filename
        paths.append(candidate if candidate.exists() else None)
    return paths


def load_optional_posteriors(paths: list[Path | None]) -> list[np.ndarray | None]:
    arrays = []
    for path in paths:
        if path is None:
            arrays.append(None)
            continue
        try:
            arrays.append(load_posterior_samples(path))
        except Exception:
            arrays.append(None)
    return arrays


# Injection value resolution
# ---------------------------------------------------------------------------

def _derived_from_masses(injections: dict, parameter: str, idx: int) -> float:
    m1 = float(np.asarray(injections["mass_1"])[idx])
    m2 = float(np.asarray(injections["mass_2"])[idx])
    q = min(m1, m2) / max(m1, m2)
    if parameter == "mass_ratio":
        return q
    if parameter == "chirp_mass":
        return (m1 * m2) ** (3 / 5) / (m1 + m2) ** (1 / 5)
    if parameter in {"chi_eff", "chi_p"}:
        a1 = float(np.asarray(injections["a_1"])[idx])
        a2 = float(np.asarray(injections["a_2"])[idx])
        tilt1 = float(np.asarray(injections["tilt_1"])[idx])
        tilt2 = float(np.asarray(injections["tilt_2"])[idx])
        if parameter == "chi_eff":
            return (m1 * a1 * np.cos(tilt1) + m2 * a2 * np.cos(tilt2)) / (m1 + m2)
        weight = q * (4 * q + 3) / (4 + 3 * q)
        return max(a1 * np.sin(tilt1), weight * a2 * np.sin(tilt2))
    raise KeyError(parameter)


def resolve_injection(injections, parameter: str, idx: int) -> tuple[float, float]:
    """Return (left_truth, right_truth). Both equal unless luminosity_distance."""
    if parameter == "luminosity_distance":
        values = np.asarray(injections[INJECTION_ALIASES[parameter]])
        return float(values[idx, 0]), float(values[idx, 1])

    key = INJECTION_ALIASES.get(parameter, parameter)

    if isinstance(injections, np.ndarray) and injections.dtype.names:
        if parameter in injections.dtype.names:
            val = injections[parameter] if injections.ndim == 0 else injections[parameter][idx]
            v = float(np.asarray(val).reshape(-1)[0])
            return v, v

    if isinstance(injections, dict):
        if key not in injections:
            if parameter in {"chirp_mass", "mass_ratio", "chi_eff", "chi_p"}:
                v = _derived_from_masses(injections, parameter, idx)
                return v, v
            raise KeyError(parameter)
        values = np.asarray(injections[key])
        if values.ndim == 0:
            v = float(values)
        elif values.ndim == 1:
            v = float(values[idx])
        else:
            v = float(values[idx, 0])
        return v, v

    raise KeyError(parameter)


# Parameter selection
# ---------------------------------------------------------------------------

def select_parameters(posterior: np.ndarray, injections, requested: list[str] | None, idx: int) -> list[str]:
    names = set(posterior.dtype.names)
    candidates = requested or DEFAULT_PARAMETERS
    selected = []
    for p in candidates:
        has_p = p in names or f"{p}^(1)" in names
        if not has_p:
            continue
        try:
            resolve_injection(injections, p, idx)
        except Exception:
            continue
        selected.append(p)
    if not selected:
        raise ValueError("No common parameters found between posterior samples and injections.")
    return selected


# Plot helpers
# ---------------------------------------------------------------------------

def _compute_kde(values: np.ndarray, gridsize: int = 200):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return None, None
    if values.size == 1 or np.allclose(values, values[0]):
        support = np.array([values[0] - 1e-9, values[0] + 1e-9])
        return support, np.array([1.0, 1.0])
    support = np.linspace(values.min(), values.max(), gridsize)
    density = gaussian_kde(values)(support)
    if density.max() > 0:
        density /= density.max()
    return support, density


def _draw_horizontal_kde_outline(
    ax,
    values,
    center,
    color,
    width,
    linestyle,
    draw_median=False,
    half=None,
):
    support, density = _compute_kde(values)
    if support is None:
        return

    half_width = width / 2
    scaled = density * half_width

    baseline = np.full_like(support, center)

    if half == "upper":
        ax.plot(support, center + scaled,
                color=color, linewidth=1.25, linestyle=linestyle, zorder=7)
        ax.plot(support, baseline,
                color=color, linewidth=0.7, linestyle=linestyle, zorder=7)
    elif half == "lower":
        ax.plot(support, center - scaled,
                color=color, linewidth=1.25, linestyle=linestyle, zorder=7)
        ax.plot(support, baseline,
                color=color, linewidth=0.7, linestyle=linestyle, zorder=7)
    else:
        ax.plot(support, center + scaled,
                color=color, linewidth=1.25, linestyle=linestyle, zorder=7)
        ax.plot(support, center - scaled,
                color=color, linewidth=1.25, linestyle=linestyle, zorder=7)

    if draw_median:
        p50 = np.percentile(values, 50)
        w = np.interp(p50, support, density) * half_width
        if half == "upper":
            ax.plot([p50, p50], [center, center + w],
                    color=color, linewidth=1.1, linestyle=":", zorder=9)
        elif half == "lower":
            ax.plot([p50, p50], [center - w, center],
                    color=color, linewidth=1.1, linestyle=":", zorder=9)
        else:
            ax.plot([p50, p50], [center - w, center + w],
                    color=color, linewidth=1.1, linestyle=":", zorder=9)


def _draw_joint_credible_interval_ticks(
    ax,
    values,
    center,
    color,
    credible_level=90.0,
    half_height=0.055,
):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return

    tail = 0.5 * (100.0 - credible_level)
    low, high = np.percentile(values, [tail, 100.0 - tail])
    for value in (low, high):
        ax.plot(
            [value, value],
            [center - half_height, center + half_height],
            color=color,
            linewidth=1.2,
            solid_capstyle="butt",
            zorder=12,
        )


def _draw_anchored_vertical_ticks(
    ax,
    values,
    y_start,
    y_end,
    color,
    linewidth,
    zorder,
):
    for value in values:
        ax.plot(
            [value, value],
            [y_start, y_end],
            color=color,
            linewidth=linewidth,
            solid_capstyle="butt",
            zorder=zorder,
        )


def _get_single_values(samples: np.ndarray | None, parameter: str):
    if samples is None:
        return None
    if parameter in samples.dtype.names:
        return np.asarray(samples[parameter], dtype=float)
    return None


def _draw_single_overlays(ax, single1, single2, parameter, row_idx):
    vals1 = _get_single_values(single1, parameter)
    vals2 = _get_single_values(single2, parameter)

    if vals1 is not None:
        _draw_horizontal_kde_outline(
            ax, vals1, row_idx, SINGLE1_COLOR, 0.68, "-.", half="lower"
        )
    if vals2 is not None:
        _draw_horizontal_kde_outline(
            ax, vals2, row_idx, SINGLE2_COLOR, 0.68, ":", half="upper"
        )


# Main plotting
# ---------------------------------------------------------------------------

def plot_paper_summary(
    posterior_arrays: list[np.ndarray],
    single1_arrays: list[np.ndarray | None],
    single2_arrays: list[np.ndarray | None],
    event_labels: list[str],
    injection_indices: list[int],
    injections,
    parameters: list[str],
    output: Path,
    title: str | None = None,
):
    n_events = len(posterior_arrays)
    n_params = len(parameters)

    fig, axes = plt.subplots(
        1, n_params,
        figsize=(2.3 * n_params + 2.8, 0.55 * 145),
        sharey=True,
        constrained_layout=False,
    )
    if n_params == 1:
        axes = [axes]

    # Reserve space at the bottom for the legend
    fig.subplots_adjust(left=0.12, right=0.98, top=0.92, bottom=0.1, wspace=0.08)

    for col_idx, parameter in enumerate(parameters):
        ax = axes[col_idx]

        for row_idx, (samples, single1, single2, inj_idx) in enumerate(
            zip(posterior_arrays, single1_arrays, single2_arrays, injection_indices)
        ):
            left_key = f"{parameter}^(1)"
            right_key = f"{parameter}^(2)"
            has_pair = left_key in samples.dtype.names and right_key in samples.dtype.names
            if has_pair:
                left_vals = np.asarray(samples[left_key], dtype=float)
                right_vals = np.asarray(samples[right_key], dtype=float)

                left_half  = "lower" if parameter == "luminosity_distance" else "upper"
                right_half = "upper" if parameter == "luminosity_distance" else "lower"

                _draw_horizontal_kde_outline(
                ax, left_vals, row_idx, JOINT_COLOR, 0.72, "-",
                draw_median=True, half=left_half,
                )
                _draw_horizontal_kde_outline(
                ax, right_vals, row_idx, JOINT_COLOR, 0.72, "-",
                draw_median=True, half=right_half,
                )
                left_tail, left_high = np.percentile(left_vals[np.isfinite(left_vals)], [5.0, 95.0])
                right_tail, right_high = np.percentile(right_vals[np.isfinite(right_vals)], [5.0, 95.0])
                _draw_anchored_vertical_ticks(
                    ax, [left_tail, left_high], row_idx, row_idx - 0.09,
                    JOINT_COLOR, 1.45, 12,
                )
                _draw_anchored_vertical_ticks(
                    ax, [right_tail, right_high], row_idx, row_idx + 0.09,
                    JOINT_COLOR, 1.45, 12,
                )
                _draw_single_overlays(ax, single1, single2, parameter, row_idx)

                left_truth, right_truth = resolve_injection(injections, parameter, inj_idx)
                _draw_anchored_vertical_ticks(
                    ax, [left_truth], row_idx, row_idx - 0.13,
                    "0.10", 1.2, 13,
                )
                _draw_anchored_vertical_ticks(
                    ax, [right_truth], row_idx, row_idx + 0.13,
                    "0.10", 1.2, 13,
                )

            else:
                vals = np.asarray(samples[parameter], dtype=float)

                _draw_horizontal_kde_outline(
                    ax, vals, row_idx, JOINT_COLOR, 0.72, "-",
                    draw_median=True, half="upper",
                )
                _draw_horizontal_kde_outline(
                    ax, vals, row_idx, JOINT_COLOR, 0.72, "-",
                    draw_median=True, half="lower",
                )
                _draw_joint_credible_interval_ticks(ax, vals, row_idx, JOINT_COLOR)
                _draw_single_overlays(ax, single1, single2, parameter, row_idx)

                truth, _ = resolve_injection(injections, parameter, inj_idx)
                ax.scatter(
                    [truth], [row_idx],
                    marker="|", s=120, linewidths=1.2, color="0.10", zorder=10,
                )

            ax.axhline(row_idx, color="0.85", linewidth=0.7, zorder=0)

        # ── Parameter label on both top and bottom ───────────────────────────
        param_label = LABEL_MAP.get(parameter, parameter)
        ax.set_xlabel(param_label, fontsize=13, labelpad=6)
        ax.set_title(param_label, fontsize=13, pad=6)

        # Ticks and tick labels on both top and bottom
        ax.xaxis.set_ticks_position("both")
        ax.xaxis.set_label_position("bottom")
        ax.tick_params(axis="x", which="both", top=True, bottom=True,
                       labeltop=True, labelbottom=True, labelsize=9)
        ax.grid(axis="x", alpha=0.22, linewidth=0.7)

        if col_idx == 0:
            ax.set_yticks(np.arange(n_events))
            ax.set_yticklabels(event_labels, fontsize=10)
        else:
            ax.tick_params(axis="y", left=False, labelleft=False)

    # ── Legend: horizontal, below all axes, 3 items in one row
    legend_handles = [
        Line2D([0, 1], [0, 0], color=JOINT_COLOR,   lw=1.8, linestyle="-",
               label="Joint"),
        Line2D([0, 1], [0, 0], color=SINGLE1_COLOR, lw=1.8, linestyle="-.",
               label="Single image 1"),
        Line2D([0, 1], [0, 0], color=SINGLE2_COLOR, lw=1.8, linestyle=":",
               label="Single image 2"),
        Line2D([0], [0], color="0.10", marker="|", linestyle="None",
               markersize=10, markeredgewidth=1.4, label="Injection"),
    ]

    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=4,
        fontsize=10,
        frameon=False,
        handlelength=3.0,       # long lines like in the reference image
        handletextpad=0.6,
        columnspacing=2.0,
        bbox_to_anchor=(0.5, 0.088),
    )

    for ax in axes:
        ax.set_ylim(n_events - 0.5, -0.5)

    if title:
        fig.suptitle(title, y=0.97, fontsize=13)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    print(f"Saved plot to: {output.resolve()}")
    return fig, axes


# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Create a LIGO-style paper summary KDE plot.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--posterior-dir", required=True, metavar="DIR",
        help="Base directory. Searches every subdirectory for --posterior-filename.",
    )
    parser.add_argument(
        "--posterior-filename", default="posterior_samples_joint_thin.h5", metavar="FILENAME",
        help="H5 filename to look for inside each subdirectory. (default: posterior_samples_joint_thin.h5)",
    )
    parser.add_argument(
        "--single1-filename", default="posterior_samples_img1_thin.h5", metavar="FILENAME",
        help="H5 filename for image-1 single posterior. (default: posterior_samples_img1_thin.h5)",
    )
    parser.add_argument(
        "--single2-filename", default="posterior_samples_img2_thin.h5", metavar="FILENAME",
        help="H5 filename for image-2 single posterior. (default: posterior_samples_img2_thin.h5)",
    )
    parser.add_argument(
        "--no-singles", action="store_true",
        help="Disable single-posterior KDE overlays.",
    )
    parser.add_argument(
        "--injections", required=True,
        help="Injection file (.h5/.hdf5/.pkl/.json/.txt).",
    )
    parser.add_argument(
        "--parameters", nargs="+",
        help="Parameters to plot. If omitted, a default set is used.",
    )
    parser.add_argument(
        "--output", default="ligo_violin_plot.png",
        help="Output image path. (default: ligo_violin_plot.png)",
    )
    parser.add_argument("--title", help="Optional plot title.")
    return parser.parse_args()


def main():
    args = parse_args()

    base_dir = Path(args.posterior_dir)
    posterior_paths = discover_posterior_files(base_dir, args.posterior_filename)
    event_labels = [p.parent.name for p in posterior_paths]

    def _index_from_label(label: str) -> int:
        digits = "".join(ch for ch in label if ch.isdigit())
        if digits:
            return int(digits) - 1
        raise ValueError(
            f"Cannot extract a numeric index from folder name '{label}'. "
            "Expected names like 'inj001', 'inj042', etc."
        )

    injection_indices = [_index_from_label(label) for label in event_labels]

    print(f"Found {len(posterior_paths)} event(s):")
    for label, path, idx in zip(event_labels, posterior_paths, injection_indices):
        print(f"  {label} (injection index {idx}): {path}")

    print("\nLoading posterior samples...")
    posterior_arrays = [load_posterior_samples(p) for p in posterior_paths]

    if args.no_singles:
        single1_arrays = [None] * len(posterior_arrays)
        single2_arrays = [None] * len(posterior_arrays)
    else:
        single1_paths = matching_single_paths(posterior_paths, args.single1_filename)
        single2_paths = matching_single_paths(posterior_paths, args.single2_filename)
        single1_arrays = load_optional_posteriors(single1_paths)
        single2_arrays = load_optional_posteriors(single2_paths)

        loaded1 = sum(arr is not None for arr in single1_arrays)
        loaded2 = sum(arr is not None for arr in single2_arrays)
        print(f"Loaded image-1 single posteriors: {loaded1}/{len(posterior_arrays)}")
        print(f"Loaded image-2 single posteriors: {loaded2}/{len(posterior_arrays)}")

    injections = load_injections(Path(args.injections))
    print("All files loaded.")

    parameters = select_parameters(
        posterior=posterior_arrays[0],
        injections=injections,
        requested=args.parameters,
        idx=injection_indices[0],
    )
    print("Parameters:", ", ".join(parameters))

    plot_paper_summary(
        posterior_arrays=posterior_arrays,
        single1_arrays=single1_arrays,
        single2_arrays=single2_arrays,
        event_labels=event_labels,
        injection_indices=injection_indices,
        injections=injections,
        parameters=parameters,
        output=Path(args.output),
        title=args.title,
    )


if __name__ == "__main__":
    main()