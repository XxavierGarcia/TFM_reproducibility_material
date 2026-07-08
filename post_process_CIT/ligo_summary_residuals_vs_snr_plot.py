"""
Plot posterior residuals vs injected network SNR.

Usage
-----
python ligo_summary_residuals_vs_snr_plot.py \
    --posterior-dir   /path/to/results/ \
    --injections-file /path/to/injections.txt \
    --outfig          /path/to/residuals_vs_snr.png

Example:
python ligo_summary_residuals_vs_snr_plot.py --posterior-dir /home/xavier.garcia-sabat/Lensing/pesummaries_merge_all/ --injections-file /home/xavier.garcia-sabat/Lensing/ler_data/n_lensed_params_bbh_filtered.txt --outfig /home/xavier.garcia-sabat/Lensing/pesummaries_merge_all/residuals_vs_snr.png
"""

from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path

import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

mpl.use("Agg")
plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "serif"
plt.rcParams["text.latex.preamble"] = r"\usepackage{amsmath}"
plt.rcParams["savefig.dpi"] = 300

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SNR_TABLE_FILENAME = "snr_all_merge_table_new.txt"

DEFAULT_PARAMETERS = [
    "chirp_mass", "mass_ratio", "chi_eff", "chi_p",
    "a_1", "a_2", "snr",
    "luminosity_distance_1", "luminosity_distance_2",
]

PARAMETER_CONFIG = {
    "chirp_mass": {
        "posterior": ["chirp_mass"],
        "injection": ["chirp_mass"],
        "label":    r"$\mathcal{M}_{\mathrm{post}} - \mathcal{M}_{\mathrm{inj}}\,[M_\odot]$",
        "relative": False,
    },
    "mass_ratio": {
        "posterior": ["mass_ratio", "q"],
        "injection": ["mass_ratio"],
        "label":    r"$q_{\mathrm{post}} - q_{\mathrm{inj}}$",
        "relative": False,
    },
    "chi_eff": {
        "posterior": ["chi_eff", "xeff"],
        "injection": ["chi_eff", "xeff"],
        "label":    r"$\chi_{\mathrm{eff,post}} - \chi_{\mathrm{eff,inj}}$",
        "relative": False,
    },
    "chi_p": {
        "posterior": ["chi_p", "xp"],
        "injection": ["chi_p", "xp"],
        "label":    r"$\chi_{\mathrm{p,post}} - \chi_{\mathrm{p,inj}}$",
        "relative": False,
    },
    "a_1": {
        "posterior": ["a_1", "a1"],
        "injection": ["a_1", "a1"],
        "label":    r"$a_{1,\mathrm{post}} - a_{1,\mathrm{inj}}$",
        "relative": False,
    },
    "a_2": {
        "posterior": ["a_2", "a2"],
        "injection": ["a_2", "a2"],
        "label":    r"$a_{2,\mathrm{post}} - a_{2,\mathrm{inj}}$",
        "relative": False,
    },
    "snr": {
        "injection": ["optimal_snr_net", "snr"],
        "label":    r"$\rho_{\mathrm{post}} - \rho_{\mathrm{inj}}$",
        "relative": False,
        "external_snr": True,
    },
    "luminosity_distance_1": {
        "posterior": ["luminosity_distance^(1)", "luminosity_distance", "d_L", "dl"],
        "injection": ["effective_luminosity_distance", "luminosity_distance", "d_L", "dl"],
        "label":    r"$(D_{L,1,\mathrm{post}} - D_{L,1,\mathrm{inj}}) / D_{L,1,\mathrm{inj}}$",
        "relative": True,
        "image_index": 1,
    },
    "luminosity_distance_2": {
        "posterior": ["luminosity_distance^(2)", "luminosity_distance", "d_L", "dl"],
        "injection": ["effective_luminosity_distance", "luminosity_distance", "d_L", "dl"],
        "label":    r"$(D_{L,2,\mathrm{post}} - D_{L,2,\mathrm{inj}}) / D_{L,2,\mathrm{inj}}$",
        "relative": True,
        "image_index": 2,
    },
}

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


def load_injections(path: Path):
    suffix = path.suffix.lower()
    if suffix in {".h5", ".hdf5"}:
        with h5py.File(path, "r") as f:
            key = list(f.keys())[0]
            obj = f[key]
            if isinstance(obj, h5py.Dataset):
                return obj[()]
            if "injections" in obj:
                return obj["injections"][()]
            return {k: obj[k][()] for k in obj.keys()}
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


def load_snr_table(path: Path) -> dict[int, dict[str, float]]:
    rows = np.genfromtxt(path, names=True, dtype=None, encoding="utf-8")
    if rows.shape == ():
        rows = np.array([rows], dtype=rows.dtype)

    table = {}
    for row in rows:
        inj = int(row["inj"])
        table[inj] = {
            "optimal_joint_p5": float(row["optimal_joint_p5"]),
            "optimal_joint_p50": float(row["optimal_joint_p50"]),
            "optimal_joint_p95": float(row["optimal_joint_p95"]),
        }
    return table

# ---------------------------------------------------------------------------
# Injection value resolution
# ---------------------------------------------------------------------------

def get_posterior_field(array: np.ndarray, candidates: list[str]) -> np.ndarray:
    for name in candidates:
        if name in array.dtype.names:
            values = np.asarray(array[name], dtype=float).reshape(-1)
            values = values[np.isfinite(values)]
            if len(values):
                return values
    raise KeyError(f"None of these fields found: {', '.join(candidates)}")


def get_injection_value(injections, candidates, idx, image_index):
    ii = image_index - 1
    if isinstance(injections, np.ndarray) and injections.dtype.names:
        for name in candidates:
            if name not in injections.dtype.names:
                continue
            values = np.asarray(injections[name])
            row = values if injections.shape == () else values[idx]
            flat = np.asarray(row).reshape(-1)
            return float(flat[ii if flat.size > 1 else 0])
    if isinstance(injections, dict):
        for name in candidates:
            if name not in injections:
                continue
            v = np.asarray(injections[name])
            if v.ndim == 0:
                return float(v)
            if v.ndim == 1:
                return float(v[idx])
            return float(v[idx, ii])
    raise KeyError(f"Could not resolve: {', '.join(candidates)}")


def _raw(injections, field, idx):
    return get_injection_value(injections, [field], idx, 1)


def get_derived_injection(parameter, injections, idx):
    m1, m2 = _raw(injections, "mass_1", idx), _raw(injections, "mass_2", idx)
    if parameter == "chirp_mass":
        return (m1 * m2) ** (3 / 5) / (m1 + m2) ** (1 / 5)
    if parameter == "mass_ratio":
        return min(m1, m2) / max(m1, m2)

    a1 = _raw(injections, "a_1", idx)
    a2 = _raw(injections, "a_2", idx)
    if parameter == "a_1":
        return a1
    if parameter == "a_2":
        return a2

    t1 = _raw(injections, "tilt_1", idx)
    t2 = _raw(injections, "tilt_2", idx)
    if parameter == "chi_eff":
        return (m1 * a1 * np.cos(t1) + m2 * a2 * np.cos(t2)) / (m1 + m2)
    if parameter == "chi_p":
        q = m2 / m1
        return max(a1 * np.sin(t1), q * (4 * q + 3) / (4 + 3 * q) * a2 * np.sin(t2))

    raise KeyError(f"No derived rule for '{parameter}'")


def infer_injection_index(event_name: str) -> int:
    match = re.search(r"inj(\d+)", event_name, flags=re.IGNORECASE)
    if not match:
        raise ValueError(f"Cannot infer injection index from '{event_name}'.")
    return int(match.group(1)) - 1

# ---------------------------------------------------------------------------
# Per-event computation
# ---------------------------------------------------------------------------

def summarize_event(posterior, injections, snr_table, parameter, event_name, credible_level=90.0):
    cfg = PARAMETER_CONFIG[parameter]
    idx = infer_injection_index(event_name)
    ii = cfg.get("image_index", 1)

    try:
        inj_value = get_injection_value(injections, cfg["injection"], idx, ii)
    except KeyError:
        inj_value = get_derived_injection(parameter, injections, idx)

    snr = get_injection_value(injections, ["optimal_snr_net"], idx, 1)

    scale = inj_value if cfg["relative"] else 1.0
    if cfg["relative"] and scale == 0:
        raise ZeroDivisionError(f"Injection value is zero for '{parameter}' in '{event_name}'.")

    if cfg.get("external_snr", False):
        inj_number = idx + 1
        row = snr_table[inj_number]
        p_low = row["optimal_joint_p5"]
        p50 = row["optimal_joint_p50"]
        p_high = row["optimal_joint_p95"]
    else:
        post_values = get_posterior_field(posterior, cfg["posterior"])
        tail = 0.5 * (100.0 - credible_level)
        p_low, p50, p_high = np.percentile(post_values, [tail, 50.0, 100.0 - tail])

    return {
        "inj": inj_value,
        "snr": snr,
        "y16": (p_low - inj_value) / scale,
        "y50": (p50 - inj_value) / scale,
        "y84": (p_high - inj_value) / scale,
    }


def collect_results(events, injections, snr_table, parameters):
    collected = {p: [] for p in parameters}
    for event_name, posterior in sorted(events.items()):
        for parameter in parameters:
            try:
                row = summarize_event(posterior, injections, snr_table, parameter, event_name)
                row["event"] = event_name
                collected[parameter].append(row)
            except Exception as exc:
                print(f"[skip] {event_name} / {parameter}: {exc}")

    output = {}
    for parameter, rows in collected.items():
        if not rows:
            continue
        rows = sorted(rows, key=lambda r: r["snr"])
        output[parameter] = {k: np.array([r[k] for r in rows]) for k in rows[0]}
    return output

# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def add_summary_text(ax, values):
    text = "\n".join([
        rf"$N={len(values)}$",
        rf"$\mathrm{{median}}={np.median(values):.3g}$",
        rf"$\sigma={np.std(values):.3g}$",
    ])
    ax.text(
        0.02, 0.97, text,
        transform=ax.transAxes, va="top", ha="left", fontsize=8,
        bbox={"facecolor": "white", "edgecolor": "0.8", "alpha": 0.9, "pad": 2.5},
    )


def plot_residuals_vs_snr(results, parameters, outfig):
    ncols = 2
    nrows = int(np.ceil(len(parameters) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(10.5, 3.6 * nrows))
    axes = np.atleast_1d(axes).ravel()

    for ax in axes[len(parameters):]:
        ax.remove()

    for ax, parameter in zip(axes, parameters):
        if parameter not in results:
            ax.set_visible(False)
            continue

        data = results[parameter]
        x = data["snr"].astype(float)
        y = data["y50"].astype(float)
        yerr = np.vstack([
            y - data["y16"].astype(float),
            data["y84"].astype(float) - y,
        ])

        ax.axhline(0.0, color="0.35", lw=1.0, ls="--", zorder=0)
        ax.errorbar(
            x, y, yerr=yerr, fmt="o", color="#2F5DA8", ecolor="#2F5DA8",
            elinewidth=0.8, alpha=0.80, ms=3.5, capsize=0, zorder=2,
        )
        ax.set_xlabel(r"Injected network SNR")
        ax.set_ylabel(PARAMETER_CONFIG[parameter]["label"])
        ax.grid(alpha=0.22, lw=0.5)
        add_summary_text(ax, y)

    fig.tight_layout()
    fig.savefig(outfig, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure to {outfig}")

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Plot posterior residuals vs injected network SNR.")
    parser.add_argument(
        "--posterior-dir", required=True, metavar="DIR",
        help="Base directory. Searches every subdirectory for --posterior-filename.",
    )
    parser.add_argument(
        "--posterior-filename", default="posterior_samples_joint_thin.h5",
        help="H5 filename inside each subdirectory. (default: posterior_samples_joint_thin.h5)",
    )
    parser.add_argument(
        "--injections-file", required=True,
        help="Injection file (.h5/.json/.txt).",
    )
    parser.add_argument(
        "--parameters", nargs="+", default=DEFAULT_PARAMETERS,
        choices=list(PARAMETER_CONFIG), help="Parameters to plot.",
    )
    parser.add_argument(
        "--outfig", default="ligo_summary_residuals_vs_snr_plot.png",
        help="Output figure path.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    posterior_paths = discover_posterior_files(Path(args.posterior_dir), args.posterior_filename)
    event_labels = [p.parent.name for p in posterior_paths]

    print(f"Found {len(posterior_paths)} event(s).")
    events = {}
    for label, path in zip(event_labels, posterior_paths):
        try:
            events[label] = load_posterior_samples(path)
        except Exception as exc:
            print(f"[skip] {label}: could not load posterior ({exc})")
    print(f"Loaded {len(events)} event(s) successfully.")

    injections = load_injections(Path(args.injections_file))

    snr_table = {}
    if "snr" in args.parameters:
        snr_path = Path(args.posterior_dir) / SNR_TABLE_FILENAME
        if not snr_path.exists():
            raise FileNotFoundError(
                f"Parameter 'snr' requested, but SNR table was not found at '{snr_path}'."
            )
        snr_table = load_snr_table(snr_path)

    results = collect_results(events, injections, snr_table, args.parameters)

    if not results:
        raise ValueError("No valid parameter/event combinations found.")

    plot_residuals_vs_snr(results, args.parameters, Path(args.outfig))


if __name__ == "__main__":
    main()