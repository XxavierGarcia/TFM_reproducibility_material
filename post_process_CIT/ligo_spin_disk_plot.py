"""
Generate LIGO-style spin-disk plots from posterior samples.

Usage
-----
python ligo_spin_disk_plot.py \
    --posterior-dir /path/to/results/ \
    --output-dir    /path/to/output/ \
    --colorbar \
    --events-per-page 30

Example: 
python ligo_spin_disk_plot.py --posterior-dir /home/xavier.garcia-sabat/Lensing/pesummaries_merge_all/ --output-dir pesummaries_merge_all --colorbar --events-per-page 9
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PatchCollection
from matplotlib.colors import LinearSegmentedColormap, hex2color, hsv_to_rgb, rgb_to_hsv
from matplotlib.patches import Wedge
from matplotlib.projections import PolarAxes
from matplotlib.transforms import Affine2D, ScaledTranslation
from mpl_toolkits.axisartist.grid_finder import MaxNLocator
import mpl_toolkits.axisartist.angle_helper as angle_helper
import mpl_toolkits.axisartist.floating_axes as floating_axes
from scipy.stats import gaussian_kde

mpl.use("Agg")

plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "serif"
plt.rcParams["text.latex.preamble"] = r"\usepackage{amsmath}"

DEFAULT_EVENT_COLORS = [
    "#8C7A7A", "#5A58A7", "#8BC7E8", "#7ED957", "#7ED9B1",
    "#C48BEA", "#C8B76A", "#D8B8EC", "#E9A441", "#6FA8DC",
]

# I/O
# ---------------------------------------------------------------------------

def discover_posterior_files(base_dir: Path, filename: str) -> list[Path]:
    found = sorted(base_dir.glob(f"*/{filename}"))
    if not found:
        raise FileNotFoundError(
            f"No files named '{filename}' found in any subdirectory of '{base_dir}'."
        )
    return found


def load_posterior_samples(path: Path) -> np.ndarray:
    with h5py.File(path, "r") as f:
        group = list(f.keys())[0]
        return f[group]["posterior_samples"][()]

# KDE
# ---------------------------------------------------------------------------

class Bounded2DKDE:
    def __init__(self, samples, *, xlow=None, xhigh=None, ylow=None, yhigh=None):
        data = np.asarray(samples, dtype=float)
        reflected = [data]
        if xlow  is not None: reflected.append(np.column_stack([2*xlow  - data[:,0], data[:,1]]))
        if xhigh is not None: reflected.append(np.column_stack([2*xhigh - data[:,0], data[:,1]]))
        if ylow  is not None: reflected.append(np.column_stack([data[:,0], 2*ylow  - data[:,1]]))
        if yhigh is not None: reflected.append(np.column_stack([data[:,0], 2*yhigh - data[:,1]]))
        if xlow  is not None and ylow  is not None: reflected.append(np.column_stack([2*xlow  - data[:,0], 2*ylow  - data[:,1]]))
        if xlow  is not None and yhigh is not None: reflected.append(np.column_stack([2*xlow  - data[:,0], 2*yhigh - data[:,1]]))
        if xhigh is not None and ylow  is not None: reflected.append(np.column_stack([2*xhigh - data[:,0], 2*ylow  - data[:,1]]))
        if xhigh is not None and yhigh is not None: reflected.append(np.column_stack([2*xhigh - data[:,0], 2*yhigh - data[:,1]]))
        self._kde = gaussian_kde(np.vstack(reflected).T)
        self.xlow, self.xhigh, self.ylow, self.yhigh = xlow, xhigh, ylow, yhigh

    def __call__(self, points):
        pts    = np.asarray(points, dtype=float)
        values = self._kde(pts.T)
        mask   = np.ones(len(pts), dtype=bool)
        if self.xlow  is not None: mask &= pts[:,0] >= self.xlow
        if self.xhigh is not None: mask &= pts[:,0] <= self.xhigh
        if self.ylow  is not None: mask &= pts[:,1] >= self.ylow
        if self.yhigh is not None: mask &= pts[:,1] <= self.yhigh
        values[~mask] = 0.0
        return values

# Spin helpers
# ---------------------------------------------------------------------------

def extract_spin_samples(samples):
    data = {n: np.asarray(samples[n], dtype=float).reshape(-1) for n in samples.dtype.names}

    def get(*names):
        for n in names:
            if n in data:
                return data[n]
        raise KeyError(f"None of these fields found: {', '.join(names)}")

    a1  = get("a_1",  "a1")
    a2  = get("a_2",  "a2")
    ct1 = get("cos_tilt_1_preferred", "cos_tilt_1", "costilt1")
    ct2 = get("cos_tilt_2_preferred", "cos_tilt_2", "costilt2")

    n    = min(len(a1), len(a2), len(ct1), len(ct2))
    pos1 = np.column_stack([a1[:n], ct1[:n]])
    pos2 = np.column_stack([a2[:n], ct2[:n]])
    mask = np.isfinite(pos1).all(axis=1) & np.isfinite(pos2).all(axis=1)
    return pos1[mask], pos2[mask]


def build_spin_pdf(pos, na=25, nt=25):
    kde   = Bounded2DKDE(pos, xlow=0.0, xhigh=0.99, ylow=-1.0, yhigh=1.0)
    rs    = np.linspace(0, 0.99, na)
    costs = np.linspace(-1, 1,   nt)
    dr    = float(abs(rs[1]    - rs[0]))
    dcost = float(abs(costs[1] - costs[0]))
    cg, rg = np.meshgrid(costs[:-1], rs[:-1])
    pdf = kde(np.column_stack([rg.ravel() + dr/2, cg.ravel() + dcost/2]))
    return rs, costs, rg, pdf.reshape(rg.shape)


def make_colormap(name, color, n=10):
    base_rgb = plt.get_cmap("Blues")(np.linspace(0.15, 0.95, n))[:, :3]
    base_hsv = np.array([rgb_to_hsv(rgb) for rgb in base_rgb])
    target   = rgb_to_hsv(hex2color(color))
    colors   = [hsv_to_rgb([target[0], np.clip(s * max(target[1], 0.25), 0, 1), v])
                for s, v in zip(base_hsv[:,1], base_hsv[:,2])]
    return LinearSegmentedColormap.from_list(name, colors)


def add_semidisk_axis(fig, rect, invert_x=False):
    transform = (Affine2D().translate(90, 0)
                 + Affine2D().scale(np.pi / 180.0, 1.0)
                 + PolarAxes.PolarTransform())
    grid_helper = floating_axes.GridHelperCurveLinear(
        transform,
        extremes=(0, 180, 0, 0.99),
        grid_locator1=angle_helper.LocatorD(7),
        grid_locator2=MaxNLocator(5),
        tick_formatter1=angle_helper.FormatterDMS(),
        tick_formatter2=None,
    )
    axis = floating_axes.FloatingSubplot(fig, rect, grid_helper=grid_helper)
    if invert_x:
        axis.invert_xaxis()
    fig.add_subplot(axis)
    return axis


def fill_semidisk(axis, rs, costs, pdf, cmap, vmax):
    dr    = float(abs(rs[1]    - rs[0]))
    dcost = float(abs(costs[1] - costs[0]))
    x = np.arccos(np.meshgrid(costs[:-1], rs[:-1])[0]) * 180 / np.pi + 90.0
    y = np.meshgrid(costs[:-1], rs[:-1])[1]
    patches, colors = [], []
    for angle, radius, value in zip(x.ravel(), y.ravel(), pdf.ravel()):
        next_angle = np.arccos(np.cos((angle - 90.0) * np.pi / 180.0) + dcost) * 180.0 / np.pi + 90.0
        patches.append(Wedge((0.0, 0.0), radius + dr, next_angle, angle, width=dr))
        colors.append(value)
    coll = PatchCollection(patches, cmap=cmap, edgecolors="face", zorder=10)
    coll.set_clim(0.0, vmax)
    coll.set_array(np.asarray(colors))
    axis.add_collection(coll)
    return coll


def render_panel(fig, left_spec, right_spec, *, samples, label, color, colorbar):
    pos1, pos2 = extract_spin_samples(samples)
    rs1, costs1, _, pdf1 = build_spin_pdf(pos1)
    rs2, costs2, _, pdf2 = build_spin_pdf(pos2)

    maxp  = float(max(np.max(pdf1), np.max(pdf2)))
    scale = np.exp(1.0)
    h1    = np.log(1.0 + scale * pdf1)
    h2    = np.log(1.0 + scale * pdf2)
    vmax  = np.log(1.0 + scale * maxp)
    cmap  = make_colormap(label, color)

    lax = add_semidisk_axis(fig, left_spec, invert_x=False)
    lax.axis["bottom"].toggle(all=False)
    lax.axis["top"].toggle(all=True)
    lax.axis["top"].major_ticks.set_tick_out(True)
    lax.axis["top"].set_axis_direction("top")
    lax.axis["top"].set_ticklabel_direction("+")
    lax.axis["left"].major_ticks.set_tick_out(True)
    lax.axis["left"].set_axis_direction("right")
    offset = ScaledTranslation(2.0 / 72.0, 0.0, fig.dpi_scale_trans)
    lax.axis["left"].major_ticklabels.set(figure=fig, transform=offset)
    coll = fill_semidisk(lax, rs1, costs1, h1, cmap, vmax)

    rax = add_semidisk_axis(fig, right_spec, invert_x=True)
    rax.axis["bottom"].toggle(all=False)
    rax.axis["top"].toggle(all=True)
    rax.axis["top"].set_axis_direction("top")
    rax.axis["top"].major_ticks.set_tick_out(True)
    rax.axis["left"].major_ticks.set_tick_out(True)
    rax.axis["left"].toggle(ticklabels=False)
    rax.axis["left"].major_ticklabels.set_visible(False)
    rax.axis["right"].major_ticks.set_tick_out(True)
    fill_semidisk(rax, rs2, costs2, h2, cmap, vmax)

    lb  = lax.get_position()
    rb  = rax.get_position()
    x_c = (lb.x0 + rb.x1) / 2.0

    title  = fig.text(x_c, max(lb.y1, rb.y1) + 0.02,
                      label, fontsize=14, ha="center", va="bottom")
    y_bot  = min(lb.y0, rb.y0) - (0.03 if colorbar else 0.03)
    s1_lbl = fig.text((lb.x0 + lb.x1) / 2.0, y_bot,
                      r"$c\vec{S}_{1}/(Gm_1^2)$", fontsize=11, ha="center")
    s2_lbl = fig.text((rb.x0 + rb.x1) / 2.0, y_bot,
                      r"$c\vec{S}_{2}/(Gm_2^2)$", fontsize=11, ha="center")

    if colorbar:
        bw  = (rb.x1 - lb.x0) * 0.62
        cax = fig.add_axes([x_c - bw / 2.0, y_bot - 0.03, bw, 0.012])
        fig.colorbar(coll, cax=cax, orientation="horizontal")

    return [title, s1_lbl, s2_lbl]


def plot_spindisk_grid(loaded, output_dir, *, colorbar, ncols=3, events_per_page=30):
    ncols  = max(1, ncols)
    pages  = [loaded[i:i + events_per_page] for i in range(0, len(loaded), events_per_page)]
    print(f"Splitting {len(loaded)} events into {len(pages)} page(s) ({events_per_page} per page).")

    output_dir.mkdir(parents=True, exist_ok=True)
    for page_idx, page in enumerate(pages, start=1):
        n_events   = len(page)
        page_ncols = max(1, min(ncols, n_events))
        nrows      = int(np.ceil(n_events / page_ncols))

        fig = plt.figure(figsize=(4.4 * page_ncols, 4.35 * nrows))
        outer = fig.add_gridspec(
            nrows, page_ncols,
            wspace=0.04, hspace=0.58,
            left=0.1, right=0.9, top=0.93, bottom=0.12,
        )
        all_artists = []
        for idx, (label, samples) in enumerate(page):
            global_idx = (page_idx - 1) * events_per_page + idx
            row, col   = divmod(idx, page_ncols)
            inner      = outer[row, col].subgridspec(1, 2, wspace=-0.01)
            color      = DEFAULT_EVENT_COLORS[global_idx % len(DEFAULT_EVENT_COLORS)]
            artists    = render_panel(fig, inner[0, 0], inner[0, 1],
                                      samples=samples, label=label,
                                      color=color, colorbar=colorbar)
            all_artists.extend(artists)

        out = output_dir / f"spin_disk_page{page_idx:03d}.png"
        fig.savefig(out, bbox_extra_artists=all_artists, dpi=300)
        plt.close(fig)
        print(f"Saved: {out.resolve()}")


def parse_args():
    parser = argparse.ArgumentParser(description="Create LIGO-style spin-disk plots.")
    parser.add_argument("--posterior-dir",      required=True, metavar="DIR",
                        help="Base directory. Searches every subdirectory for --posterior-filename.")
    parser.add_argument("--posterior-filename", default="posterior_samples_joint_thin.h5", metavar="FILENAME",
                        help="H5 filename inside each subdirectory. (default: posterior_samples_joint_thin.h5)")
    parser.add_argument("--output-dir",         default="spin_disk_plots",
                        help="Output directory. (default: spin_disk_plots)")
    parser.add_argument("--colorbar",           action="store_true",
                        help="Include a horizontal colorbar.")
    parser.add_argument("--events-per-page",    type=int, default=30,
                        help="Max events per output PNG. (default: 30)")
    return parser.parse_args()


def main():
    args = parse_args()

    base_dir        = Path(args.posterior_dir)
    posterior_paths = discover_posterior_files(base_dir, args.posterior_filename)
    event_labels    = [p.parent.name for p in posterior_paths]

    print(f"Found {len(posterior_paths)} event(s).")
    print("Loading posterior samples...")
    loaded = [(label, load_posterior_samples(path))
              for label, path in zip(event_labels, posterior_paths)]
    print("All files loaded.")

    plot_spindisk_grid(
        loaded,
        output_dir=Path(args.output_dir),
        colorbar=args.colorbar,
        events_per_page=args.events_per_page,
    )


if __name__ == "__main__":
    main()