#python ligo_sky_area_vs_snr_plot.py --sky-area sky_area_90_table.csv --snr snr_all_merge_table.txt --outfig sky_area_vs_snr_hl.png

import argparse

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

mpl.use("Agg")
plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "serif"
plt.rcParams["text.latex.preamble"] = r"\usepackage{amsmath}"
plt.rcParams["savefig.dpi"] = 300


def inj_id(x):
    x = str(x).strip()
    digits = "".join(c for c in x if c.isdigit())
    return f"inj{int(digits):03d}" if digits else x.lower()


parser = argparse.ArgumentParser()
parser.add_argument("--sky-area", required=True)
parser.add_argument("--snr", required=True)
parser.add_argument("--outfig", default="ligo_sky_area_vs_snr_plot.png")
parser.add_argument("--title", default=None)
parser.add_argument("--single-snr-kind", choices=["optimal", "matched"], default="optimal")
parser.add_argument("--joint-snr-kind", choices=["optimal", "matched"], default="optimal")
args = parser.parse_args()

sky = {}
df = pd.read_csv(args.sky_area, sep=None, engine="python")
df.columns = df.columns.str.strip()
if "Unnamed: 0" in df.columns:
    df = df.drop(columns="Unnamed: 0")

for _, row in df.iterrows():
    if pd.isna(row["inj"]) or pd.isna(row["img1_area90"]) or pd.isna(row["img2_area90"]) or pd.isna(row["joint_area90"]):
        continue
    sky[inj_id(row["inj"])] = {
        "img1": float(row["img1_area90"]),
        "img2": float(row["img2_area90"]),
        "joint": float(row["joint_area90"]),
    }

snr = {}
header = None
with open(args.snr, encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        if line.startswith("#"):
            cols = line[1:].split()
            if "inj" in cols:
                header = cols
            continue
        if header is None:
            header = line.split()
            continue
        row = dict(zip(header, line.split()))
        snr[inj_id(row["inj"])] = {k: float(v) for k, v in row.items() if k != "inj"}

inj = sorted(set(sky) & set(snr))
if not inj:
    raise ValueError("No common injections found between sky-area and SNR files.")

fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2), constrained_layout=True, sharey=True)
for ax, i in zip(axes, [1, 2]):
    single = np.array([snr[k][f"{args.single_snr_kind}_img{i}"] for k in inj])
    joint = np.array([snr[k][f"{args.joint_snr_kind}_joint"] for k in inj])
    area_single = np.array([sky[k][f"img{i}"] for k in inj])
    area_joint = np.array([sky[k]["joint"] for k in inj])

    ax.scatter(single, area_single, marker="+", s=55, linewidths=1.3, color="#1f77b4", label="HL single")
    ax.scatter(joint, area_joint, marker="x", s=55, linewidths=1.3, color="#ff7f0e", label="HL joint")
    ax.set_xlabel(f"network SNR (img{i})")
    #ax.set_xlim(0,50)
    #ax.set_ylim(0,1000)
    ax.set_ylabel(r"sky area 90\% [$\mathrm{deg}^2$]")
    #ax.set_title(fr"$\mathrm{{img}}_{{{i}}}$")
    ax.grid(alpha=0.25, linewidth=0.6)

axes[1].legend(loc="upper right")
if args.title:
    fig.suptitle(args.title)
fig.savefig(args.outfig, bbox_inches="tight")
plt.close(fig)
