#python ligo_top_constellation_vs_snr_plot.py --constellation dominant_constellations_table.csv --snr snr_all_merge_table.txt --outfig top_constellation_vs_snr_hl.png


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
parser.add_argument("--constellation", required=True)
parser.add_argument("--snr", required=True)
parser.add_argument("--outfig", default="ligo_top_constellation_vs_snr_plot.png")
parser.add_argument("--title", default=None)
parser.add_argument("--single-snr-kind", choices=["optimal", "matched"], default="optimal")
parser.add_argument("--joint-snr-kind", choices=["optimal", "matched"], default="optimal")
args = parser.parse_args()

df = pd.read_csv(args.constellation, sep=None, engine="python")
df.columns = df.columns.str.strip()
if "Unnamed: 0" in df.columns:
    df = df.drop(columns="Unnamed: 0")

const = {}
for _, row in df.iterrows():
    if pd.isna(row["inj"]) or pd.isna(row["img1_percentage"]) or pd.isna(row["img2_percentage"]) or pd.isna(row["joint_percentage"]):
        continue
    const[inj_id(row["inj"])] = {
        "img1": float(row["img1_percentage"]),
        "img2": float(row["img2_percentage"]),
        "joint": float(row["joint_percentage"]),
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

inj = sorted(set(const) & set(snr))
if not inj:
    raise ValueError("No common injections found between constellation and SNR files.")

fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2), constrained_layout=True, sharey=True)
for ax, i in zip(axes, [1, 2]):
    single = np.array([snr[k][f"{args.single_snr_kind}_img{i}"] for k in inj])
    joint = np.array([snr[k][f"{args.joint_snr_kind}_joint"] for k in inj])
    perc_single = np.array([const[k][f"img{i}"] for k in inj])
    perc_joint = np.array([const[k]["joint"] for k in inj])

    ax.scatter(single, perc_single, marker="+", s=55, linewidths=1.3, color="#1f77b4", label="HL single")
    ax.scatter(joint, perc_joint, marker="x", s=55, linewidths=1.3, color="#ff7f0e", label="HL joint")
    ax.set_xlabel(f"network SNR (img{i})")
    ax.set_ylabel(r"top constellation [\%]")
    #ax.set_title(fr"$\mathrm{{img}}_{{{i}}}$")
    #ax.set_ylim(0, 100)
    ax.set_xlim(0, 62)
    ax.grid(alpha=0.25, linewidth=0.6)

axes[1].legend(loc="lower right")
if args.title:
    fig.suptitle(args.title)
fig.savefig(args.outfig, bbox_inches="tight")
plt.close(fig)
