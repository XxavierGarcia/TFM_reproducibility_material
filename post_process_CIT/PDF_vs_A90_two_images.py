import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "serif"

# Load data 
df = pd.read_csv("sky_area_90_table.csv")

# Drop rows with missing values 
df = df.dropna(subset=["img1_area90", "img2_area90", "joint_area90"])

# Compute the two series 
best = np.minimum(df["img1_area90"].values, df["img2_area90"].values)
joint = df["joint_area90"].values

n1 = len(best)
n2 = len(joint)

# Log-spaced bins (same for both)
bins = np.logspace(np.log10(1), np.log10(2e4), 50)

# Compute probability density histograms
def log_density(values, bins):
    counts, edges = np.histogram(values, bins=bins)
    widths = np.diff(edges)
    density = counts / (values.size * widths)
    return edges, density

e1, d1 = log_density(best, bins)
e2, d2 = log_density(joint, bins)

fig, ax = plt.subplots(figsize=(7, 5))

ax.stairs(d1, e1, color="#1f77b4", linewidth=1.5,
          label=f"Best image (n={n1})")
ax.stairs(d2, e2, color="#ff7f0e", linewidth=1.5,
          label=f"Joint area (n={n2})")
ax.axvline(x=10, color="black", linestyle="--", linewidth=1.2,
           label=r"$10\,\mathrm{deg}^2$ threshold")

ax.set_xscale("log")
ax.set_yscale("log")

ax.set_xlabel(r"90\% area (deg$^2$)", fontsize=12)
ax.set_ylabel("Probability density", fontsize=12)
#ax.set_title("2-image systems", fontsize=12)

ax.set_xlim(1, 2e4)

ax.legend(fontsize=10, loc="upper right")
ax.grid(True, which="both", linestyle="--", linewidth=0.4, alpha=0.6)

plt.tight_layout()
plt.savefig("pdf_90_area.png", dpi=300)
plt.show()