import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "serif"

# Load data 
df = pd.read_csv("sky_area_90_table.csv")

# Drop rows with missing values in relevant columns 
df = df.dropna(subset=["img1_area90", "img2_area90", "joint_area90"])

# Compute the two series 
best_image = np.minimum(df["img1_area90"].values, df["img2_area90"].values)
joint    = df["joint_area90"].values

n1 = len(best_image)
n2 = len(joint)

# CDF function 
def compute_cdf(values):
    sorted_vals = np.sort(values)
    cdf = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)
    return sorted_vals, cdf

x1, y1 = compute_cdf(best_image)
x2, y2 = compute_cdf(joint)

fig, ax = plt.subplots(figsize=(7, 5))

ax.step(x1, y1, where="post", color="#1f77b4", linewidth=1.5,
        label=f"Best image (n={n1})")
ax.step(x2, y2, where="post", color="#ff7f0e", linewidth=1.5,
        label=f"Joint area (n={n2})")
ax.axvline(x=10, color="black", linestyle="--", linewidth=1.2,
           label=r"$10\,\mathrm{deg}^2$ threshold")

ax.set_xscale("log")
ax.set_yscale("log")

ax.set_xlabel(r"90\% area (deg$^2$)", fontsize=12)
ax.set_ylabel("CDF", fontsize=12)
#ax.set_title(r"CDF of 90\% area: 2-image systems", fontsize=12)

#ax.set_xlim(1, 2e4)
#ax.set_ylim(3e-4, 1.5)

ax.legend(fontsize=10, loc="lower right")
ax.grid(True, which="both", linestyle="--", linewidth=0.4, alpha=0.6)

plt.tight_layout()
plt.savefig("cdf_90_area.png", dpi=300)  
plt.show()