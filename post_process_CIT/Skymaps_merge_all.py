#code to create skymaps and save them all in the same file
#run it with secondary screen, it's not necessary to do it with condor

import os
import subprocess
import glob

# Base directory
base_dir = "/home/xavier.garcia-sabat/Lensing/pesummaries_merge_all"

# Output directory for all plots
output_dir = os.path.join(base_dir, "all_skymap_plots")
os.makedirs(output_dir, exist_ok=True)

# Find all skymap.fits files
fits_files = glob.glob(os.path.join(base_dir, "inj*/skyloc_*/skymap/skymap.fits"))

fits_files.sort()

print(f"Found {len(fits_files)} skymaps\n")

for fits in fits_files:

    parts = fits.split("/")
    inj = parts[-4]          # injXXX
    skyloc = parts[-3]       # skyloc_img1 / skyloc_img2 / skyloc_joint

    name = f"{inj}_{skyloc}.png"
    output = os.path.join(output_dir, name)

    print(f"Plotting {fits}")

    cmd = [
        "ligo-skymap-plot",
        fits,
        "--contour", "50", "90",
        "--annotate",
        "--output", output
    ]

    subprocess.run(cmd)

print("\nAll skymaps plotted in:")
print(output_dir)