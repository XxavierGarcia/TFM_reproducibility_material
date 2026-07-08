import glob
import os
import numpy as np

BASE = "/mnt/home/users/uib54_res/resh000464/lenskyloc/lenskyloc_runs"

files = sorted(glob.glob(f"{BASE}/inj*/snr_inj*.txt"))

catalog_path = f"{BASE}/snr_all_merge_table.txt"

rows = []

for f in files:
    inj_name = os.path.basename(f).split(".")[0] # snr_injxxx_merge without txt
    inj_number = inj_name.split("_")[1].replace("inj", "") #001/002...

    # Initialize values
    optimal_img1 = None
    optimal_img2 = None
    optimal_joint = None

    matched_img1 = None
    matched_img2 = None
    matched_joint = None

    with open(f, "r") as file:
        lines = file.readlines()

    current_block = None
    current_image = None

    for line in lines:

        line = line.strip()

        if "optimal_snr" in line:
            current_block = "optimal"

        elif "matched_filter_snr" in line:
            current_block = "matched"

        elif line.startswith("Image 1"):
            current_image = "img1"

        elif line.startswith("Image 2"):
            current_image = "img2"

        elif "Combined network SNR (image 1)" in line:
            value = float(line.split("=")[1])
            if current_block == "optimal":
                optimal_img1 = value
            elif current_block == "matched":
                matched_img1 = value

        elif "Combined network SNR (image 2)" in line:
            value = float(line.split("=")[1])
            if current_block == "optimal":
                optimal_img2 = value
            elif current_block == "matched":
                matched_img2 = value

        elif "Total combined SNR (all images)" in line:
            value = float(line.split("=")[1])
            if current_block == "optimal":
                optimal_joint = value
            elif current_block == "matched":
                matched_joint = value

    rows.append([
        int(inj_number),
        optimal_img1,
        optimal_img2,
        optimal_joint,
        matched_img1,
        matched_img2,
        matched_joint
    ])

rows = np.array(rows)

np.savetxt(
    catalog_path,
    rows,
    header="inj optimal_img1 optimal_img2 optimal_joint matched_img1 matched_img2 matched_joint",
    fmt=["%d"] + ["%.6f"]*6
)

print("Table created:", catalog_path)