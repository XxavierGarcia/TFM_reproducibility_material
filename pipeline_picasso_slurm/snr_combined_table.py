import glob
import os
import numpy as np

BASE = "/mnt/home/users/uib54_res/resh000464/lenskyloc/lenskyloc_runs"

files = sorted(glob.glob(f"{BASE}/inj*/snr_inj*.txt"))

catalog_path = f"{BASE}/snr_all_merge_table.txt"

rows = []

for f in files:
    inj_name = os.path.basename(f).split(".")[0]  # snr_injxxx_merge without txt
    inj_number = inj_name.split("_")[1].replace("inj", "")  # 001/002...

    # Initialize values
    optimal_img1 = None
    optimal_img2 = None
    optimal_joint = None
    optimal_joint_p5 = None
    optimal_joint_p50 = None
    optimal_joint_p95 = None

    matched_img1 = None
    matched_img2 = None
    matched_joint = None
    matched_joint_p5 = None
    matched_joint_p50 = None
    matched_joint_p95 = None

    with open(f, "r") as file:
        lines = file.readlines()
    current_block = None

    for line in lines:
        line = line.strip()

        if not line:
            continue

        if "optimal_snr" in line:
            current_block = "optimal"

        elif "matched_filter_snr" in line:
            current_block = "matched"

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
        
        elif "=" in line:
            key, value = line.split("=", 1)
            key = key.strip()
            value = float(value.strip())

            if key == "optimal_joint_p5":
                optimal_joint_p5 = value
            elif key == "optimal_joint_p50":
                optimal_joint_p50 = value
            elif key == "optimal_joint_p95":
                optimal_joint_p95 = value
            elif key == "matched_joint_p5":
                matched_joint_p5 = value
            elif key == "matched_joint_p50":
                matched_joint_p50 = value
            elif key == "matched_joint_p95":
                matched_joint_p95 = value
        
    rows.append([
        int(inj_number),
        optimal_img1,
        optimal_img2,
        optimal_joint,
        matched_img1,
        matched_img2,
        matched_joint,
        optimal_joint_p5,
        optimal_joint_p50,
        optimal_joint_p95,
        matched_joint_p5,
        matched_joint_p50,
        matched_joint_p95,
    ])

rows = np.array(rows, dtype=float)
rows = rows[np.argsort(rows[:, 0])]

np.savetxt(
    catalog_path,
    rows,
    header=(
        "inj "
        "optimal_img1 optimal_img2 optimal_joint "
        "matched_img1 matched_img2 matched_joint "
        "optimal_joint_p5 optimal_joint_p50 optimal_joint_p95 "
        "matched_joint_p5 matched_joint_p50 matched_joint_p95"
    ),
    fmt=["%d"] + ["%.6f"] * 12
)

print("Table created:", catalog_path)
