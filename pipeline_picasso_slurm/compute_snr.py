import sys
import json
import numpy as np

if len(sys.argv) != 3:
    raise RuntimeError("Usage: python compute_snr.py input.json output.txt")

json_path = sys.argv[1]
out_path = sys.argv[2]

with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

content = data["posterior"]["content"]

detectors = ["H1", "L1"]
images = [1, 2]
snr_types = ["optimal_snr", "matched_filter_snr"]


def extract_snr_samples(obj):
    if isinstance(obj, list) and len(obj) > 0 and isinstance(obj[0], dict) and "__complex__" in obj[0]:
        return np.array([x["real"] for x in obj], dtype=float)
    if isinstance(obj, list) and len(obj) > 0 and isinstance(obj[0], (int, float)):
        return np.array(obj, dtype=float)
    if isinstance(obj, dict) and "samples" in obj:
        return np.array(obj["samples"], dtype=float)
    raise TypeError("Unrecognized SNR format")


results = {}


for snr_type in snr_types:
    snr_joint_sq = None
    image_medians = {}

    for img in images:
        snr_net_sq_img = None

        for det in detectors:
            key = f"{det}_{snr_type}^({img})"
            samples = extract_snr_samples(content[key])
            if snr_net_sq_img is None:
                snr_net_sq_img = samples**2
            else:
                snr_net_sq_img += samples**2

            if snr_joint_sq is None:
                snr_joint_sq = samples**2
            else:
                snr_joint_sq += samples**2

        snr_net_img = np.sqrt(snr_net_sq_img)
        image_medians[img] = float(np.median(snr_net_img))

    snr_joint = np.sqrt(snr_joint_sq)
    p5, p50, p95 = np.percentile(snr_joint, [5, 50, 95])

    results[snr_type] = {
        "img1_median": image_medians[1],
        "img2_median": image_medians[2],
        "joint_median": float(np.median(snr_joint)),
        "joint_p5": float(p5),
        "joint_p50": float(p50),
        "joint_p95": float(p95),
    }   

with open(out_path, "w", encoding="utf-8") as fout:
    for snr_type in snr_types:
        fout.write(f"{snr_type}\n")

        if snr_type == "optimal_snr":
            fout.write("\nImage 1\n")
            fout.write(f"Combined network SNR (image 1) = {results[snr_type]['img1_median']:.6f}\n")

            fout.write("\nImage 2\n")
            fout.write(f"Combined network SNR (image 2) = {results[snr_type]['img2_median']:.6f}\n")

            fout.write(f"\nTotal combined SNR (all images) = {results[snr_type]['joint_median']:.6f}\n")
            fout.write(f"optimal_joint_p5={results[snr_type]['joint_p5']:.6f}\n")
            fout.write(f"optimal_joint_p50={results[snr_type]['joint_p50']:.6f}\n")
            fout.write(f"optimal_joint_p95={results[snr_type]['joint_p95']:.6f}\n\n")
    
        elif snr_type == "matched_filter_snr":
            fout.write("\nImage 1\n")
            fout.write(f"Combined network SNR (image 1) = {results[snr_type]['img1_median']:.6f}\n")

            fout.write("\nImage 2\n")
            fout.write(f"Combined network SNR (image 2) = {results[snr_type]['img2_median']:.6f}\n")

            fout.write(f"\nTotal combined SNR (all images) = {results[snr_type]['joint_median']:.6f}\n")
            fout.write(f"matched_joint_p5={results[snr_type]['joint_p5']:.6f}\n")
            fout.write(f"matched_joint_p50={results[snr_type]['joint_p50']:.6f}\n")
            fout.write(f"matched_joint_p95={results[snr_type]['joint_p95']:.6f}\n\n")
