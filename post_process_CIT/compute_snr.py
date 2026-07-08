import sys
import json
import numpy as np

if len(sys.argv) != 3:
    raise RuntimeError("Usage: python compute_snr.py input.json output.txt")

json_path = sys.argv[1]
out_path = sys.argv[2]

with open(json_path, "r") as f:
    data = json.load(f)

content = data["posterior"]["content"]

detectors = ["H1", "L1"]
images = [1, 2]
snr_types = ["optimal_snr", "matched_filter_snr"]

def extract_snr_samples(obj):
    if isinstance(obj, list) and isinstance(obj[0], dict) and "__complex__" in obj[0]:
        return np.array([x["real"] for x in obj])
    if isinstance(obj, list) and isinstance(obj[0], (int, float)):
        return np.array(obj)
    if isinstance(obj, dict) and "samples" in obj:
        return np.array(obj["samples"])
    raise TypeError("Unrecognized SNR format")

with open(out_path, "w") as fout:

    for snr_type in snr_types:
        fout.write(f"{snr_type}\n")

        snr_joint_sq = None

        for img in images:
            fout.write(f"\nImage {img}\n")

            snr_net_sq_img = None

            for det in detectors:
                key = f"{det}_{snr_type}^({img})"
                samples = extract_snr_samples(content[key])

                median = np.median(samples)

                fout.write(
                    f"{det} SNR = {median:.2f}\n"
                )

                if snr_net_sq_img is None:
                    snr_net_sq_img = samples**2
                else:
                    snr_net_sq_img += samples**2

                if snr_joint_sq is None:
                    snr_joint_sq = samples**2
                else:
                    snr_joint_sq += samples**2

            snr_net_img = np.sqrt(snr_net_sq_img)
            median_net = np.median(snr_net_img)

            fout.write(
                f"Combined network SNR (image {img}) = {median_net:.2f}\n"
            )

        snr_joint = np.sqrt(snr_joint_sq)
        median_joint = np.median(snr_joint)

        fout.write(
            f"\nTotal combined SNR (all images) = {median_joint:.2f}\n\n"
        )