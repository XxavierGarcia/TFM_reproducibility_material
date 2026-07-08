import os
import glob
import subprocess
import sys

# Get the process ID from Condor (0 to N-1) and offset it to 1-N
process_id = int(sys.argv[1])
i = process_id + 1
inj = f"{i:03d}"

# Define directory paths
BASE_JSON = "/home/xavier.garcia-sabat/Lensing/pesummary_jsons_all"
BASE_OUT = "/home/xavier.garcia-sabat/Lensing/pesummaries_merge_all"

# Locate JSON files using glob patterns
img1 = glob.glob(f"{BASE_JSON}/*inj_{inj}_hanabi_single_img1*_result.json")
joint = glob.glob(f"{BASE_JSON}/*inj_{inj}_hanabi_joint*_result.json")
img2 = glob.glob(f"{BASE_JSON}/*inj_{inj}_hanabi_single_img2*_result.json")

# Execute PESummary only if all required files are present
if img1 and joint and img2:
    outdir = f"{BASE_OUT}/inj{inj}"
    os.makedirs(outdir, exist_ok=True)

    print(f"--- Starting PESummary for Injection {inj} ---")

    cmd = [
        "summarypages",
        "--webdir", outdir,
        "--samples", img1[0], joint[0], img2[0],
        "--labels", f"inj{inj}_image1_merge", f"inj{inj}_joint_merge", f"inj{inj}_image2_merge",
        "--gw",
        "--no_ligo_skymap",
        "--disable_interactive"
    ]
    
    # Run the command and catch potential errors
    subprocess.run(cmd, check=True)
else:
    print(f"Skipping Injection {inj}: Required files are missing.")