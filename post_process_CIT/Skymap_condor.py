#!/usr/bin/env python3

import os
import subprocess
import h5py
import numpy as np
from numpy.lib import recfunctions
import sys

BASE = "/home/xavier.garcia-sabat/Lensing/pesummaries_merge_all"

def run(cmd):
    print("RUNNING:", " ".join(cmd), flush=True)
    subprocess.run(cmd)

#def run(cmd):
    #print(" ".join(cmd))
    #subprocess.run(cmd, check=True)

# Get injection number from Condor
process_id = int(sys.argv[1])
i = process_id + 1
inj = f"inj{i:03d}"
inj_id = f"{i:03d}"

print(f"\n========== Processing {inj} ==========\n")

INJDIR = os.path.join(BASE, inj)

cases = {
    "img1": f"{inj}_image1_merge",
    "img2": f"{inj}_image2_merge",
    "joint": f"{inj}_joint_merge",
}

for tag, group in cases.items():

    infile = f"{BASE}/{inj}/samples/posterior_samples.h5"
    print("Checking:", infile)
    print("Exists?", os.path.exists(infile))
    outfile = f"{INJDIR}/posterior_samples_{tag}_thin.h5"

    if not os.path.exists(infile):
        print(f"posterior_samples.h5 not found: {infile}")
        continue

    with h5py.File(infile, "r") as f, h5py.File(outfile, "w") as g:

        if group not in f:
            print(f"Group '{group}' not found in {infile}")
            continue

        data = f[group]["posterior_samples"][:]
        thin = data[:]

        if "time" not in thin.dtype.names:

            gps_time = None

            if tag in ["img1", "img2"]:
                gps_time = thin["geocent_time"]

            elif tag == "joint" and "geocent_time^(1)" in thin.dtype.names:
                gps_time = thin["geocent_time^(1)"]

            if gps_time is None:
                print(f"No suitable geocent_time found for {group}")
                continue

            thin = recfunctions.append_fields(
                thin,
                "time",
                gps_time,
                usemask=False
            )

        ggrp = g.create_group(group)
        ggrp.create_dataset("posterior_samples", data=thin)

    print(f"Written: {outfile} with {len(thin)} samples")

    outbase = os.path.join(INJDIR, f"skyloc_{tag}")
    sky_dir = os.path.join(outbase, "skymap")
    os.makedirs(sky_dir, exist_ok=True)

    run([
        "ligo-skymap-from-samples",
        "--path", group,
        "--tablename", "posterior_samples",
        "--disable-distance-map",
        "--trials", "1",
        "--jobs", "1",
        "--outdir", sky_dir,
        outfile
    ])

    run([
        "ligo-skymap-stats",
        "--contour", "90",
        "--output", os.path.join(sky_dir, "skymap_stats.dat"),
        "--modes", os.path.join(sky_dir, "skymap.fits")
    ])

    run([
        "ligo-skymap-constellations",
        "--output", os.path.join(sky_dir, "constellations.dat"),
        os.path.join(sky_dir, "skymap.fits")
    ])

print(f"\nFinished {inj}")