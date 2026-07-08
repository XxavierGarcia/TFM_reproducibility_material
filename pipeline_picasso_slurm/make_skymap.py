#!/usr/bin/env python3

import os
import subprocess
import h5py
import sys
import numpy as np
from numpy.lib import recfunctions


# Arguments

if len(sys.argv) != 2:
    raise RuntimeError("Usage: make_skymap.py injXXX")

inj = sys.argv[1]          # injxxx
inj_id = inj.replace("inj", "") #xxx

BASE = "/mnt/home/users/uib54_res/resh000464/lenskyloc/lenskyloc_runs"
INJDIR = os.path.join(BASE, inj)

# Helpers

def run(cmd):
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)

# What we want skymaps for
cases = {
    "img1": f"{inj}_image1",
    "img2": f"{inj}_image2",
    "joint": f"{inj}_joint",
}


# Loop over img1 / img2 / joint

for tag, group in cases.items():

        
    infile = f"{BASE}/inj{inj_id}/pesummaries_{inj}/{inj}_final/samples/posterior_samples.h5"
    outfile = f"{INJDIR}/posterior_samples_{tag}_thin.h5"

    if not os.path.exists(infile):
        raise RuntimeError(f"posterior_samples.h5 not found: {infile}")

    with h5py.File(infile, "r") as f, h5py.File(outfile, "w") as g:

        if group not in f:
            raise RuntimeError(f"Group '{group}' not found in {infile}")

        data = f[group]["posterior_samples"][:]
        thin = data[:]

        # automatic GPS time 
        if "time" not in thin.dtype.names:

            gps_time = None

            if tag == "img1":
                gps_time = thin["geocent_time"]

            elif tag == "img2":
                gps_time = thin["geocent_time"]

            elif tag == "joint":
                if "geocent_time^(1)" in thin.dtype.names:
                    gps_time = thin["geocent_time^(1)"]

            if gps_time is None:
                raise RuntimeError(
                    f"No suitable geocent_time found for {group}. "
                    f"Available fields: {thin.dtype.names}"
                )

            thin = thin.copy()
            thin = recfunctions.append_fields(
                thin,
                "time",
                gps_time,
                usemask=False
            )

        
        ggrp = g.create_group(group)
        ggrp.create_dataset("posterior_samples", data=thin)

    print(f"Written: {outfile} with {len(thin)} samples")



    # SKYMAP 
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

print("\nAll skymaps generated successfully.")
