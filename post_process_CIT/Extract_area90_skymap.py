import os
import pandas as pd

BASE = "/home/xavier.garcia-sabat/Lensing/pesummaries_merge_all"

rows = []

for i in range(1,158):

    inj = f"inj{i:03d}"
    areas = {}

    for tag in ["img1", "img2", "joint"]:

        stats_file = os.path.join(
            BASE,
            inj,
            f"skyloc_{tag}",
            "skymap",
            "skymap_stats.dat"
        )

        if not os.path.exists(stats_file):
            print("Missing:", stats_file)
            areas[tag] = None
            continue

        df = pd.read_csv(
            stats_file,
            delim_whitespace=True,
            comment="#"
        )

        areas[tag] = df["area(90)"].iloc[0]

    rows.append({
        "inj": inj,
        "img1_area90": areas["img1"],
        "img2_area90": areas["img2"],
        "joint_area90": areas["joint"]
    })

table = pd.DataFrame(rows)

outfile = os.path.join(BASE, "sky_area_90_table.csv")
table.to_csv(outfile, index=False)

print("\nSaved to:", outfile)