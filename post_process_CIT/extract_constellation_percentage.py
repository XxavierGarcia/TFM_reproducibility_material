import os
import pandas as pd

BASE = "/home/xavier.garcia-sabat/Lensing/pesummaries_merge_all"

rows = []

for i in range(1, 158):

    inj = f"inj{i:03d}"
    data = {}

    for tag in ["img1", "img2", "joint"]:

        const_file = os.path.join(
            BASE,
            inj,
            f"skyloc_{tag}",
            "skymap",
            "constellations.dat"
        )

        if not os.path.exists(const_file):
            print("Missing:", const_file)
            data[f"{tag}_constellation"] = None
            data[f"{tag}_percentage"] = None
            continue

        parsed_data = []

        with open(const_file, "r") as f:
            next(f)  # skip header line
            for line in f:
                line = line.strip()
                if not line:
                    continue

                parts = line.split(maxsplit=1)

                if len(parts) != 2:
                    continue

                prob = float(parts[0])
                constellation = parts[1]

                parsed_data.append((prob, constellation))

        if not parsed_data:
            data[f"{tag}_constellation"] = None
            data[f"{tag}_percentage"] = None
            continue

        df = pd.DataFrame(parsed_data, columns=["prob", "constellation"])

        # Find dominant constellation
        idx = df["prob"].idxmax()
        dominant_const = df.loc[idx, "constellation"]
        dominant_percentage = df.loc[idx, "prob"] * 100

        data[f"{tag}_constellation"] = dominant_const
        data[f"{tag}_percentage"] = dominant_percentage

    rows.append({
        "inj": inj,
        **data
    })

table = pd.DataFrame(rows)

outfile = os.path.join(BASE, "dominant_constellations_table.csv")
table.to_csv(outfile, index=False)

print("\nSaved to:", outfile)