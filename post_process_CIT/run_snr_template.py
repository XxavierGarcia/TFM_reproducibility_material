#!/bin/bash
#SBATCH --job-name=snr_injXXX
#SBATCH --output=/mnt/home/users/uib54_res/resh000464/lenskyloc/lenskyloc_runs/injXXX/snr_injXXX.out
#SBATCH --error=/mnt/home/users/uib54_res/resh000464/lenskyloc/lenskyloc_runs/injXXX/snr_injXXX.err
#SBATCH --time=00:05:00
#SBATCH --mem=2G
#SBATCH --cpus-per-task=1

INJ=injXXX
BASE=/mnt/home/users/uib54_res/resh000464/lenskyloc/lenskyloc_runs

INJDIR=$BASE/$INJ
ID=${INJ#inj}

JSON=${INJDIR}/lenskyloc_inj_${ID}_hanabi_joint_postprocess.json
OUT=${INJDIR}/snr_${INJ}.txt

echo "Computing SNRs for $INJ"
echo "Input JSON: $JSON"
echo "Output file: $OUT"

python /mnt/home/users/uib54_res/resh000464/lenskyloc/compute_snr.py "$JSON" "$OUT"

echo "Done."
