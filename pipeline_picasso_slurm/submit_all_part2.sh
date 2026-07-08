#!/bin/bash

BASE=/mnt/home/users/uib54_res/resh000464/lenskyloc/lenskyloc_runs

for injdir in $BASE/inj*/ ; do
    inj=$(basename $injdir)
    id=${inj#inj}

    echo "Submitting FINAL + POST + SNR for $inj"

    joint_submit_dir=$BASE/$inj/${inj}_joint/submit

    # FINAL RESULT
    final_script=$joint_submit_dir/lenskyloc_inj_${id}_hanabi_joint_final_merge_result.sh
    if [ ! -f "$final_script" ]; then
        echo "  ERROR: final_result script not found for $inj"
        continue
    fi

    jid_final=$(sbatch $final_script | awk '{print $4}')
    echo "  FINAL_RESULT jobid: $jid_final"

    # POSTPROCESS
    jid_post=$(sbatch --dependency=afterok:$jid_final $BASE/$inj/run_postprocess_$inj | awk '{print $4}')
    echo "  POSTPROCESS jobid: $jid_post"

    # SUMMARYPAGES
    #jid_sum=$(sbatch --dependency=afterok:$jid_post $BASE/$inj/run_summarypages_$inj | awk '{print $4}')
    #echo "  SUMMARYPAGES jobid: $jid_sum"

    # SKYMAP
    #jid_sky=$(sbatch --dependency=afterok:$jid_sum $BASE/$inj/run_skymap_$inj | awk '{print $4}')
    #echo "  SKYMAP jobid: $jid_sky"

    # SNR COMPUTATION
    jid_snr=$(sbatch --dependency=afterok:$jid_post $BASE/$inj/run_snr_$inj | awk '{print $4}')
    echo "  SNR jobid: $jid_snr"

done
