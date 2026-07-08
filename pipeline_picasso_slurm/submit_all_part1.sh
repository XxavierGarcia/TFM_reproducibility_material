#!/bin/bash

BASE=/mnt/home/users/uib54_res/resh000464/lenskyloc/lenskyloc_runs

for injdir in $BASE/inj*/ ; do
    inj=$(basename $injdir)
    id=${inj#inj}

    echo "Submitting SINGLE + JOINT for $inj"
    
    # SINGLE IMG 1
    cd $injdir || exit 1
    bilby_pipe single_event_${id}_image1.ini
    s1_script=$(ls ${inj}_img1/submit/slurm_*single_img1*_master.sh)
    jid_s1=$(sbatch $s1_script | awk '{print $4}')
    echo "  IMG1 jobid: $jid_s1"

    # SINGLE IMG 2
    cd $injdir || exit 1
    bilby_pipe single_event_${id}_image2.ini
    s2_script=$(ls ${inj}_img2/submit/slurm_*single_img2*_master.sh)
    jid_s2=$(sbatch $s2_script | awk '{print $4}')
    echo "  IMG2 jobid: $jid_s2"

    # JOINT
    cd $injdir || exit 1
    hanabi_joint_pipe joint_event_${id}.ini
    joint_script=$(ls ${inj}_joint/submit/slurm_*joint*_master.sh)
    jid_joint=$(sbatch --dependency=afterok:$jid_s1:$jid_s2 $joint_script | awk '{print $4}')
    echo "  JOINT jobid: $jid_joint"

done

