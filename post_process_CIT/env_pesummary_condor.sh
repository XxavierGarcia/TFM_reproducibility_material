#!/bin/bash


source /cvmfs/oasis.opensciencegrid.org/ligo/sw/conda/etc/profile.d/conda.sh
conda activate igwn

cd /home/xavier.garcia-sabat/Lensing

python Pesummaries_automated_condor_all.py $1