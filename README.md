# TFM_reproducibility_material

This repository contains the material associated with the MSc thesis "Parameter Estimation and Sky Localisation of Lensed Image Pairs of Binary Black Hole Mergers", provided to ensure the reproducibility of the results.

## Repository structure

```
TFM_reproducibility_material/
├── environment_full.yml       # exact conda environment used on Picasso (hanabi_bilby260)
├── picasso/                   # SLURM-based parameter estimation pipeline
│   ├── Generator_inis_lenskyloc.py
│   ├── lenskyloc_inj_hanabi_single.ini
│   ├── lenskyloc_inj_hanabi_joint.ini
│   ├── submit_all_part1.sh / submit_all_part2.sh
│   ├── final_result_template, postprocess_template, run_*_template
│   ├── compute_snr.py, snr_combined_table.py
│   └── n_lensed_params_bbh_filtered.txt
└── post_process_CIT/          # HTCondor-based post-processing on the CIT cluster
    ├── Pesummaries_automated_condor_all.py, Skymap_condor.py
    ├── env_pesummary_condor.sh, env_skymap_condor.sh
    ├── pesummary_condor.submit, skymap.submit
    ├── ligo_*_plot.py              # plotting scripts used for the thesis figures
    ├── Extract_area90_skymap.py, extract_constellation_percentage.py
    ├── compute_snr.py, snr_combined_table.py
    └── Generator_inis_lenskyloc.py, Lensing_calculator.ipynb
```



## Environment

```markdown
The exact conda environment used to run the pipeline on the Picasso HPC cluster (environment name `hanabi_bilby260`) is provided in [`environment_full.yml`](environment_full.yml). To recreate it:

```bash
conda env create -f environment_full.yml

Key packages include bilby, bilby_pipe, hanabi, and ligo.skymap.

Pipeline overview
The full pipeline is described in Chapter 5 of the thesis ("Parameter estimation pipeline"):

picasso/: single-image analysis with bilby_pipe and joint analysis with hanabi_joint_pipe, automated on the Picasso HPC cluster via SLURM.
post_process_CIT/: sky map generation and PESummary post-processing, run on LIGO's computing cluster at Caltech (CIT) via HTCondor.
A representative pair of single-image/joint .ini configuration files is reproduced in Appendix C of the thesis.




