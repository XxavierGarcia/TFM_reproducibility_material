#!/usr/bin/env python
# coding: utf-8

"""
Script to automatically generate:
- .ini files (single and joint)
- SLURM scripts (final_result, summarypages, postprocess, skymap, snr)

from a JSON file containing lensed BBH injection parameters.
"""

# Imports
# ------------------------------------------------------------

import os
import re
import json
import numpy as np
import lal
import lalsimulation as LS
from bilby.gw.utils import calculate_time_to_merger


# Paths and Base Configuration
# ------------------------------------------------------------

json_path = "n_lensed_params_bbh_filtered.txt"

single_template_path = "ini_templates/lenskyloc_inj_hanabi_single.ini"
joint_template_path = "ini_templates/lenskyloc_inj_hanabi_joint.ini"

base_outdir = "/mnt/home/users/uib54_res/resh000464/lenskyloc/lenskyloc_runs_prova"
templates_dir = "/mnt/home/users/uib54_res/resh000464/lenskyloc"
root = "/mnt/home/users/uib54_res/resh000464"

os.makedirs(base_outdir, exist_ok=True)


# Utility Functions
# ------------------------------------------------------------

def load_text(path):
    """Load a full text file into memory."""
    with open(path) as f:
        return f.read()


# Load templates
# ------------------------------------------------------------

single_template = load_text(single_template_path)
joint_template = load_text(joint_template_path)

final_result_template = load_text("final_result_template")
summary_template = load_text("run_summarypages_template")
postprocess_template = load_text("postprocess_template")


# Injection Parameter Handling
# ------------------------------------------------------------

def replace_injection_params(template_text, params_dict):
    """
    Replace values inside injection-dict in the .ini template
    using parameters from the JSON file.
    """

    # Mapping between JSON keys and .ini keys
    param_map = {
        "effective_luminosity_distance": "luminosity_distance",
        "mass_1": "mass_1",
        "mass_2": "mass_2",
        "theta_jn": "theta_jn",
        "psi": "psi",
        "a_1": "a_1",
        "a_2": "a_2",
        "tilt_1": "tilt_1",
        "tilt_2": "tilt_2",
        "phase": "phase",
        "ra": "ra",
        "dec": "dec",
        "phi_12": "phi_12",
        "phi_jl": "phi_jl",
        "effective_geocent_time": "geocent_time",
        "image_type": "image_type",
    }

    # Extract injection-dict={...} block
    match = re.search(r"injection-dict=(\{.*?\})", template_text, re.DOTALL)
    injection_block = match.group(1)
    new_block = injection_block

    # Replace each key:value inside the dictionary
    for json_key, ini_key in param_map.items():
        if json_key in params_dict:
            value = params_dict[json_key]
            new_block = re.sub(
                rf"'{ini_key}'\s*:\s*[^,}}]+",
                f"'{ini_key}': {value}",
                new_block,
            )

    # Reconstruct full text
    updated_text = (
        template_text[:match.start(1)]
        + new_block
        + template_text[match.end(1):]
    )

    return updated_text


# Function definitions
# ------------------------------------------------------------

def estimate_seglen(fLow, m_total_min, q, chi1z, chi2z):
    """
    Estimate a safe data segment duration based on:
      - Merger time
      - Ringdown time
      - Waveform duration
    """

    m1 = m_total_min * q / (1.0 + q) * lal.MSUN_SI
    m2 = m_total_min / (1.0 + q) * lal.MSUN_SI

    s = LS.SimInspiralFinalBlackHoleSpinBound(chi1z, chi2z)

    tmerge = (
        LS.SimInspiralMergeTimeBound(m1, m2)
        + LS.SimInspiralRingdownTimeBound(m1 + m2, s)
    )

    try:
        wf_len = LS.SimIMRPhenomXASDuration(m1, m2, chi1z, chi2z, fLow)
    except Exception:
        wf_len = 0

    safe_len = (wf_len + tmerge) * 1.03 + 3.0

    # Round up to next power of 2
    return int(2 ** (np.floor(np.log2(safe_len)) + 1))


def update_duration(text, duration):
    """Update duration inside the .ini file."""
    return re.sub(
        r"duration\s*=\s*[\d\.]+",
        f"duration={duration}",
        text,
    )


def update_chirp_prior(text, chirp_mass):
    """
    Adjust the chirp_mass prior range to ±30% around the computed value.
    """

    delta = 0.3 * chirp_mass
    chirp_min = max(0, chirp_mass - delta)
    chirp_max = chirp_mass + delta

    pattern = r"chirp-mass:\s*bilby\.gw\.prior\.UniformInComponentsChirpMass\(.*?\)"

    replacement = (
        "chirp-mass: bilby.gw.prior.UniformInComponentsChirpMass("
        f"minimum={chirp_min:.3f}, maximum={chirp_max:.3f}, "
        "name='chirp_mass', boundary=None)"
    )

    return re.sub(pattern, replacement, text, flags=re.DOTALL)


def format_geocent_tag(geocent):
    """
    Convert:
        1241598347.4801836
    into:
        1241598347-4801836
    """

    geocent_str = str(geocent)

    if "." not in geocent_str:
        raise ValueError(f"Invalid geocent time: {geocent_str}")

    return geocent_str.replace(".", "-")


def generate_slurm_from_template(template_name, output_name, inj_id, event_dir):
    """
    Generate a SLURM script by replacing 'XXX' with the injection ID.
    """

    template_path = os.path.join(templates_dir, template_name)
    out_script = os.path.join(event_dir, output_name)

    with open(template_path) as f:
        text = f.read().replace("XXX", inj_id)

    with open(out_script, "w") as f:
        f.write(text)


# Load JSON Data
# ------------------------------------------------------------

with open(json_path, "r") as f:
    data = json.load(f)

num_events = len(data["geocent_time"])


# Main Event Loop
# ------------------------------------------------------------

for i in range(num_events):

    event_id = i + 1
    inj_id = f"{event_id:03d}"

    
    # Create event directory 

    event_dir = os.path.join(base_outdir, f"inj{inj_id}")
    os.makedirs(event_dir, exist_ok=True)

    param_image1 = {}
    param_image2 = {}

    
    # Extract parameters for both images

    for key, values in data.items():

        val = values[i]

        if isinstance(val, list) and len(val) >= 2:
            param_image1[key] = val[0]
            param_image2[key] = val[1]
        else:
            param_image1[key] = val
            param_image2[key] = val

    geocent1 = str(param_image1["effective_geocent_time"])
    geocent2 = str(param_image2["effective_geocent_time"])

    m1 = float(param_image1["mass_1"])
    m2 = float(param_image1["mass_2"])

    
    # Physical calculations

    q_real = max(m1, m2) / min(m1, m2)
    chirp_mass = (m1 * m2)**(3/5) / (m1 + m2)**(1/5)

    chi1z = float(param_image1["a_1"])
    chi2z = float(param_image1["a_2"])

    duration = estimate_seglen(
        fLow=20,
        m_total_min=m1 + m2,
        q=q_real,
        chi1z=chi1z,
        chi2z=chi2z,
    )

    print(
        f"Event {event_id}: "
        f"m1={m1:.2f}, m2={m2:.2f}, "
        f"TotalM={m1+m2:.2f}, "
        f"chirp_mass={chirp_mass:.3f}, "
        f"ratio={q_real:.3f}, "
        f"duration={duration:.3f}"
    )

    # Prepare single .ini files

    def prepare_single_ini(template, params, duration, chirp_mass):
        text = replace_injection_params(template, params)
        text = update_duration(text, duration)
        return update_chirp_prior(text, chirp_mass)

    single1_text = prepare_single_ini(
        single_template, param_image1, duration, chirp_mass
    )

    single2_text = prepare_single_ini(
        single_template, param_image2, duration, chirp_mass
    )

    single1_name = f"single_event_{inj_id}_image1.ini"
    single2_name = f"single_event_{inj_id}_image2.ini"
    joint_name = f"joint_event_{inj_id}.ini"

    # Update label, outdir and trigger-time
    
    for n, (text, geocent, name) in enumerate(
        [(single1_text, geocent1, single1_name),
         (single2_text, geocent2, single2_name)],
        start=1,
    ):

        text = re.sub(
            r"^label\s*=.*$",
            f"label=lenskyloc_inj_{inj_id}_hanabi_single_img{n}",
            text,
            flags=re.MULTILINE,
        )

        text = re.sub(
            r"^outdir\s*=.*$",
            f"outdir={event_dir}/inj{inj_id}_img{n}",
            text,
            flags=re.MULTILINE,
        )

        text = re.sub(
            r"^trigger-time\s*=.*$",
            f"trigger-time={geocent}",
            text,
            flags=re.MULTILINE,
        )

        with open(os.path.join(event_dir, name), "w") as f:
            f.write(text)

    # Create joint.ini

    joint_text = joint_template.replace("xxx", f"{inj_id}")

    with open(os.path.join(event_dir, joint_name), "w") as f:
        f.write(joint_text)

    # Generate final_result SLURM script

    joint_result = (
        f"{event_dir}/inj{inj_id}_joint/result/"
        f"lenskyloc_inj_{inj_id}_hanabi_joint_hanabi_joint_analysis_merge_result.json"
    )

    joint_final_dir = f"{event_dir}/inj{inj_id}_joint/final_result"

    final_result_script = (
        f"{event_dir}/inj{inj_id}_joint/submit/"
        f"lenskyloc_inj_{inj_id}_hanabi_joint_final_merge_result.sh"
    )

    os.makedirs(os.path.dirname(final_result_script), exist_ok=True)

    with open(final_result_script, "w") as f:
        f.write(
            final_result_template.format(
                inj_id=inj_id,
                joint_result=joint_result,
                joint_final_dir=joint_final_dir,
            )
        )

    # Generate summarypages 

    geocent_tag1 = format_geocent_tag(geocent1)
    geocent_tag2 = format_geocent_tag(geocent2)

    img1_result = (
        f"{event_dir}/inj{inj_id}_img1/final_result/"
        f"lenskyloc_inj_{inj_id}_hanabi_single_img1_data0_"
        f"{geocent_tag1}_analysis_H1L1_merge_result.json"
    )

    img2_result = (
        f"{event_dir}/inj{inj_id}_img2/final_result/"
        f"lenskyloc_inj_{inj_id}_hanabi_single_img2_data0_"
        f"{geocent_tag2}_analysis_H1L1_merge_result.json"
    )

    joint_final_result = (
        f"{event_dir}/inj{inj_id}_joint/final_result/"
        f"lenskyloc_inj_{inj_id}_hanabi_joint_hanabi_joint_analysis_merge_result.json"
    )

    with open(
        os.path.join(event_dir, f"run_summarypages_inj{inj_id}"),
        "w",
    ) as f:
        f.write(
            summary_template.format(
                inj_id=inj_id,
                img1_result=img1_result,
                img2_result=img2_result,
                joint_result=joint_final_result,
            )
        )

    # Generate postprocess 

    img1_data_dump = (
        f"{event_dir}/inj{inj_id}_img1/data/"
        f"lenskyloc_inj_{inj_id}_hanabi_single_img1_data0_"
        f"{geocent_tag1}_generation_data_dump.pickle"
    )

    img2_data_dump = (
        f"{event_dir}/inj{inj_id}_img2/data/"
        f"lenskyloc_inj_{inj_id}_hanabi_single_img2_data0_"
        f"{geocent_tag2}_generation_data_dump.pickle"
    )

    postprocess_output = (
        f"{event_dir}/lenskyloc_inj_{inj_id}_hanabi_joint_postprocess_merge.json"
    )

    with open(
        os.path.join(event_dir, f"run_postprocess_inj{inj_id}"),
        "w",
    ) as f:
        f.write(
            postprocess_template.format(
                inj_id=inj_id,
                single1_ini=f"{event_dir}/{single1_name}",
                single2_ini=f"{event_dir}/{single2_name}",
                img1_data_dump=img1_data_dump,
                img2_data_dump=img2_data_dump,
                postprocess_output=postprocess_output,
                joint_result=joint_final_result,
            )
        )

    # Generate skymap and SNR 

    generate_slurm_from_template(
        "run_skymap_template",
        f"run_skymap_inj{inj_id}",
        inj_id,
        event_dir,
    )

    generate_slurm_from_template(
        "run_snr_template.py",
        f"run_snr_inj{inj_id}",
        inj_id,
        event_dir,
    )
