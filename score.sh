#!/bin/bash
#
# score.sh - standalone scorer for a single protein-ligand complex against a single density map:
# per-residue RSCC (score_scripts/calc_rscc.py, qfit's transformer) plus gnina CNN docking-pose
# scoring (score_scripts/complex_to_pdbqt.sh + run_gnina_score.sh) for every instance of
# <ligand_resname> in <structure_file>, merged into one scores.csv (score_scripts/merge_scores.py).
#
# Everything - rscc.csv, receptor.pdb/.pdbqt, ligand_<label>.pdb/.pdbqt per ligand instance,
# gnina_scores.csv, and the final merged scores.csv - is written into <output_dir>.
#
# Usage:
#   score.sh <map_file> <structure_file> <output_dir> <ligand_resname>
#            [-em] [--resolution <float>] [--bfactor <float>] [--label <F,PHI>]
#            [--cnn <model>] [--gnina-image <image>]

set -uo pipefail

usage() {
    cat <<EOF
Usage: $0 <map_file> <structure_file> <output_dir> <ligand_resname>
           [-em] [--resolution <float>] [--bfactor <float>] [--label <F,PHI>]
           [--cnn <model>] [--gnina-image <image>] [--smi <SMILES>]

  <map_file>        Density map: .ccp4/.mrc/.map (real-space) or .mtz (structure factors).
  <structure_file>  Protein-ligand complex structure: .pdb.
  <output_dir>      Directory to write every output to (created if missing) - rscc.csv,
                     receptor.pdb/.pdbqt, ligand_<label>.pdb/.pdbqt (one per ligand instance),
                     gnina_scores.csv, and the final merged scores.csv.
  <ligand_resname>  Residue name of the ligand to score, e.g. LIG - every instance/altloc in
                     <structure_file> matching this resname is scored (both RSCC and gnina).

  -em                    Mark <map_file> as a cryo-EM map: electron (Mott-Bethe) scattering
                          factors are used for RSCC's model density (instead of X-ray
                          scattering factors), the map is treated as one non-periodic P1 box,
                          and the RSCC mask radius always uses the static 1.5 Å fallback below
                          (regardless of --resolution) - a cryo-EM map's single global
                          resolution figure is a much weaker proxy for local mask radius than
                          it is for X-ray data.
  --resolution <float>   Map resolution (Å), for RSCC. Optional - if omitted (or if -em is
                          given), the RSCC mask radius falls back to a static 1.5 Å instead of
                          the resolution-derived 0.5 + resolution/3.0 heuristic (matching qfit's
                          own resolution-unaware default - see qfit.py's QFitBase.__init__).
  --bfactor <float>      Constant B-factor to use for every atom when generating each ligand
                          instance's model density for RSCC. Optional - if omitted (the
                          default), each atom's own B-factor from <structure_file> is used
                          instead of one constant value for the whole instance.
  --label <F,PHI>        MTZ amplitude/phase column labels (only used for a .mtz map file).
                          Default: FWT,PHWT
  --cnn <model>          gnina --cnn model name for docking-pose scoring. Default:
                          crossdock_default2018
  --gnina-image <image>  Docker image to run gnina from. Default: gnina/gnina:latest
  --smi <SMILES>         SMILES for <ligand_resname>, used as an RDKit template to assign each
                          ligand instance's bond orders/aromaticity (AllChem.
                          AssignBondOrdersFromTemplate) when converting it for gnina docking-pose
                          scoring - see score_scripts/apply_smiles_template.py. Optional - if
                          omitted, obabel's own geometry-based bond/aromaticity perception is used
                          instead, which can fail to produce a structure meeko/RDKit can kekulize
                          for some heteroaromatic ligands.
EOF
    exit 1
}

if [ $# -eq 0 ]; then
    usage
fi

# --- Positional args ---
map_file=""
structure_file=""
output_dir=""
ligand_resname=""

# --- Flags ---
em=0
resolution=""
bfactor=""
label="FWT,PHWT"
cnn_model="crossdock_default2018"
gnina_image="gnina/gnina:latest"
ligand_smiles=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        -em|--em)
            em=1
            shift
            ;;
        --resolution)
            resolution="$2"
            shift 2
            ;;
        --bfactor)
            bfactor="$2"
            shift 2
            ;;
        --label)
            label="$2"
            shift 2
            ;;
        --cnn)
            cnn_model="$2"
            shift 2
            ;;
        --gnina-image)
            gnina_image="$2"
            shift 2
            ;;
        --smi)
            ligand_smiles="$2"
            shift 2
            ;;
        -h|--help)
            usage
            ;;
        -*)
            echo "Unknown option: $1" >&2
            usage
            ;;
        *)
            if [ -z "$map_file" ]; then
                map_file="$1"
            elif [ -z "$structure_file" ]; then
                structure_file="$1"
            elif [ -z "$output_dir" ]; then
                output_dir="$1"
            elif [ -z "$ligand_resname" ]; then
                ligand_resname="$1"
            else
                echo "Unexpected argument: $1" >&2
                usage
            fi
            shift
            ;;
    esac
done

if [ -z "$map_file" ] || [ -z "$structure_file" ] || [ -z "$output_dir" ] || [ -z "$ligand_resname" ]; then
    echo "Error: <map_file>, <structure_file>, <output_dir>, and <ligand_resname> are all required." >&2
    usage
fi
if [ ! -f "$map_file" ]; then
    echo "Error: map file not found: ${map_file}" >&2
    exit 1
fi
if [ ! -f "$structure_file" ]; then
    echo "Error: structure file not found: ${structure_file}" >&2
    exit 1
fi

# --- User-specified configuration: edit these for your environment ---
CONDA_SH="/home/ngupta/miniconda3/etc/profile.d/conda.sh"
CONDA_ENV_QFIT="nikhils_program_exp"
CONDA_ENV_OBABEL="openbabel"
SCORE_SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/score_scripts"

# --- Derived paths ---
CALC_RSCC_PY="${SCORE_SCRIPTS_DIR}/calc_rscc.py"
SPLIT_COMPLEX_PDBQT_PY="${SCORE_SCRIPTS_DIR}/split_complex_pdbqt.py"
DETECT_UNTEMPLATED_PY="${SCORE_SCRIPTS_DIR}/detect_untemplated_hetero_residues.py"
APPLY_SMILES_TEMPLATE_PY="${SCORE_SCRIPTS_DIR}/apply_smiles_template.py"
COMPLEX_TO_PDBQT_SH="${SCORE_SCRIPTS_DIR}/complex_to_pdbqt.sh"
RUN_GNINA_SH="${SCORE_SCRIPTS_DIR}/run_gnina_score.sh"
MERGE_SCORES_PY="${SCORE_SCRIPTS_DIR}/merge_scores.py"

for f in "$CALC_RSCC_PY" "$SPLIT_COMPLEX_PDBQT_PY" "$DETECT_UNTEMPLATED_PY" \
         "$APPLY_SMILES_TEMPLATE_PY" "$COMPLEX_TO_PDBQT_SH" "$RUN_GNINA_SH" "$MERGE_SCORES_PY"; do
    if [ ! -f "$f" ]; then
        echo "Error: required file not found: ${f}" >&2
        exit 1
    fi
done

conda_activate() {
    set +u
    source "$CONDA_SH"
    conda activate "$1" > /dev/null
    set -u
}

conda_deactivate() {
    set +u
    conda deactivate > /dev/null
    set -u
}

run_step() {
    local desc="$1"
    shift
    echo ""
    echo "========= ${desc} ========="
    "$@"
    local status=$?
    if [ $status -ne 0 ]; then
        echo "ERROR: ${desc} failed with exit code ${status}" >&2
        exit "$status"
    fi
}

mkdir -p "$output_dir"

overall_start=$(date +%s)

calc_rscc_step() {
    local em_args=()
    [ "$em" -eq 1 ] && em_args+=(--em)
    local resolution_args=()
    [ -n "$resolution" ] && resolution_args+=(--resolution "$resolution")
    local bfactor_args=()
    [ -n "$bfactor" ] && bfactor_args+=(--bfactor "$bfactor")
    conda_activate "$CONDA_ENV_QFIT"
    python "$CALC_RSCC_PY" "$structure_file" "$map_file" "${output_dir}/rscc.csv" \
        --label "$label" --ligand-resname "$ligand_resname" \
        "${em_args[@]}" "${resolution_args[@]}" "${bfactor_args[@]}"
    local status=$?
    conda_deactivate
    return $status
}
run_step "Calculate ligand RSCC" calc_rscc_step

run_step "Prepare receptor/ligand pdbqt files (meeko + obabel)" \
    bash "$COMPLEX_TO_PDBQT_SH" "$structure_file" "$ligand_resname" "$output_dir" "$CONDA_SH" "$CONDA_ENV_OBABEL" "$ligand_smiles"

run_step "Score docking pose(s) with gnina" \
    bash "$RUN_GNINA_SH" "$output_dir" "$cnn_model" "$gnina_image"

run_step "Merge RSCC + gnina scores" \
    python3 "$MERGE_SCORES_PY" "${output_dir}/rscc.csv" "${output_dir}/gnina_scores.csv" "${output_dir}/scores.csv"

echo ""
echo "========= score.sh complete ========="
echo "Final scores: ${output_dir}/scores.csv"
overall_end=$(date +%s)
elapsed=$((overall_end - overall_start))
printf "Total time: %02d:%02d:%02d (HH:MM:SS)\n" $((elapsed/3600)) $(((elapsed%3600)/60)) $((elapsed%60))
