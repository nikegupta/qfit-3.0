#!/bin/bash
#
# run_gnina_score.sh - runs gnina's CNN --score_only scoring for every ligand_<label>.pdbqt in
# <output_dir> against that same directory's receptor.pdbqt (both written by
# complex_to_pdbqt.sh), inside the gnina/gnina docker (podman) container. No GPU is required
# (falls back to CPU, just slower - see the container's own "No GPU detected" warning).
#
# Writes one row per ligand instance to <output_dir>/gnina_scores.csv
# (residue,affinity_kcal_mol,cnnscore,cnnaffinity) - 'residue' matches calc_rscc.py's own residue
# label convention (e.g. 'A502' or 'A502-B'), so the two csvs can be joined directly on that
# column (see merge_scores.py).
#
# Usage:
#   run_gnina_score.sh <output_dir> [cnn_model] [gnina_image]
#
#   <output_dir>   Directory containing receptor.pdbqt and one or more ligand_<label>.pdbqt files
#                  (written by complex_to_pdbqt.sh).
#   [cnn_model]    gnina --cnn model name. Default: crossdock_default2018
#   [gnina_image]  Docker image to run gnina from. Default: gnina/gnina:latest

set -uo pipefail

if [ $# -lt 1 ] || [ $# -gt 3 ]; then
    echo "Usage: $0 <output_dir> [cnn_model] [gnina_image]" >&2
    exit 1
fi
output_dir="$1"
cnn_model="${2:-crossdock_default2018}"
gnina_image="${3:-gnina/gnina:latest}"

receptor_pdbqt="${output_dir}/receptor.pdbqt"
if [ ! -f "$receptor_pdbqt" ]; then
    echo "Error: receptor pdbqt not found: ${receptor_pdbqt}" >&2
    exit 1
fi

shopt -s nullglob
ligand_pdbqts=("${output_dir}"/ligand_*.pdbqt)
shopt -u nullglob
if [ ${#ligand_pdbqts[@]} -eq 0 ]; then
    echo "Error: no ligand_*.pdbqt files found in ${output_dir}" >&2
    exit 1
fi

output_dir_abs="$(cd "$output_dir" && pwd)"
receptor_base="$(basename "$receptor_pdbqt")"
output_csv="${output_dir}/gnina_scores.csv"
echo "residue,affinity_kcal_mol,cnnscore,cnnaffinity" > "$output_csv"

for lig_pdbqt in "${ligand_pdbqts[@]}"; do
    label="$(basename "$lig_pdbqt" .pdbqt | sed 's/^ligand_//')"
    ligand_base="$(basename "$lig_pdbqt")"

    echo ""
    echo "========= gnina --score_only ${label} (${cnn_model}) ========="
    raw_output=$(docker run --rm \
        -v "${output_dir_abs}:/data:ro" \
        "$gnina_image" \
        gnina --receptor "/data/${receptor_base}" \
              --ligand "/data/${ligand_base}" \
              --score_only \
              --cnn "$cnn_model" 2>&1)
    status=$?
    echo "$raw_output"
    if [ $status -ne 0 ]; then
        echo "ERROR: gnina failed for ${label} with exit code ${status}" >&2
        exit "$status"
    fi

    affinity=$(echo "$raw_output" | grep -m1 '^Affinity:' | awk '{print $2}')
    cnnscore=$(echo "$raw_output" | grep -m1 '^CNNscore:' | awk '{print $2}')
    cnnaffinity=$(echo "$raw_output" | grep -m1 '^CNNaffinity:' | awk '{print $2}')
    echo "${label},${affinity},${cnnscore},${cnnaffinity}" >> "$output_csv"
done

echo ""
echo "${#ligand_pdbqts[@]} ligand instance(s) scored -> ${output_csv}"
