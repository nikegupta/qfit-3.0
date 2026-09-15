#!/bin/bash
#
# protein_to_mol2.sh - protonates/assigns partial charges to a protein structure pdb via
# pdb2pqr30, then converts the result to mol2, both written beside the input pdb with the
# same basename. Sibling of pdb_to_mol2.sh (ligand conversion) - together these are the two
# structure -> mol2 conversion scripts the DESPOT scoring workflow uses.
#
# Steps (both in obabel_env - pdb2pqr30 and obabel are installed side by side there, see
# openbabel.yml):
#   1. pdb2pqr30: protonates <protein_pdb> at pH 7.4 (propka titration state method, AMBER
#      forcefield), writes <protein_pdb basename>.pqr beside the input pdb.
#   2. obabel: converts that pqr to <protein_pdb basename>.mol2.
#
# conda_sh/obabel_env are passed in (not hardcoded here) so the caller - program.sh - is the
# single place that names conda environments.
#
# Usage:
#   protein_to_mol2.sh <protein_pdb> <conda_sh> <obabel_env>

set -uo pipefail

if [[ $# -ne 3 ]]; then
    echo "Usage: $0 <protein_pdb> <conda_sh> <obabel_env>" >&2
    exit 1
fi

protein_pdb="$1"
CONDA_SH="$2"
CONDA_ENV_OBABEL="$3"

if [[ ! -f "$protein_pdb" ]]; then
    echo "Error: protein pdb not found: ${protein_pdb}" >&2
    exit 1
fi

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

base="${protein_pdb%.pdb}"
pqr_file="${base}.pqr"
mol2_file="${base}.mol2"

conda_activate "$CONDA_ENV_OBABEL"

echo "Protonating/assigning charges: ${protein_pdb} -> ${pqr_file}"
pdb2pqr30 --ff=AMBER --with-ph=7.4 --titration-state-method=propka --log-level WARNING "$protein_pdb" "$pqr_file"
status=$?
if [[ $status -ne 0 ]]; then
    echo "Error: pdb2pqr30 failed for ${protein_pdb}" >&2
    conda_deactivate
    exit "$status"
fi

echo "Converting to mol2: ${pqr_file} -> ${mol2_file}"
# obabel's PQR-format reader plugin unconditionally prints " charge : N" /
# " radius : N" to stderr for every atom (a debug leftover that bypasses its
# normal -q/--errorlevel message-level controls entirely, so it can't be
# silenced with a flag) - filter just those lines out, leave everything else
# on stderr untouched.
obabel "$pqr_file" -O "$mol2_file" 2> >(grep -vE '^ (charge|radius) : ' >&2)
status=$?
conda_deactivate
if [[ $status -ne 0 ]]; then
    echo "Error: obabel failed converting ${pqr_file}" >&2
    exit "$status"
fi

echo "Done: ${mol2_file}"
