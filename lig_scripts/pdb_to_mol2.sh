#!/bin/bash
#
# pdb_to_mol2.sh - assigns correct bond orders to one or more single-instance ligand pdbs
# (from their shared SMILES) and converts the combined result to one mol2, both written as
# <output_base>.sdf/<output_base>.mol2.
#
# Steps:
#   1. (rdkit_env) assign_bond_orders.py: reads each ligand instance's 3D coordinates from
#      its own <ligand_pdb> and their shared bond orders/connectivity from <smiles> (rdkit's
#      AssignBondOrdersFromTemplate; hydrogens are ignored throughout), writes one combined
#      <output_base>.sdf (one entry per instance, named after that instance's own pdb
#      basename - see assign_bond_orders.py).
#   2. (obabel_env) obabel: converts that sdf to <output_base>.mol2.
#
# conda_sh/rdkit_env/obabel_env/assign_bond_orders_py are passed in (not hardcoded here) so
# the caller - program.sh - is the single place that names conda environments and script
# locations (assign_bond_orders.py lives in this same lig_scripts/ directory - see
# program.sh's ASSIGN_BOND_ORDERS_PY).
#
# Usage:
#   pdb_to_mol2.sh <output_base> <smiles> <conda_sh> <rdkit_env> <obabel_env> \
#                  <assign_bond_orders_py> <ligand_pdb>...

set -uo pipefail

if [[ $# -lt 7 ]]; then
    echo "Usage: $0 <output_base> <smiles> <conda_sh> <rdkit_env> <obabel_env> " \
         "<assign_bond_orders_py> <ligand_pdb>..." >&2
    exit 1
fi

output_base="$1"
smiles="$2"
CONDA_SH="$3"
CONDA_ENV_RDKIT="$4"
CONDA_ENV_OBABEL="$5"
ASSIGN_BOND_ORDERS_PY="$6"
shift 6
ligand_pdbs=("$@")

for ligand_pdb in "${ligand_pdbs[@]}"; do
    if [[ ! -f "$ligand_pdb" ]]; then
        echo "Error: ligand pdb not found: ${ligand_pdb}" >&2
        exit 1
    fi
done

if [[ ! -f "$ASSIGN_BOND_ORDERS_PY" ]]; then
    echo "Error: required script not found: ${ASSIGN_BOND_ORDERS_PY}" >&2
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

sdf_file="${output_base}.sdf"
mol2_file="${output_base}.mol2"

echo "Assigning bond orders from SMILES: ${ligand_pdbs[*]} -> ${sdf_file}"
conda_activate "$CONDA_ENV_RDKIT"
python "$ASSIGN_BOND_ORDERS_PY" "${ligand_pdbs[@]}" "$smiles" "$sdf_file"
status=$?
conda_deactivate
if [[ $status -ne 0 ]]; then
    echo "Error: assign_bond_orders.py failed for ${ligand_pdbs[*]}" >&2
    exit "$status"
fi

echo "Converting to mol2: ${sdf_file} -> ${mol2_file}"
conda_activate "$CONDA_ENV_OBABEL"
obabel "$sdf_file" -O "$mol2_file"
status=$?
conda_deactivate
if [[ $status -ne 0 ]]; then
    echo "Error: obabel failed converting ${sdf_file}" >&2
    exit "$status"
fi

echo "Done: ${mol2_file}"
