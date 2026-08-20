#!/bin/bash
#
# pdb_to_mol2.sh - assigns correct bond orders to a single ligand pdb (from its SMILES) and
# converts the result to a mol2, both written beside the input pdb with the same basename.
#
# Steps:
#   1. (rdkit_env) assign_bond_orders.py: reads the ligand's 3D coordinates from
#      <ligand_pdb> and its bond orders/connectivity from <smiles> (rdkit's
#      AssignBondOrdersFromTemplate; hydrogens are ignored throughout), writes <ligand_pdb
#      basename>.sdf beside the input pdb.
#   2. (obabel_env) obabel: converts that sdf to <ligand_pdb basename>.mol2.
#
# conda_sh/rdkit_env/obabel_env/assign_bond_orders_py are passed in (not hardcoded here) so
# the caller - program.sh - is the single place that names conda environments and script
# locations (assign_bond_orders.py lives in this same lig_scripts/ directory - see
# program.sh's ASSIGN_BOND_ORDERS_PY).
#
# Usage:
#   pdb_to_mol2.sh <ligand_pdb> <smiles> <conda_sh> <rdkit_env> <obabel_env> <assign_bond_orders_py>

set -uo pipefail

if [[ $# -ne 6 ]]; then
    echo "Usage: $0 <ligand_pdb> <smiles> <conda_sh> <rdkit_env> <obabel_env> <assign_bond_orders_py>" >&2
    exit 1
fi

ligand_pdb="$1"
smiles="$2"
CONDA_SH="$3"
CONDA_ENV_RDKIT="$4"
CONDA_ENV_OBABEL="$5"
ASSIGN_BOND_ORDERS_PY="$6"

if [[ ! -f "$ligand_pdb" ]]; then
    echo "Error: ligand pdb not found: ${ligand_pdb}" >&2
    exit 1
fi

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

base="${ligand_pdb%.pdb}"
sdf_file="${base}.sdf"
mol2_file="${base}.mol2"

echo "Assigning bond orders from SMILES: ${ligand_pdb} -> ${sdf_file}"
conda_activate "$CONDA_ENV_RDKIT"
python "$ASSIGN_BOND_ORDERS_PY" "$ligand_pdb" "$smiles" "$sdf_file"
status=$?
conda_deactivate
if [[ $status -ne 0 ]]; then
    echo "Error: assign_bond_orders.py failed for ${ligand_pdb}" >&2
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
