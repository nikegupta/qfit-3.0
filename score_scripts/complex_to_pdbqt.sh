#!/bin/bash
#
# complex_to_pdbqt.sh - splits a protein-ligand complex PDB (via split_complex_pdbqt.py) into a
# receptor.pdbqt and one ligand_<label>.pdbqt per ligand instance, for docking-pose scoring (e.g.
# gnina - see run_gnina_score.sh), using meeko (mk_prepare_ligand.py/mk_prepare_receptor.py) plus
# openbabel.
#
# Each ligand instance (every atom whose resname matches <ligand_resname>, split per-altloc where
# applicable - see split_complex_pdbqt.py) is converted to SDF - either from a SMILES template, if
# one was given, via RDKit's AssignBondOrdersFromTemplate (apply_smiles_template.py), or otherwise
# via obabel's own geometry-based bond/aromaticity perception, which adds hydrogens in the
# process. Either way, mk_prepare_ligand.py then builds a proper AutoDock torsion-tree pdbqt with
# gasteiger charges. SDF (explicit bond orders), not MOL2 (generic aromatic flags), is used for
# this handoff: mk_prepare_ligand.py uses RDKit internally, and obabel's MOL2 'ar' bond flags for
# some heteroaromatic rings have no Kekule structure RDKit will accept, aborting the conversion
# with "Can't kekulize mol" even though the same ring converts fine via SDF or a SMILES template.
#
# Receptor (everything else, e.g. the rest of the protein plus cofactors like HEM): prepped via
# mk_prepare_receptor.py, EXCEPT for any HETATM cofactor resname meeko has no built-in residue
# template for. meeko can't build a working ad hoc template for those from a bare SDF via
# --add_templates - its residue matcher needs the input residue's own bonds, and its PDB/ProDy
# readers never perceive intra-residue bonds for an unrecognized resname (confirmed empirically:
# tried --read_pdb and --read_with_prody, both fail FindMCS-based matching identically regardless
# of the supplied template). So each such cofactor instance is instead detected up front
# (detect_untemplated_hetero_residues.py), deleted from the receptor before meeko runs on it, and
# prepped independently as a rigid block via obabel's own PDBQT writer (which handles Fe/metal
# centers correctly with the 'eem' charge model - gasteiger/mmff94 silently fail on structures
# containing Fe), then appended onto the protein-only receptor.pdbqt that mk_prepare_receptor.py
# writes for every recognized residue.
#
# Usage:
#   complex_to_pdbqt.sh <complex_pdb> <ligand_resname> <output_dir> <conda_sh> <obabel_env>
#                        [ligand_smiles]
#
#   [ligand_smiles]  Optional. A SMILES for <ligand_resname>, used as an RDKit template
#                     (AllChem.AssignBondOrdersFromTemplate) to assign each ligand instance's
#                     bond orders/aromaticity directly from known-correct chemistry instead of
#                     obabel's geometry-based perception (see apply_smiles_template.py). Falls
#                     back to obabel (`obabel ... -O ligand.sdf -h`) when omitted.
#
# Writes into <output_dir>:
#   receptor.pdb, ligand_<label>.pdb (from split_complex_pdbqt.py)
#   receptor.pdbqt, ligand_<label>.pdbqt (one per ligand instance)
#   cofactor_<resname>_<chain><resnum>.pdb/.pdbqt, untemplated_residues.json

set -uo pipefail

if [ $# -lt 5 ] || [ $# -gt 6 ]; then
    echo "Usage: $0 <complex_pdb> <ligand_resname> <output_dir> <conda_sh> <obabel_env> [ligand_smiles]" >&2
    exit 1
fi
complex_pdb="$1"
ligand_resname="$2"
output_dir="$3"
conda_sh="$4"
obabel_env="$5"
ligand_smiles="${6:-}"

if [ ! -f "$complex_pdb" ]; then
    echo "Error: complex pdb not found: ${complex_pdb}" >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SPLIT_PY="${SCRIPT_DIR}/split_complex_pdbqt.py"
DETECT_PY="${SCRIPT_DIR}/detect_untemplated_hetero_residues.py"
SMILES_TEMPLATE_PY="${SCRIPT_DIR}/apply_smiles_template.py"

mkdir -p "$output_dir"

set +u
source "$conda_sh"
conda activate "$obabel_env"
set -u

run_step() {
    local desc="$1"
    shift
    echo ""
    echo "========= ${desc} ========="
    "$@"
    local status=$?
    if [ $status -ne 0 ]; then
        echo "ERROR: ${desc} failed with exit code ${status}" >&2
        set +u; conda deactivate; set -u
        exit "$status"
    fi
}

# --- Split complex into receptor.pdb / ligand_<label>.pdb ---
run_step "Split complex by ligand resname" \
    python "$SPLIT_PY" "$complex_pdb" "$ligand_resname" "$output_dir"

# --- Ligand branch: one pdbqt per ligand instance ---
shopt -s nullglob
ligand_pdbs=("${output_dir}"/ligand_*.pdb)
shopt -u nullglob
if [ ${#ligand_pdbs[@]} -eq 0 ]; then
    echo "Error: no ligand_*.pdb files found in ${output_dir} (split_complex_pdbqt.py should have failed already if none matched)" >&2
    set +u; conda deactivate; set -u
    exit 1
fi
for lig_pdb in "${ligand_pdbs[@]}"; do
    label="$(basename "$lig_pdb" .pdb | sed 's/^ligand_//')"
    lig_sdf="${output_dir}/ligand_${label}.sdf"
    lig_pdbqt="${output_dir}/ligand_${label}.pdbqt"
    if [ -n "$ligand_smiles" ]; then
        run_step "Assign bond orders for ligand ${label} from SMILES template (RDKit)" \
            python "$SMILES_TEMPLATE_PY" "$lig_pdb" "$ligand_smiles" "$lig_sdf"
    else
        run_step "Convert ligand ${label} to sdf (obabel, add H)" \
            obabel "$lig_pdb" -O "$lig_sdf" -h
    fi
    run_step "Prepare ligand_${label}.pdbqt (meeko)" \
        mk_prepare_ligand.py -i "$lig_sdf" -o "$lig_pdbqt"
done

# --- Receptor branch ---
untemplated_json="${output_dir}/untemplated_residues.json"
run_step "Detect cofactor residues meeko can't template natively" \
    python "$DETECT_PY" "${output_dir}/receptor.pdb" "$untemplated_json"

delete_arg=""
cofactor_keys=()
while IFS=$'\t' read -r chain resnum resname; do
    [ -z "$chain" ] && [ -z "$resnum" ] && continue
    if [ -z "$delete_arg" ]; then
        delete_arg="${chain}:${resnum}"
    else
        delete_arg="${delete_arg},${chain}:${resnum}"
    fi
    cofactor_keys+=("${chain}:${resnum}:${resname}")
done < <(python -c "
import json
d = json.load(open('${untemplated_json}'))
for chain, resnum, resname in d:
    print(f'{chain}\t{resnum}\t{resname}')
")

receptor_args=(--read_pdb "${output_dir}/receptor.pdb" -o "${output_dir}/receptor" -p)
if [ -n "$delete_arg" ]; then
    receptor_args+=(-d "$delete_arg")
fi
run_step "Prepare receptor.pdbqt (meeko, standard residues)" \
    mk_prepare_receptor.py "${receptor_args[@]}"

for key in "${cofactor_keys[@]}"; do
    IFS=':' read -r chain resnum resname <<< "$key"
    inst_pdb="${output_dir}/cofactor_${resname}_${chain}${resnum}.pdb"
    inst_pdbqt="${output_dir}/cofactor_${resname}_${chain}${resnum}.pdbqt"
    awk -v c="$chain" -v r="$resnum" \
        '$0 ~ /^HETATM/ && substr($0,22,1)==c && substr($0,23,4)+0==r' \
        "${output_dir}/receptor.pdb" > "$inst_pdb"
    echo "END" >> "$inst_pdb"
    run_step "Prepare cofactor ${resname} ${chain}${resnum} (obabel)" \
        obabel "$inst_pdb" -O "$inst_pdbqt" -h --partialcharge eem -xr
    grep -E '^(ATOM|HETATM)' "$inst_pdbqt" >> "${output_dir}/receptor.pdbqt"
done

echo ""
echo "========= complex_to_pdbqt.sh complete ========="
echo "Receptor: ${output_dir}/receptor.pdbqt"
for lig_pdb in "${ligand_pdbs[@]}"; do
    label="$(basename "$lig_pdb" .pdb | sed 's/^ligand_//')"
    echo "Ligand ${label}: ${output_dir}/ligand_${label}.pdbqt"
done

set +u
conda deactivate
set -u
