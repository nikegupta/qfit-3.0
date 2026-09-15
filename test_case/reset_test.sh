#!/bin/bash
#
# reset_test.sh - deletes every output test.sh produces, so test_case can be run again
# from a clean state. Paths are relative to this script's own directory (not the caller's
# cwd), so it can be invoked from anywhere.
#
# Removes:
#   datasets/x00407-1/run_1   - all pipeline stage output for the dataset
#   graphs                    - all pooled/per-stage plots
#   logs                      - all run logs
#   ligands/DSI_1_G22/DSI_1_G22.{mol2,sdf} - convert_ligs' generated conversions
#     (DSI_1_G22.pdb/.cif and obabel_DSI_1_G22.log are original inputs, not outputs -
#     left alone)
#   datasets/x00407-1/x00407-1-aligned-structure_rscc.csv - calc_apo_rscc's output (stage 0b;
#     dataset-scoped, not nested under run_1, so the run_1 removal above doesn't catch it)
#   reference_set/x00407-1/x00407-1-pandda-model_rscc.csv - calc_ref_set_rscc's output (stage
#     0c; likewise dataset-scoped, lives in reference_set/ rather than datasets/)
#   reference_set/x00407-1/{despot_log.txt,expanded.pdb,expanded.pqr,expanded.mol2,ligs,
#     ligs.sdf,ligs.mol2,x00407-1_DESPOT.csv} - ref_set_despot's outputs (stage 0d), all written
#     directly into reference_set/x00407-1/ alongside the reference structure itself -
#     x00407-1-pandda-model.pdb is the original input there and is left alone
#
# Usage:
#   ./reset_test.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

rm_path() {
    local path="$1"
    if [ -e "$path" ]; then
        echo "Removing ${path}"
        rm -rf "$path"
    fi
}

rm_path "${SCRIPT_DIR}/datasets/x00407-1/run_1"
rm_path "${SCRIPT_DIR}/graphs"
rm_path "${SCRIPT_DIR}/logs"
rm_path "${SCRIPT_DIR}/ligands/DSI_1_G22/DSI_1_G22.mol2"
rm_path "${SCRIPT_DIR}/ligands/DSI_1_G22/DSI_1_G22.sdf"

# calc_apo_rscc (stage 0b)
rm_path "${SCRIPT_DIR}/datasets/x00407-1/x00407-1-aligned-structure_rscc.csv"

# calc_ref_set_rscc (stage 0c)
rm_path "${SCRIPT_DIR}/reference_set/x00407-1/x00407-1-pandda-model_rscc.csv"

# ref_set_despot (stage 0d) - x00407-1-pandda-model.pdb (the input) is left alone
rm_path "${SCRIPT_DIR}/reference_set/x00407-1/despot_log.txt"
rm_path "${SCRIPT_DIR}/reference_set/x00407-1/expanded.pdb"
rm_path "${SCRIPT_DIR}/reference_set/x00407-1/expanded.pqr"
rm_path "${SCRIPT_DIR}/reference_set/x00407-1/expanded.mol2"
rm_path "${SCRIPT_DIR}/reference_set/x00407-1/ligs"
rm_path "${SCRIPT_DIR}/reference_set/x00407-1/ligs.sdf"
rm_path "${SCRIPT_DIR}/reference_set/x00407-1/ligs.mol2"
rm_path "${SCRIPT_DIR}/reference_set/x00407-1/x00407-1_DESPOT.csv"

echo "test_case reset."
