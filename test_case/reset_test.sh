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

echo "test_case reset."
