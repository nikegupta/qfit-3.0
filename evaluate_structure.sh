#!/bin/bash
#
# evaluate_structure.sh - per-atom EDIA of an experimental structure's ligand(s) placed into a
# reference structure, ranked in a null distribution. Scored as generate_null.sh scores the null
# conformers (run_edia: EDIAscorer 1.3, one call per molecule against the model cropped around
# it, in a podman container), and ranked by the same summary, mean atom EDIA, by default.
#
#   1. The experimental structure is superposed onto the reference on all shared CA atoms
#      (matched by chain and residue number; the two must have the same sequence).
#   2. chimera.pdb: the reference without its LIG, HOH and DMS residues (--exclude_resn), plus
#      the experimental structure's LIG residues moved into the reference's frame.
#   3. Each ligand (one per altloc), in the atom order and bond orders of the pocket's template
#      ligand (--pocket, default the distribution's folder), is scored by EDIAscorer 1.3 against
#      the reference without ligands cropped to --crop_radius Å (default run_edia's, 5), and the
#      map (in the reference's frame); resolution from --resolution or the reference's REMARK 3.
#   4. Each ligand's --metric (default mean_atom_edia) is ranked in the distribution's column of
#      that name (e.g. a pocket's edia_scores.csv): percentile = 100 * (n_below + 0.5 * n_equal)
#      / n.
#
# Output, in <output_dir>: ligand_edia.csv (EDIA scores and percentile per ligand),
# ligand_atom_scores.csv (per-atom EDIA), chimera.pdb, edia_target.pdb, evaluate_params.json,
# evaluate_structure_log.txt, and edia_atom_failed/ if a call failed.
# See evaluate_structure --help (src/qfit/command_line/evaluate_structure.py) for details.

set -uo pipefail

usage() {
    cat <<EOF
Usage: $0 <reference_pdb> <experimental_pdb> <map.ccp4> <distribution.csv> <output_dir>
           [--pocket <dir>] [--metric <column>] [--resolution <float>] [--ligand_resn <resn>]
           [--exclude_resn <resn[,resn...]>] [--ediascorer <path>] [--crop_radius <float>]
           [--workers <int>]

Options:
  --pocket <dir>           Pocket folder with the template ligand (ligand.pdb,
                            conformer_summary.json). Default: the distribution's folder.
  --metric <column>        Score ranked in the distribution, a column of run_edia's
                            edia_scores.csv. Default: mean_atom_edia (as generate_null.sh's plots).
  --resolution <float>     Resolution (Å) given to EDIA. Default: the reference's REMARK 3
                            RESOLUTION RANGE HIGH.
  --ligand_resn <resn>     Residue name of the ligand(s). Default: LIG.
  --exclude_resn <resn[,resn...]>  Comma-separated residue names removed from the reference
                            besides the ligand's. Default: HOH,DMS.
  --ediascorer <path>      EDIAscorer 1.3 executable, run in the ${EDIA_IMAGE}
                            podman container. Default: ${EDIASCORER}.
  --crop_radius <float>    Each ligand is scored against the residues with an atom within this
                            distance (Å). Default: run_edia's (5), as generate_null.sh.
  --workers <int>          EDIAscorer calls run at once. Default: 1.
EOF
}

# --- User-specified configuration: edit these for your environment ---
CONDA_SH="/home/ngupta/miniconda3/etc/profile.d/conda.sh"
CONDA_ENV_NULL="snEDIAm"
EDIASCORER="/home/ngupta/EDIAscorer_1.3.0/EDIAscorer"
EDIA_IMAGE="registry.access.redhat.com/ubi9/ubi"

positional=()
pocket=""
metric=""
resolution=""
ligand_resn=""
exclude_resn=""
crop_radius=""
workers=""
while [ $# -gt 0 ]; do
    case "$1" in
        -h|--help)
            usage
            exit 0
            ;;
        --pocket)
            pocket="$2"
            shift 2
            ;;
        --metric)
            metric="$2"
            shift 2
            ;;
        --resolution)
            resolution="$2"
            shift 2
            ;;
        --ligand_resn)
            ligand_resn="$2"
            shift 2
            ;;
        --exclude_resn)
            exclude_resn="$2"
            shift 2
            ;;
        --ediascorer)
            EDIASCORER="$2"
            shift 2
            ;;
        --crop_radius)
            crop_radius="$2"
            shift 2
            ;;
        --workers)
            workers="$2"
            shift 2
            ;;
        -*)
            echo "Error: unknown option $1" >&2
            usage >&2
            exit 1
            ;;
        *)
            positional+=("$1")
            shift
            ;;
    esac
done
if [ ${#positional[@]} -ne 5 ]; then
    echo "Error: expected 5 positional arguments, got ${#positional[@]}" >&2
    usage >&2
    exit 1
fi
reference_pdb="${positional[0]}"
experimental_pdb="${positional[1]}"
map="${positional[2]}"
distribution="${positional[3]}"
output_dir="${positional[4]}"

for f in "$reference_pdb" "$experimental_pdb" "$map" "$distribution" "$CONDA_SH"; do
    if [ ! -f "$f" ]; then
        echo "Error: required file not found: ${f}" >&2
        exit 1
    fi
done
if [ ! -x "$EDIASCORER" ]; then
    echo "Error: EDIAscorer not found or not executable: ${EDIASCORER}" >&2
    exit 1
fi
if ! podman image exists "$EDIA_IMAGE"; then
    echo "Error: ${EDIA_IMAGE} is not in the local podman image store (podman pull it first)." >&2
    exit 1
fi

extra_args=()
[ -n "$pocket" ] && extra_args+=(--pocket "$pocket")
[ -n "$metric" ] && extra_args+=(--metric "$metric")
[ -n "$resolution" ] && extra_args+=(--resolution "$resolution")
[ -n "$ligand_resn" ] && extra_args+=(--ligand-resn "$ligand_resn")
[ -n "$exclude_resn" ] && extra_args+=(--exclude-resn ${exclude_resn//,/ })
[ -n "$crop_radius" ] && extra_args+=(--crop-radius "$crop_radius")
[ -n "$workers" ] && extra_args+=(--workers "$workers")

# conda's activate.d hooks may reference unset variables; discard their stdout.
set +u
source "$CONDA_SH"
conda activate "$CONDA_ENV_NULL" > /dev/null
set -u

mkdir -p "$output_dir"
evaluate_structure "$reference_pdb" "$experimental_pdb" "$map" "$distribution" "$output_dir" \
    --ediascorer "$EDIASCORER" --container-image "$EDIA_IMAGE" "${extra_args[@]}" 2>&1 \
    | tee "${output_dir}/evaluate_structure_log.txt"
exit "${PIPESTATUS[0]}"
