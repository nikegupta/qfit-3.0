#!/bin/bash
#
# generate_null.sh - driver for the null-distribution construction pipeline (random ligand
# placements around each modeled ligand, scored against the density).
# Takes positional args corresponding to the stages of the pipeline:
#   conformer_run_name
#
# Runs, in order:
#   STAGE_1: Null conformer sampling
#   1a. generate_conformers             -> <dataset>/<conformer_run_name>/LIG_<chain><resi>_<alt>/
#                                          conformers.pdb, conformer_summary.json,
#                                          ensemble_mask.ccp4
#   1b. plot_ensemble_volumes (pooled)  -> GRAPHS_DIR/<conformer_run_name>/
#   1c. plot_conformer_acceptance (pooled) -> GRAPHS_DIR/<conformer_run_name>/
#   1d. plot_reference_coverage (pooled)   -> GRAPHS_DIR/<conformer_run_name>/
#   STAGE_2: per-atom EDIA scoring of the null conformers and reference ligand (per ligand null)
#   2a. run_edia (twice per null)       -> <dataset>/<conformer_run_name>/LIG_<chain><resi>_<alt>/
#                                          conformers.pdb: distribution_residue_scores.csv,
#                                            distribution_atom_scores.csv (+ _params.json)
#                                          ligand.pdb: reference_residue_scores.csv,
#                                            reference_atom_scores.csv (+ _params.json)
#   2b. plot_edia_histogram             -> <dataset>/<conformer_run_name>/LIG_<chain><resi>_<alt>/
#                                          mean_atom_edia_histogram.png
#   STAGE_3: Evaluation (pooled)
#   3.  plot_edia_evaluation            -> GRAPHS_DIR/<conformer_run_name>/
#                                          structure_mean_atom_edia,
#                                          null_median_mean_atom_edia,
#                                          null_mean_mean_atom_edia,
#                                          structure_mean_atom_edia_percentile (.png/.csv)
#
# EDIA (stage 2) scores against the dataset's density map
# (DATASETS_DIR/<dataset>/MAP_NAME_PATTERN, default {dataset}_2fofc.ccp4; any CCP4 map in the
# structure's frame - made outside this pipeline, e.g. by claude_scripts/make_2fofc_maps.py),
# with LIG and the --exclude_resn residues (default HOH, DMS) removed from the model, and the
# unit cell, space group and resolution always read here from CSV_FILE and passed to run_edia
# (--cell, --space-group, --resolution; the structure's own CRYST1 and REMARK records are not
# used).
# Stage 2: run_edia (a general scorer: ligand PDB, target, map) is run twice per null: on
# conformers.pdb (the null distribution) and on ligand.pdb (the null's reference ligand
# conformation; written by generate_conformers, or extracted here from the structure for nulls
# sampled before it was). Each molecule is scored in its own EDIAscorer 1.3 call (the only way
# it writes per-atom scores) against the dataset's structure without its LIG residues (every
# altloc kept: a multiconformer model's apo state contributes density too), cropped to the
# residues within --edia_crop_radius Å of the molecule (the map is never cropped), bond orders
# from the dataset's SMILES - so the reference always shares the null's map and settings.
# EDIAscorer 1.3 needs a newer glibc than this host's, so each call set runs in one podman
# container (EDIA_IMAGE), --edia_workers at a time: stage 2 uses -p x --edia_workers cores.
# Per-molecule scores (EDIAm, mean atom EDIA, ...; one row per conformer, model = MODEL number
# in conformers.pdb) go to distribution_residue_scores.csv and reference_residue_scores.csv,
# per-atom scores to distribution_atom_scores.csv and reference_atom_scores.csv, and a
# histogram of the null's mean atom EDIA (0.01 bins) to mean_atom_edia_histogram.png. A failed
# call is recorded (status failed, files in <name>_edia_failed/) and doesn't stop the null.
# Datasets without a map are skipped with a warning; an interrupted null resumes from its
# finished calls (<name>_edia_work/).
# Stage 3 pools stage 2 across every dataset into four histograms, each with its CSV: the
# reference ligands' mean atom EDIA (one point per LIG residue altloc), each null's median and
# mean null mean atom EDIA (one point per null), and each reference's mean atom EDIA
# percentile within its null distribution (ties counted as half).
#
# Per-ligand output: every conformation of every modeled ligand (resname LIG; each altloc letter
# of each residue, e.g. altloc B, or C and D) gets its own null ensemble, in
# <dataset>/<conformer_run_name>/LIG_<chain><resi>_<altloc>/ (e.g. .../run_21/LIG_A701_B); a
# ligand with no altloc letter at all gets LIG_<chain><resi>. That conformation is the reference
# ligand: the atom-naming template, the sampling centre, and the one reference scored in stage 2.
#
# Structures: DATASETS_DIR/<dataset>/STRUCTURE_PDB_PATTERN, default {dataset}_001.pdb: the
# multiconformer model, apo state as altloc A and bound state as B (ligands B only). Its LIG
# residues are expected to carry the {dataset}.pdb chains and numbers, with secondary-site copies
# removed (claude_scripts/filter_001_ligands.py).
#
# Sampling (stage 1a): each placement is a random torsion conformer (RDKit ETKDG library from the
# SMILES), its centroid drawn evenly by volume within --sampling_radius Å (default 3) of the
# reference conformation's centroid, at a uniformly random orientation. Clashes are checked
# against the dataset's structure itself: a placement is dropped if it clashes with any of its
# heavy atoms except LIG, --exclude_resn residues and the --ignore_altlocs altlocs (default A,
# the apo state) - it must fit the bound state and may overlap the apo state. No crystal
# symmetry mates are used. Placements are drawn until --n_conformers are kept.
#
# Modularity: each stage nests under the previous stage's run-name folder, so a new name
# branches off without touching prior results.
#
# Idempotency: every step checks whether its own output already exists - per ligand null (main
# pipeline steps) or per run (graphing steps) - and skips if so. Pass --overwrite to force every
# step to redo, or --replot for just the graphing steps.
#
# Dataset scoping: by default every stage runs over every dataset in DATASETS_FILE
# (datasets.txt). Pass --dataset <id[,id...]> to restrict the whole invocation to just those
# datasets. Pooled (cross-dataset) graphing steps are skipped entirely when --dataset is given,
# so a partial run never overwrites a full-run plot.

set -uo pipefail

# One thread per process: numpy's OpenBLAS otherwise starts a thread per core and spreads large
# matrix products over all of them (slower here, and N x cores threads under -p N). Parallelism
# comes from -p. Inherited by every stage.
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

usage() {
    cat <<EOF
Usage: $0 <conformer_run_name>
           [-p <num_parallel>] [--overwrite] [--replot] [--dataset <id[,id...]>]
           [--n_conformers <int>] [--n_torsion_confs <int>]
           [--radius_step <float>] [--angle_step <float>] [--clash_scale <float>]
           [--polar_clash_scale <float>] [--exclude_resn <resn[,resn...]>]
           [--max_candidates_per_conformer <int>] [--seed <int>] [--mask_spacing <float>]
           [--sampling_radius <float>] [--ignore_altlocs <letters>]
           [--map_pattern <pattern>]
           [--edia_crop_radius <float>] [--edia_workers <int>]

Only <conformer_run_name> is required.

Options:
  -p <num_parallel>        Number of datasets processed in parallel. Default: 1
  --overwrite              Force every requested step to re-run in place, even if its output
                            already exists (normally such a step is skipped - see "Idempotency"
                            in the header comment). Applies to every stage, including the
                            graphing steps.
  --replot                 Force just the graphing steps (1b-1d, 2b, 3) to re-run in place, even
                            if their output already exists. Does not affect generate_conformers
                            or EDIA scoring.
  --dataset <id[,id...]>   Run only on this dataset, or comma-separated list of datasets
                            (e.g. x00001-1 or x00001-1,x00002-1), instead of every dataset
                            listed in DATASETS_FILE (datasets.txt). Graphing steps are
                            skipped.
  --n_conformers <int>     Conformers: number of non-clashing placements kept per ligand null.
                            Default: 1000.
  --n_torsion_confs <int>  Conformers: size of the RDKit ETKDG torsion conformer library.
                            Default: 200.
  --radius_step <float>    Conformers: translation distances are rounded to this step (Å).
                            Default: 0.1.
  --angle_step <float>     Conformers: translation polar/azimuth angles are rounded to this step
                            (degrees). Default: 1.
  --clash_scale <float>    Conformers: a ligand-protein atom pair clashes below this fraction of
                            the sum of their vdW radii, for every element pair. Default: 0.5.
  --polar_clash_scale <float>  Conformers: a different threshold for N-N, N-O and O-O pairs only.
                            Default: --clash_scale (one threshold for all pairs).
  --exclude_resn <resn[,resn...]>  Conformers and EDIA: comma-separated residue names left
                            out of the clash obstacles and of the EDIA target (besides
                            hydrogens and every LIG residue). Default: HOH,DMS.
  --max_candidates_per_conformer <int>  Conformers: stop after this many candidate placements
                            per conformer asked for (i.e. this x --n_conformers per ligand
                            null) even if fewer than --n_conformers were kept. Default: 1000000.
  --seed <int>             Conformers: random seed. Default: a fresh seed per ligand null from the
                            clock, recorded in each conformer_summary.json (params.seed).

  --mask_spacing <float>   Conformers: grid spacing (Å) of ensemble_mask.ccp4, the mask of every
                            voxel the kept placements occupy. Default: 0.25.
  --sampling_radius <float>  Conformers: placement centroids are drawn evenly by volume within
                            this distance (Å) of the reference ligand's centroid. Default: 3.
  --ignore_altlocs <letters>  Conformers: altloc letters (e.g. A or A,C) whose atoms are not clash
                            obstacles - the apo state of the multiconformer structures; "none"
                            makes every altloc an obstacle. EDIA always keeps every altloc.
                            Default: A.
  --map_pattern <pattern>  EDIA: density map file inside each dataset's folder, {dataset}
                            replaced by the dataset name. Default: {dataset}_2fofc.ccp4.
  --edia_crop_radius <float>  EDIA: each molecule is scored against the residues with an atom
                            within this distance (Å) of it. Default: run_edia's (5).
  --edia_workers <int>     EDIA: EDIAscorer calls run at once per dataset; stage 2 uses
                            -p x this many cores. Default: 1.

Examples:
  $0 conformers_1
  $0 conformers_1 -p 8
  $0 conformers_1 --dataset x01325-1
  $0 conformers_2 --n_conformers 20000 --sampling_radius 2
  $0 conformers_1 --map_pattern "{dataset}-event_1_1-BDC_0.3_map.native.ccp4"
  $0 conformers_1 -p 4 --edia_workers 4
EOF
    exit 1
}

# --- User-specified configuration: edit these for your environment ---
BASE_DIR="${BASE_DIR:-/home/ngupta/snEDIAm}"
CSV_FILE="${BASE_DIR}/pxr_fragments.csv"
DATASETS_DIR="${BASE_DIR}/datasets"
DATASETS_FILE="${BASE_DIR}/datasets_test.txt"
GRAPHS_DIR="${BASE_DIR}/graphs"
SNEDIAM_ANALYSIS_SCRIPTS_DIR="${BASE_DIR}/qfit-3.0/src/snediam_analysis_scripts"
CONDA_SH="/home/ngupta/miniconda3/etc/profile.d/conda.sh"
CONDA_ENV_NULL="snEDIAm"

# Per-dataset inputs, inside DATASETS_DIR/<dataset>/: the structure holding the modeled
# ligand(s) - the multiconformer model (apo state altloc A, bound state B).
STRUCTURE_PDB_PATTERN="{dataset}_001.pdb"
# Residue name of the modeled ligands; every conformation of each gets a null ensemble.
LIGAND_RESN="LIG"
# Density map EDIA scores against, inside DATASETS_DIR/<dataset>/ (override: --map_pattern).
MAP_NAME_PATTERN="{dataset}_refined_2fofc.ccp4"
# EDIAscorer 1.3 executable, and the podman image (glibc >= 2.29, in the local image store) it
# runs in.
EDIASCORER="/home/ngupta/EDIAscorer_1.3.0/EDIAscorer"
EDIA_IMAGE="registry.access.redhat.com/ubi9/ubi"

# --- Derived paths: assumed to live at fixed locations under BASE_DIR ---
PLOT_ENSEMBLE_VOLUMES_PY="${SNEDIAM_ANALYSIS_SCRIPTS_DIR}/plot_ensemble_volumes.py"
PLOT_CONFORMER_ACCEPTANCE_PY="${SNEDIAM_ANALYSIS_SCRIPTS_DIR}/plot_conformer_acceptance.py"
PLOT_REFERENCE_COVERAGE_PY="${SNEDIAM_ANALYSIS_SCRIPTS_DIR}/plot_reference_coverage.py"
PLOT_EDIA_HISTOGRAM_PY="${SNEDIAM_ANALYSIS_SCRIPTS_DIR}/plot_edia_histogram.py"
PLOT_EDIA_EVALUATION_PY="${SNEDIAM_ANALYSIS_SCRIPTS_DIR}/plot_edia_evaluation.py"

for f in "$DATASETS_FILE" "$CSV_FILE" "$CONDA_SH" "$PLOT_ENSEMBLE_VOLUMES_PY" \
         "$PLOT_CONFORMER_ACCEPTANCE_PY" "$PLOT_REFERENCE_COVERAGE_PY" \
         "$PLOT_EDIA_HISTOGRAM_PY" "$PLOT_EDIA_EVALUATION_PY"; do
    if [ ! -f "$f" ]; then
        echo "Error: required file not found: ${f}" >&2
        exit 1
    fi
done
if [ ! -x "$EDIASCORER" ]; then
    echo "Error: EDIAscorer not found or not executable: ${EDIASCORER}" >&2
    exit 1
fi
if [ ! -d "$DATASETS_DIR" ]; then
    echo "Error: required directory not found: ${DATASETS_DIR}" >&2
    exit 1
fi

# --- Argument parsing ---
if [ $# -eq 0 ]; then
    usage
fi

orig_args=("$@")

conformer_run_name=""

num_parallel=""
overwrite=0
replot=0
dataset_arg=""
DATASET_OVERRIDE_FILE=""

# generate_conformers tunables, left empty by default so generate_conformers' own argparse
# defaults apply (--n-conformers 1000, --n-torsion-confs 200,
# --radius-step 0.1, --angle-step 1, --clash-scale 0.5, --polar-clash-scale = clash scale,
# --exclude-resn HOH DMS, --max-candidates-per-conformer 1000000, --seed from the clock,
# --mask-spacing 0.25, --sampling-radius 3).
n_conformers=""
n_torsion_confs=""
radius_step=""
angle_step=""
clash_scale=""
polar_clash_scale=""
exclude_resn=""
max_candidates_per_conformer=""
seed=""
mask_spacing=""
sampling_radius=""
ignore_altlocs=""
map_pattern=""
# run_edia tunables, empty for its argparse defaults (--crop-radius 5, --workers 1).
edia_crop_radius=""
edia_workers=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        -p)
            num_parallel="$2"
            shift 2
            ;;
        --overwrite)
            overwrite=1
            shift
            ;;
        --replot)
            replot=1
            shift
            ;;
        --dataset)
            dataset_arg="$2"
            shift 2
            ;;
        --n_conformers)
            n_conformers="$2"
            shift 2
            ;;
        --n_torsion_confs)
            n_torsion_confs="$2"
            shift 2
            ;;
        --radius_step)
            radius_step="$2"
            shift 2
            ;;
        --angle_step)
            angle_step="$2"
            shift 2
            ;;
        --clash_scale)
            clash_scale="$2"
            shift 2
            ;;
        --polar_clash_scale)
            polar_clash_scale="$2"
            shift 2
            ;;
        --exclude_resn)
            exclude_resn="$2"
            shift 2
            ;;
        --max_candidates_per_conformer)
            max_candidates_per_conformer="$2"
            shift 2
            ;;
        --seed)
            seed="$2"
            shift 2
            ;;
        --mask_spacing)
            mask_spacing="$2"
            shift 2
            ;;
        --sampling_radius)
            sampling_radius="$2"
            shift 2
            ;;
        --ignore_altlocs)
            ignore_altlocs="$2"
            shift 2
            ;;
        --map_pattern)
            map_pattern="$2"
            shift 2
            ;;
        --edia_crop_radius)
            edia_crop_radius="$2"
            shift 2
            ;;
        --edia_workers)
            edia_workers="$2"
            shift 2
            ;;
        -h|--help)
            usage
            ;;
        -*)
            echo "Unknown option: $1"
            usage
            ;;
        *)
            if [ -z "$conformer_run_name" ]; then
                conformer_run_name="$1"
            else
                echo "Unexpected argument: $1"
                usage
            fi
            shift
            ;;
    esac
done

if [ -z "$conformer_run_name" ]; then
    echo "Error: <conformer_run_name> is required."
    usage
fi

# --- Full-run logging: everything from here on is teed into
# BASE_DIR/logs/<conformer_run_name>/log.txt (log_2.txt, log_3.txt, ... if it already exists,
# never overwriting a prior run's log).
log_dir="${BASE_DIR}/logs/${conformer_run_name}"
mkdir -p "$log_dir"

log_file="${log_dir}/log.txt"
if [ -f "$log_file" ]; then
    log_n=2
    while [ -f "${log_dir}/log_${log_n}.txt" ]; do
        log_n=$((log_n + 1))
    done
    log_file="${log_dir}/log_${log_n}.txt"
fi

exec > >(tee "$log_file") 2>&1
echo "Logging full run output to: ${log_file}"

echo "========= Run parameters ========="
echo "Invocation: $0 ${orig_args[*]}"
echo "conformer_run_name: ${conformer_run_name}"
echo "num_parallel: ${num_parallel:-1}"
echo "overwrite: ${overwrite}"
echo "replot: ${replot}"
echo "dataset: ${dataset_arg:-(all in ${DATASETS_FILE})}"
echo "n_conformers: ${n_conformers:-(not set - generate_conformers default 1000)}"
echo "n_torsion_confs: ${n_torsion_confs:-(not set - generate_conformers default 200)}"
echo "radius_step: ${radius_step:-(not set - generate_conformers default 0.1)}"
echo "angle_step: ${angle_step:-(not set - generate_conformers default 1)}"
echo "clash_scale: ${clash_scale:-(not set - generate_conformers default 0.5)}"
echo "polar_clash_scale: ${polar_clash_scale:-(not set - same as clash_scale)}"
echo "exclude_resn: ${exclude_resn:-(not set - generate_conformers default HOH,DMS)}"
echo "max_candidates_per_conformer: ${max_candidates_per_conformer:-(not set - generate_conformers default 1000000)}"
echo "seed: ${seed:-(not set - generate_conformers picks one per ligand null from the clock)}"
echo "mask_spacing: ${mask_spacing:-(not set - generate_conformers default 0.25)}"
echo "sampling_radius: ${sampling_radius:-(not set - generate_conformers default 3)}"
echo "ignore_altlocs: ${ignore_altlocs:-(not set - generate_conformers default A)}"
[ -n "$map_pattern" ] && MAP_NAME_PATTERN="$map_pattern"
echo "map_pattern: ${MAP_NAME_PATTERN}"
echo "edia_crop_radius: ${edia_crop_radius:-(not set - run_edia default 5)}"
echo "edia_workers: ${edia_workers:-(not set - run_edia default 1)}"

# --dataset override: repoint DATASETS_FILE at a generated temp file listing just the requested
# dataset(s).
if [ -n "$dataset_arg" ]; then
    DATASET_OVERRIDE_FILE=$(mktemp)
    IFS=',' read -ra _cli_datasets <<< "$dataset_arg"
    for _cli_dataset in "${_cli_datasets[@]}"; do
        _cli_dataset="$(echo -n "$_cli_dataset" | xargs)"
        [ -z "$_cli_dataset" ] && continue
        if [ ! -d "${DATASETS_DIR}/${_cli_dataset}" ]; then
            echo "Error: --dataset given but dataset directory not found: ${DATASETS_DIR}/${_cli_dataset}" >&2
            exit 1
        fi
        echo "$_cli_dataset" >> "$DATASET_OVERRIDE_FILE"
    done
    unset _cli_datasets _cli_dataset

    if [ ! -s "$DATASET_OVERRIDE_FILE" ]; then
        echo "Error: --dataset given but no valid dataset IDs were parsed from '${dataset_arg}'" >&2
        exit 1
    fi

    DATASETS_FILE="$DATASET_OVERRIDE_FILE"
    echo "--dataset given: restricting run to $(tr '\n' ' ' < "$DATASETS_FILE")"
fi
trap 'rm -f "$DATASET_OVERRIDE_FILE"' EXIT

# Canonical list of datasets for this run, read from DATASETS_FILE exactly once.
mapfile -t DATASETS < <(grep -v '^[[:space:]]*$' "$DATASETS_FILE" | tr -d '\r')
if [ ${#DATASETS[@]} -eq 0 ]; then
    echo "Error: no datasets found in ${DATASETS_FILE}" >&2
    exit 1
fi

NUM_PARALLEL_DEFAULT=${num_parallel:-1}

# Everything the per-dataset *_process_dataset functions need, exported so GNU parallel's
# subshells can see it.
export conformer_run_name overwrite replot
export n_conformers n_torsion_confs radius_step angle_step clash_scale
export polar_clash_scale exclude_resn max_candidates_per_conformer seed mask_spacing
export sampling_radius ignore_altlocs
export BASE_DIR DATASETS_DIR DATASETS_FILE GRAPHS_DIR CSV_FILE CONDA_SH CONDA_ENV_NULL
export STRUCTURE_PDB_PATTERN LIGAND_RESN
export MAP_NAME_PATTERN EDIASCORER EDIA_IMAGE edia_crop_radius edia_workers PLOT_EDIA_HISTOGRAM_PY

# --- Helpers ---

# files_exist <path...>: true (0) iff every given path exists as a regular file.
files_exist() {
    local f
    for f in "$@"; do
        [ -f "$f" ] || return 1
    done
    return 0
}
export -f files_exist

# list_ligands <pdb_file>: prints one "<chain> <resnum> <altloc>" line per conformation of each
# LIGAND_RESN residue in the file - one line per altloc letter of the residue's heavy atoms, in
# file order; a residue with no altloc letter gives one line with altloc "-".
list_ligands() {
    local pdb_file=$1
    awk -v resn="$LIGAND_RESN" '
        ($1 == "ATOM" || $1 == "HETATM") {
            r = substr($0, 18, 3); gsub(/ /, "", r)
            if (r != resn) next
            el = substr($0, 77, 2); gsub(/ /, "", el)
            if (el == "") { el = substr($0, 13, 4); gsub(/ /, "", el); el = substr(el, 1, 1) }
            if (toupper(el) == "H" || toupper(el) == "D") next
            chain = substr($0, 22, 1); gsub(/ /, "", chain)
            resnum = substr($0, 23, 4); gsub(/ /, "", resnum)
            alt = substr($0, 17, 1)
            key = chain " " resnum
            if (!(key in seen)) { seen[key] = 1; order[++n] = key }
            if (alt != " " && !((key, alt) in has)) { has[key, alt] = 1; alts[key] = alts[key] alt }
        }
        END {
            for (i = 1; i <= n; i++) {
                key = order[i]
                if (alts[key] == "") { print key " -"; continue }
                m = split(alts[key], letters, "")
                for (j = 1; j <= m; j++) print key " " letters[j]
            }
        }
    ' "$pdb_file"
}
export -f list_ligands

# write_params_txt <output_file> <name=value> [<name=value> ...]
# An empty value means the corresponding flag wasn't given, so the underlying script's own
# argparse default applied instead.
write_params_txt() {
    local output_file="$1"
    shift
    local kv name value
    for kv in "$@"; do
        name="${kv%%=*}"
        value="${kv#*=}"
        if [ -z "$value" ]; then
            echo "${name}: (not set - script default used)"
        else
            echo "${name}: ${value}"
        fi
    done > "$output_file"
}
export -f write_params_txt

# run_step <description> <command...>
run_step() {
    local desc="$1"
    shift
    echo ""
    echo ""
    echo "========= ${desc} ========="
    "$@"
    local status=$?
    if [ $status -ne 0 ]; then
        echo "ERROR: ${desc} failed with exit code ${status}" >&2
        exit "$status"
    fi
}

# run_step_pooled_replot <description> <check_fn> <run_fn>
# For pooled (cross-dataset) graphing steps. Skipped entirely when --dataset was given: pooled
# plots write into one shared GRAPHS_DIR location keyed only by run name, so running one against
# a --dataset subset would overwrite the full-run plot with a partial one (this wins even under
# --replot). Otherwise skips run_fn when check_fn (a function name taking no args) returns
# success ("all of this step's outputs already exist"), unless --replot or --overwrite was given.
run_step_pooled_replot() {
    local desc="$1" check_fn="$2" run_fn="$3"
    if [ -n "$dataset_arg" ]; then
        echo ""
        echo "Skipping ${desc} (pooled plot; --dataset was given, would overwrite the full-run plot with a partial one)."
        return 0
    fi
    if [ "$overwrite" -ne 1 ] && [ "$replot" -ne 1 ] && "$check_fn"; then
        echo ""
        echo "Skipping ${desc} (outputs already exist; pass --replot or --overwrite to redo)."
        return 0
    fi
    run_step "$desc" "$run_fn"
}

# conda_activate <env_name> / conda_deactivate: relax nounset for activate.d/deactivate.d hooks
# that reference unset variables, and discard hook stdout.
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
export -f conda_activate conda_deactivate

# print_elapsed <start_epoch_seconds> [label]
print_elapsed() {
    local start_time="$1"
    local label="${2:-Script}"
    local elapsed=$(( $(date +%s) - start_time ))
    printf "%s took %02d:%02d:%02d (HH:MM:SS)\n" "$label" \
        $((elapsed / 3600)) $(((elapsed % 3600) / 60)) $((elapsed % 60))
}
export -f print_elapsed

######################################################################
# Stage 1a: generate_conformers
######################################################################
# For every conformation (altloc) of every ligand of the dataset's structure, samples
# non-clashing placements (torsion conformer + translation + rotation) within --sampling_radius
# of that conformation's centroid, into LIG_<chain><resi>_<altloc>/.

conformers_process_dataset() {
    local dataset=$1
    local dataset_dir="${DATASETS_DIR}/${dataset}"
    local structure="${dataset_dir}/${STRUCTURE_PDB_PATTERN//\{dataset\}/${dataset}}"

    if [ ! -f "$structure" ]; then
        echo "Warning [${dataset}]: structure not found: ${structure}, skipping."
        return 1
    fi

    local ligands=()
    mapfile -t ligands < <(list_ligands "$structure")
    if [ ${#ligands[@]} -eq 0 ]; then
        echo "Warning [${dataset}]: no ${LIGAND_RESN} residues found in ${structure}, skipping."
        return 1
    fi

    conda_activate "$CONDA_ENV_NULL"

    local lig chain resnum altloc out_dir label
    for lig in "${ligands[@]}"; do
        read -r chain resnum altloc <<< "$lig"
        if [ "$altloc" = "-" ]; then
            altloc=""
            label="${LIGAND_RESN}_${chain}${resnum}"
        else
            label="${LIGAND_RESN}_${chain}${resnum}_${altloc}"
        fi
        out_dir="${dataset_dir}/${conformer_run_name}/${label}"
        if [ "$overwrite" -ne 1 ] && files_exist "${out_dir}/conformer_summary.json"; then
            echo "Skipping [${dataset} ${label}]: conformers already complete (${out_dir}/conformer_summary.json exists)."
            continue
        fi

        mkdir -p "$out_dir"
        write_params_txt "${out_dir}/conformer_params.txt" \
            "structure=${structure}" \
            "chain=${chain}" \
            "resi=${resnum}" \
            "altloc=${altloc}" \
            "sampling_radius=${sampling_radius}" \
            "n_conformers=${n_conformers}" \
            "n_torsion_confs=${n_torsion_confs}" \
            "radius_step=${radius_step}" \
            "angle_step=${angle_step}" \
            "clash_scale=${clash_scale}" \
            "polar_clash_scale=${polar_clash_scale}" \
            "exclude_resn=${exclude_resn}" \
            "max_candidates_per_conformer=${max_candidates_per_conformer}" \
            "seed=${seed}" \
            "mask_spacing=${mask_spacing}" \
            "ignore_altlocs=${ignore_altlocs}"

        local extra_args=()
        [ -n "$n_conformers" ] && extra_args+=(--n-conformers "$n_conformers")
        [ -n "$n_torsion_confs" ] && extra_args+=(--n-torsion-confs "$n_torsion_confs")
        [ -n "$radius_step" ] && extra_args+=(--radius-step "$radius_step")
        [ -n "$angle_step" ] && extra_args+=(--angle-step "$angle_step")
        [ -n "$clash_scale" ] && extra_args+=(--clash-scale "$clash_scale")
        [ -n "$polar_clash_scale" ] && extra_args+=(--polar-clash-scale "$polar_clash_scale")
        [ -n "$exclude_resn" ] && extra_args+=(--exclude-resn ${exclude_resn//,/ })
        [ -n "$max_candidates_per_conformer" ] && extra_args+=(--max-candidates-per-conformer "$max_candidates_per_conformer")
        [ -n "$seed" ] && extra_args+=(--seed "$seed")
        [ -n "$mask_spacing" ] && extra_args+=(--mask-spacing "$mask_spacing")
        [ -n "$sampling_radius" ] && extra_args+=(--sampling-radius "$sampling_radius")
        [ -n "$altloc" ] && extra_args+=(--altloc "$altloc")
        if [ "$ignore_altlocs" = "none" ]; then
            extra_args+=(--ignore-altlocs)
        elif [ -n "$ignore_altlocs" ]; then
            extra_args+=(--ignore-altlocs ${ignore_altlocs//,/ })
        fi

        echo "Processing [${dataset} ${label}]: sampling around ${LIGAND_RESN} ${chain} ${resnum}${altloc:+ altloc ${altloc}}"
        generate_conformers "$structure" "$CSV_FILE" "$dataset" "$chain" "$resnum" "$out_dir" \
            "${extra_args[@]}" \
            > "${out_dir}/generate_conformers_log.txt" 2>&1
        local status=$?
        if [ $status -ne 0 ]; then
            echo "ERROR [${dataset} ${label}]: generate_conformers failed with exit code ${status} (see ${out_dir}/generate_conformers_log.txt)"
            continue
        fi
        echo "Completed [${dataset} ${label}]: $(tail -n 1 "${out_dir}/generate_conformers_log.txt")"
    done

    conda_deactivate
}
export -f conformers_process_dataset

do_conformers() {
    echo "Starting run"
    local start_time=$(date +%s)
    printf '%s\n' "${DATASETS[@]}" | parallel -j "$NUM_PARALLEL_DEFAULT" conformers_process_dataset {}
    echo "All jobs completed"
    print_elapsed "$start_time"
}

######################################################################
# Stage 2: per-atom EDIA scoring of the null conformers and reference ligands (2a) and mean
# atom EDIA histogram (2b), per ligand null
######################################################################
# For every LIG_* null of the dataset's run folder that has conformers: run_edia scores
# conformers.pdb (-> distribution_residue_scores.csv, distribution_atom_scores.csv) and then
# ligand.pdb, the reference ligand conformation (-> reference_residue_scores.csv,
# reference_atom_scores.csv), against the dataset's density map, then plots the null's mean
# atom EDIA histogram (plot_edia_histogram.py). A null counts as scored once all four CSVs and
# both params JSONs exist; run_edia itself skips a part already scored with the same inputs and
# resumes an interrupted one.

# crystal_info <dataset>: the dataset's row of CSV_FILE as "resolution a b c alpha beta gamma
# space_group" (columns found by header name; plain comma-separated fields), nothing if absent.
crystal_info() {
    awk -F, -v ds="$1" '
        NR == 1 { for (i = 1; i <= NF; i++) { h = $i; gsub(/[ \r"]/, "", h); col[h] = i }; next }
        { gsub(/\r/, "") }
        $col["dataset"] == ds {
            sg = $col["space_group"]; gsub(/^ +| +$/, "", sg)
            print $col["resolution"] + 0, $col["a"] + 0, $col["b"] + 0, $col["c"] + 0,
                  $col["alpha"] + 0, $col["beta"] + 0, $col["gamma"] + 0, sg
            exit
        }' "$CSV_FILE"
}
export -f crystal_info

# extract_ligand <structure_pdb> <chain> <resi> [altloc]: the LIG residue's heavy atoms (the
# blank-altloc atoms plus the altloc's; every atom if no altloc is given), altloc column blanked,
# as a PDB on stdout - the same conformation generate_conformers' read_ligand takes.
extract_ligand() {
    awk -v resn="$LIGAND_RESN" -v chain="$2" -v resi="$3" -v alt="$4" '
        /^(ATOM  |HETATM)/ {
            r = substr($0, 18, 3); gsub(/ /, "", r)
            a = substr($0, 17, 1)
            el = substr($0, 77, 2); gsub(/ /, "", el)
            if (el == "") { n = substr($0, 13, 4); gsub(/ /, "", n); el = substr(n, 1, 1) }
            el = toupper(el)
            if (r == resn && substr($0, 22, 1) == chain && substr($0, 23, 4) + 0 == resi + 0 \
                    && el != "H" && el != "D" && (alt == "" || a == " " || a == alt))
                print substr($0, 1, 16) " " substr($0, 18)
        }
        END { print "END" }' "$1"
}
export -f extract_ligand

edia_process_dataset() {
    local dataset=$1
    local dataset_dir="${DATASETS_DIR}/${dataset}"
    local structure="${dataset_dir}/${STRUCTURE_PDB_PATTERN//\{dataset\}/${dataset}}"
    local map="${dataset_dir}/${MAP_NAME_PATTERN//\{dataset\}/${dataset}}"

    if [ ! -f "$map" ]; then
        echo "Warning [${dataset}]: density map not found: ${map}, skipping EDIA."
        return 1
    fi
    # Unit cell, space group and resolution: always from CSV_FILE, passed to run_edia.
    local resolution cell_a cell_b cell_c cell_alpha cell_beta cell_gamma space_group
    read -r resolution cell_a cell_b cell_c cell_alpha cell_beta cell_gamma space_group \
        < <(crystal_info "$dataset")
    if [ -z "$space_group" ]; then
        echo "ERROR [${dataset}]: no row (or no space_group) in ${CSV_FILE}, skipping EDIA."
        return 1
    fi
    local pockets=()
    mapfile -t pockets < <(find "${dataset_dir}/${conformer_run_name}" -mindepth 1 -maxdepth 1 \
        -type d -name "${LIGAND_RESN}_*" 2>/dev/null | sort)
    if [ ${#pockets[@]} -eq 0 ]; then
        echo "Warning [${dataset}]: no ${conformer_run_name}/${LIGAND_RESN}_* ligand nulls, skipping EDIA."
        return 1
    fi

    conda_activate "$CONDA_ENV_NULL"

    local pocket label status scored smiles lig_chain lig_resi lig_alt
    for pocket in "${pockets[@]}"; do
        label="${pocket##*/}"
        scored=0
        if [ ! -f "${pocket}/conformers.pdb" ]; then
            echo "Skipping [${dataset} ${label}]: no conformers.pdb."
            continue
        fi

        if [ "$overwrite" -ne 1 ] && files_exist \
                "${pocket}/distribution_residue_scores.csv" "${pocket}/distribution_atom_scores.csv" \
                "${pocket}/distribution_residue_scores_params.json" \
                "${pocket}/reference_residue_scores.csv" "${pocket}/reference_atom_scores.csv" \
                "${pocket}/reference_residue_scores_params.json"; then
            echo "Skipping [${dataset} ${label}]: EDIA already complete (distribution_* and reference_* scores exist)."
        else
            # The reference ligand conformation: ligand.pdb from generate_conformers, or, for
            # nulls sampled before it wrote one, extracted from the structure (the folder
            # label's residue: blank-altloc plus its altloc's heavy atoms, altloc blanked).
            if [ ! -s "${pocket}/ligand.pdb" ]; then
                if [[ ! "$label" =~ ^${LIGAND_RESN}_(.)(-?[0-9]+)(_(.))?$ ]]; then
                    echo "ERROR [${dataset} ${label}]: no ligand.pdb, and the folder name is not ${LIGAND_RESN}_<chain><resi>[_<altloc>]"
                    continue
                fi
                lig_chain="${BASH_REMATCH[1]}" lig_resi="${BASH_REMATCH[2]}" lig_alt="${BASH_REMATCH[4]}"
                extract_ligand "$structure" "$lig_chain" "$lig_resi" "$lig_alt" > "${pocket}/ligand.pdb"
                if ! grep -q '^HETATM\|^ATOM' "${pocket}/ligand.pdb"; then
                    echo "ERROR [${dataset} ${label}]: no ${LIGAND_RESN} ${lig_chain}${lig_resi}${lig_alt:+ alt ${lig_alt}} heavy atoms in ${structure}"
                    rm -f "${pocket}/ligand.pdb"
                    continue
                fi
                echo "Wrote ${pocket}/ligand.pdb from ${structure##*/}"
            fi
            smiles=$(python -c 'import json, sys; print(json.load(open(sys.argv[1]))["ligand"]["smiles"])' \
                "${pocket}/conformer_summary.json" 2>/dev/null)
            if [ -z "$smiles" ]; then
                echo "ERROR [${dataset} ${label}]: no ligand SMILES in ${pocket}/conformer_summary.json"
                continue
            fi
            write_params_txt "${pocket}/edia_params.txt" \
                "map=${map}" \
                "map_pattern=${MAP_NAME_PATTERN}" \
                "structure=${structure}" \
                "fragments_csv=${CSV_FILE}" \
                "cell=${cell_a} ${cell_b} ${cell_c} ${cell_alpha} ${cell_beta} ${cell_gamma}" \
                "space_group=${space_group}" \
                "resolution=${resolution}" \
                "smiles=${smiles}" \
                "exclude_resn=${LIGAND_RESN}${exclude_resn:+,${exclude_resn}}" \
                "ediascorer=${EDIASCORER}" \
                "container_image=${EDIA_IMAGE}" \
                "crop_radius=${edia_crop_radius}" \
                "workers=${edia_workers}"
            local common_args=(--cell "$cell_a" "$cell_b" "$cell_c" "$cell_alpha" "$cell_beta" "$cell_gamma"
                               --space-group "$space_group" --resolution "$resolution"
                               --smiles "$smiles" --resn "$LIGAND_RESN"
                               --exclude-resn "$LIGAND_RESN" ${exclude_resn//,/ }
                               --ediascorer "$EDIASCORER" --container-image "$EDIA_IMAGE")
            [ "$overwrite" -eq 1 ] && common_args+=(--overwrite)
            [ -n "$edia_crop_radius" ] && common_args+=(--crop-radius "$edia_crop_radius")
            [ -n "$edia_workers" ] && common_args+=(--workers "$edia_workers")
            echo "Processing [${dataset} ${label}]: per-atom EDIA of conformers.pdb and ligand.pdb against ${map##*/}"
            # The log is appended to: an interrupted null resumes, and its earlier progress
            # stays readable.
            status=0
            local what
            for what in distribution reference; do
                local ligand_pdb="${pocket}/conformers.pdb"
                [ "$what" = reference ] && ligand_pdb="${pocket}/ligand.pdb"
                run_edia "$ligand_pdb" "$structure" "$map" "$pocket" \
                    --residue-scores "${what}_residue_scores.csv" \
                    --atom-scores "${what}_atom_scores.csv" \
                    "${common_args[@]}" >> "${pocket}/run_edia_log.txt" 2>&1
                status=$?
                if [ $status -ne 0 ]; then
                    echo "ERROR [${dataset} ${label}]: run_edia (${what}, ${ligand_pdb##*/}) failed with exit code ${status} (see ${pocket}/run_edia_log.txt)"
                    break
                fi
                echo "Completed [${dataset} ${label}] ${what}: $(tail -n 1 "${pocket}/run_edia_log.txt")"
            done
            [ $status -ne 0 ] && continue
            scored=1
        fi

        # Replotted whenever the pocket was (re)scored just now.
        if [ "$overwrite" -ne 1 ] && [ "$replot" -ne 1 ] && [ "$scored" -ne 1 ] \
                && files_exist "${pocket}/mean_atom_edia_histogram.png"; then
            echo "Skipping [${dataset} ${label}]: mean atom EDIA histogram exists (pass --replot to redo)."
        elif ! python "$PLOT_EDIA_HISTOGRAM_PY" "$dataset_dir" "$conformer_run_name" "$label"; then
            echo "ERROR [${dataset} ${label}]: plot_edia_histogram failed"
        fi
    done

    conda_deactivate
}
export -f edia_process_dataset

do_edia() {
    if ! podman image exists "$EDIA_IMAGE"; then
        echo "Error: ${EDIA_IMAGE} is not in the local podman image store (podman pull it first)." >&2
        return 1
    fi
    echo "Starting run"
    local start_time=$(date +%s)
    printf '%s\n' "${DATASETS[@]}" | parallel -j "$NUM_PARALLEL_DEFAULT" edia_process_dataset {}
    echo "All jobs completed"
    print_elapsed "$start_time"
}

######################################################################
# Stages 1b-1d and 3: pooled plots
######################################################################
# run_pooled_plot <script> [extra args...]: runs one snediam_analysis_scripts plot over every
# dataset of the run.
run_pooled_plot() {
    local script="$1"
    shift
    conda_activate "$CONDA_ENV_NULL"
    echo "Starting run"
    local start_time=$(date +%s)
    python "$script" \
        "$conformer_run_name" \
        --datasets-dir "$DATASETS_DIR" --datasets-file "$DATASETS_FILE" \
        --graphs-dir "${GRAPHS_DIR}/${conformer_run_name}" \
        "$@"
    local status=$?
    echo "All jobs completed"
    print_elapsed "$start_time"
    conda_deactivate
    return $status
}

ensemble_volumes_outputs_exist() {
    files_exist "${GRAPHS_DIR}/${conformer_run_name}/ensemble_volumes.png"
}
do_plot_ensemble_volumes() {
    run_pooled_plot "$PLOT_ENSEMBLE_VOLUMES_PY"
}

conformer_acceptance_outputs_exist() {
    files_exist "${GRAPHS_DIR}/${conformer_run_name}/conformer_acceptance.png"
}
do_plot_conformer_acceptance() {
    run_pooled_plot "$PLOT_CONFORMER_ACCEPTANCE_PY"
}

# Stage 3: pooled evaluation of the EDIA outputs (stage 2).
edia_evaluation_outputs_exist() {
    local name
    for name in structure_mean_atom_edia null_median_mean_atom_edia null_mean_mean_atom_edia \
                structure_mean_atom_edia_percentile; do
        files_exist "${GRAPHS_DIR}/${conformer_run_name}/${name}.png" || return 1
    done
}
do_plot_edia_evaluation() {
    run_pooled_plot "$PLOT_EDIA_EVALUATION_PY"
}

reference_coverage_outputs_exist() {
    files_exist "${GRAPHS_DIR}/${conformer_run_name}/reference_coverage.png"
}
do_plot_reference_coverage() {
    run_pooled_plot "$PLOT_REFERENCE_COVERAGE_PY"
}

# --- Drive the requested stages, in order ---
overall_start=$(date +%s)

run_step "Stage 1a: generate_conformers (${conformer_run_name})" do_conformers
run_step_pooled_replot "Stage 1b: plot_ensemble_volumes (${conformer_run_name})" \
    ensemble_volumes_outputs_exist do_plot_ensemble_volumes
run_step_pooled_replot "Stage 1c: plot_conformer_acceptance (${conformer_run_name})" \
    conformer_acceptance_outputs_exist do_plot_conformer_acceptance
run_step_pooled_replot "Stage 1d: plot_reference_coverage (${conformer_run_name})" \
    reference_coverage_outputs_exist do_plot_reference_coverage
run_step "Stage 2: per-atom EDIA scoring of the null conformers and reference ligands, mean atom EDIA histograms (${conformer_run_name}, map ${MAP_NAME_PATTERN})" do_edia
run_step_pooled_replot "Stage 3: EDIA evaluation plots (${conformer_run_name})" \
    edia_evaluation_outputs_exist do_plot_edia_evaluation

echo "========= generate_null.sh complete ========="
print_elapsed "$overall_start" "Total"
