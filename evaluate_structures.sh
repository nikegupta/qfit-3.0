#!/bin/bash
#
# evaluate_structures.sh - driver for evaluating predicted structures against the null
# distributions of generate_null.sh (collected by prep_distribution.sh): for every prediction
# matching PREDICTION_PATTERN in each dataset's test folder, which reference ligand it
# predicts, its ligand's EDIA in the reference's density, and that score's percentile in the
# reference ligand's null distribution.
# Takes two positional args: the generate_null.sh run whose nulls are evaluated against (as
# collected by prep_distribution.sh), and the name of this evaluation run:
#   distribution_run run_name
#
# Inputs (configuration block below):
#   TEST_STRUCTURES_DIR/<dataset>/<PREDICTION_PATTERN>   the predictions ({dataset} replaced,
#                                                         then globbed)
#   DISTRIBUTION_DIR/<dataset>/                           prep_distribution.sh output: the
#       <dataset>_001.pdb, <dataset>_refined_2fofc.ccp4   reference structure and map, and
#       <DISTRIBUTION_RUN>/LIG_<chain><resi>_<alt>/       the run's nulls:
#           distribution_residue_scores.csv, reference_residue_scores.csv
#   CSV_FILE                                              unit cell, space group, resolution
#                                                         and SMILES of each dataset
#
# Runs, in order (per prediction:
# TEST_STRUCTURES_DIR/<dataset>/<distribution_run>/<run_name>/<prediction stem>/):
#   STAGE_1: match_ligand          -> match.txt (e.g. LIG_A701_B), match.json, ligand.pdb
#       The prediction is CA-superposed onto each reference chain with a LIG residue (sequence
#       alignment; residue numbers need not match) and its ligand matched to the reference
#       ligand conformation with a null whose centroid is nearest the superposed ligand's;
#       ligand.pdb is the prediction's ligand in the reference's frame.
#   STAGE_2: run_edia              -> prediction_residue_scores.csv, prediction_atom_scores.csv
#                                     (+ _params.json), run_edia_log.txt
#       ligand.pdb scored per atom against the reference structure (LIG and --exclude_resn
#       residues removed, every altloc kept) and map, as generate_null.sh stage 2 scores the
#       nulls: cell, space group and resolution from CSV_FILE, bond orders from its SMILES.
#   STAGE_3: plot_prediction_evaluation (pooled) -> GRAPHS_DIR/<distribution_run>/<run_name>/
#       evaluation.csv (one row per prediction: match, score, percentile in the matched null,
#       the reference ligand's score and percentile), prediction_<metric>_percentile.png/.csv
#
# Idempotent: a prediction whose stage outputs exist is skipped (--overwrite redoes them), and
# stage 3 is skipped if its outputs exist (--replot redoes it). Every run's output is logged to
# TEST_DIR/logs/<distribution_run>/<run_name>/log.txt (log_2.txt, ... if it exists).

set -uo pipefail

usage() {
    cat <<EOF
Usage: $0 <distribution_run> <run_name> [-p <num_parallel>] [--dataset <id[,id...]>]
           [--prediction_pattern <pattern>] [--metric <column>] [--overwrite] [--replot]
           [--exclude_resn <resn[,resn...]>] [--edia_crop_radius <float>] [--edia_workers <int>]

Positional:
  distribution_run         generate_null.sh run whose nulls (collected by prep_distribution.sh
                            into DISTRIBUTION_DIR/<dataset>/<distribution_run>/) the predictions
                            are evaluated against.
  run_name                 Name of this evaluation run: outputs go to
                            TEST_STRUCTURES_DIR/<dataset>/<distribution_run>/<run_name>/<prediction>/
                            and GRAPHS_DIR/<distribution_run>/<run_name>/.

Options:
  -p <num_parallel>        Datasets processed at once (GNU parallel). Default: 1.
  --dataset <id[,id...]>   Only these datasets (comma-separated) instead of DATASETS_FILE.
                            Skips the pooled stage 3.
  --prediction_pattern <pattern>  Prediction file names inside TEST_STRUCTURES_DIR/<dataset>/,
                            {dataset} replaced, then globbed. Default: ${PREDICTION_PATTERN}
  --metric <column>        Score ranked in the null distribution. Default: mean_atom_edia.
  --overwrite              Redo every stage's outputs.
  --replot                 Redo stage 3 even if its outputs exist.
  --exclude_resn <resn[,resn...]>  Residues removed from the EDIA target besides LIG.
                            Default: HOH,DMS.
  --edia_crop_radius <float>  EDIA: each ligand is scored against the residues with an atom
                            within this distance (Å). Default: run_edia's (5).
  --edia_workers <int>     EDIA: EDIAscorer calls run at once. Default: 1.

Paths (edit the configuration block of this script):
  TEST_STRUCTURES_DIR  ${TEST_STRUCTURES_DIR}
  DISTRIBUTION_DIR     ${DISTRIBUTION_DIR}
  GRAPHS_DIR           ${GRAPHS_DIR}
  DATASETS_FILE        ${DATASETS_FILE}
  CSV_FILE             ${CSV_FILE}

Example:
  $0 run_21 eval_1 -p 4 --prediction_pattern '{dataset}_seed_42_sample_*_model.pdb'
EOF
    exit 1
}

# --- User-specified configuration: edit these for your environment ---
BASE_DIR="${BASE_DIR:-/home/ngupta/snEDIAm}"
TEST_DIR="${BASE_DIR}/test"
# The structures to test: TEST_STRUCTURES_DIR/<dataset>/<PREDICTION_PATTERN>.
TEST_STRUCTURES_DIR="${TEST_DIR}/datasets"
# prep_distribution.sh output: DISTRIBUTION_DIR/<dataset>/ (structure, map) and
# DISTRIBUTION_DIR/<dataset>/<distribution_run>/ (LIG_* nulls; distribution_run, the first
# positional argument, is the generate_null.sh run).
DISTRIBUTION_DIR="${BASE_DIR}/distribution"
GRAPHS_DIR="${TEST_DIR}/graphs"
DATASETS_FILE="${BASE_DIR}/datasets_test.txt"
CSV_FILE="${BASE_DIR}/pxr_fragments.csv"
SNEDIAM_ANALYSIS_SCRIPTS_DIR="${BASE_DIR}/qfit-3.0/src/snediam_analysis_scripts"
CONDA_SH="/home/ngupta/miniconda3/etc/profile.d/conda.sh"
CONDA_ENV_NULL="snEDIAm"

# Prediction file names inside TEST_STRUCTURES_DIR/<dataset>/ (override: --prediction_pattern).
PREDICTION_PATTERN="{dataset}_seed_42_sample_*_model.pdb"
# The reference structure and map inside DISTRIBUTION_DIR/<dataset>/ (as prep_distribution.sh
# copies them).
STRUCTURE_PDB_PATTERN="{dataset}_001.pdb"
MAP_NAME_PATTERN="{dataset}_refined_2fofc.ccp4"
LIGAND_RESN="LIG"
# EDIAscorer 1.3 executable, and the podman image (glibc >= 2.29, in the local image store) it
# runs in.
EDIASCORER="/home/ngupta/EDIAscorer_1.3.0/EDIAscorer"
EDIA_IMAGE="registry.access.redhat.com/ubi9/ubi"

# --- Derived paths ---
PLOT_PREDICTION_EVALUATION_PY="${SNEDIAM_ANALYSIS_SCRIPTS_DIR}/plot_prediction_evaluation.py"

# --- Argument parsing ---
if [ $# -eq 0 ]; then
    usage
fi
orig_args=("$@")

DISTRIBUTION_RUN=""
run_name=""
num_parallel=""
overwrite=0
replot=0
dataset_arg=""
DATASET_OVERRIDE_FILE=""
prediction_pattern=""
metric=""
exclude_resn=""
# run_edia tunables, empty for its argparse defaults (--crop-radius 5, --workers 1).
edia_crop_radius=""
edia_workers=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        -p)
            num_parallel="$2"
            shift 2
            ;;
        --dataset)
            dataset_arg="$2"
            shift 2
            ;;
        --prediction_pattern)
            prediction_pattern="$2"
            shift 2
            ;;
        --metric)
            metric="$2"
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
        --exclude_resn)
            exclude_resn="$2"
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
            if [ -z "$DISTRIBUTION_RUN" ]; then
                DISTRIBUTION_RUN="$1"
            elif [ -z "$run_name" ]; then
                run_name="$1"
            else
                echo "Unexpected argument: $1"
                usage
            fi
            shift
            ;;
    esac
done

if [ -z "$DISTRIBUTION_RUN" ] || [ -z "$run_name" ]; then
    echo "Error: <distribution_run> and <run_name> are required."
    usage
fi
[ -n "$prediction_pattern" ] && PREDICTION_PATTERN="$prediction_pattern"
# This run's outputs, under each dataset's test folder, GRAPHS_DIR and the logs.
RUN_SUBDIR="${DISTRIBUTION_RUN}/${run_name}"
metric="${metric:-mean_atom_edia}"

for f in "$DATASETS_FILE" "$CSV_FILE" "$CONDA_SH" "$PLOT_PREDICTION_EVALUATION_PY"; do
    if [ ! -f "$f" ]; then
        echo "Error: required file not found: ${f}" >&2
        exit 1
    fi
done
for d in "$TEST_STRUCTURES_DIR" "$DISTRIBUTION_DIR"; do
    if [ ! -d "$d" ]; then
        echo "Error: required directory not found: ${d}$([ "$d" = "$DISTRIBUTION_DIR" ] && echo ' (run prep_distribution.sh first)')" >&2
        exit 1
    fi
done
if [ ! -x "$EDIASCORER" ]; then
    echo "Error: EDIAscorer not found or not executable: ${EDIASCORER}" >&2
    exit 1
fi

# --- Full-run logging: TEST_DIR/logs/<distribution_run>/<run_name>/log.txt (log_2.txt, ...
# never overwriting).
log_dir="${TEST_DIR}/logs/${RUN_SUBDIR}"
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
echo "distribution_run: ${DISTRIBUTION_RUN}"
echo "run_name: ${run_name}"
echo "num_parallel: ${num_parallel:-1}"
echo "overwrite: ${overwrite}"
echo "replot: ${replot}"
echo "dataset: ${dataset_arg:-(all in ${DATASETS_FILE})}"
echo "test_structures_dir: ${TEST_STRUCTURES_DIR}"
echo "prediction_pattern: ${PREDICTION_PATTERN}"
echo "distribution_dir: ${DISTRIBUTION_DIR}/<dataset>/${DISTRIBUTION_RUN}"
echo "outputs: ${TEST_STRUCTURES_DIR}/<dataset>/${RUN_SUBDIR}/<prediction>/"
echo "graphs_dir: ${GRAPHS_DIR}/${RUN_SUBDIR}"
echo "metric: ${metric}"
echo "exclude_resn: ${exclude_resn:-(not set - HOH,DMS)}"
echo "edia_crop_radius: ${edia_crop_radius:-(not set - run_edia default 5)}"
echo "edia_workers: ${edia_workers:-(not set - run_edia default 1)}"

# --dataset override: a temp file listing just the requested dataset(s).
if [ -n "$dataset_arg" ]; then
    DATASET_OVERRIDE_FILE=$(mktemp)
    IFS=',' read -ra _cli_datasets <<< "$dataset_arg"
    for _cli_dataset in "${_cli_datasets[@]}"; do
        _cli_dataset="$(echo -n "$_cli_dataset" | xargs)"
        [ -z "$_cli_dataset" ] && continue
        if [ ! -d "${TEST_STRUCTURES_DIR}/${_cli_dataset}" ]; then
            echo "Error: --dataset given but test structure directory not found: ${TEST_STRUCTURES_DIR}/${_cli_dataset}" >&2
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

mapfile -t DATASETS < <(grep -v '^[[:space:]]*$' "$DATASETS_FILE" | tr -d '\r')
if [ ${#DATASETS[@]} -eq 0 ]; then
    echo "Error: no datasets found in ${DATASETS_FILE}" >&2
    exit 1
fi
NUM_PARALLEL_DEFAULT=${num_parallel:-1}

# Everything the per-dataset *_process_dataset functions need, exported for GNU parallel.
export run_name RUN_SUBDIR overwrite replot metric exclude_resn edia_crop_radius edia_workers
export BASE_DIR TEST_DIR TEST_STRUCTURES_DIR DISTRIBUTION_DIR DISTRIBUTION_RUN GRAPHS_DIR DATASETS_FILE CSV_FILE
export CONDA_SH CONDA_ENV_NULL PREDICTION_PATTERN STRUCTURE_PDB_PATTERN MAP_NAME_PATTERN
export LIGAND_RESN EDIASCORER EDIA_IMAGE

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

# write_params_txt <output_file> <name=value> [<name=value> ...]; an empty value means the
# flag wasn't given, so the underlying script's default applied.
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

# run_step_pooled_replot <description> <check_fn> <run_fn>: a pooled stage, skipped under
# --dataset (it would overwrite the full-run outputs with partial ones), and when check_fn says
# its outputs exist, unless --replot or --overwrite.
run_step_pooled_replot() {
    local desc="$1" check_fn="$2" run_fn="$3"
    if [ -n "$dataset_arg" ]; then
        echo ""
        echo "Skipping ${desc} (pooled; --dataset was given, would overwrite the full-run outputs with partial ones)."
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

# csv_field <dataset> <column>: the dataset's value of a CSV_FILE column (found by header name;
# plain comma-separated fields), nothing if absent.
csv_field() {
    awk -F, -v ds="$1" -v name="$2" '
        NR == 1 { for (i = 1; i <= NF; i++) { h = $i; gsub(/[ \r"]/, "", h); col[h] = i }; next }
        { gsub(/\r/, "") }
        (col["dataset"] && $col["dataset"] == ds && col[name]) {
            v = $col[name]; gsub(/^ +| +$/, "", v); print v; exit
        }' "$CSV_FILE"
}
export -f csv_field

# predictions <dataset>: the dataset's prediction files, one per line, sorted.
predictions() {
    local dataset=$1
    local pattern="${PREDICTION_PATTERN//\{dataset\}/${dataset}}"
    local f
    for f in "${TEST_STRUCTURES_DIR}/${dataset}"/${pattern}; do
        [ -f "$f" ] && echo "$f"
    done
}
export -f predictions

# run_per_dataset <function>: runs a *_process_dataset function over every dataset, -p at once.
run_per_dataset() {
    local fn="$1"
    echo "Starting run"
    local start_time=$(date +%s)
    printf '%s\n' "${DATASETS[@]}" | parallel -j "$NUM_PARALLEL_DEFAULT" "$fn" {}
    echo "All jobs completed"
    print_elapsed "$start_time"
}

######################################################################
# Stage 1: match_ligand
######################################################################
# For every prediction: the reference ligand conformation (with a null in
# DISTRIBUTION_DIR/<dataset>/<DISTRIBUTION_RUN>/) it predicts, and its ligand in the reference's frame.

match_process_dataset() {
    local dataset=$1
    local dist_dir="${DISTRIBUTION_DIR}/${dataset}/${DISTRIBUTION_RUN}"
    local reference="${DISTRIBUTION_DIR}/${dataset}/${STRUCTURE_PDB_PATTERN//\{dataset\}/${dataset}}"
    if [ ! -f "$reference" ]; then
        echo "Warning [${dataset}]: no reference structure ${reference} (run prep_distribution.sh), skipping."
        return 1
    fi
    local files=()
    mapfile -t files < <(predictions "$dataset")
    if [ ${#files[@]} -eq 0 ]; then
        echo "Warning [${dataset}]: no predictions matching ${TEST_STRUCTURES_DIR}/${dataset}/${PREDICTION_PATTERN//\{dataset\}/${dataset}}, skipping."
        return 1
    fi

    conda_activate "$CONDA_ENV_NULL"
    local f stem out_dir status
    for f in "${files[@]}"; do
        stem=$(basename "$f" .pdb)
        out_dir="${TEST_STRUCTURES_DIR}/${dataset}/${RUN_SUBDIR}/${stem}"
        if [ "$overwrite" -ne 1 ] && files_exist "${out_dir}/match.txt" "${out_dir}/match.json" \
                "${out_dir}/ligand.pdb"; then
            echo "Skipping [${dataset} ${stem}]: already matched ($(cat "${out_dir}/match.txt"))."
            continue
        fi
        mkdir -p "$out_dir"
        rm -f "${out_dir}/match.txt"
        python -m qfit.command_line.match_ligand "$f" "$reference" "$out_dir" \
            --distribution-dir "$dist_dir" --ligand-resn "$LIGAND_RESN" \
            > "${out_dir}/match_ligand_log.txt" 2>&1
        status=$?
        if [ $status -ne 0 ]; then
            echo "ERROR [${dataset} ${stem}]: match_ligand failed with exit code ${status}: $(tail -n 1 "${out_dir}/match_ligand_log.txt")"
            continue
        fi
        echo "Matched [${dataset} ${stem}]: $(tail -n 1 "${out_dir}/match_ligand_log.txt")"
    done
    conda_deactivate
}
export -f match_process_dataset

######################################################################
# Stage 2: run_edia
######################################################################
# For every matched prediction: per-atom EDIA of its superposed ligand (ligand.pdb) against the
# reference structure and map, as generate_null.sh stage 2 scores the nulls.

edia_process_dataset() {
    local dataset=$1
    local dist_dir="${DISTRIBUTION_DIR}/${dataset}/${DISTRIBUTION_RUN}"
    local reference="${DISTRIBUTION_DIR}/${dataset}/${STRUCTURE_PDB_PATTERN//\{dataset\}/${dataset}}"
    local map="${DISTRIBUTION_DIR}/${dataset}/${MAP_NAME_PATTERN//\{dataset\}/${dataset}}"
    if [ ! -f "$reference" ] || [ ! -f "$map" ]; then
        echo "Warning [${dataset}]: no reference structure or map in ${DISTRIBUTION_DIR}/${dataset} (run prep_distribution.sh), skipping."
        return 1
    fi
    # Unit cell, space group, resolution and SMILES: always from CSV_FILE, passed to run_edia.
    local resolution cell smiles space_group
    resolution=$(csv_field "$dataset" resolution)
    cell=()
    local k
    for k in a b c alpha beta gamma; do
        cell+=("$(csv_field "$dataset" "$k")")
    done
    space_group=$(csv_field "$dataset" space_group)
    smiles=$(csv_field "$dataset" smiles)
    if [ -z "$resolution" ] || [ -z "$space_group" ] || [ -z "$smiles" ] || [ ${#cell[@]} -ne 6 ] \
            || [[ " ${cell[*]} " == *"  "* ]]; then
        echo "ERROR [${dataset}]: incomplete row (resolution, a-gamma, space_group, smiles) in ${CSV_FILE}, skipping."
        return 1
    fi
    local files=()
    mapfile -t files < <(predictions "$dataset")

    conda_activate "$CONDA_ENV_NULL"
    local excluded=()
    IFS=',' read -ra excluded <<< "${exclude_resn:-HOH,DMS}"
    local common_args=(--cell "${cell[@]}" --space-group "$space_group" --resolution "$resolution"
                       --smiles "$smiles" --resn "$LIGAND_RESN"
                       --exclude-resn "$LIGAND_RESN" "${excluded[@]}"
                       --ediascorer "$EDIASCORER" --container-image "$EDIA_IMAGE")
    [ "$overwrite" -eq 1 ] && common_args+=(--overwrite)
    [ -n "$edia_crop_radius" ] && common_args+=(--crop-radius "$edia_crop_radius")
    [ -n "$edia_workers" ] && common_args+=(--workers "$edia_workers")
    local f stem out_dir status
    for f in "${files[@]}"; do
        stem=$(basename "$f" .pdb)
        out_dir="${TEST_STRUCTURES_DIR}/${dataset}/${RUN_SUBDIR}/${stem}"
        if [ ! -f "${out_dir}/match.txt" ] || [ ! -f "${out_dir}/ligand.pdb" ]; then
            echo "Skipping [${dataset} ${stem}]: not matched (stage 1)."
            continue
        fi
        if [ "$overwrite" -ne 1 ] && files_exist "${out_dir}/prediction_residue_scores.csv" \
                "${out_dir}/prediction_atom_scores.csv" \
                "${out_dir}/prediction_residue_scores_params.json"; then
            echo "Skipping [${dataset} ${stem}]: EDIA already complete."
            continue
        fi
        write_params_txt "${out_dir}/edia_params.txt" \
            "reference=${reference}" "map=${map}" "fragments_csv=${CSV_FILE}" \
            "cell=${cell[*]}" "space_group=${space_group}" "resolution=${resolution}" \
            "smiles=${smiles}" "exclude_resn=${LIGAND_RESN},${exclude_resn:-HOH,DMS}" \
            "ediascorer=${EDIASCORER}" "container_image=${EDIA_IMAGE}" \
            "crop_radius=${edia_crop_radius}" "workers=${edia_workers}"
        echo "Processing [${dataset} ${stem}]: per-atom EDIA of its $(cat "${out_dir}/match.txt") match against ${map##*/}"
        run_edia "${out_dir}/ligand.pdb" "$reference" "$map" "$out_dir" \
            --residue-scores prediction_residue_scores.csv \
            --atom-scores prediction_atom_scores.csv \
            "${common_args[@]}" >> "${out_dir}/run_edia_log.txt" 2>&1
        status=$?
        if [ $status -ne 0 ]; then
            echo "ERROR [${dataset} ${stem}]: run_edia failed with exit code ${status} (see ${out_dir}/run_edia_log.txt)"
            continue
        fi
        echo "Completed [${dataset} ${stem}]: $(tail -n 1 "${out_dir}/run_edia_log.txt")"
    done
    conda_deactivate
}
export -f edia_process_dataset

do_match() {
    run_per_dataset match_process_dataset
}

do_edia() {
    if ! podman image exists "$EDIA_IMAGE"; then
        echo "Error: ${EDIA_IMAGE} is not in the local podman image store (podman pull it first)." >&2
        return 1
    fi
    run_per_dataset edia_process_dataset
}

######################################################################
# Stage 3: pooled evaluation
######################################################################
evaluation_outputs_exist() {
    files_exist "${GRAPHS_DIR}/${RUN_SUBDIR}/evaluation.csv" \
        "${GRAPHS_DIR}/${RUN_SUBDIR}/prediction_${metric,,}_percentile.png" \
        "${GRAPHS_DIR}/${RUN_SUBDIR}/prediction_vs_reference_${metric,,}.png" \
        "${GRAPHS_DIR}/${RUN_SUBDIR}/prediction_${metric,,}_by_reference.png"
}

do_evaluation() {
    conda_activate "$CONDA_ENV_NULL"
    local start_time=$(date +%s)
    python "$PLOT_PREDICTION_EVALUATION_PY" "$run_name" \
        --test-dir "$TEST_STRUCTURES_DIR" --distribution-dir "$DISTRIBUTION_DIR" \
        --distribution-run "$DISTRIBUTION_RUN" \
        --datasets-file "$DATASETS_FILE" --graphs-dir "${GRAPHS_DIR}/${RUN_SUBDIR}" \
        --metric "$metric"
    local status=$?
    print_elapsed "$start_time"
    conda_deactivate
    return $status
}

# --- Main ---
script_start=$(date +%s)
run_step "Stage 1: match_ligand (${RUN_SUBDIR}, ${PREDICTION_PATTERN})" do_match
run_step "Stage 2: per-atom EDIA of the predicted ligands (${RUN_SUBDIR}, map ${MAP_NAME_PATTERN})" do_edia
run_step_pooled_replot "Stage 3: percentiles in the matched nulls (${RUN_SUBDIR})" \
    evaluation_outputs_exist do_evaluation
print_elapsed "$script_start" "evaluate_structures.sh"
