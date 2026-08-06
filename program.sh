#!/bin/bash
#
# program.sh - combined driver for the full ligand-fitting/PLACER/RSR pipeline.
#takes in 6 positional args corresponding to the stages of the pipeline:
#run_name, placer_run_name, filter_run_name, placer2_run_name, filter2_run_name, final_run_name
#
# Runs, in order:
#   0. calc_apo_rscc                    -> <dataset>/<dataset>-aligned-structure_rscc.csv
#      + calc_ref_set_rscc (only with -c) -> REF_SET/<dataset>/<REF_SET_PDB_PATTERN%.pdb>_rscc.csv
#   1. fit_ligand                       -> <run_name>/
#   2. placer + rsr_placer              -> <run_name>/<placer_run_name>/
#   3. filter + rsr_backbone
#      + calc_backbone_refined_rscc     -> .../<filter_run_name>/
#      + (only with -c) plot_lig_vs_ref_filter1, plot_residues_vs_ref_backbone
#                                        -> GRAPHS_DIR/<run_name>/.../<filter_run_name>/
#   4. placer2 + rsr_placer2            -> .../<placer2_run_name>/
#   5. filter2                          -> .../<filter2_run_name>/
#      + (only with -c) plot_lig_vs_ref_filter2
#                                        -> GRAPHS_DIR/<run_name>/.../<filter2_run_name>/
#   6. build_final + rsr_final
#      + calc_final_refined_rscc        -> .../<final_run_name>/
#      + (only with -c) plot_residues_vs_ref_final
#                                        -> GRAPHS_DIR/<run_name>/.../<final_run_name>/
#   7. analysis_scripts/*.py            -> .../<final_run_name>/graphs/
#      (cluster-rep and per-residue RSCC plots, computed independently per
#      dataset only runs once <final_run_name> is given and stage 6's output exists for every dataset)
#
#
# Modularity: pass only as many of the six run-name arguments as you want to
# run through (e.g. just <run_name> <placer_run_name> <filter_run_name> stops
# after stage 3). Each stage's directory tree is nested under the previous
# stage's, so re-running with a new name at any point (e.g. a new
# filter_run_name under an existing run_name/placer_run_name) naturally
# branches off the old results without touching them. Stage 0 is
# dataset-scoped rather than run-name-scoped, so it always runs (skipping
# per-dataset once that dataset's apo RSCC csv exists) regardless of which
# run-name arguments are given.
#
# Idempotency: before running a stage, its output directory
# (<run_name>/.../<stage_run_name>) is checked for every dataset listed in
# datasets.txt. If it already exists for all of them, the stage (and its
# associated RSR/RSCC sub-steps) is skipped entirely so previous runs are
# never overwritten. If you want to redo a stage, use a new *_run_name for
# it (and everything downstream will naturally run fresh too, since its
# nested path is new).

set -uo pipefail

usage() {
    cat <<EOF
Usage: $0 <run_name> [placer_run_name [filter_run_name [placer2_run_name [filter2_run_name [final_run_name]]]]]
           [-n <num_placer_confs>] [-n2 <num_placer2_confs>] [-g <gpu_ids>] [-p <num_parallel>] [-c]

Only <run_name> is required. Supplying fewer than all six names runs only
that many stages of the pipeline (see header comment for the stage list).

Options:
  -n <num_placer_confs>    Number of PLACER conformers for round 1 (placer -n). Default: 1000
  -n2 <num_placer2_confs>  Number of PLACER conformers for round 2 (placer2 -n). Default: 1000
  -g <gpu_ids>             Comma-separated GPU ids for both PLACER rounds. Default: 0
  -p <num_parallel>        CPU parallelism for every non-PLACER stage (calc_apo_rscc, fit_ligand,
                            rsr_placer, filter, rsr_backbone, calc_backbone_refined_rscc, rsr_placer2,
                            filter2, build_final, rsr_final, calc_final_refined_rscc). Default: 1
  -c                       Also compare results to the reference set (REF_SET). Runs
                            calc_ref_set_rscc as stage 0b: per dataset, computes RSCC of
                            REF_SET/<dataset>/<REF_SET_PDB_PATTERN>, skipping any dataset whose
                            output csv already exists.

Examples:
  $0 run_1 placer_1 filter_1 placer2_1 filter2_1 final_1
  $0 run_1 placer_1 filter_1
  $0 run_1 placer_1 filter_2 placer2_1 filter2_1 final_1 -n 1000 -n2 500 -g 0,1
  $0 run_1 placer_1 filter_1 placer2_1 filter2_1 final_1 -c
EOF
    exit 1
}

# --- User-specified configuration: edit these for your environment ---
BASE_DIR="/home/ngupta/main/program_claude"
CSV_FILE="${BASE_DIR}/pxr_fragments.csv"
LIG_PDB_DIR="${BASE_DIR}/pdb_final_geometry"
CONDA_SH="/home/ngupta/miniconda3/etc/profile.d/conda.sh"
CONDA_ENV_QFIT="qfit_bs"
CONDA_ENV_PLACER="placer_env"
CONDA_ENV_RSR="rsr"
CONDA_ENV_EVAL="biotite"
RUN_PLACER_PY="/home/ngupta/PLACER/PLACER/run_PLACER.py"

# Only used when -c is given: reference_set/<dataset>/ subfolders (one per
# datasets.txt entry) holding a reference structure to compare RSCC against.
REF_SET="${BASE_DIR}/reference_set"
REF_SET_PDB_PATTERN="{dataset}-pandda-model.pdb"

# --- Derived paths: assumed to live at fixed locations under BASE_DIR ---
DATASETS_DIR="${BASE_DIR}/datasets"
DATASETS_FILE="${BASE_DIR}/datasets.txt"
RSR_SCRIPTS_DIR="${BASE_DIR}/rsr_scripts"
RSR_SCRIPT_LIGAND="${RSR_SCRIPTS_DIR}/real_space_refine.py"
RSR_SCRIPT_PROTEIN="${RSR_SCRIPTS_DIR}/real_space_refine_protein.py"
RSR_SCRIPT_FINAL="${RSR_SCRIPTS_DIR}/real_space_refine_final.py"
ANALYSIS_SCRIPTS_DIR="${BASE_DIR}/analysis_scripts"
PLOT_CLUSTER_REPS_PY="${ANALYSIS_SCRIPTS_DIR}/plot_cluster_reps_rscc.py"
AGGREGATE_PROTEIN_RSCC_PY="${ANALYSIS_SCRIPTS_DIR}/aggregate_protein_rscc.py"
AGGREGATE_LIG_RSCC_PY="${ANALYSIS_SCRIPTS_DIR}/aggregate_lig_rscc.py"
PLOT_LIG_VS_REF_FILTER1_PY="${ANALYSIS_SCRIPTS_DIR}/plot_lig_vs_ref_filter1.py"
PLOT_LIG_VS_REF_FILTER2_PY="${ANALYSIS_SCRIPTS_DIR}/plot_lig_vs_ref_filter2.py"
PLOT_RESIDUES_VS_REF_BACKBONE_PY="${ANALYSIS_SCRIPTS_DIR}/plot_residues_vs_ref_backbone.py"
PLOT_RESIDUES_VS_REF_FINAL_PY="${ANALYSIS_SCRIPTS_DIR}/plot_residues_vs_ref_final.py"
GRAPHS_DIR="${BASE_DIR}/graphs"

for f in "$DATASETS_FILE" "$CSV_FILE" "$RSR_SCRIPT_LIGAND" "$RSR_SCRIPT_PROTEIN" "$RSR_SCRIPT_FINAL" \
         "$RUN_PLACER_PY" "$PLOT_CLUSTER_REPS_PY" "$AGGREGATE_PROTEIN_RSCC_PY" "$AGGREGATE_LIG_RSCC_PY" \
         "$PLOT_LIG_VS_REF_FILTER1_PY" "$PLOT_LIG_VS_REF_FILTER2_PY" \
         "$PLOT_RESIDUES_VS_REF_BACKBONE_PY" "$PLOT_RESIDUES_VS_REF_FINAL_PY"; do
    if [ ! -f "$f" ]; then
        echo "Error: required file not found: ${f}" >&2
        exit 1
    fi
done
for d in "$DATASETS_DIR" "$LIG_PDB_DIR"; do
    if [ ! -d "$d" ]; then
        echo "Error: required directory not found: ${d}" >&2
        exit 1
    fi
done

# --- Argument parsing ---
if [ $# -eq 0 ]; then
    usage
fi

run_name=""
placer_run_name=""
filter_run_name=""
placer2_run_name=""
filter2_run_name=""
final_run_name=""

num_placer_confs=100
num_placer2_confs=100
gpu_ids=""
num_parallel=""
compare_ref_set=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        -n)
            num_placer_confs="$2"
            shift 2
            ;;
        -n2)
            num_placer2_confs="$2"
            shift 2
            ;;
        -g)
            gpu_ids="$2"
            shift 2
            ;;
        -p)
            num_parallel="$2"
            shift 2
            ;;
        -c)
            compare_ref_set=1
            shift
            ;;
        -h|--help)
            usage
            ;;
        -*)
            echo "Unknown option: $1"
            usage
            ;;
        *)
            if [ -z "$run_name" ]; then
                run_name="$1"
            elif [ -z "$placer_run_name" ]; then
                placer_run_name="$1"
            elif [ -z "$filter_run_name" ]; then
                filter_run_name="$1"
            elif [ -z "$placer2_run_name" ]; then
                placer2_run_name="$1"
            elif [ -z "$filter2_run_name" ]; then
                filter2_run_name="$1"
            elif [ -z "$final_run_name" ]; then
                final_run_name="$1"
            else
                echo "Unexpected argument: $1"
                usage
            fi
            shift
            ;;
    esac
done

if [ -z "$run_name" ]; then
    echo "Error: <run_name> is required."
    usage
fi

if [ "$compare_ref_set" -eq 1 ] && [ ! -d "$REF_SET" ]; then
    echo "Error: -c given but reference set directory not found: ${REF_SET}" >&2
    exit 1
fi

NUM_PARALLEL_DEFAULT=${num_parallel:-1}

IFS=',' read -ra GPU_IDS_ARR <<< "$gpu_ids"
NUM_GPUS=${#GPU_IDS_ARR[@]}

# Everything below this point (dataset names, tunables, consolidated paths)
# needs to be visible inside the per-dataset *_process_dataset functions even
# when GNU parallel forks them into new subshells, so it all gets exported.
export run_name placer_run_name filter_run_name placer2_run_name filter2_run_name final_run_name
export num_placer_confs num_placer2_confs compare_ref_set
export BASE_DIR DATASETS_DIR DATASETS_FILE CSV_FILE LIG_PDB_DIR
export RSR_SCRIPT_LIGAND RSR_SCRIPT_PROTEIN RSR_SCRIPT_FINAL
export ANALYSIS_SCRIPTS_DIR PLOT_CLUSTER_REPS_PY AGGREGATE_PROTEIN_RSCC_PY AGGREGATE_LIG_RSCC_PY
export PLOT_LIG_VS_REF_FILTER1_PY PLOT_LIG_VS_REF_FILTER2_PY
export PLOT_RESIDUES_VS_REF_BACKBONE_PY PLOT_RESIDUES_VS_REF_FINAL_PY GRAPHS_DIR
export REF_SET REF_SET_PDB_PATTERN
export CONDA_SH CONDA_ENV_QFIT CONDA_ENV_RSR CONDA_ENV_PLACER CONDA_ENV_EVAL
export RUN_PLACER_PY

# --- Shared lookup file: dataset -> ligand_name resolution, built once from
# CSV_FILE and reused by every stage that needs it (fit_ligand, filter,
# filter2, build_final, calc_backbone_refined_rscc, calc_final_refined_rscc). ---
LOOKUP_FILE=$(mktemp)
trap 'rm -f "$LOOKUP_FILE"' EXIT
tail -n +2 "$CSV_FILE" | while IFS=',' read -r dataset resolution ligand_name; do
    dataset="${dataset//$'\r'/}"
    resolution="${resolution//$'\r'/}"
    ligand_name="${ligand_name//$'\r'/}"
    [ -z "$dataset" ] && continue
    echo "${dataset} ${ligand_name} ${resolution}" >> "$LOOKUP_FILE"
done
export LOOKUP_FILE

# Parses a PDB file's ATOM/HETATM records to find the chain and residue
# number of the LIG residue, and prints it as "CHAIN-LIG-RESNUM"
# (e.g. "C-LIG-1"), matching what --predict_ligand expects. Shared by both
# PLACER rounds.
get_lig_id() {
    local pdb_file=$1
    awk '
        ($1 == "ATOM" || $1 == "HETATM") {
            resname = substr($0, 18, 3); gsub(/ /, "", resname)
            if (resname == "LIG") {
                chain = substr($0, 22, 1); gsub(/ /, "", chain)
                resnum = substr($0, 23, 4); gsub(/ /, "", resnum)
                print chain "-LIG-" resnum
                exit
            }
        }
    ' "$pdb_file"
}
export -f get_lig_id

# --- Helpers ---

# stage_complete <relative_path_under_dataset_dir>
# Returns success (0) only if that path exists as a directory for every
# dataset listed in datasets.txt.
stage_complete() {
    local rel_path="$1"
    local dataset
    while IFS= read -r dataset; do
        [ -z "$dataset" ] && continue
        if [ ! -d "${DATASETS_DIR}/${dataset}/${rel_path}" ]; then
            return 1
        fi
    done < "$DATASETS_FILE"
    return 0
}

# run_step <description> <command...>
run_step() {
    local desc="$1"
    shift
    echo "=== ${desc} ==="
    "$@"
    local status=$?
    if [ $status -ne 0 ]; then
        echo "ERROR: ${desc} failed with exit code ${status}" >&2
        exit "$status"
    fi
}

# conda_activate <env_name>
# Some conda environments (e.g. ones with compiler packages like
# binutils_linux-64) install activate.d hooks that reference variables
# (ADDR2LINE, etc.) without defaults. Those hooks are fine under an
# interactive shell (no `set -u`) but abort this script's `set -uo
# pipefail`. Temporarily relax nounset just for the source/activate calls.
conda_activate() {
    set +u
    source "$CONDA_SH"
    conda activate "$1"
    set -u
}

# conda_deactivate: same nounset relaxation as conda_activate, for
# deactivate.d hooks that restore saved variables.
conda_deactivate() {
    set +u
    conda deactivate
    set -u
}
export -f conda_activate conda_deactivate

# print_elapsed <start_epoch_seconds>
print_elapsed() {
    local start_time="$1"
    local end_time
    end_time=$(date +%s)
    local elapsed=$((end_time - start_time))
    local hours=$((elapsed / 3600))
    local minutes=$(((elapsed % 3600) / 60))
    local seconds=$((elapsed % 60))
    printf "Script took %02d:%02d:%02d (HH:MM:SS)\n" $hours $minutes $seconds
}

######################################################################
# Stage 0: calc_apo_rscc
######################################################################
# Computes the per-residue RSCC of each dataset's baseline
# {dataset}-aligned-structure.pdb (no PLACER/RSR involved) so later analysis
# scripts have an apo baseline to compare backbone/final refined RSCC
# against. This is dataset-scoped, not run-name-scoped, so it runs once per
# dataset regardless of run_name and is skipped per-dataset (not gated by
# stage_complete) whenever its output csv already exists.

calc_apo_rscc_process_dataset() {
    conda_activate "$CONDA_ENV_QFIT"
    shopt -s nullglob

    local dataset=$1
    local dataset_dir="${DATASETS_DIR}/${dataset}"

    local structure="${dataset_dir}/${dataset}-aligned-structure.pdb"
    local output_csv="${dataset_dir}/${dataset}-aligned-structure_rscc.csv"

    if [ -f "$output_csv" ]; then
        echo "Skipping [${dataset}]: ${output_csv} already exists."
        return 0
    fi

    if [ ! -f "$structure" ]; then
        echo "Warning [${dataset}]: aligned structure not found: ${structure}, skipping."
        return 1
    fi

    local lookup=$(grep "^${dataset} " "$LOOKUP_FILE")
    if [ -z "$lookup" ]; then
        echo "Warning: No match found for dataset ${dataset}, skipping."
        return 1
    fi
    local resolution=$(echo "$lookup" | awk '{print $3}')

    echo "Processing ${dataset}: resolution=${resolution}"

    local event_maps=("${dataset_dir}/${dataset}-event_"*)
    if [ ${#event_maps[@]} -eq 0 ]; then
        echo "Warning [${dataset}]: no event maps found matching ${dataset_dir}/${dataset}-event_*, skipping."
        return 1
    fi

    calc_rscc "${structure}" "${event_maps[@]}" "${resolution}" "${output_csv}"

    local calc_exit=$?
    if [ $calc_exit -ne 0 ]; then
        echo "ERROR [${dataset}]: calc_rscc failed on ${structure} with exit code ${calc_exit}"
        return 1
    fi

    echo "Completed [${dataset}]: ${structure} -> ${output_csv}"
}
export -f calc_apo_rscc_process_dataset

do_calc_apo_rscc() {
    conda_activate "$CONDA_ENV_QFIT"

    echo "Starting run"
    local start_time=$(date +%s)
    cat "$DATASETS_FILE" | parallel -j "$NUM_PARALLEL_DEFAULT" calc_apo_rscc_process_dataset {}
    echo "All jobs completed"
    print_elapsed "$start_time"
}

######################################################################
# Stage 0b: calc_ref_set_rscc (only runs when -c is given)
######################################################################
# Computes the per-residue RSCC of each dataset's reference-set structure
# (REF_SET/<dataset>/<REF_SET_PDB_PATTERN>), so later analysis can compare
# the pipeline's results against it. Dataset-scoped like calc_apo_rscc, and
# skipped per-dataset whenever its output csv already exists.

calc_ref_set_rscc_process_dataset() {
    conda_activate "$CONDA_ENV_QFIT"
    shopt -s nullglob

    local dataset=$1
    local dataset_dir="${DATASETS_DIR}/${dataset}"
    local reference_dataset_dir="${REF_SET}/${dataset}"

    if [ ! -d "$reference_dataset_dir" ]; then
        echo "Warning [${dataset}]: reference set folder ${reference_dataset_dir} not found, skipping."
        return 1
    fi

    local pdb_pattern="${REF_SET_PDB_PATTERN//\{dataset\}/${dataset}}"
    local structure="${reference_dataset_dir}/${pdb_pattern}"
    local output_csv="${structure%.pdb}_rscc.csv"

    if [ -f "$output_csv" ]; then
        echo "Skipping [${dataset}]: ${output_csv} already exists."
        return 0
    fi

    if [ ! -f "$structure" ]; then
        echo "Warning [${dataset}]: reference structure not found: ${structure}, skipping."
        return 1
    fi

    local lookup=$(grep "^${dataset} " "$LOOKUP_FILE")
    if [ -z "$lookup" ]; then
        echo "Warning: No match found for dataset ${dataset}, skipping."
        return 1
    fi
    local resolution=$(echo "$lookup" | awk '{print $3}')

    echo "Processing ${dataset}: resolution=${resolution}"

    local event_maps=("${dataset_dir}/${dataset}-event_"*)
    if [ ${#event_maps[@]} -eq 0 ]; then
        echo "Warning [${dataset}]: no event maps found matching ${dataset_dir}/${dataset}-event_*, skipping."
        return 1
    fi

    calc_rscc "${structure}" "${event_maps[@]}" "${resolution}" "${output_csv}"

    local calc_exit=$?
    if [ $calc_exit -ne 0 ]; then
        echo "ERROR [${dataset}]: calc_rscc failed on ${structure} with exit code ${calc_exit}"
        return 1
    fi

    echo "Completed [${dataset}]: ${structure} -> ${output_csv}"
}
export -f calc_ref_set_rscc_process_dataset

do_calc_ref_set_rscc() {
    conda_activate "$CONDA_ENV_QFIT"

    echo "Starting run"
    local start_time=$(date +%s)
    cat "$DATASETS_FILE" | parallel -j "$NUM_PARALLEL_DEFAULT" calc_ref_set_rscc_process_dataset {}
    echo "All jobs completed"
    print_elapsed "$start_time"
}

######################################################################
# Stage 1: fit_ligand
######################################################################

fit_ligand_process_dataset() {
    local dataset=$1

    conda_activate "$CONDA_ENV_QFIT"

    local lookup=$(grep "^${dataset} " "$LOOKUP_FILE")
    if [ -z "$lookup" ]; then
        echo "Warning: No match found for dataset ${dataset}"
        return 1
    fi

    local fragment_id=$(echo "$lookup" | awk '{print $2}')
    local resolution=$(echo "$lookup" | awk '{print $3}')

    echo "Processing ${dataset}: fragment_id=${fragment_id}, resolution=${resolution}"

    local pdb_dirs=()
    while IFS= read -r -d '' dir; do
        local dir_name=$(basename "$dir")
        if [[ "$dir_name" == "$fragment_id" || "$dir_name" == "${fragment_id}_"* ]]; then
            pdb_dirs+=("$dir")
        fi
    done < <(find "$LIG_PDB_DIR" -maxdepth 1 -mindepth 1 -type d -name "${fragment_id}*" -print0 | sort -z)

    if [[ ${#pdb_dirs[@]} -eq 0 ]]; then
        echo "Warning: No directories found matching '${fragment_id}' for dataset ${dataset}"
        return 1
    fi

    echo "  Found ${#pdb_dirs[@]} matching director(ies) for ${dataset} (fragment_id=${fragment_id})"

    for pdb_dir in "${pdb_dirs[@]}"; do
        local dir_name=$(basename "$pdb_dir")
        local pdb_file="${pdb_dir}/${dir_name}.pdb"

        if [[ ! -f "$pdb_file" ]]; then
            echo "  Warning: Expected PDB file not found: ${pdb_file}, skipping."
            continue
        fi

        local out_dir="${DATASETS_DIR}/${dataset}/${run_name}"
        mkdir -p "${out_dir}"

        echo "  Running fit_ligand: PDB=${dir_name}, sampling=${run_name}"

        fit_ligand "${DATASETS_DIR}/${dataset}" \
            "${pdb_file}" \
            -r ${resolution} \
            --sampling ${run_name} \
            > "${out_dir}/ligandfit_${dir_name}.txt" 2>&1

        echo "  Completed: ${dataset} / ${dir_name}"
    done
}
export -f fit_ligand_process_dataset

do_fit_ligand() {
    echo "Starting run"
    local start_time=$(date +%s)
    cat "$DATASETS_FILE" | parallel -j "$NUM_PARALLEL_DEFAULT" fit_ligand_process_dataset {}
    echo "All jobs completed"
    print_elapsed "$start_time"
}

######################################################################
# Stage 2a: placer (round 1)
######################################################################

placer_process_dataset() {
    local dataset=$1
    local gpu_id=$2

    export CUDA_VISIBLE_DEVICES=$gpu_id

    echo "=== Dataset: ${dataset} (GPU ${gpu_id}) ==="

    local dataset_dir="${DATASETS_DIR}/${dataset}"
    local run_dir="${dataset_dir}/${run_name}"
    local manifest_file="${run_dir}/fit_ligand_manifest.csv"

    if [[ ! -f "$manifest_file" ]]; then
        echo "  Warning: manifest not found: ${manifest_file}, skipping."
        return
    fi

    local out_dir="${run_dir}/${placer_run_name}"
    mkdir -p "${out_dir}"

    # Manifest columns: dataset,ligand_name,ligand_file,peak_index,output_pdb
    tail -n +2 "$manifest_file" | tr -d '\r' | while IFS=',' read -r m_dataset m_ligand_name m_ligand_file m_peak_index m_output_pdb; do
        [ -z "$m_dataset" ] && continue

        if [[ ! -f "$m_output_pdb" ]]; then
            echo "  Warning: output_pdb not found: ${m_output_pdb}, skipping."
            continue
        fi

        local ligand_mol2="${m_ligand_file%.pdb}.mol2"
        if [[ ! -f "$ligand_mol2" ]]; then
            echo "  Warning: No matching .mol2 for ligand '${m_ligand_name}' (expected ${ligand_mol2}), skipping."
            continue
        fi

        local pdb_name=$(basename "${m_output_pdb%.pdb}")

        local lig_id=$(get_lig_id "$m_output_pdb")
        if [ -z "$lig_id" ]; then
            echo "  Warning: could not find a LIG residue in ${m_output_pdb}, skipping."
            continue
        fi

        echo "  Running PLACER on: ${pdb_name}.pdb (ligand: $(basename "$ligand_mol2"), predict_ligand=${lig_id})"

        python "$RUN_PLACER_PY" \
            --ifile "${m_output_pdb}" \
            --odir "${out_dir}/." \
            -n ${num_placer_confs} \
            --ligand_file "LIG:${ligand_mol2}" \
            --predict_ligand "${lig_id}" \
            --ignore_ligand_hydrogens
    done
}
export -f placer_process_dataset

do_placer() {
    conda_activate "$CONDA_ENV_PLACER"

    echo "Starting run on GPU(s): ${GPU_IDS_ARR[*]}"
    local start_time=$(date +%s)

    local idx=0
    while IFS= read -r dataset; do
        [ -z "$dataset" ] && continue
        local gpu_id=${GPU_IDS_ARR[$((idx % NUM_GPUS))]}
        echo "${dataset} ${gpu_id}"
        idx=$((idx + 1))
    done < "$DATASETS_FILE" | parallel -j "$NUM_GPUS" --colsep ' ' placer_process_dataset {1} {2}

    echo "All jobs completed"
    print_elapsed "$start_time"
    conda deactivate
}

######################################################################
# Stage 2b: rsr_placer
######################################################################

rsr_placer_process_dataset() {
    conda_activate "$CONDA_ENV_RSR"

    local dataset=$1
    local dataset_dir="${DATASETS_DIR}/${dataset}"
    local placer_dir="${dataset_dir}/${run_name}/${placer_run_name}"

    echo "Processing ${dataset}..."

    local map_file
    map_file=$(find "$dataset_dir" -maxdepth 1 -name "${dataset}-event_1*" | head -1)
    if [ -z "$map_file" ]; then
        echo "ERROR [${dataset}]: No event map found matching ${dataset}-event_1*"
        return 1
    fi

    local manifest_file="${dataset_dir}/${run_name}/fit_ligand_manifest.csv"
    if [ ! -f "$manifest_file" ]; then
        echo "ERROR [${dataset}]: Manifest not found: ${manifest_file}"
        return 1
    fi

    local -A key_to_cif
    local csv_dataset ligand_name ligand_file peak_index output_pdb
    while IFS=, read -r csv_dataset ligand_name ligand_file peak_index output_pdb; do
        csv_dataset="${csv_dataset//$'\r'/}"
        ligand_name="${ligand_name//$'\r'/}"
        ligand_file="${ligand_file//$'\r'/}"
        peak_index="${peak_index//$'\r'/}"
        output_pdb="${output_pdb//$'\r'/}"

        ligand_file="$(echo -n "$ligand_file" | sed -e 's/^[[:space:]"'"'"']*//' -e 's/[[:space:]"'"'"']*$//')"
        output_pdb="$(echo -n "$output_pdb" | sed -e 's/^[[:space:]"'"'"']*//' -e 's/[[:space:]"'"'"']*$//')"

        [ -z "$output_pdb" ] && continue

        local key
        key="$(basename "$output_pdb")"
        key="${key%.pdb}"

        local cif_path="${ligand_file%.pdb}.cif"
        key_to_cif["$key"]="$cif_path"
    done < <(tail -n +2 "$manifest_file")

    local pdb_files
    mapfile -t pdb_files < <(find "$placer_dir" -maxdepth 1 -name "*_model.pdb")

    if [ ${#pdb_files[@]} -eq 0 ]; then
        echo "ERROR [${dataset}]: No *_model.pdb files found in $placer_dir"
        return 1
    fi

    local any_failed=0

    for input_pdb in "${pdb_files[@]}"; do
        local key
        key="$(basename "$input_pdb")"
        key="${key%_model.pdb}"

        local cif_path="${key_to_cif[$key]}"

        if [ -z "$cif_path" ]; then
            echo "ERROR [${dataset}]: No manifest entry found for key '${key}' (from $input_pdb)"
            any_failed=1
            continue
        fi

        if [ ! -f "$cif_path" ]; then
            echo "ERROR [${dataset}]: CIF not found: ${cif_path}"
            any_failed=1
            continue
        fi

        local output_pdb="${input_pdb%_model.pdb}_refined.pdb"

        echo "[${dataset}] Input:  $input_pdb"
        echo "[${dataset}] Output: $output_pdb"
        echo "[${dataset}] Map:    $map_file"
        echo "[${dataset}] CIF:    $cif_path"

        python "$RSR_SCRIPT_LIGAND" \
            "$input_pdb" \
            "$map_file" \
            "$output_pdb" \
            --cif-restraints "$cif_path"
        local exit_code=$?
        if [ $exit_code -ne 0 ]; then
            echo "ERROR [${dataset}]: Refinement failed for $input_pdb with exit code $exit_code"
            any_failed=1
        else
            echo "Completed: ${dataset} / $(basename "$input_pdb")"
        fi
    done

    return $any_failed
}
export -f rsr_placer_process_dataset

do_rsr_placer() {
    echo "Starting RSR run"
    local start_time=$(date +%s)
    cat "$DATASETS_FILE" | parallel -j "$NUM_PARALLEL_DEFAULT" rsr_placer_process_dataset {}
    echo "All jobs completed"
    print_elapsed "$start_time"
}

######################################################################
# Stage 3a: filter
######################################################################

filter_process_dataset() {
    conda_activate "$CONDA_ENV_QFIT"

    local dataset=$1
    local dataset_dir="${DATASETS_DIR}/${dataset}"

    local lookup=$(grep "^${dataset} " "$LOOKUP_FILE")
    if [ -z "$lookup" ]; then
        echo "Warning: No match found for dataset ${dataset}, skipping."
        return 1
    fi

    local fragment_id=$(echo "$lookup" | awk '{print $2}')
    local resolution=$(echo "$lookup" | awk '{print $3}')

    echo "Processing ${dataset}: fragment_id=${fragment_id}, resolution=${resolution}"

    filter ${dataset_dir} \
        "${dataset_dir}/${run_name}/${placer_run_name}/*_refined.pdb" \
        "${dataset_dir}/${run_name}/*.pdb" \
        $run_name/${placer_run_name}/${filter_run_name} \
        -r ${resolution}

    local filter_exit=$?
    if [ $filter_exit -ne 0 ]; then
        echo "ERROR [${dataset}]: filter failed with exit code ${filter_exit}"
        return 1
    fi

    # --- Post-hoc: annotate cluster_reps.csv with a cif_restraints_file column ---
    local cluster_csv="${dataset_dir}/${run_name}/${placer_run_name}/${filter_run_name}/cluster_reps.csv"
    if [ ! -f "$cluster_csv" ]; then
        echo "Warning [${dataset}]: cluster_reps.csv not found at ${cluster_csv}, skipping CIF annotation."
        return 0
    fi

    local manifest_file="${dataset_dir}/${run_name}/fit_ligand_manifest.csv"
    if [ ! -f "$manifest_file" ]; then
        echo "ERROR [${dataset}]: Manifest not found: ${manifest_file}, cannot annotate CIF restraints."
        return 1
    fi

    local -A key_to_cif
    local csv_dataset ligand_name ligand_file peak_index output_pdb
    while IFS=, read -r csv_dataset ligand_name ligand_file peak_index output_pdb; do
        csv_dataset="${csv_dataset//$'\r'/}"
        ligand_file="${ligand_file//$'\r'/}"
        output_pdb="${output_pdb//$'\r'/}"

        ligand_file="$(echo -n "$ligand_file" | sed -e 's/^[[:space:]"'"'"']*//' -e 's/[[:space:]"'"'"']*$//')"
        output_pdb="$(echo -n "$output_pdb" | sed -e 's/^[[:space:]"'"'"']*//' -e 's/[[:space:]"'"'"']*$//')"

        [ -z "$output_pdb" ] && continue

        local key
        key="$(basename "$output_pdb")"
        key="${key%.pdb}"

        local cif_path="${ligand_file%.pdb}.cif"
        key_to_cif["$key"]="$cif_path"
    done < <(tail -n +2 "$manifest_file")

    local tmp_csv
    tmp_csv="$(mktemp "${cluster_csv}.XXXXXX")"

    {
        local header
        IFS= read -r header
        echo "${header},cif_restraints_file"

        local placer_file index mse cluster rscc num_members
        while IFS=, read -r placer_file index mse cluster rscc num_members; do
            [ -z "$placer_file" ] && continue
            placer_file="${placer_file//$'\r'/}"
            num_members="${num_members//$'\r'/}"

            local key
            key="$(basename "$placer_file")"
            key="${key%_refined.pdb}"

            local cif_path="${key_to_cif[$key]}"
            if [ -z "$cif_path" ]; then
                echo "Warning [${dataset}]: No manifest entry found for key '${key}' (from ${placer_file})" >&2
                cif_path="NA"
            fi

            echo "${placer_file},${index},${mse},${cluster},${rscc},${num_members},${cif_path}"
        done
    } < "$cluster_csv" > "$tmp_csv"

    mv "$tmp_csv" "$cluster_csv"

    echo "Completed: ${dataset}"
}
export -f filter_process_dataset

do_filter() {
    conda_activate "$CONDA_ENV_QFIT"

    echo "Starting run"
    local start_time=$(date +%s)
    cat "$DATASETS_FILE" | parallel -j "$NUM_PARALLEL_DEFAULT" filter_process_dataset {}
    echo "All jobs completed"
    print_elapsed "$start_time"
}

######################################################################
# Stage 3b: rsr_backbone
######################################################################

rsr_backbone_process_dataset() {
    local dataset=$1
    local dataset_dir="${DATASETS_DIR}/${dataset}"

    conda_activate "$CONDA_ENV_RSR"

    local map_file
    map_file=$(find "${dataset_dir}" -maxdepth 1 -name "${dataset}-event_1*" | head -1)
    if [ -z "$map_file" ]; then
        echo "ERROR [${dataset}]: No event map found matching ${dataset}-event_1* in ${dataset_dir}"
        return 1
    fi

    local run_dir="${dataset_dir}/${run_name}/${placer_run_name}/${filter_run_name}"
    local cluster_reps_csv="${run_dir}/cluster_reps.csv"

    if [ ! -f "$cluster_reps_csv" ]; then
        echo "ERROR [${dataset}]: cluster_reps.csv not found: $cluster_reps_csv"
        return 1
    fi

    local header
    IFS=, read -r header < "$cluster_reps_csv"

    local cif_col_index=-1
    local i=0
    local col
    IFS=, read -r -a header_cols <<< "$header"
    for col in "${header_cols[@]}"; do
        col="${col//$'\r'/}"
        if [ "$col" = "cif_restraints_file" ]; then
            cif_col_index=$i
        fi
        i=$((i + 1))
    done

    if [ $cif_col_index -lt 0 ]; then
        echo "ERROR [${dataset}]: cif_restraints_file column not found in ${cluster_reps_csv}"
        return 1
    fi

    local cif_paths=()
    local any_cif_missing=0
    local row_num=0
    while IFS=, read -r -a row_cols; do
        row_num=$((row_num + 1))
        [ $row_num -eq 1 ] && continue
        [ -z "${row_cols[0]}" ] && continue

        local cif_path="${row_cols[$cif_col_index]}"
        cif_path="${cif_path//$'\r'/}"
        cif_path="$(echo -n "$cif_path" | sed -e 's/^[[:space:]"'"'"']*//' -e 's/[[:space:]"'"'"']*$//')"

        if [ -z "$cif_path" ]; then
            echo "ERROR [${dataset}]: Empty cif_restraints_file value on row ${row_num} of ${cluster_reps_csv}"
            any_cif_missing=1
            continue
        fi
        if [ ! -f "$cif_path" ]; then
            echo "ERROR [${dataset}]: CIF not found: ${cif_path} (row ${row_num} of ${cluster_reps_csv})"
            any_cif_missing=1
            continue
        fi
        cif_paths+=("$cif_path")
    done < "$cluster_reps_csv"

    if [ $any_cif_missing -ne 0 ]; then
        echo "ERROR [${dataset}]: One or more CIF restraint files were missing, aborting."
        return 1
    fi

    if [ ${#cif_paths[@]} -eq 0 ]; then
        echo "ERROR [${dataset}]: No data rows found in $cluster_reps_csv"
        return 1
    fi

    local cif_list
    cif_list=$(IFS=,; echo "${cif_paths[*]}")

    local apo_pdb="${dataset_dir}/${dataset}-aligned-structure.pdb"
    local multimodel_pdb="${run_dir}/cluster_rep_models.pdb"
    local output_pdb="${run_dir}/${dataset}_backbone_refined.pdb"

    if [ ! -f "$multimodel_pdb" ]; then
        echo "ERROR [${dataset}]: multimodel_pdb not found: ${multimodel_pdb}"
        return 1
    fi
    if [ ! -f "$apo_pdb" ]; then
        echo "ERROR [${dataset}]: apo_pdb not found: ${apo_pdb}"
        return 1
    fi

    echo "[${dataset}] Map: $map_file"
    echo "[${dataset}] Using CIF restraints list (${#cif_paths[@]} entries): $cif_list"

    python "$RSR_SCRIPT_PROTEIN" \
        "$multimodel_pdb" \
        "$apo_pdb" \
        "$map_file" \
        "$output_pdb" \
        --cif-list "$cif_list"
    local exit_code=$?
    if [ $exit_code -ne 0 ]; then
        echo "ERROR [${dataset}]: Refinement failed with exit code $exit_code"
        return 1
    fi

    echo "Completed: ${dataset}"
}
export -f rsr_backbone_process_dataset

do_rsr_backbone() {
    conda_activate "$CONDA_ENV_RSR"

    # Prevent native libraries underneath coot_headless_api (FFTW, OpenMP-based
    # geometry minimization, any linked BLAS) from each spawning one thread per
    # core on the machine. Scoped to just this stage (exported here, unset
    # below) so it doesn't affect other stages' parallelism.
    export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1

    echo "Starting run"
    local start_time=$(date +%s)
    cat "$DATASETS_FILE" | parallel -j "$NUM_PARALLEL_DEFAULT" rsr_backbone_process_dataset {}
    echo "All jobs completed"
    print_elapsed "$start_time"

    unset OMP_NUM_THREADS OPENBLAS_NUM_THREADS MKL_NUM_THREADS NUMEXPR_NUM_THREADS VECLIB_MAXIMUM_THREADS
}

######################################################################
# Stage 3c: calc_backbone_refined_rscc
######################################################################

calc_backbone_refined_rscc_process_dataset() {
    conda_activate "$CONDA_ENV_QFIT"
    shopt -s nullglob

    local dataset=$1
    local dataset_dir="${DATASETS_DIR}/${dataset}"

    local lookup=$(grep "^${dataset} " "$LOOKUP_FILE")
    if [ -z "$lookup" ]; then
        echo "Warning: No match found for dataset ${dataset}, skipping."
        return 1
    fi

    local resolution=$(echo "$lookup" | awk '{print $3}')

    echo "Processing ${dataset}: resolution=${resolution}"

    local event_maps=("${dataset_dir}/${dataset}-event_"*)
    if [ ${#event_maps[@]} -eq 0 ]; then
        echo "Warning [${dataset}]: no event maps found matching ${dataset_dir}/${dataset}-event_*, skipping."
        return 1
    fi

    local run_dir="${dataset_dir}/${run_name}/${placer_run_name}/${filter_run_name}"
    local structures=("${run_dir}/${dataset}_backbone_refined_"*.pdb)
    if [ ${#structures[@]} -eq 0 ]; then
        echo "Warning [${dataset}]: no structures found matching ${run_dir}/${dataset}_backbone_refined_*.pdb, skipping."
        return 1
    fi

    local structure output_csv
    for structure in "${structures[@]}"; do
        output_csv="${structure%.pdb}_rscc.csv"

        calc_rscc "${structure}" "${event_maps[@]}" "${resolution}" "${output_csv}"

        local calc_exit=$?
        if [ $calc_exit -ne 0 ]; then
            echo "ERROR [${dataset}]: calc_rscc failed on ${structure} with exit code ${calc_exit}"
            continue
        fi
        echo "Completed [${dataset}]: ${structure} -> ${output_csv}"
    done

    echo "Completed: ${dataset}"
}
export -f calc_backbone_refined_rscc_process_dataset

do_calc_backbone_rscc() {
    conda_activate "$CONDA_ENV_QFIT"

    echo "Starting run"
    local start_time=$(date +%s)
    cat "$DATASETS_FILE" | parallel -j "$NUM_PARALLEL_DEFAULT" calc_backbone_refined_rscc_process_dataset {}
    echo "All jobs completed"
    print_elapsed "$start_time"
}

######################################################################
# Stage 3d/3e: reference-set comparison (only runs when -c is given)
######################################################################
# Pooled (cross-dataset) plots under GRAPHS_DIR/<run>/<placer>/<filter>/:
# ligand RSCC (filter_run_name/cluster_reps.csv vs reference, matched by
# centroid) and per-residue RSCC (backbone-refined vs reference, matched by
# residue label). No RSCC is computed by either script - only cached values
# already written by calc_backbone_refined_rscc/calc_ref_set_rscc are read.

do_plot_lig_vs_ref_filter1() {
    conda_activate "$CONDA_ENV_EVAL"

    local out_dir="${GRAPHS_DIR}/${run_name}/${placer_run_name}/${filter_run_name}"
    echo "Starting run"
    local start_time=$(date +%s)
    python "$PLOT_LIG_VS_REF_FILTER1_PY" \
        "$run_name" "$placer_run_name" "$filter_run_name" \
        --datasets-dir "$DATASETS_DIR" --datasets-file "$DATASETS_FILE" \
        --ref-set "$REF_SET" --ref-pdb-pattern "$REF_SET_PDB_PATTERN" --graphs-dir "$out_dir"
    echo "All jobs completed"
    print_elapsed "$start_time"
}

do_plot_residues_vs_ref_backbone() {
    conda_activate "$CONDA_ENV_EVAL"

    local out_dir="${GRAPHS_DIR}/${run_name}/${placer_run_name}/${filter_run_name}"
    echo "Starting run"
    local start_time=$(date +%s)
    python "$PLOT_RESIDUES_VS_REF_BACKBONE_PY" \
        "$run_name" "$placer_run_name" "$filter_run_name" \
        --datasets-dir "$DATASETS_DIR" --datasets-file "$DATASETS_FILE" \
        --ref-set "$REF_SET" --ref-pdb-pattern "$REF_SET_PDB_PATTERN" --graphs-dir "$out_dir"
    echo "All jobs completed"
    print_elapsed "$start_time"
}

######################################################################
# Stage 4a: placer2 (round 2)
######################################################################

placer2_process_dataset() {
    local dataset=$1
    local gpu_id=$2

    export CUDA_VISIBLE_DEVICES=$gpu_id

    echo "=== Dataset: ${dataset} (GPU ${gpu_id}) ==="

    local dataset_dir="${DATASETS_DIR}/${dataset}"
    local filter_dir="${dataset_dir}/${run_name}/${placer_run_name}/${filter_run_name}"

    if [[ ! -d "$filter_dir" ]]; then
        echo "  Warning: filter run directory not found: ${filter_dir}, skipping."
        return
    fi

    local cluster_reps_csv="${filter_dir}/cluster_reps.csv"
    if [[ ! -f "$cluster_reps_csv" ]]; then
        echo "  Warning: cluster_reps.csv not found: ${cluster_reps_csv}, skipping."
        return
    fi

    # The i-th backbone-refined pdb (${dataset}_backbone_refined_{i}.pdb) corresponds to
    # the i-th (1-indexed) DATA row of cluster_reps.csv, so rows are read in order and
    # position (not any value in the row) recovers which ligand goes with which
    # backbone-refined model. The ligand's .mol2 file is derived directly from that
    # row's cif_restraints_file column (same directory, .cif swapped for .mol2)
    # rather than being parsed out of the placer_file column's filename.
    local header=""
    IFS=, read -r header < "$cluster_reps_csv"

    local cif_col_index=-1
    local i=0
    local header_cols
    IFS=, read -r -a header_cols <<< "$header"
    for col in "${header_cols[@]}"; do
        col="${col//$'\r'/}"
        if [ "$col" = "cif_restraints_file" ]; then
            cif_col_index=$i
        fi
        i=$((i + 1))
    done

    if [ $cif_col_index -lt 0 ]; then
        echo "  Warning: cif_restraints_file column not found in ${cluster_reps_csv}, skipping."
        return
    fi

    local -a cif_paths=()
    local row_num=0
    while IFS=, read -r -a row_cols; do
        row_num=$((row_num + 1))
        [ $row_num -eq 1 ] && continue
        [ -z "${row_cols[0]}" ] && continue

        local cif_path="${row_cols[$cif_col_index]}"
        cif_path="${cif_path//$'\r'/}"
        cif_path="$(echo -n "$cif_path" | sed -e 's/^[[:space:]"'"'"']*//' -e 's/[[:space:]"'"'"']*$//')"

        cif_paths+=("$cif_path")
    done < "$cluster_reps_csv"

    if [ ${#cif_paths[@]} -eq 0 ]; then
        echo "  Warning: no data rows found in ${cluster_reps_csv}, skipping."
        return
    fi

    local out_dir="${filter_dir}/${placer2_run_name}"
    mkdir -p "${out_dir}"

    for i in "${!cif_paths[@]}"; do
        local model_idx=$((i + 1))
        local pdb_file="${filter_dir}/${dataset}_backbone_refined_${model_idx}.pdb"

        if [[ ! -f "$pdb_file" ]]; then
            echo "  Warning: backbone-refined pdb not found for model ${model_idx}: ${pdb_file}, skipping."
            continue
        fi

        local cif_path="${cif_paths[$i]}"
        if [ -z "$cif_path" ]; then
            echo "  Warning: Empty cif_restraints_file for model ${model_idx}, skipping."
            continue
        fi

        local lig_name
        lig_name=$(basename "${cif_path%.cif}")
        local ligand_file="${cif_path%.cif}.mol2"
        if [[ ! -f "$ligand_file" ]]; then
            echo "  Warning: No matching .mol2 for ligand '${lig_name}' (expected ${ligand_file}), skipping model ${model_idx}."
            continue
        fi

        local lig_id=$(get_lig_id "$pdb_file")
        if [ -z "$lig_id" ]; then
            echo "  Warning: could not find a LIG residue in ${pdb_file}, skipping."
            continue
        fi

        echo "  Running PLACER on: ${dataset}_backbone_refined_${model_idx}.pdb (ligand: ${lig_name}.mol2, predict_ligand=${lig_id})"

        python "$RUN_PLACER_PY" \
            --ifile "${pdb_file}" \
            --odir "${out_dir}/." \
            -n ${num_placer2_confs} \
            --ligand_file "LIG:${ligand_file}" \
            --predict_ligand "${lig_id}" \
            --ignore_ligand_hydrogens
    done
}
export -f placer2_process_dataset

do_placer2() {
    conda_activate "$CONDA_ENV_PLACER"

    echo "Starting run on GPU(s): ${GPU_IDS_ARR[*]}"
    local start_time=$(date +%s)

    local idx=0
    for folder in "${DATASETS_DIR}"/*/; do
        local dataset
        dataset=$(basename "${folder%/}")
        local gpu_id=${GPU_IDS_ARR[$((idx % NUM_GPUS))]}
        echo "${dataset} ${gpu_id}"
        idx=$((idx + 1))
    done | parallel -j "$NUM_GPUS" --colsep ' ' placer2_process_dataset {1} {2}

    echo "All jobs completed"
    print_elapsed "$start_time"
    conda deactivate
}

######################################################################
# Stage 4b: rsr_placer2
######################################################################

rsr_placer2_process_dataset() {
    conda_activate "$CONDA_ENV_RSR"

    local dataset=$1
    local dataset_dir="${DATASETS_DIR}/${dataset}"
    local filter_dir="${dataset_dir}/${run_name}/${placer_run_name}/${filter_run_name}"
    local placer2_dir="${filter_dir}/${placer2_run_name}"

    echo "Processing ${dataset}..."

    local map_file
    map_file=$(find "$dataset_dir" -maxdepth 1 -name "${dataset}-event_1*" | head -1)
    if [ -z "$map_file" ]; then
        echo "ERROR [${dataset}]: No event map found matching ${dataset}-event_1*"
        return 1
    fi

    local cluster_reps_csv="${filter_dir}/cluster_reps.csv"
    if [ ! -f "$cluster_reps_csv" ]; then
        echo "ERROR [${dataset}]: cluster_reps.csv not found: ${cluster_reps_csv}"
        return 1
    fi

    # The i-th backbone-refined pdb ({dataset}_backbone_refined_{i}.pdb) -- and
    # therefore its PLACER2 output, {dataset}_backbone_refined_{i}_model.pdb --
    # corresponds to the i-th (1-indexed) DATA row of cluster_reps.csv, so rows
    # are read in order and position (not any value in the row) recovers which
    # CIF restraints file goes with which model.
    local header=""
    IFS=, read -r header < "$cluster_reps_csv"

    local cif_col_index=-1
    local i=0
    local header_cols
    IFS=, read -r -a header_cols <<< "$header"
    for col in "${header_cols[@]}"; do
        col="${col//$'\r'/}"
        if [ "$col" = "cif_restraints_file" ]; then
            cif_col_index=$i
        fi
        i=$((i + 1))
    done

    if [ $cif_col_index -lt 0 ]; then
        echo "ERROR [${dataset}]: cif_restraints_file column not found in $cluster_reps_csv"
        return 1
    fi

    local cif_paths=()
    local row_num=0
    while IFS=, read -r -a row_cols; do
        row_num=$((row_num + 1))
        [ $row_num -eq 1 ] && continue
        [ -z "${row_cols[0]}" ] && continue

        local cif_path="${row_cols[$cif_col_index]}"
        cif_path="${cif_path//$'\r'/}"
        cif_path="$(echo -n "$cif_path" | sed -e 's/^[[:space:]"'"'"']*//' -e 's/[[:space:]"'"'"']*$//')"

        cif_paths+=("$cif_path")
    done < "$cluster_reps_csv"

    if [ ${#cif_paths[@]} -eq 0 ]; then
        echo "ERROR [${dataset}]: No data rows found in $cluster_reps_csv"
        return 1
    fi

    local pdb_files
    mapfile -t pdb_files < <(find "$placer2_dir" -maxdepth 1 -name "${dataset}_backbone_refined_*_model.pdb")

    if [ ${#pdb_files[@]} -eq 0 ]; then
        echo "ERROR [${dataset}]: No ${dataset}_backbone_refined_*_model.pdb files found in $placer2_dir"
        return 1
    fi

    local any_failed=0

    for input_pdb in "${pdb_files[@]}"; do
        local basename
        basename=$(basename "$input_pdb" _model.pdb)

        local model_idx="${basename#${dataset}_backbone_refined_}"
        if ! [[ "$model_idx" =~ ^[0-9]+$ ]]; then
            echo "ERROR [${dataset}]: Could not parse model index from $(basename "$input_pdb")"
            any_failed=1
            continue
        fi

        local row_idx=$((model_idx - 1))
        if [ "$row_idx" -lt 0 ] || [ "$row_idx" -ge ${#cif_paths[@]} ]; then
            echo "ERROR [${dataset}]: Model index ${model_idx} has no corresponding row in $cluster_reps_csv"
            any_failed=1
            continue
        fi

        local cif_path="${cif_paths[$row_idx]}"
        if [ -z "$cif_path" ]; then
            echo "ERROR [${dataset}]: Empty cif_restraints_file for model ${model_idx}"
            any_failed=1
            continue
        fi

        if [ ! -f "$cif_path" ]; then
            echo "ERROR [${dataset}]: CIF not found for model ${model_idx}: ${cif_path}"
            any_failed=1
            continue
        fi

        local output_pdb="${input_pdb%_model.pdb}_refined.pdb"

        echo "[${dataset}] Input:  $input_pdb"
        echo "[${dataset}] Output: $output_pdb"
        echo "[${dataset}] Map:    $map_file"
        echo "[${dataset}] CIF:    $cif_path"

        python "$RSR_SCRIPT_LIGAND" \
            "$input_pdb" \
            "$map_file" \
            "$output_pdb" \
            --cif-restraints "$cif_path"
        local exit_code=$?
        if [ $exit_code -ne 0 ]; then
            echo "ERROR [${dataset}]: Refinement failed for $input_pdb with exit code $exit_code"
            any_failed=1
        else
            echo "Completed: ${dataset} / $(basename "$input_pdb")"
        fi
    done

    return $any_failed
}
export -f rsr_placer2_process_dataset

do_rsr_placer2() {
    echo "Starting RSR run"
    local start_time=$(date +%s)
    cat "$DATASETS_FILE" | parallel -j "$NUM_PARALLEL_DEFAULT" rsr_placer2_process_dataset {}
    echo "All jobs completed"
    print_elapsed "$start_time"
}

######################################################################
# Stage 5: filter2
######################################################################

filter2_process_dataset() {
    conda_activate "$CONDA_ENV_QFIT"

    local dataset=$1
    local dataset_dir="${DATASETS_DIR}/${dataset}"

    local lookup=$(grep "^${dataset} " "$LOOKUP_FILE")
    if [ -z "$lookup" ]; then
        echo "Warning: No match found for dataset ${dataset}, skipping."
        return 1
    fi

    local fragment_id=$(echo "$lookup" | awk '{print $2}')
    local resolution=$(echo "$lookup" | awk '{print $3}')

    echo "Processing ${dataset}: fragment_id=${fragment_id}, resolution=${resolution}"

    filter_all "${dataset_dir}" \
        "${dataset_dir}/${run_name}/${placer_run_name}/${filter_run_name}/${placer2_run_name}/*_refined.pdb" \
        "${dataset_dir}/${run_name}/${placer_run_name}/${filter_run_name}/*_refined_*.pdb" \
        ${run_name}/${placer_run_name}/${filter_run_name}/${placer2_run_name}/${filter2_run_name} \
        -r ${resolution}

    local filter_exit=$?
    if [ $filter_exit -ne 0 ]; then
        echo "ERROR [${dataset}]: filter_all failed with exit code ${filter_exit}"
        return 1
    fi

    # --- Post-hoc: carry the cif_restraints_file column over from filter_run_name's
    # cluster_reps.csv into filter2_run_name's cluster_reps.csv ---
    local filter_csv="${dataset_dir}/${run_name}/${placer_run_name}/${filter_run_name}/cluster_reps.csv"
    local filter2_dir="${dataset_dir}/${run_name}/${placer_run_name}/${filter_run_name}/${placer2_run_name}/${filter2_run_name}"
    local filter2_csv="${filter2_dir}/cluster_reps.csv"

    if [ ! -f "$filter2_csv" ]; then
        echo "Warning [${dataset}]: cluster_reps.csv not found at ${filter2_csv}, skipping CIF annotation."
        return 0
    fi

    if [ ! -f "$filter_csv" ]; then
        echo "ERROR [${dataset}]: filter_run_name cluster_reps.csv not found: ${filter_csv}, cannot annotate CIF restraints."
        return 1
    fi

    local filter_header=""
    IFS= read -r filter_header < "$filter_csv"

    local filter_cif_col_index=-1
    local i=0
    local filter_header_cols
    IFS=, read -r -a filter_header_cols <<< "$filter_header"
    for col in "${filter_header_cols[@]}"; do
        col="${col//$'\r'/}"
        if [ "$col" = "cif_restraints_file" ]; then
            filter_cif_col_index=$i
        fi
        i=$((i + 1))
    done

    if [ $filter_cif_col_index -lt 0 ]; then
        echo "ERROR [${dataset}]: cif_restraints_file column not found in ${filter_csv}, cannot annotate."
        return 1
    fi

    local -a filter_cif_paths=()
    local row_num=0
    while IFS=, read -r -a row_cols; do
        row_num=$((row_num + 1))
        [ $row_num -eq 1 ] && continue
        [ -z "${row_cols[0]}" ] && continue

        local cif_val="${row_cols[$filter_cif_col_index]}"
        cif_val="${cif_val//$'\r'/}"
        filter_cif_paths+=("$cif_val")
    done < "$filter_csv"

    if [ ${#filter_cif_paths[@]} -eq 0 ]; then
        echo "ERROR [${dataset}]: No data rows found in ${filter_csv}, cannot annotate."
        return 1
    fi

    local tmp_csv
    tmp_csv="$(mktemp "${filter2_csv}.XXXXXX")"

    {
        local header
        IFS= read -r header
        echo "${header},cif_restraints_file"

        local placer_file rest_of_row
        while IFS=, read -r placer_file rest_of_row; do
            [ -z "$placer_file" ] && continue
            placer_file="${placer_file//$'\r'/}"

            local model_idx=""
            if [[ "$(basename "$placer_file")" =~ backbone_refined_([0-9]+)_refined\.pdb$ ]]; then
                model_idx="${BASH_REMATCH[1]}"
            fi

            local cif_path="NA"
            if [ -z "$model_idx" ]; then
                echo "Warning [${dataset}]: Could not parse backbone_refined index from ${placer_file}" >&2
            else
                local row_idx=$((model_idx - 1))
                if [ "$row_idx" -lt 0 ] || [ "$row_idx" -ge ${#filter_cif_paths[@]} ]; then
                    echo "Warning [${dataset}]: Model index ${model_idx} has no corresponding row in ${filter_csv}" >&2
                else
                    cif_path="${filter_cif_paths[$row_idx]}"
                    [ -z "$cif_path" ] && cif_path="NA"
                fi
            fi

            echo "${placer_file},${rest_of_row},${cif_path}"
        done
    } < "$filter2_csv" > "$tmp_csv"

    mv "$tmp_csv" "$filter2_csv"

    echo "Completed: ${dataset}"
}
export -f filter2_process_dataset

do_filter2() {
    conda_activate "$CONDA_ENV_QFIT"

    echo "Starting run"
    local start_time=$(date +%s)
    cat "$DATASETS_FILE" | parallel -j "$NUM_PARALLEL_DEFAULT" filter2_process_dataset {}
    echo "All jobs completed"
    print_elapsed "$start_time"
}

######################################################################
# Stage 5b: reference-set comparison (only runs when -c is given)
######################################################################
# Same lig-vs-reference comparison as stage 3d, one round later: pooled
# plot under GRAPHS_DIR/<run>/<placer>/<filter>/<placer2>/<filter2>/.

do_plot_lig_vs_ref_filter2() {
    conda_activate "$CONDA_ENV_EVAL"

    local out_dir="${GRAPHS_DIR}/${run_name}/${placer_run_name}/${filter_run_name}/${placer2_run_name}/${filter2_run_name}"
    echo "Starting run"
    local start_time=$(date +%s)
    python "$PLOT_LIG_VS_REF_FILTER2_PY" \
        "$run_name" "$placer_run_name" "$filter_run_name" "$placer2_run_name" "$filter2_run_name" \
        --datasets-dir "$DATASETS_DIR" --datasets-file "$DATASETS_FILE" \
        --ref-set "$REF_SET" --ref-pdb-pattern "$REF_SET_PDB_PATTERN" --graphs-dir "$out_dir"
    echo "All jobs completed"
    print_elapsed "$start_time"
}

######################################################################
# Stage 6a: build_final
######################################################################

build_final_process_dataset() {
    conda_activate "$CONDA_ENV_QFIT"

    local dataset=$1
    local dataset_dir="${DATASETS_DIR}/${dataset}"

    local lookup=$(grep "^${dataset} " "$LOOKUP_FILE")
    if [ -z "$lookup" ]; then
        echo "Warning: No match found for dataset ${dataset}, skipping."
        return 1
    fi

    local fragment_id=$(echo "$lookup" | awk '{print $2}')
    local resolution=$(echo "$lookup" | awk '{print $3}')

    echo "Processing ${dataset}: fragment_id=${fragment_id}, resolution=${resolution}"

    local placer2_dir="${dataset_dir}/${run_name}/${placer_run_name}/${filter_run_name}/${placer2_run_name}"
    local filter2_dir="${placer2_dir}/${filter2_run_name}"
    local apo_structure="${dataset_dir}/${dataset}-aligned-structure.pdb"

    if [ ! -f "$apo_structure" ]; then
        echo "ERROR [${dataset}]: apo structure not found: ${apo_structure}"
        return 1
    fi

    build_final_model "${dataset_dir}" \
        "${placer2_dir}/*_refined.pdb" \
        "${filter2_dir}/cluster_rep_models.pdb" \
        "${apo_structure}" \
        ${run_name}/${placer_run_name}/${filter_run_name}/${placer2_run_name}/${filter2_run_name}/${final_run_name} \
        -r ${resolution}

    local build_exit=$?
    if [ $build_exit -ne 0 ]; then
        echo "ERROR [${dataset}]: build_final_model failed with exit code ${build_exit}"
        return 1
    fi

    echo "Completed: ${dataset}"
}
export -f build_final_process_dataset

do_build_final() {
    conda_activate "$CONDA_ENV_QFIT"

    echo "Starting run"
    local start_time=$(date +%s)
    cat "$DATASETS_FILE" | parallel -j "$NUM_PARALLEL_DEFAULT" build_final_process_dataset {}
    echo "All jobs completed"
    print_elapsed "$start_time"
}

######################################################################
# Stage 6b: rsr_final
######################################################################

rsr_final_process_dataset() {
    conda_activate "$CONDA_ENV_RSR"

    local dataset=$1
    local dataset_dir="${DATASETS_DIR}/${dataset}"

    local map_file
    map_file=$(find "${dataset_dir}" -maxdepth 1 -name "${dataset}-event_1*" | head -1)
    if [ -z "$map_file" ]; then
        echo "ERROR [${dataset}]: No event map found matching ${dataset}-event_1* in ${dataset_dir}"
        return 1
    fi

    local filter2_dir="${dataset_dir}/${run_name}/${placer_run_name}/${filter_run_name}/${placer2_run_name}/${filter2_run_name}"
    local cluster_reps_csv="${filter2_dir}/cluster_reps.csv"
    local final_dir="${filter2_dir}/${final_run_name}"

    if [ ! -f "$cluster_reps_csv" ]; then
        echo "ERROR [${dataset}]: cluster_reps.csv not found: $cluster_reps_csv"
        return 1
    fi

    # The i-th LIG residue in final_model.pdb (resid i, 1-based) corresponds
    # to the i-th (1-indexed) DATA row of filter2's cluster_reps.csv.
    local header=""
    IFS= read -r header < "$cluster_reps_csv"

    local cif_col_index=-1
    local i=0
    local header_cols
    IFS=, read -r -a header_cols <<< "$header"
    for col in "${header_cols[@]}"; do
        col="${col//$'\r'/}"
        if [ "$col" = "cif_restraints_file" ]; then
            cif_col_index=$i
        fi
        i=$((i + 1))
    done

    if [ $cif_col_index -lt 0 ]; then
        echo "ERROR [${dataset}]: cif_restraints_file column not found in $cluster_reps_csv"
        return 1
    fi

    local cif_paths=()
    local row_num=0
    local any_cif_missing=0
    while IFS=, read -r -a row_cols; do
        row_num=$((row_num + 1))
        [ $row_num -eq 1 ] && continue
        [ -z "${row_cols[0]}" ] && continue

        local cif_path="${row_cols[$cif_col_index]}"
        cif_path="${cif_path//$'\r'/}"
        cif_path="$(echo -n "$cif_path" | sed -e 's/^[[:space:]"'"'"']*//' -e 's/[[:space:]"'"'"']*$//')"

        if [ -z "$cif_path" ] || [ ! -f "$cif_path" ]; then
            echo "ERROR [${dataset}]: CIF not found for row ${row_num} of ${cluster_reps_csv}: '${cif_path}'"
            any_cif_missing=1
            continue
        fi

        cif_paths+=("$cif_path")
    done < "$cluster_reps_csv"

    if [ $any_cif_missing -ne 0 ]; then
        echo "ERROR [${dataset}]: One or more CIF restraint files were missing, aborting."
        return 1
    fi

    if [ ${#cif_paths[@]} -eq 0 ]; then
        echo "ERROR [${dataset}]: No data rows found in $cluster_reps_csv"
        return 1
    fi

    local cif_list
    cif_list=$(IFS=,; echo "${cif_paths[*]}")

    local final_pdb="${final_dir}/final_model.pdb"
    local residues_csv="${final_dir}/residues_with_placer_conformers.csv"
    local output_pdb="${final_dir}/final_model_refined.pdb"

    if [ ! -f "$final_pdb" ]; then
        echo "ERROR [${dataset}]: final_pdb not found: ${final_pdb}"
        return 1
    fi
    if [ ! -f "$residues_csv" ]; then
        echo "ERROR [${dataset}]: residues_csv not found: ${residues_csv}"
        return 1
    fi

    echo "[${dataset}] Map: $map_file"
    echo "[${dataset}] Using CIF restraints list (${#cif_paths[@]} entries): $cif_list"

    python "$RSR_SCRIPT_FINAL" \
        "$final_pdb" \
        "$residues_csv" \
        "$map_file" \
        "$output_pdb" \
        --cif-list "$cif_list"
    local exit_code=$?
    if [ $exit_code -ne 0 ]; then
        echo "ERROR [${dataset}]: Refinement failed with exit code $exit_code"
        return 1
    fi

    echo "Completed: ${dataset}"
}
export -f rsr_final_process_dataset

do_rsr_final() {
    conda_activate "$CONDA_ENV_RSR"
    export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1

    echo "Starting run"
    local start_time=$(date +%s)
    cat "$DATASETS_FILE" | parallel -j "$NUM_PARALLEL_DEFAULT" rsr_final_process_dataset {}
    echo "All jobs completed"
    print_elapsed "$start_time"

    unset OMP_NUM_THREADS OPENBLAS_NUM_THREADS MKL_NUM_THREADS NUMEXPR_NUM_THREADS VECLIB_MAXIMUM_THREADS
}

######################################################################
# Stage 6c: calc_final_refined_rscc
######################################################################

calc_final_refined_rscc_process_dataset() {
    conda_activate "$CONDA_ENV_QFIT"
    shopt -s nullglob

    local dataset=$1
    local dataset_dir="${DATASETS_DIR}/${dataset}"

    local lookup=$(grep "^${dataset} " "$LOOKUP_FILE")
    if [ -z "$lookup" ]; then
        echo "Warning: No match found for dataset ${dataset}, skipping."
        return 1
    fi

    local resolution=$(echo "$lookup" | awk '{print $3}')

    echo "Processing ${dataset}: resolution=${resolution}"

    local event_maps=("${dataset_dir}/${dataset}-event_"*)
    if [ ${#event_maps[@]} -eq 0 ]; then
        echo "Warning [${dataset}]: no event maps found matching ${dataset_dir}/${dataset}-event_*, skipping."
        return 1
    fi

    local final_dir="${dataset_dir}/${run_name}/${placer_run_name}/${filter_run_name}/${placer2_run_name}/${filter2_run_name}/${final_run_name}"
    local structure="${final_dir}/final_model_refined.pdb"

    if [ ! -f "$structure" ]; then
        echo "Warning [${dataset}]: final_model_refined.pdb not found: ${structure}, skipping."
        return 1
    fi

    local output_csv="${structure%.pdb}_rscc.csv"

    calc_rscc "${structure}" "${event_maps[@]}" "${resolution}" "${output_csv}"

    local calc_exit=$?
    if [ $calc_exit -ne 0 ]; then
        echo "ERROR [${dataset}]: calc_rscc failed on ${structure} with exit code ${calc_exit}"
        return 1
    fi

    echo "Completed [${dataset}]: ${structure} -> ${output_csv}"
}
export -f calc_final_refined_rscc_process_dataset

do_calc_final_rscc() {
    conda_activate "$CONDA_ENV_QFIT"

    echo "Starting run"
    local start_time=$(date +%s)
    cat "$DATASETS_FILE" | parallel -j "$NUM_PARALLEL_DEFAULT" calc_final_refined_rscc_process_dataset {}
    echo "All jobs completed"
    print_elapsed "$start_time"
}

######################################################################
# Stage 6d: reference-set comparison (only runs when -c is given)
######################################################################
# Per-residue RSCC comparison of final_model_refined vs reference, pooled
# across datasets, into GRAPHS_DIR/<run>/.../<final_run_name>/.

do_plot_residues_vs_ref_final() {
    conda_activate "$CONDA_ENV_EVAL"

    local out_dir="${GRAPHS_DIR}/${run_name}/${placer_run_name}/${filter_run_name}/${placer2_run_name}/${filter2_run_name}/${final_run_name}"
    echo "Starting run"
    local start_time=$(date +%s)
    python "$PLOT_RESIDUES_VS_REF_FINAL_PY" \
        "$run_name" "$placer_run_name" "$filter_run_name" "$placer2_run_name" "$filter2_run_name" "$final_run_name" \
        --datasets-dir "$DATASETS_DIR" --datasets-file "$DATASETS_FILE" \
        --ref-set "$REF_SET" --ref-pdb-pattern "$REF_SET_PDB_PATTERN" --graphs-dir "$out_dir"
    echo "All jobs completed"
    print_elapsed "$start_time"
}

######################################################################
# Stage 7a: plot_cluster_reps_rscc
######################################################################
# Pooled histograms of the cluster-rep RSCC values already written into
# cluster_reps.csv by filter/filter2 - no RSCC values are computed here.

do_plot_cluster_reps_rscc() {
    conda_activate "$CONDA_ENV_EVAL"

    echo "Starting run"
    local start_time=$(date +%s)
    python "$PLOT_CLUSTER_REPS_PY" \
        "$run_name" "$placer_run_name" "$filter_run_name" \
        "$placer2_run_name" "$filter2_run_name" "$final_run_name" \
        --datasets-dir "$DATASETS_DIR" --datasets-file "$DATASETS_FILE"
    echo "All jobs completed"
    print_elapsed "$start_time"
}

######################################################################
# Stage 7b: aggregate_protein_rscc
######################################################################
# Scatter plots comparing every protein residue's RSCC (apo vs backbone vs
# final), pooling the per-residue csvs already written by calc_apo_rscc,
# calc_backbone_refined_rscc, and calc_final_refined_rscc.

do_aggregate_protein_rscc() {
    conda_activate "$CONDA_ENV_EVAL"

    echo "Starting run"
    local start_time=$(date +%s)
    python "$AGGREGATE_PROTEIN_RSCC_PY" \
        "$run_name" "$placer_run_name" "$filter_run_name" \
        "$placer2_run_name" "$filter2_run_name" "$final_run_name" \
        --datasets-dir "$DATASETS_DIR" --datasets-file "$DATASETS_FILE"
    echo "All jobs completed"
    print_elapsed "$start_time"
}

######################################################################
# Stage 7c: aggregate_lig_rscc
######################################################################
# Same comparisons as stage 7b, restricted to the fitted LIG residue.

do_aggregate_lig_rscc() {
    conda_activate "$CONDA_ENV_EVAL"

    echo "Starting run"
    local start_time=$(date +%s)
    python "$AGGREGATE_LIG_RSCC_PY" \
        "$run_name" "$placer_run_name" "$filter_run_name" \
        "$placer2_run_name" "$filter2_run_name" "$final_run_name" \
        --datasets-dir "$DATASETS_DIR" --datasets-file "$DATASETS_FILE"
    echo "All jobs completed"
    print_elapsed "$start_time"
}

######################################################################
# Stage orchestration (unchanged shape: check -> run_step -> label)
######################################################################

stage0_apo_rscc() {
    run_step "Stage 0a: calc_apo_rscc" do_calc_apo_rscc
    if [ "$compare_ref_set" -eq 1 ]; then
        run_step "Stage 0b: calc_ref_set_rscc" do_calc_ref_set_rscc
    fi
}

stage1_run() {
    local rel_path="${run_name}"
    if stage_complete "$rel_path"; then
        echo "Skipping stage 1 (fit_ligand): ${rel_path} already exists for all datasets."
        return
    fi
    run_step "Stage 1: fit_ligand (${run_name})" do_fit_ligand
}

stage2_placer() {
    local rel_path="${run_name}/${placer_run_name}"
    if stage_complete "$rel_path"; then
        echo "Skipping stage 2 (placer/rsr_placer): ${rel_path} already exists for all datasets."
        return
    fi
    run_step "Stage 2a: placer (${placer_run_name})" do_placer
    run_step "Stage 2b: rsr_placer (${placer_run_name})" do_rsr_placer
}

stage3_filter() {
    local rel_path="${run_name}/${placer_run_name}/${filter_run_name}"
    if stage_complete "$rel_path"; then
        echo "Skipping stage 3 (filter/rsr_backbone/calc_backbone_refined_rscc): ${rel_path} already exists for all datasets."
    else
        run_step "Stage 3a: filter (${filter_run_name})" do_filter
        run_step "Stage 3b: rsr_backbone (${filter_run_name})" do_rsr_backbone
        run_step "Stage 3c: calc_backbone_refined_rscc (${filter_run_name})" do_calc_backbone_rscc
    fi
    # Reference-set comparison runs whenever -c is given, even if the stage's
    # main sub-steps above were skipped (e.g. this run/placer/filter already
    # existed from an earlier invocation without -c).
    if [ "$compare_ref_set" -eq 1 ]; then
        run_step "Stage 3d: plot_lig_vs_ref_filter1 (${filter_run_name})" do_plot_lig_vs_ref_filter1
        run_step "Stage 3e: plot_residues_vs_ref_backbone (${filter_run_name})" do_plot_residues_vs_ref_backbone
    fi
}

stage4_placer2() {
    local rel_path="${run_name}/${placer_run_name}/${filter_run_name}/${placer2_run_name}"
    if stage_complete "$rel_path"; then
        echo "Skipping stage 4 (placer2/rsr_placer2): ${rel_path} already exists for all datasets."
        return
    fi
    run_step "Stage 4a: placer2 (${placer2_run_name})" do_placer2
    run_step "Stage 4b: rsr_placer2 (${placer2_run_name})" do_rsr_placer2
}

stage5_filter2() {
    local rel_path="${run_name}/${placer_run_name}/${filter_run_name}/${placer2_run_name}/${filter2_run_name}"
    if stage_complete "$rel_path"; then
        echo "Skipping stage 5 (filter2): ${rel_path} already exists for all datasets."
    else
        run_step "Stage 5a: filter2 (${filter2_run_name})" do_filter2
    fi
    if [ "$compare_ref_set" -eq 1 ]; then
        run_step "Stage 5b: plot_lig_vs_ref_filter2 (${filter2_run_name})" do_plot_lig_vs_ref_filter2
    fi
}

stage6_final() {
    local rel_path="${run_name}/${placer_run_name}/${filter_run_name}/${placer2_run_name}/${filter2_run_name}/${final_run_name}"
    if stage_complete "$rel_path"; then
        echo "Skipping stage 6 (build_final/rsr_final/calc_final_refined_rscc): ${rel_path} already exists for all datasets."
    else
        run_step "Stage 6a: build_final (${final_run_name})" do_build_final
        run_step "Stage 6b: rsr_final (${final_run_name})" do_rsr_final
        run_step "Stage 6c: calc_final_refined_rscc (${final_run_name})" do_calc_final_rscc
    fi
    if [ "$compare_ref_set" -eq 1 ]; then
        run_step "Stage 6d: plot_residues_vs_ref_final (${final_run_name})" do_plot_residues_vs_ref_final
    fi
}

stage7_analysis() {
    local rel_path="${run_name}/${placer_run_name}/${filter_run_name}/${placer2_run_name}/${filter2_run_name}/${final_run_name}"
    if ! stage_complete "$rel_path"; then
        echo "Skipping stage 7 (analysis): ${rel_path} is not complete for all datasets yet."
        return
    fi
    run_step "Stage 7a: plot_cluster_reps_rscc (${final_run_name})" do_plot_cluster_reps_rscc
    run_step "Stage 7b: aggregate_protein_rscc (${final_run_name})" do_aggregate_protein_rscc
    run_step "Stage 7c: aggregate_lig_rscc (${final_run_name})" do_aggregate_lig_rscc
}

# --- Drive the requested stages, in order, stopping after the last name given ---
overall_start=$(date +%s)

stage0_apo_rscc
stage1_run

if [ -n "$placer_run_name" ]; then
    stage2_placer
fi

if [ -n "$filter_run_name" ]; then
    stage3_filter
fi

if [ -n "$placer2_run_name" ]; then
    stage4_placer2
fi

if [ -n "$filter2_run_name" ]; then
    stage5_filter2
fi

if [ -n "$final_run_name" ]; then
    stage6_final
    stage7_analysis
fi

overall_end=$(date +%s)
elapsed=$((overall_end - overall_start))
hours=$((elapsed / 3600))
minutes=$(((elapsed % 3600) / 60))
seconds=$((elapsed % 60))
echo "=== program.sh complete ==="
printf "Total time: %02d:%02d:%02d (HH:MM:SS)\n" $hours $minutes $seconds

