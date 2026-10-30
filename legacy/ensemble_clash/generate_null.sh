#!/bin/bash
#
# generate_null.sh - driver for the null-distribution construction pipeline (random ligand
# placements in each chain's binding pocket, scored against the density).
# Takes positional args corresponding to the stages of the pipeline:
#   conformer_run_name
#
# Runs, in order:
#   STAGE_1: Null conformer sampling
#   1a. build_clash_ensemble (once per run, before the parallel step)
#                                       -> logs/<conformer_run_name>/clash_ensemble.pkl
#   1b. generate_conformers             -> <dataset>/<conformer_run_name>/chain<X>/
#   1c. plot_ensemble_volumes (pooled)  -> GRAPHS_DIR/<conformer_run_name>/
#   1d. plot_conformer_acceptance (pooled) -> GRAPHS_DIR/<conformer_run_name>/
#   1e. plot_reference_coverage (pooled)   -> GRAPHS_DIR/<conformer_run_name>/
#   STAGE_2: per-atom EDIA scoring of the null conformers and reference ligands (per chain pocket)
#   2a. run_edia                        -> <dataset>/<conformer_run_name>/chain<X>/
#                                          edia_scores.csv, conformers/conformer_<n>.csv,
#                                          reference_edia.csv, reference_atom_scores.csv
#   2b. plot_edia_histogram             -> <dataset>/<conformer_run_name>/chain<X>/
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
# with the --exclude_resn residues (default HOH, DMS) removed from the model and resolution and
# CRYST1 from CSV_FILE.
# Stage 2: every pocket's conformers and the chain's reference ligands (each LIG residue of the
# dataset's structure, one molecule per altloc) are scored one molecule per EDIAscorer 1.3 call
# (the only way it writes per-atom scores), each against the dataset's structure with its LIG
# residues removed, cropped to the residues within --edia_crop_radius Å of the molecule (the map
# is never cropped) - so the references always share the run's map and settings. EDIAscorer
# 1.3 needs a newer glibc than this host's, so each pocket's calls run in one podman container
# (EDIA_IMAGE), --edia_workers at a time: stage 2 uses -p x --edia_workers cores. Each
# conformer's per-atom EDIA goes to conformers/conformer_<n>.csv (n = MODEL number in
# conformers.pdb), its EDIAm and mean atom EDIA to edia_scores.csv (the null distribution;
# references excluded), each reference's to reference_edia.csv and reference_atom_scores.csv,
# and a histogram of the pocket's null mean atom EDIA (0.01 bins) to
# mean_atom_edia_histogram.png. A failed call is recorded (status failed, files in
# edia_atom_failed/) and doesn't stop the pocket. Datasets without a map are skipped with a
# warning. Pockets scored by the earlier batched EDIAscorer 1.1 version (EDIAm only) are
# rescored; an interrupted pocket resumes from its finished calls (edia_atom_work/).
# Stage 3 pools stage 2 across every dataset into four histograms, each with its CSV: the
# reference ligands' mean atom EDIA (one point per LIG residue altloc), each pocket's median and
# mean null mean atom EDIA (one point per pocket), and each reference's mean atom EDIA
# percentile within its pocket's null distribution (ties counted as half).
#
# Per-pocket output: the null ensemble belongs to a chain's binding pocket, not to a modeled
# ligand. Every chain holding at least one ligand (resname LIG) gets one ensemble, sampled within
# a sphere around the observed ligand centroids of that chain (see below), in
# <dataset>/<conformer_run_name>/chain<X>/ (e.g. .../conformers_1/chainA, .../chainB); all of the
# chain's LIG residues are its reference ligands (the first is the atom-naming template).
#
# Clash ensemble and pocket volume: every dataset in the full DATASETS_FILE (even under
# --dataset) contributes its structure. Stage 1a builds, once per run, into
# logs/<conformer_run_name>/ (model list: ensemble_models.txt):
#   - pocket_residues.csv: binding-site residue numbers (chain agnostic), residues with a heavy
#     atom within --binding_site_cutoff Å (default 4) of a LIG residue in any structure;
#   - pocket_volume_<chain>.ccp4: per chain, every structure superposed onto REFERENCE_PDB on all
#     CA atoms, all their LIG heavy atoms in that chain made spheres of --pocket_atom_radius Å
#     (default 2, a cushion around the observed positions), with a probe of
#     --pocket_probe_radius Å (default 2) rolled over them: the volume the probe can't enter
#     (gaps narrower than the probe filled, no outward growth) (0.25 Å grid, REFERENCE_PDB
#     frame);
#   - each chain's sampling sphere: the minimum enclosing sphere of all structures' ligand
#     centroids in that chain (centre in REFERENCE_PDB's frame, moved into each sampled
#     structure's frame by the CA superposition), radius + --centroid_margin Å (default 2);
#     written to pocket_spheres.json;
#   - clash_ensemble.pkl: all of the above, plus each binding-site residue's distinct side-chain
#     conformations across all structures, each superposed onto the residue's backbone frame
#     (N, CA, C of REFERENCE_PDB);
#   - ensemble_aligned/: every structure superposed onto REFERENCE_PDB, for viewing.
# Every generate_conformers run loads the pkl. A placement is kept only if all its heavy atoms are
# inside the chain's pocket volume (the sampled structure superposed onto REFERENCE_PDB on all
# CA). Clashes: the sampled structure's own backbone is always a rigid obstacle; binding-site
# side chains come from the ensemble, grafted onto that backbone, and a placement is dropped only
# if, for some residue, it clashes with every one of its conformations; all other residues keep
# the sampled structure's own side chains (altlocs as conformations). No crystal symmetry mates
# are used.
#
# Modularity: each stage nests under the previous stage's run-name folder, so a new name
# branches off without touching prior results.
#
# Idempotency: every step checks whether its own output already exists - per chain (main
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
           [--pocket_atom_radius <float>] [--pocket_probe_radius <float>]
           [--centroid_margin <float>]
           [--binding_site_cutoff <float>]
           [--rotamer_tolerance <float>] [--map_pattern <pattern>]
           [--edia_crop_radius <float>] [--edia_workers <int>]

Only <conformer_run_name> is required.

Options:
  -p <num_parallel>        Number of datasets processed in parallel. Default: 1
  --overwrite              Force every requested step to re-run in place, even if its output
                            already exists (normally such a step is skipped - see "Idempotency"
                            in the header comment). Applies to every stage, including the
                            graphing steps.
  --replot                 Force just the graphing steps (1c-1e, 2b, 3) to re-run in place, even
                            if their output already exists. Does not affect generate_conformers
                            or EDIA scoring.
  --dataset <id[,id...]>   Run only on this dataset, or comma-separated list of datasets
                            (e.g. x00001-1 or x00001-1,x00002-1), instead of every dataset
                            listed in DATASETS_FILE (datasets.txt). Graphing steps are
                            skipped.
  --n_conformers <int>     Conformers: number of non-clashing placements kept per chain pocket.
                            Default: 1000.
  --n_torsion_confs <int>  Conformers: size of the RDKit ETKDG torsion conformer library.
                            Default: 200.
  --radius_step <float>    Conformers: translation distances are rounded to this step (Å).
                            Default: 0.1.
  --angle_step <float>     Conformers: translation polar/azimuth angles are rounded to this step
                            (degrees). Default: 1.
  --clash_scale <float>    Conformers: a ligand-protein atom pair clashes below this fraction of
                            the sum of their vdW radii. Default: 0.75.
  --polar_clash_scale <float>  Conformers: the same, for N-N, N-O and O-O pairs. Default: 0.6.
  --exclude_resn <resn[,resn...]>  Conformers and EDIA: comma-separated residue names left
                            out of the clash obstacles and of the EDIA target (besides
                            hydrogens and every LIG residue). Default: HOH,DMS.
  --max_candidates_per_conformer <int>  Conformers: stop after this many candidate placements
                            per conformer asked for (i.e. this x --n_conformers per chain
                            pocket) even if fewer than --n_conformers were kept. Default: 1000000.
  --seed <int>             Conformers: random seed. Default: a fresh seed per chain pocket from the
                            clock, recorded in each conformer_summary.json (params.seed).

  --mask_spacing <float>   Conformers: grid spacing (Å) of ensemble_mask.ccp4, the mask of every
                            voxel the kept placements occupy. Default: 0.25.
  --pocket_atom_radius <float>  Clash ensemble: each chain's pocket volume starts from the
                            ensemble's ligand heavy atoms as spheres of this radius (Å, every
                            element alike). Default: 2.
  --pocket_probe_radius <float>  Clash ensemble: radius (Å) of the probe rolled over those
                            spheres; gaps narrower than it are filled (0: just the spheres).
                            Default: 0.5.
  --centroid_margin <float>  Clash ensemble: the ligand centroid is sampled within the observed
                            ligand centroids' enclosing sphere, its radius grown by this much
                            (Å). Default: 2.
  --binding_site_cutoff <float>  Clash ensemble: residues with a heavy atom within this distance
                            (Å) of a LIG residue in any structure are binding-site residues
                            (ensemble side chains); all others keep the sampled structure's
                            own. Default: 4.
  --rotamer_tolerance <float>  Conformers: conformations of a residue across the clash ensemble
                            within this heavy-atom RMSD (Å) count as one. Default: 0.3.
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
  $0 conformers_2 --n_conformers 20000 --pocket_atom_radius 1.5
  $0 conformers_1 --map_pattern "{dataset}-event_1_1-BDC_0.3_map.native.ccp4"
  $0 conformers_1 -p 4 --edia_workers 4
EOF
    exit 1
}

# --- User-specified configuration: edit these for your environment ---
BASE_DIR="${BASE_DIR:-/home/ngupta/snEDIAm}"
CSV_FILE="${BASE_DIR}/pxr_fragments.csv"
DATASETS_DIR="${BASE_DIR}/datasets"
DATASETS_FILE="${BASE_DIR}/datasets.txt"
GRAPHS_DIR="${BASE_DIR}/graphs"
SNEDIAM_ANALYSIS_SCRIPTS_DIR="${BASE_DIR}/qfit-3.0/src/snediam_analysis_scripts"
CONDA_SH="/home/ngupta/miniconda3/etc/profile.d/conda.sh"
CONDA_ENV_NULL="snEDIAm"

# Per-dataset inputs, inside DATASETS_DIR/<dataset>/: the structure holding the modeled
# ligand(s).
STRUCTURE_PDB_PATTERN="{dataset}.pdb"
# Residue name of the modeled ligands; every chain holding one gets a null ensemble.
LIGAND_RESN="LIG"
# Structure whose residues define the backbone frames of the clash ensemble's side chains.
REFERENCE_PDB="${BASE_DIR}/reference.pdb"
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

for f in "$DATASETS_FILE" "$CSV_FILE" "$CONDA_SH" "$REFERENCE_PDB" "$PLOT_ENSEMBLE_VOLUMES_PY" \
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
# --radius-step 0.1, --angle-step 1, --clash-scale 0.75, --polar-clash-scale 0.6,
# --exclude-resn HOH DMS, --max-candidates-per-conformer 1000000, --seed from the clock,
# --mask-spacing 0.25).
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
pocket_atom_radius=""
pocket_probe_radius=""
centroid_margin=""
binding_site_cutoff=""
rotamer_tolerance=""
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
        --pocket_atom_radius)
            pocket_atom_radius="$2"
            shift 2
            ;;
        --pocket_probe_radius)
            pocket_probe_radius="$2"
            shift 2
            ;;
        --centroid_margin)
            centroid_margin="$2"
            shift 2
            ;;
        --binding_site_cutoff)
            binding_site_cutoff="$2"
            shift 2
            ;;
        --rotamer_tolerance)
            rotamer_tolerance="$2"
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
echo "clash_scale: ${clash_scale:-(not set - generate_conformers default 0.75)}"
echo "polar_clash_scale: ${polar_clash_scale:-(not set - generate_conformers default 0.6)}"
echo "exclude_resn: ${exclude_resn:-(not set - generate_conformers default HOH,DMS)}"
echo "max_candidates_per_conformer: ${max_candidates_per_conformer:-(not set - generate_conformers default 1000000)}"
echo "seed: ${seed:-(not set - generate_conformers picks one per chain pocket from the clock)}"
echo "mask_spacing: ${mask_spacing:-(not set - generate_conformers default 0.25)}"
echo "pocket_atom_radius: ${pocket_atom_radius:-(not set - build_clash_ensemble default 2)}"
echo "pocket_probe_radius: ${pocket_probe_radius:-(not set - build_clash_ensemble default 2)}"
echo "centroid_margin: ${centroid_margin:-(not set - build_clash_ensemble default 2)}"
echo "binding_site_cutoff: ${binding_site_cutoff:-(not set - build_clash_ensemble default 4)}"
echo "rotamer_tolerance: ${rotamer_tolerance:-(not set - build_clash_ensemble default 0.3)}"
[ -n "$map_pattern" ] && MAP_NAME_PATTERN="$map_pattern"
echo "map_pattern: ${MAP_NAME_PATTERN}"
echo "edia_crop_radius: ${edia_crop_radius:-(not set - run_edia default 5)}"
echo "edia_workers: ${edia_workers:-(not set - run_edia default 1)}"

# Clash ensemble: the structure of every dataset in the full DATASETS_FILE, written before any
# --dataset override so partial runs use the same ensemble.
ENSEMBLE_LIST="${log_dir}/ensemble_models.txt"
CLASH_ENSEMBLE="${log_dir}/clash_ensemble.pkl"
grep -v '^[[:space:]]*$' "$DATASETS_FILE" | tr -d '\r' | while read -r _ds; do
    _pdb="${DATASETS_DIR}/${_ds}/${STRUCTURE_PDB_PATTERN//\{dataset\}/${_ds}}"
    if [ -f "$_pdb" ]; then
        echo "$_pdb"
    else
        echo "Warning: ${_pdb} not found; left out of the clash ensemble" >&2
    fi
done > "$ENSEMBLE_LIST"
echo "clash ensemble: $(wc -l < "$ENSEMBLE_LIST") structures (${ENSEMBLE_LIST})"

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
export pocket_atom_radius pocket_probe_radius centroid_margin binding_site_cutoff rotamer_tolerance ENSEMBLE_LIST CLASH_ENSEMBLE
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

# list_ligands <pdb_file>: prints one "<chain> <resnum>" line per LIGAND_RESN residue in the
# file, in file order.
list_ligands() {
    local pdb_file=$1
    awk -v resn="$LIGAND_RESN" '
        ($1 == "ATOM" || $1 == "HETATM") {
            r = substr($0, 18, 3); gsub(/ /, "", r)
            if (r != resn) next
            chain = substr($0, 22, 1); gsub(/ /, "", chain)
            resnum = substr($0, 23, 4); gsub(/ /, "", resnum)
            key = chain " " resnum
            if (!(key in seen)) { seen[key] = 1; print key }
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
# Stage 1b: generate_conformers
######################################################################
# For every chain of the dataset's structure that holds a ligand, samples non-clashing placements
# (torsion conformer + translation + rotation) within that chain's sampling sphere and pocket
# volume (both from the clash ensemble). One ensemble per
# chain, however many LIG residues the chain has (they're all passed as reference ligands).

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

    # Chains holding a ligand, in file order, with their LIG residue numbers.
    local chains=() chain resnum
    declare -A chain_resis=()
    for lig in "${ligands[@]}"; do
        read -r chain resnum <<< "$lig"
        if [ -z "${chain_resis[$chain]+x}" ]; then
            chains+=("$chain")
            chain_resis[$chain]="$resnum"
        else
            chain_resis[$chain]+=" $resnum"
        fi
    done

    conda_activate "$CONDA_ENV_NULL"

    local out_dir label
    for chain in "${chains[@]}"; do
        label="chain${chain}"
        out_dir="${dataset_dir}/${conformer_run_name}/${label}"
        if [ "$overwrite" -ne 1 ] && files_exist "${out_dir}/conformer_summary.json"; then
            echo "Skipping [${dataset} ${label}]: conformers already complete (${out_dir}/conformer_summary.json exists)."
            continue
        fi

        mkdir -p "$out_dir"
        write_params_txt "${out_dir}/conformer_params.txt" \
            "structure=${structure}" \
            "chain=${chain}" \
            "reference_ligands=${chain_resis[$chain]}" \
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
            "clash_ensemble=${CLASH_ENSEMBLE}" \
            "rotamer_tolerance=${rotamer_tolerance}" \
            "pocket_atom_radius=${pocket_atom_radius}" \
            "pocket_probe_radius=${pocket_probe_radius}" \
            "centroid_margin=${centroid_margin}" \
            "binding_site_cutoff=${binding_site_cutoff}"

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

        echo "Processing [${dataset} ${label}]: sampling sphere from the clash ensemble, reference ${LIGAND_RESN} ${chain} ${chain_resis[$chain]}"
        generate_conformers "$structure" "$CSV_FILE" "$dataset" "$chain" "$out_dir" \
            --reference-resi ${chain_resis[$chain]} \
            --clash-ensemble "$CLASH_ENSEMBLE" \
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

######################################################################
# Stage 1a: build_clash_ensemble
######################################################################
# Once per run, before the parallel step: saves the binding-site residues, each binding-site
# residue's distinct side-chain conformations across every ensemble structure (in REFERENCE_PDB's
# residue frames), each chain's pocket volume and sampling sphere; every generate_conformers run
# loads it.

do_build_clash_ensemble() {
    if [ "$overwrite" -ne 1 ] && [ -f "$CLASH_ENSEMBLE" ]; then
        echo "Skipping: clash ensemble already built (${CLASH_ENSEMBLE} exists; pass --overwrite to rebuild)."
        return 0
    fi
    local extra_args=()
    [ -n "$rotamer_tolerance" ] && extra_args+=(--rotamer-tolerance "$rotamer_tolerance")
    [ -n "$pocket_atom_radius" ] && extra_args+=(--pocket-atom-radius "$pocket_atom_radius")
    [ -n "$pocket_probe_radius" ] && extra_args+=(--pocket-probe-radius "$pocket_probe_radius")
    [ -n "$centroid_margin" ] && extra_args+=(--centroid-margin "$centroid_margin")
    [ -n "$binding_site_cutoff" ] && extra_args+=(--binding-site-cutoff "$binding_site_cutoff")
    [ -n "$exclude_resn" ] && extra_args+=(--exclude-resn ${exclude_resn//,/ })
    conda_activate "$CONDA_ENV_NULL"
    build_clash_ensemble "$REFERENCE_PDB" "$ENSEMBLE_LIST" "$CLASH_ENSEMBLE" \
        --aligned-dir "${log_dir}/ensemble_aligned" "${extra_args[@]}"
    local status=$?
    conda_deactivate
    return $status
}

do_conformers() {
    echo "Starting run"
    local start_time=$(date +%s)
    printf '%s\n' "${DATASETS[@]}" | parallel -j "$NUM_PARALLEL_DEFAULT" conformers_process_dataset {}
    echo "All jobs completed"
    print_elapsed "$start_time"
}

######################################################################
# Stage 2: per-atom EDIA scoring of the null conformers and reference ligands (2a) and mean
# atom EDIA histogram (2b), per chain pocket
######################################################################
# For every chain<X> pocket of the dataset's run folder that has conformers: scores each of its
# conformers and the chain's reference ligands alone (run_edia) against the dataset's density
# map, then plots the pocket's null mean atom EDIA histogram (plot_edia_histogram.py). A pocket
# counts as scored only if its edia_params.json records per-atom scoring; run_edia decides
# between full scoring, resuming, and scoring just the references.

edia_process_dataset() {
    local dataset=$1
    local dataset_dir="${DATASETS_DIR}/${dataset}"
    local structure="${dataset_dir}/${STRUCTURE_PDB_PATTERN//\{dataset\}/${dataset}}"
    local map="${dataset_dir}/${MAP_NAME_PATTERN//\{dataset\}/${dataset}}"

    if [ ! -f "$map" ]; then
        echo "Warning [${dataset}]: density map not found: ${map}, skipping EDIA."
        return 1
    fi
    local pockets=()
    mapfile -t pockets < <(find "${dataset_dir}/${conformer_run_name}" -mindepth 1 -maxdepth 1 \
        -type d -name 'chain*' 2>/dev/null | sort)
    if [ ${#pockets[@]} -eq 0 ]; then
        echo "Warning [${dataset}]: no ${conformer_run_name}/chain* pockets, skipping EDIA."
        return 1
    fi

    conda_activate "$CONDA_ENV_NULL"

    local pocket chain label status scored
    for pocket in "${pockets[@]}"; do
        label="${pocket##*/}"
        chain="${label#chain}"
        scored=0
        if [ ! -f "${pocket}/conformers.pdb" ]; then
            echo "Skipping [${dataset} ${label}]: no conformers.pdb."
            continue
        fi

        if [ "$overwrite" -ne 1 ] && files_exist "${pocket}/edia_scores.csv" "${pocket}/reference_edia.csv" \
                && grep -q '"mode": "per_atom"' "${pocket}/edia_params.json" 2>/dev/null; then
            echo "Skipping [${dataset} ${label}]: EDIA already complete (edia_scores.csv and reference_edia.csv exist, scored per atom)."
        else
            write_params_txt "${pocket}/edia_params.txt" \
                "map=${map}" \
                "map_pattern=${MAP_NAME_PATTERN}" \
                "structure=${structure}" \
                "fragments_csv=${CSV_FILE}" \
                "exclude_resn=${exclude_resn}" \
                "ediascorer=${EDIASCORER}" \
                "container_image=${EDIA_IMAGE}" \
                "crop_radius=${edia_crop_radius}" \
                "workers=${edia_workers}"
            local exclude_args=() overwrite_args=() tunable_args=()
            [ -n "$exclude_resn" ] && exclude_args=(--exclude-resn ${exclude_resn//,/ })
            [ "$overwrite" -eq 1 ] && overwrite_args=(--overwrite)
            [ -n "$edia_crop_radius" ] && tunable_args+=(--crop-radius "$edia_crop_radius")
            [ -n "$edia_workers" ] && tunable_args+=(--workers "$edia_workers")
            echo "Processing [${dataset} ${label}]: per-atom EDIA against ${map##*/}"
            # The log is appended to: an interrupted pocket resumes, and its earlier progress
            # stays readable.
            run_edia "$dataset_dir" "$conformer_run_name" "$chain" \
                --map "$map" --structure "$structure" --fragments-csv "$CSV_FILE" \
                --ediascorer "$EDIASCORER" --container-image "$EDIA_IMAGE" \
                "${tunable_args[@]}" "${overwrite_args[@]}" "${exclude_args[@]}" \
                >> "${pocket}/run_edia_log.txt" 2>&1
            status=$?
            if [ $status -ne 0 ]; then
                echo "ERROR [${dataset} ${label}]: run_edia failed with exit code ${status} (see ${pocket}/run_edia_log.txt)"
                continue
            fi
            echo "Completed [${dataset} ${label}]: $(tail -n 1 "${pocket}/run_edia_log.txt")"
            scored=1
        fi

        # Replotted whenever the pocket was (re)scored just now.
        if [ "$overwrite" -ne 1 ] && [ "$replot" -ne 1 ] && [ "$scored" -ne 1 ] \
                && files_exist "${pocket}/mean_atom_edia_histogram.png"; then
            echo "Skipping [${dataset} ${label}]: mean atom EDIA histogram exists (pass --replot to redo)."
        elif ! python "$PLOT_EDIA_HISTOGRAM_PY" "$dataset_dir" "$conformer_run_name" "$chain"; then
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
# Stages 1c-1e and 3: pooled plots
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

run_step "Stage 1a: build_clash_ensemble (${conformer_run_name})" do_build_clash_ensemble
run_step "Stage 1b: generate_conformers (${conformer_run_name})" do_conformers
run_step_pooled_replot "Stage 1c: plot_ensemble_volumes (${conformer_run_name})" \
    ensemble_volumes_outputs_exist do_plot_ensemble_volumes
run_step_pooled_replot "Stage 1d: plot_conformer_acceptance (${conformer_run_name})" \
    conformer_acceptance_outputs_exist do_plot_conformer_acceptance
run_step_pooled_replot "Stage 1e: plot_reference_coverage (${conformer_run_name})" \
    reference_coverage_outputs_exist do_plot_reference_coverage
run_step "Stage 2: per-atom EDIA scoring of the null conformers and reference ligands, mean atom EDIA histograms (${conformer_run_name}, map ${MAP_NAME_PATTERN})" do_edia
run_step_pooled_replot "Stage 3: EDIA evaluation plots (${conformer_run_name})" \
    edia_evaluation_outputs_exist do_plot_edia_evaluation

echo "========= generate_null.sh complete ========="
print_elapsed "$overall_start" "Total"
