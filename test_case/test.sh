#!/bin/bash
#
# program.sh - combined driver for the full ligand-fitting/PLACER/RSR pipeline.
#takes in 8 positional args corresponding to the stages of the pipeline:
#run_name, placer_run_name, filter_run_name, placer2_run_name, filter2_run_name, final_run_name, rotamer_run_name, despot_run_name
#
# Runs, in order:
#   0a. convert_ligs                     -> LIG_PDB_DIR/<ligand_name>*/<ligand_name>*.mol2
#   0b. calc_apo_rscc                    -> <dataset>/<dataset>-aligned-structure_rscc.csv
#   0c. calc_ref_set_rscc (only with -c) -> REF_SET/<dataset>/<REF_SET_PDB_PATTERN%.pdb>_rscc.csv
#   0d. ref_set_despot (only with -c and <despot_run_name>): symmetry_expand + mol2 conversion +
#       DESPOT score_complex.py on the reference structure
#                                        -> REF_SET/<dataset>/<dataset>_DESPOT.csv
#   STAGE_1: Fit_ligand
#   1a. fit_ligand                      -> <run_name>/
#   1b. plot_fit_ligand_counts (always) -> GRAPHS_DIR/<run_name>/
#   1c. centroid_rmsd_all (only with -c) -> GRAPHS_DIR/<run_name>/
#   STAGE_2: PLACER
#   2a. placer                          -> <run_name>/<placer_run_name>/
#   2b. rsr_placer                      -> <run_name>/<placer_run_name>/
#   2c. calc_placer_sampling (refined + unrefined, only with -c)
#                                        -> GRAPHS_DIR/<run_name>/<placer_run_name>/
#   STAGE_3: Filter
#   3a. filter                          -> .../<filter_run_name>/
#   3b. rsr_backbone                    -> .../<filter_run_name>/
#   3c. calc_backbone_refined_rscc      -> .../<filter_run_name>/
#   3d. plot_lig_vs_ref_filter1, plot_residues_vs_ref_backbone (only with -c)
#                                        -> GRAPHS_DIR/<run_name>/.../<filter_run_name>/
#   STAGE_4: PLACER2
#   4a. placer2                         -> .../<placer2_run_name>/
#   4b. rsr_placer2                     -> .../<placer2_run_name>/
#   4c. calc_placer_sampling (refined + unrefined, only with -c)
#                                        -> GRAPHS_DIR/<run_name>/.../<placer2_run_name>/
#   STAGE_5: Filter2
#   5a. filter2 (runs the same `filter` script as stage 3a, not `filter_all`)
#                                        -> .../<filter2_run_name>/
#   5b. plot_lig_vs_ref_filter2 (only with -c)
#                                        -> GRAPHS_DIR/<run_name>/.../<filter2_run_name>/
#   STAGE_6: Build final (old name)
#   6a. build_final                     -> .../<final_run_name>/
#   6b. rsr_final                       -> .../<final_run_name>/
#   6c. calc_final_refined_rscc         -> .../<final_run_name>/
#   6d. plot_residues_vs_ref_final (only with -c)
#                                        -> GRAPHS_DIR/<run_name>/.../<final_run_name>/
#   6e. aggregate_clash_groups (always): concatenates every dataset's sidechain_clash_groups.csv
#                                        -> GRAPHS_DIR/<run_name>/.../<final_run_name>/
#                                           sidechain_clash_groups_combined.csv
#   STAGE_7: Rotamer_optimize
#   7a. rotamer_optimize (only with <rotamer_run_name>): resamples chi/aromatic rotamers for
#       low-RSCC residues in final_model.pdb
#                                        -> .../<final_run_name>/<rotamer_run_name>/
#                                           rotamer_optimized.pdb + fitted.pdb + residue_rscc.csv
#   7b. rsr_rotamer                     -> .../<final_run_name>/<rotamer_run_name>/rotamer_refined.pdb
#   7c. calc_rotamer_refined_rscc       -> .../<final_run_name>/<rotamer_run_name>/
#                                           rotamer_refined_rscc.csv
#   7d. select_optimized_residues: reverts a residue to final_model_refined's conformation unless
#       rotamer_refined_rscc.csv beats it by at least REVERT_MIN_DIFF (0.1)
#                                        -> .../<final_run_name>/<rotamer_run_name>/
#                                           optimized.pdb + optimized_rscc.csv + reverted_residues.csv
#   7e. plot_residues_vs_ref_rotamer (only with -c)
#                                        -> GRAPHS_DIR/<run_name>/.../<final_run_name>/
#                                           <rotamer_run_name>/rotamer_refined_vs_reference_rscc_restricted.png
#   7f. aggregate_rotamer_worse_residues (only with -c): pooled csv of residues whose optimized
#       RSCC is >0.1 worse than reference or final_model_refined
#                                        -> GRAPHS_DIR/<run_name>/.../<final_run_name>/
#                                           <rotamer_run_name>/rotamer_refined_worse_residues.csv
#   STAGE_8: Despot
#   8a. despot (only with <rotamer_run_name> and <despot_run_name>): pools placer2 conformers,
#       expands optimized.pdb around them, converts to mol2, scores with DESPOT's
#       score_complex.py, then despot_filter reselects the per-cluster winner and resets any
#       residue left unbacked by a rejected ligand to apo
#                                        -> .../<final_run_name>/<rotamer_run_name>/<despot_run_name>/
#                                           despot_filtered.pdb + despot_filtered_scores.csv +
#                                           cluster_reps.csv + modified_residues.csv
#   8b. plot_lig_vs_ref_despot (only with -c)
#                                        -> GRAPHS_DIR/<run_name>/.../<final_run_name>/<rotamer_run_name>/<despot_run_name>/
#   8c. plot_despot_vs_ref (only with -c)
#                                        -> GRAPHS_DIR/<run_name>/.../<final_run_name>/<rotamer_run_name>/<despot_run_name>/
#   8d. plot_rscc_despot_tradeoff (only with -c)
#                                        -> GRAPHS_DIR/<run_name>/.../<final_run_name>/<rotamer_run_name>/<despot_run_name>/
#   8e. plot_residues_vs_ref_despot (only with -c): restricted to despot_run_name/
#       modified_residues.csv instead of 7e's residues_with_placer_conformers.csv
#                                        -> GRAPHS_DIR/<run_name>/.../<final_run_name>/<rotamer_run_name>/<despot_run_name>/
#   STAGE_9: Internal analysis
#   9.  analysis_scripts/*.py: cluster-rep/per-residue RSCC plots + pooled counterparts, once
#       <final_run_name> is given; plus plot_rotamer_vs_pipeline (moved from the old 7f) once
#       <rotamer_run_name> is given; plus plot_despot_energies+pooled and
#       plot_despot_ligand_summary+single (moved from the old 8b/8d) once <despot_run_name> is
#       also given - one idempotent unit, not sub-lettered. Nests under
#       .../<rotamer_run_name>/<despot_run_name>/ if that directory exists, else
#       .../<rotamer_run_name>/ if only rotamer_run_name is given, else .../<final_run_name>/.
#                                        -> GRAPHS_DIR/<run_name>/.../<final_run_name>/[<rotamer_run_name>/[<despot_run_name>/]]
#
# Modularity: pass only as many of the eight run-name arguments as you want to run through.
# Each stage nests under the previous stage's directory, so a new name at any point branches off
# without touching prior results. Stage 0 is dataset-scoped, not run-name-scoped, so it always
# runs. Stage 7 only runs once <rotamer_run_name> is given; stage 8 only once <despot_run_name>
# is also given (error if given without <rotamer_run_name>); stage 9 always runs once
# <final_run_name> is given, independent of stages 7/8, doing more as they become available.
#
# Idempotency: every step checks whether its own output already exists - per dataset (main
# pipeline steps 0b-0d, 7a-7c, 8a) or per run (graphing steps 1b, 1c, 2c, 3d, 4c, 5b, 6d, 7e, 7f,
# 8b, 8c, 8d, 8e, 9) - and skips if so. PLACER/RSR steps use a loose "at least one output exists"
# check. Pass --overwrite to force every step to redo, or --replot for just the graphing steps.
#
# Dataset scoping: by default every stage runs over every dataset in DATASETS_FILE
# (datasets.txt). Pass --dataset <id[,id...]> to restrict the whole invocation to just those
# datasets. Every graphing step is skipped entirely when --dataset is given.

set -uo pipefail

usage() {
    cat <<EOF
Usage: $0 <run_name> [placer_run_name [filter_run_name [placer2_run_name [filter2_run_name [final_run_name [rotamer_run_name [despot_run_name]]]]]]]
           [-n <num_placer_confs>] [-n2 <num_placer2_confs>] [-g <gpu_ids>] [-p <num_parallel>] [-c] [--overwrite] [--replot]
           [--dataset <id[,id...]>]
           [--z_threshold <float>] [--num_peaks <int>]
           [--f1_filter_proportion <float>] [--f1_min_cluster_proportion <float>]
           [--f1_rscc_cutoff <float>] [--f1_clustering_mode <all-atom|centroid>]
           [--f1_clustering_cutoff <float>]
           [--f2_filter_proportion <float>] [--f2_min_cluster_proportion <float>]
           [--f2_rscc_cutoff <float>] [--f2_clustering_mode <all-atom|centroid>]
           [--f2_clustering_cutoff <float>] [--f1_clash_vdw_scale <float>] [--f2_clash_vdw_scale <float>]
           [--despot_threshold <float>]
           [--despot_rscc_threshold <float>] [--despot_rscc_weight <float>]
           [--fit_ligand_rmsd_cutoff <float>]
           [--clash_vdw_scale <float>] [--hbond_clash_vdw_scale <float>]
           [--max_clash_group_size <int>] [--max_clash_group_expansions <int>]
           [--clash_domain_top_k <int>] [--clash_solve_node_budget <int>]
           [--rotamer_rscc_threshold <float>] [--rotamer_rscc_improvement_threshold <float>]
           [--revert_min_diff <float>] [--bfactor <float>] [--expand_distance_cutoff <float>]
           [--rsr_n_cycles <int>] [--rsr_map_weight <float>]
           [--rsr_backbone_cutoff <float>] [--rsr_moved_threshold <float>]

Only <run_name> is required. Supplying fewer than all eight names runs only
that many stages of the pipeline (see header comment for the stage list).

Options:
  -n <num_placer_confs>    Number of PLACER conformers for round 1 (placer -n). Default: 1000
  -n2 <num_placer2_confs>  Number of PLACER conformers for round 2 (placer2 -n). Default: 1000
  -g <gpu_ids>             Comma-separated GPU ids for both PLACER rounds. Default: 0
  -p <num_parallel>        CPU parallelism for every non-PLACER stage 
  -c                       Also compare results to the reference set (REF_SET). 
  --overwrite              Force every requested step to re-run in place, even if its output
                            already exists (normally such a step is skipped - see "Idempotency"
                            in the header comment). Applies to every stage, including the
                            graphing steps. 
  --replot                 Force just the graphing steps (1b, 1c, 2c, 3d, 4c, 5b, 6d, 7e, 7f, 8b, 8c, 8d, 8e, 9) to
                            re-run in place, even if their output already exists. Does not
                            affect the main pipeline steps (use --overwrite for those too).
  --dataset <id[,id...]>   Run only on this dataset, or comma-separated list of datasets
                            (e.g. x00001-1 or x00001-1,x00002-1), instead of every dataset
                            listed in DATASETS_FILE (datasets.txt). 
  --z_threshold <float>            Fit_ligand: Z-score threshold for peak detection. Default: 4.
  --num_peaks <int>                Fit_ligand: number of peaks to find. Default: 100.
  --fit_ligand_rmsd_cutoff <float> Fit_ligand: RMSD below which two candidate peaks are treated
                                    as the same peak. Default: 2.
  --rsr_n_cycles <int>              Real-space refinement cycles. PLACER, Filter, PLACER2,
                                    Build final, Rotamer_optimize. Default: 1000.
  --rsr_map_weight <float>         Real-space refinement map-vs-geometry weight. PLACER, Filter,
                                    PLACER2, Build final, Rotamer_optimize. Default: 50.0.
  --f1_filter_proportion <float>       Filter: proportion of conformers kept. Default: 0.25.
  --f1_min_cluster_proportion <float>  Filter: min cluster-size proportion to keep. Default: 0.1.
  --f1_rscc_cutoff <float>             Filter: minimum RSCC to keep a cluster rep. Default: 0.6.
  --f1_clustering_mode <all-atom|centroid>  Filter: clustering distance metric. Default: centroid.
  --f1_clustering_cutoff <float>       Filter: clustering distance cutoff. Default: 2.0.
  --f1_clash_vdw_scale <float>         Filter: VDW-radius scale for cluster-rep clash detection.
                                    Default: 0.75.
  --rsr_backbone_cutoff <float>    Filter: distance from LIG used to pick residues to
                                    real-space refine. Default: 10.0.
  --rsr_moved_threshold <float>    Real-space refinement: minimum displacement to log a residue
                                    as "moved". Filter, Build final, Rotamer_optimize. Default: 0.01.
  --f2_filter_proportion <float>       Filter2: proportion of conformers kept. Default: 0.25.
  --f2_min_cluster_proportion <float>  Filter2: min cluster-size proportion to keep. Default: 0.1.
  --f2_rscc_cutoff <float>             Filter2: minimum RSCC to keep a cluster rep. Default: 0.6.
  --f2_clustering_mode <all-atom|centroid>  Filter2: clustering distance metric. Default: centroid.
  --f2_clustering_cutoff <float>       Filter2: clustering distance cutoff. Default: 2.0.
  --f2_clash_vdw_scale <float>         Filter2: VDW-radius scale for cluster-rep clash detection.
                                    Default: 0.75.
  --clash_vdw_scale <float>        Sidechain clash VDW-radius scale, shared by Build final and
                                    Rotamer_optimize. Default: 0.75.
  --hbond_clash_vdw_scale <float>  Same as --clash_vdw_scale for an (N, O) hydrogen-bonded atom
                                    pair. Build final, Rotamer_optimize. Default: 0.6.
  --max_clash_group_size <int>     Max residues absorbed into one clash group. Build final,
                                    Rotamer_optimize. Default: 8.
  --max_clash_group_expansions <int>   Max rounds absorbing new external clashes into a clash
                                    group. Build final, Rotamer_optimize. Default: 10.
  --clash_domain_top_k <int>       Candidates considered per residue during clash-group solving.
                                    Build final, Rotamer_optimize. Default: 25.
  --clash_solve_node_budget <int>  Branch-and-bound search nodes before falling back to ICM.
                                    Build final, Rotamer_optimize. Default: 200000.
  --rotamer_rscc_threshold <float> Rotamer_optimize: RSCC below which a residue is resampled.
                                    Default: 0.5.
  --rotamer_rscc_improvement_threshold <float>  Rotamer_optimize: minimum RSCC gain to accept a
                                    resampled rotamer. Default: 0.1.
  --revert_min_diff <float>        Rotamer_optimize: minimum RSCC gap to revert a residue to its
                                    pre-optimization conformation. Default: 0.1.
  --bfactor <float>                B-factor used for RSCC model density. Filter, Build final,
                                    Rotamer_optimize, Despot. Default: 20.
  --expand_distance_cutoff <float> Despot: symmetry-mate distance cutoff (Å) from a ligand atom.
                                    Default: 10.
  --despot_threshold <float>       Despot: max per-heavy-atom-normalized DESPOT score a winning
                                    pose may have to survive. Default: -1.0.
  --despot_rscc_threshold <float>  Despot: minimum RSCC a winning pose must have to survive.
                                    Default: 0.6.
  --despot_rscc_weight <float>     Despot: weight on normalized DESPOT score when picking the
                                    Pareto-front winner. Default: 0.05.

Examples:
  $0 run_1 placer_1 filter_1 placer2_1 filter2_1 final_1
  $0 run_1 placer_1 filter_1
  $0 run_1 placer_1 filter_2 placer2_1 filter2_1 final_1 -n 1000 -n2 500 -g 0,1
  $0 run_1 placer_1 filter_1 --overwrite
  $0 run_1 placer_1 filter_1 placer2_1 filter2_1 final_1 --replot
  $0 run_1 placer_1 filter_1 placer2_1 filter2_1 final_1 -c
  $0 run_1 placer_1 filter_1 --z_threshold 5 --num_peaks 50
  $0 run_1 placer_1 filter_1 placer2_1 filter2_1 final_1 --f1_rscc_cutoff 0.5 --f2_rscc_cutoff 0.7
  $0 run_1 placer_1 filter_1 --dataset x00001-1
  $0 run_1 placer_1 filter_1 --dataset x00001-1,x00002-1,x00003-1
  $0 run_1 placer_1 filter_1 placer2_1 filter2_1 final_1 rotamer_1
  $0 run_1 placer_1 filter_1 placer2_1 filter2_1 final_1 rotamer_1 despot_1
EOF
    exit 1
}

# --- User-specified configuration: edit these for your environment ---
BASE_DIR="/home/ngupta/main/program_rotamer/qfit-3.0/test_case"
CSV_FILE="${BASE_DIR}/pxr_fragments.csv"
LIG_PDB_DIR="${BASE_DIR}/ligands"
CONDA_SH="/home/ngupta/miniconda3/etc/profile.d/conda.sh"
CONDA_ENV_QFIT="nikhils_program"
CONDA_ENV_PLACER="placer_env"
CONDA_ENV_RSR="nikhils_program"
CONDA_ENV_EVAL="nikhils_program"
CONDA_ENV_DESPOT="DESPOT"
RUN_PLACER_PY="/home/ngupta/PLACER/PLACER/run_PLACER.py"
DESPOT_SCRIPT="/home/ngupta/DESPOT/scripts/score_complex.py"
DESPOT_DATABASE="CROWN"
DATASETS_DIR="${BASE_DIR}/datasets"
DATASETS_FILE="${BASE_DIR}/datasets.txt"
RSR_SCRIPTS_DIR="/home/ngupta/main/program_rotamer/qfit-3.0/src/rsr_scripts"
ANALYSIS_SCRIPTS_DIR="/home/ngupta/main/program_rotamer/qfit-3.0/src/analysis_scripts"
LIG_SCRIPTS_DIR="/home/ngupta/main/program_rotamer/qfit-3.0/src/lig_scripts"
GRAPHS_DIR="${BASE_DIR}/graphs"

# Only used when -c is given: reference_set/<dataset>/ subfolders (one per
# datasets.txt entry) holding a reference structure to compare RSCC against.
REF_SET="${BASE_DIR}/reference_set"
REF_SET_PDB_PATTERN="{dataset}-pandda-model.pdb"

# --- Derived paths: assumed to live at fixed locations under BASE_DIR ---
RSR_SCRIPT_LIGAND="${RSR_SCRIPTS_DIR}/real_space_refine.py"
RSR_SCRIPT_PROTEIN="${RSR_SCRIPTS_DIR}/real_space_refine_protein.py"
RSR_SCRIPT_FINAL="${RSR_SCRIPTS_DIR}/real_space_refine_final.py"
PLOT_CLUSTER_REPS_PY="${ANALYSIS_SCRIPTS_DIR}/plot_cluster_reps_rscc.py"
AGGREGATE_PROTEIN_RSCC_PY="${ANALYSIS_SCRIPTS_DIR}/aggregate_protein_rscc.py"
AGGREGATE_LIG_RSCC_PY="${ANALYSIS_SCRIPTS_DIR}/aggregate_lig_rscc.py"
PLOT_LIG_VS_REF_FILTER1_PY="${ANALYSIS_SCRIPTS_DIR}/plot_lig_vs_ref_filter1.py"
PLOT_LIG_VS_REF_FILTER2_PY="${ANALYSIS_SCRIPTS_DIR}/plot_lig_vs_ref_filter2.py"
PLOT_RESIDUES_VS_REF_BACKBONE_PY="${ANALYSIS_SCRIPTS_DIR}/plot_residues_vs_ref_backbone.py"
PLOT_RESIDUES_VS_REF_FINAL_PY="${ANALYSIS_SCRIPTS_DIR}/plot_residues_vs_ref_final.py"
PLOT_RESIDUES_VS_REF_ROTAMER_PY="${ANALYSIS_SCRIPTS_DIR}/plot_residues_vs_ref_rotamer.py"
PLOT_ROTAMER_VS_PIPELINE_PY="${ANALYSIS_SCRIPTS_DIR}/plot_rotamer_vs_pipeline.py"
AGGREGATE_ROTAMER_WORSE_RESIDUES_PY="${ANALYSIS_SCRIPTS_DIR}/aggregate_rotamer_worse_residues.py"
AGGREGATE_CLASH_GROUPS_PY="${ANALYSIS_SCRIPTS_DIR}/aggregate_clash_groups.py"
CENTROID_RMSD_ALL_PY="${ANALYSIS_SCRIPTS_DIR}/centroid_rmsd_all.py"
CALC_PLACER_SAMPLING_PY="${ANALYSIS_SCRIPTS_DIR}/calc_placer_sampling.py"
CALC_PLACER_SAMPLING_UNREFINED_PY="${ANALYSIS_SCRIPTS_DIR}/calc_placer_sampling_unrefined.py"
PLOT_FIT_LIGAND_COUNTS_PY="${ANALYSIS_SCRIPTS_DIR}/plot_fit_ligand_counts.py"
PLOT_CLUSTER_REPS_POOLED_PY="${ANALYSIS_SCRIPTS_DIR}/plot_cluster_reps_rscc_pooled.py"
PLOT_PROTEIN_RSCC_POOLED_PY="${ANALYSIS_SCRIPTS_DIR}/plot_protein_rscc_pooled.py"
PLOT_DESPOT_ENERGIES_PY="${ANALYSIS_SCRIPTS_DIR}/plot_despot_energies.py"
PLOT_DESPOT_ENERGIES_POOLED_PY="${ANALYSIS_SCRIPTS_DIR}/plot_despot_energies_pooled.py"
PLOT_LIG_VS_REF_DESPOT_PY="${ANALYSIS_SCRIPTS_DIR}/plot_lig_vs_ref_despot.py"
PLOT_DESPOT_LIGAND_SUMMARY_PY="${ANALYSIS_SCRIPTS_DIR}/plot_despot_ligand_summary.py"
PLOT_DESPOT_LIGAND_SUMMARY_SINGLE_PY="${ANALYSIS_SCRIPTS_DIR}/plot_despot_ligand_summary_single.py"
PLOT_DESPOT_VS_REF_PY="${ANALYSIS_SCRIPTS_DIR}/plot_despot_vs_ref.py"
PLOT_RSCC_DESPOT_TRADEOFF_PY="${ANALYSIS_SCRIPTS_DIR}/plot_rscc_despot_tradeoff.py"
PLOT_RESIDUES_VS_REF_DESPOT_PY="${ANALYSIS_SCRIPTS_DIR}/plot_residues_vs_ref_despot.py"
ASSIGN_BOND_ORDERS_PY="${LIG_SCRIPTS_DIR}/assign_bond_orders.py"
PDB_TO_MOL2_SH="${LIG_SCRIPTS_DIR}/pdb_to_mol2.sh"
PROTEIN_TO_MOL2_SH="${LIG_SCRIPTS_DIR}/protein_to_mol2.sh"

for f in "$DATASETS_FILE" "$CSV_FILE" "$RSR_SCRIPT_LIGAND" "$RSR_SCRIPT_PROTEIN" "$RSR_SCRIPT_FINAL" \
         "$RUN_PLACER_PY" "$PLOT_CLUSTER_REPS_PY" "$AGGREGATE_PROTEIN_RSCC_PY" "$AGGREGATE_LIG_RSCC_PY" \
         "$PLOT_LIG_VS_REF_FILTER1_PY" "$PLOT_LIG_VS_REF_FILTER2_PY" \
         "$PLOT_RESIDUES_VS_REF_BACKBONE_PY" "$PLOT_RESIDUES_VS_REF_FINAL_PY" "$PLOT_RESIDUES_VS_REF_ROTAMER_PY" \
         "$PLOT_ROTAMER_VS_PIPELINE_PY" "$AGGREGATE_ROTAMER_WORSE_RESIDUES_PY" \
         "$CENTROID_RMSD_ALL_PY" "$CALC_PLACER_SAMPLING_PY" "$CALC_PLACER_SAMPLING_UNREFINED_PY" \
         "$PLOT_FIT_LIGAND_COUNTS_PY" "$ASSIGN_BOND_ORDERS_PY" "$PLOT_CLUSTER_REPS_POOLED_PY" \
         "$PLOT_PROTEIN_RSCC_POOLED_PY" \
         "$PLOT_DESPOT_ENERGIES_PY" "$PLOT_DESPOT_ENERGIES_POOLED_PY" \
         "$PLOT_LIG_VS_REF_DESPOT_PY" "$PLOT_DESPOT_LIGAND_SUMMARY_PY" "$PLOT_DESPOT_LIGAND_SUMMARY_SINGLE_PY" \
         "$PLOT_DESPOT_VS_REF_PY" "$PLOT_RSCC_DESPOT_TRADEOFF_PY" "$PLOT_RESIDUES_VS_REF_DESPOT_PY" \
         "$PDB_TO_MOL2_SH" "$PROTEIN_TO_MOL2_SH" "$DESPOT_SCRIPT"; do
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

# test.sh (test_case's copy of program.sh) always runs the full pipeline against
# test_case's single dataset (x00407-1), including the reference-set comparison stages
# (-c) and --f2_filter_proportion 1 - any arguments actually passed on the command line
# (e.g. -g <gpu_ids>) are appended after these fixed ones, so they still take effect.
set -- run_1 placer_1 filter_1 placer2_1 filter2_1 final_1 rotamer_1 despot_1 -c --f2_filter_proportion 1 "$@"

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
rotamer_run_name=""
despot_run_name=""

num_placer_confs=100
num_placer2_confs=100
gpu_ids="0"
num_parallel=""
compare_ref_set=0
overwrite=0
replot=0

# --dataset <id[,id...]>: run only on this subset of datasets instead of
# reading DATASETS_FILE. dataset_arg holds the raw CLI value; if set, it's
# expanded into DATASET_OVERRIDE_FILE (a generated temp file, one dataset
# per line) which DATASETS_FILE is then repointed to - see below.
dataset_arg=""
DATASET_OVERRIDE_FILE=""

# fit_ligand tunables (stage 1a). Left empty by default so fit_ligand's own
# argparse defaults (-z/--z_threshold=4, -n/--num_peaks=100, --rmsd_cutoff=2) apply; only
# passed through when explicitly set here.
z_threshold=""
num_peaks=""
fit_ligand_rmsd_cutoff=""

# filter tunables (stage 3a, filter_run_name), left empty by default so
# filter's own argparse defaults apply.
f1_filter_proportion=""
f1_min_cluster_proportion=""
f1_rscc_cutoff=""
f1_clustering_mode=""
f1_clustering_cutoff=""
f1_clash_vdw_scale=""

# filter tunables (stage 5a, filter2_run_name) - same underlying `filter`
# script as f1_*, set independently.
f2_filter_proportion=""
f2_min_cluster_proportion=""
f2_rscc_cutoff=""
f2_clustering_mode=""
f2_clustering_cutoff=""
f2_clash_vdw_scale=""

# Sidechain-sidechain clash resolution tunables, shared identically by build_final_model.py
# (stage 6a) and rotamer_optimize.py (stage 7a) - both now use the same
# qfit.command_line.sidechain_clash engine, so one set of values drives both. Left empty by
# default so each script's own argparse defaults apply (clash_vdw_scale=0.75,
# hbond_clash_vdw_scale=0.6, max_clash_group_size=8, max_clash_group_expansions=10,
# clash_domain_top_k=25, clash_solve_node_budget=200000).
clash_vdw_scale=""
hbond_clash_vdw_scale=""
max_clash_group_size=""
max_clash_group_expansions=""
clash_domain_top_k=""
clash_solve_node_budget=""

# rotamer_optimize tunables (stage 7a), left empty by default so rotamer_optimize's own
# argparse defaults apply (--rscc_threshold=0.5, --rscc_improvement_threshold=0.1).
rotamer_rscc_threshold=""
rotamer_rscc_improvement_threshold=""

# select_optimized_residues tunable (stage 7d), left empty by default so its own argparse
# default applies (--revert_min_diff=0.1).
revert_min_diff=""

# calc_rscc's shared --bfactor, applied identically everywhere calc_rscc or despot_filter's
# internal RSCC scoring runs (stages 0b/0c/3c/6c/7c/8a). Left empty by default so each script's
# own argparse default applies (calc_rscc/despot_filter both default to 20).
bfactor=""

# symmetry_expand's distance_cutoff (stages 0d/8a) - a symmetry mate is only kept within this
# distance (Å) of a ligand atom. Was previously a fixed, non-configurable constant.
expand_distance_cutoff="10"

# Real-space refinement tunables, applied identically everywhere the corresponding flag exists
# on that RSR variant (see real_space_refine*.py --help) - rsr_n_cycles/rsr_map_weight apply to
# all 5 RSR call sites (rsr_placer, rsr_backbone, rsr_placer2, rsr_final, rsr_rotamer);
# rsr_backbone_cutoff only to rsr_backbone (real_space_refine_protein.py's own --cutoff);
# rsr_moved_threshold to rsr_backbone/rsr_final/rsr_rotamer (the 3 variants that report it).
# Left empty by default so each RSR script's own argparse defaults apply (--n-cycles=1000,
# --map-weight=50.0, --cutoff=10.0, --moved-threshold=0.01).
rsr_n_cycles=""
rsr_map_weight=""
rsr_backbone_cutoff=""
rsr_moved_threshold=""

# despot_filter tunables (stage 7a), left empty by default so despot_filter's own argparse
# defaults apply (--despot-threshold -1.0, --rscc-threshold 0.6, --rscc-weight 0.05).
despot_threshold=""
despot_rscc_threshold=""
despot_rscc_weight=""

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
        --z_threshold)
            z_threshold="$2"
            shift 2
            ;;
        --num_peaks)
            num_peaks="$2"
            shift 2
            ;;
        --f1_filter_proportion)
            f1_filter_proportion="$2"
            shift 2
            ;;
        --f1_min_cluster_proportion)
            f1_min_cluster_proportion="$2"
            shift 2
            ;;
        --f1_rscc_cutoff)
            f1_rscc_cutoff="$2"
            shift 2
            ;;
        --f1_clustering_mode)
            f1_clustering_mode="$2"
            shift 2
            ;;
        --f1_clustering_cutoff)
            f1_clustering_cutoff="$2"
            shift 2
            ;;
        --f2_filter_proportion)
            f2_filter_proportion="$2"
            shift 2
            ;;
        --f2_min_cluster_proportion)
            f2_min_cluster_proportion="$2"
            shift 2
            ;;
        --f2_rscc_cutoff)
            f2_rscc_cutoff="$2"
            shift 2
            ;;
        --f2_clustering_mode)
            f2_clustering_mode="$2"
            shift 2
            ;;
        --f2_clustering_cutoff)
            f2_clustering_cutoff="$2"
            shift 2
            ;;
        --despot_threshold)
            despot_threshold="$2"
            shift 2
            ;;
        --despot_rscc_threshold)
            despot_rscc_threshold="$2"
            shift 2
            ;;
        --despot_rscc_weight)
            despot_rscc_weight="$2"
            shift 2
            ;;
        --fit_ligand_rmsd_cutoff)
            fit_ligand_rmsd_cutoff="$2"
            shift 2
            ;;
        --f1_clash_vdw_scale)
            f1_clash_vdw_scale="$2"
            shift 2
            ;;
        --f2_clash_vdw_scale)
            f2_clash_vdw_scale="$2"
            shift 2
            ;;
        --clash_vdw_scale)
            clash_vdw_scale="$2"
            shift 2
            ;;
        --hbond_clash_vdw_scale)
            hbond_clash_vdw_scale="$2"
            shift 2
            ;;
        --max_clash_group_size)
            max_clash_group_size="$2"
            shift 2
            ;;
        --max_clash_group_expansions)
            max_clash_group_expansions="$2"
            shift 2
            ;;
        --clash_domain_top_k)
            clash_domain_top_k="$2"
            shift 2
            ;;
        --clash_solve_node_budget)
            clash_solve_node_budget="$2"
            shift 2
            ;;
        --rotamer_rscc_threshold)
            rotamer_rscc_threshold="$2"
            shift 2
            ;;
        --rotamer_rscc_improvement_threshold)
            rotamer_rscc_improvement_threshold="$2"
            shift 2
            ;;
        --revert_min_diff)
            revert_min_diff="$2"
            shift 2
            ;;
        --bfactor)
            bfactor="$2"
            shift 2
            ;;
        --expand_distance_cutoff)
            expand_distance_cutoff="$2"
            shift 2
            ;;
        --rsr_n_cycles)
            rsr_n_cycles="$2"
            shift 2
            ;;
        --rsr_map_weight)
            rsr_map_weight="$2"
            shift 2
            ;;
        --rsr_backbone_cutoff)
            rsr_backbone_cutoff="$2"
            shift 2
            ;;
        --rsr_moved_threshold)
            rsr_moved_threshold="$2"
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
            elif [ -z "$rotamer_run_name" ]; then
                rotamer_run_name="$1"
            elif [ -z "$despot_run_name" ]; then
                despot_run_name="$1"
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

if [ -n "$despot_run_name" ] && [ -z "$rotamer_run_name" ]; then
    echo "Error: <despot_run_name> was given without <rotamer_run_name> - stage 8 (despot) now" >&2
    echo "scores optimized.pdb (stage 7d's output), so <rotamer_run_name> is required" >&2
    echo "whenever <despot_run_name> is given." >&2
    usage
fi

# --- Full-run logging: every line this script (and everything it calls) prints from here on is
# teed into BASE_DIR/logs/<run_name>/<placer_run_name>/.../<deepest run-name given>/log.txt -
# the same hierarchical nesting convention as every dataset's own output tree. If that log.txt
# already exists (a previous run at this same run-name path), this run's output goes to
# log_2.txt instead, log_3.txt if that also exists, and so on - never overwriting a prior run's
# log. Per-dataset `exec > >(tee ...) 2>&1` redirections (e.g. despot_process_dataset's own
# despot_log) nest fine underneath this: each dataset job's stdout/stderr, relayed back through
# parallel, still flows through to this top-level tee.
log_dir="${BASE_DIR}/logs/${run_name}"
for _log_run_name in "$placer_run_name" "$filter_run_name" "$placer2_run_name" \
                     "$filter2_run_name" "$final_run_name" "$rotamer_run_name" "$despot_run_name"; do
    [ -n "$_log_run_name" ] && log_dir="${log_dir}/${_log_run_name}"
done
unset _log_run_name
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

if [ "$compare_ref_set" -eq 1 ] && [ ! -d "$REF_SET" ]; then
    echo "Error: -c given but reference set directory not found: ${REF_SET}" >&2
    exit 1
fi

# --dataset override: repoint DATASETS_FILE at a generated temp file listing
# just the requested dataset(s), instead of the full DATASETS_FILE. Every
# stage below reads datasets exclusively via $DATASETS_FILE, so this alone
# scopes the whole run.
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

# Canonical, in-memory list of datasets for this run - read from
# DATASETS_FILE exactly once, here (whichever it currently points to:
# datasets.txt by default, or the --dataset override above). Every stage
# that enumerates datasets directly in this shell (stage9_outputs_exist,
# do_placer, do_placer2, and every parallel-driving
# do_* function below) iterates this
# array instead of separately re-reading DATASETS_FILE or - as do_placer2
# previously did - deriving its own list some other way. That means
# overriding DATASETS_FILE (e.g. via --dataset) above is guaranteed to scope
# every stage consistently, since they all read from this one array.
mapfile -t DATASETS < <(grep -v '^[[:space:]]*$' "$DATASETS_FILE")
if [ ${#DATASETS[@]} -eq 0 ]; then
    echo "Error: no datasets found in ${DATASETS_FILE}" >&2
    exit 1
fi

NUM_PARALLEL_DEFAULT=${num_parallel:-1}

IFS=',' read -ra GPU_IDS_ARR <<< "$gpu_ids"
NUM_GPUS=${#GPU_IDS_ARR[@]}

# Everything below this point (dataset names, tunables, consolidated paths)
# needs to be visible inside the per-dataset *_process_dataset functions even
# when GNU parallel forks them into new subshells, so it all gets exported.
export run_name placer_run_name filter_run_name placer2_run_name filter2_run_name final_run_name rotamer_run_name despot_run_name
export num_placer_confs num_placer2_confs compare_ref_set overwrite replot
export z_threshold num_peaks fit_ligand_rmsd_cutoff
export f1_filter_proportion f1_min_cluster_proportion f1_rscc_cutoff \
       f1_clustering_mode f1_clustering_cutoff f1_clash_vdw_scale
export f2_filter_proportion f2_min_cluster_proportion f2_rscc_cutoff \
       f2_clustering_mode f2_clustering_cutoff f2_clash_vdw_scale
export clash_vdw_scale hbond_clash_vdw_scale max_clash_group_size \
       max_clash_group_expansions clash_domain_top_k clash_solve_node_budget
export rotamer_rscc_threshold rotamer_rscc_improvement_threshold
export revert_min_diff
export bfactor expand_distance_cutoff
export rsr_n_cycles rsr_map_weight rsr_backbone_cutoff rsr_moved_threshold
export despot_threshold despot_rscc_threshold despot_rscc_weight
export BASE_DIR DATASETS_DIR DATASETS_FILE CSV_FILE LIG_PDB_DIR ASSIGN_BOND_ORDERS_PY
export RSR_SCRIPT_LIGAND RSR_SCRIPT_PROTEIN RSR_SCRIPT_FINAL
export ANALYSIS_SCRIPTS_DIR PLOT_CLUSTER_REPS_PY AGGREGATE_PROTEIN_RSCC_PY AGGREGATE_LIG_RSCC_PY
export PLOT_LIG_VS_REF_FILTER1_PY PLOT_LIG_VS_REF_FILTER2_PY
export PLOT_RESIDUES_VS_REF_BACKBONE_PY PLOT_RESIDUES_VS_REF_FINAL_PY PLOT_RESIDUES_VS_REF_ROTAMER_PY GRAPHS_DIR
export PLOT_ROTAMER_VS_PIPELINE_PY AGGREGATE_ROTAMER_WORSE_RESIDUES_PY
export AGGREGATE_CLASH_GROUPS_PY
export CENTROID_RMSD_ALL_PY CALC_PLACER_SAMPLING_PY CALC_PLACER_SAMPLING_UNREFINED_PY
export PLOT_FIT_LIGAND_COUNTS_PY
export PLOT_CLUSTER_REPS_POOLED_PY PLOT_PROTEIN_RSCC_POOLED_PY
export PLOT_DESPOT_ENERGIES_PY PLOT_DESPOT_ENERGIES_POOLED_PY PLOT_LIG_VS_REF_DESPOT_PY PLOT_DESPOT_LIGAND_SUMMARY_PY
export PLOT_DESPOT_LIGAND_SUMMARY_SINGLE_PY
export PLOT_DESPOT_VS_REF_PY
export PLOT_RSCC_DESPOT_TRADEOFF_PY
export PLOT_RESIDUES_VS_REF_DESPOT_PY
export PDB_TO_MOL2_SH PROTEIN_TO_MOL2_SH DESPOT_SCRIPT DESPOT_DATABASE
export REF_SET REF_SET_PDB_PATTERN
export CONDA_SH CONDA_ENV_QFIT CONDA_ENV_RSR CONDA_ENV_PLACER CONDA_ENV_EVAL CONDA_ENV_DESPOT
export RUN_PLACER_PY

# --- Shared lookup files, built once from CSV_FILE and reused by every stage that needs
# them. CSV_FILE's columns are:
# dataset,resolution,ligand_name,a,b,c,alpha,beta,gamma,space_group,smiles
#   LOOKUP_FILE: "dataset ligand_name resolution" (fit_ligand, filter, filter2, build_final,
#                calc_backbone_refined_rscc, calc_final_refined_rscc)
#   LIG_SMILES_LOOKUP_FILE: "dataset smiles" (convert_ligs, despot)
#   DESPOT_CELL_LOOKUP_FILE: "dataset a b c alpha beta gamma space_group" (despot)
# NOTE: `read` must consume every CSV_FILE column here, even the ones unused below - with
# fewer read variables than fields, IFS=',' read would dump every remaining column into the
# last variable (ligand_name), silently corrupting it once CSV_FILE gained its a/b/c/.../smiles
# columns.
LOOKUP_FILE=$(mktemp)
LIG_SMILES_LOOKUP_FILE=$(mktemp)
DESPOT_CELL_LOOKUP_FILE=$(mktemp)
trap 'rm -f "$LOOKUP_FILE" "$LIG_SMILES_LOOKUP_FILE" "$DESPOT_CELL_LOOKUP_FILE" "$DATASET_OVERRIDE_FILE"' EXIT
tail -n +2 "$CSV_FILE" | while IFS=',' read -r dataset resolution ligand_name a b c alpha beta gamma space_group smiles; do
    dataset="${dataset//$'\r'/}"
    resolution="${resolution//$'\r'/}"
    ligand_name="${ligand_name//$'\r'/}"
    smiles="${smiles//$'\r'/}"
    space_group="${space_group//$'\r'/}"
    [ -z "$dataset" ] && continue
    echo "${dataset} ${ligand_name} ${resolution}" >> "$LOOKUP_FILE"
    echo "${dataset} ${smiles}" >> "$LIG_SMILES_LOOKUP_FILE"
    echo "${dataset} ${a} ${b} ${c} ${alpha} ${beta} ${gamma} ${space_group}" >> "$DESPOT_CELL_LOOKUP_FILE"
done
export LOOKUP_FILE LIG_SMILES_LOOKUP_FILE DESPOT_CELL_LOOKUP_FILE

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

# files_exist <path...>
# True (0) iff every given path exists as a regular file. Used throughout
# the per-dataset *_process_dataset functions below to decide whether that
# dataset's work for a step is already done.
files_exist() {
    local f
    for f in "$@"; do
        [ -f "$f" ] || return 1
    done
    return 0
}
export -f files_exist

# stage9_graphs_dir: prints the pooled GRAPHS_DIR base every stage 9 plot writes into -
# <final_run_name>/<rotamer_run_name>/<despot_run_name> when despot_run_name is given (despot_run_name
# is the deepest folder in the pipeline once both rotamer_optimize and despot have run),
# <final_run_name>/<rotamer_run_name> when only rotamer_run_name is given (unchanged from
# plot_rotamer_vs_pipeline's own prior behavior), else just <final_run_name> as before.
stage9_graphs_dir() {
    local base="${GRAPHS_DIR}/${run_name}/${placer_run_name}/${filter_run_name}/${placer2_run_name}/${filter2_run_name}/${final_run_name}"
    if [ -n "$despot_run_name" ]; then
        echo "${base}/${rotamer_run_name}/${despot_run_name}"
    elif [ -n "$rotamer_run_name" ]; then
        echo "${base}/${rotamer_run_name}"
    else
        echo "$base"
    fi
}
export -f stage9_graphs_dir

# dataset_stage9_graphs_dir <dataset>: prints the per-dataset graphs/ folder stage9_outputs_exist
# (and, via python's own dataset_graphs_dir, the per-dataset do_* functions) check/write to -
# <final_run_name>/<rotamer_run_name>/<despot_run_name>/graphs/ when that directory actually
# exists for this specific dataset, else <final_run_name>/graphs/ as before. Mirrors
# rscc_common.py's dataset_graphs_dir despot_subpath fallback.
dataset_stage9_graphs_dir() {
    local dataset="$1"
    local final_dir="${DATASETS_DIR}/${dataset}/${run_name}/${placer_run_name}/${filter_run_name}/${placer2_run_name}/${filter2_run_name}/${final_run_name}"
    if [ -n "$rotamer_run_name" ] && [ -n "$despot_run_name" ] && [ -d "${final_dir}/${rotamer_run_name}/${despot_run_name}" ]; then
        echo "${final_dir}/${rotamer_run_name}/${despot_run_name}/graphs"
    else
        echo "${final_dir}/graphs"
    fi
}
export -f dataset_stage9_graphs_dir

# glob_nonempty <pattern>
# True (0) iff the given glob pattern matches at least one file. Used for
# steps that produce a variable number of outputs per dataset (PLACER/RSR
# rounds), where "at least one matching output exists" is the completion
# signal (a loose check - see the --replot restructuring plan for why: a
# partially-failed dataset would be treated as done and needs --overwrite to
# resume, same tradeoff every other loose existence check here already makes).
glob_nonempty() {
    local pattern="$1"
    shopt -s nullglob
    local matches=($pattern)
    shopt -u nullglob
    [ ${#matches[@]} -gt 0 ]
}
export -f glob_nonempty

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

# run_step_pooled <description> <command...>
# Same as run_step, but skipped entirely when --dataset was given (i.e.
# dataset_arg is non-empty). Pooled (cross-dataset) plots always write into
# one shared GRAPHS_DIR location keyed only by run-name (not by which
# datasets were involved), so running one against a --dataset subset would
# silently overwrite the full-run pooled plot with a partial one.
run_step_pooled() {
    if [ -n "$dataset_arg" ]; then
        echo ""
        echo "Skipping ${1} (pooled plot; --dataset was given, would overwrite the full-run plot with a partial one)."
        return 0
    fi
    run_step "$@"
}

# run_step_replot <description> <check_fn> <run_fn>
# For graphing steps (1b, 1c, 2c, 3d, 4c, 5b, 6d, 7d, 7e, 7f, 8b, 8c, 8d, 8e, 9): skips run_fn (via
# run_step) when check_fn - a function name taking no args - returns success
# (0, "all of this step's outputs already exist"), UNLESS --replot or
# --overwrite was given.
run_step_replot() {
    local desc="$1" check_fn="$2" run_fn="$3"
    if [ "$overwrite" -ne 1 ] && [ "$replot" -ne 1 ] && "$check_fn"; then
        echo ""
        echo "Skipping ${desc} (outputs already exist; pass --replot or --overwrite to redo)."
        return 0
    fi
    run_step "$desc" "$run_fn"
}

# run_step_pooled_replot <description> <check_fn> <run_fn>
# Composes run_step_pooled's --dataset guard with run_step_replot's
# existence check. The --dataset guard wins even under --replot - a partial
# --dataset run must never touch a full-run pooled plot.
run_step_pooled_replot() {
    local desc="$1"
    if [ -n "$dataset_arg" ]; then
        echo ""
        echo "Skipping ${desc} (pooled plot; --dataset was given, would overwrite the full-run plot with a partial one)."
        return 0
    fi
    run_step_replot "$@"
}

# conda_activate <env_name>
# Some conda environments (e.g. ones with compiler packages like
# binutils_linux-64) install activate.d hooks that reference variables
# (ADDR2LINE, etc.) without defaults. Those hooks are fine under an
# interactive shell (no `set -u`) but abort this script's `set -uo
# pipefail`. Temporarily relax nounset just for the source/activate calls.
#
# Some packages (e.g. coot-headless) also install activate.d/deactivate.d
# hooks that unconditionally `echo` every variable they set/unset
# ("COOT_PREFIX set to ...", "COOT_PREFIX unset", etc). With this script
# activating/deactivating envs once per dataset (many times per run), that
# floods stdout, so hook stdout is discarded here; stderr is left alone so
# real hook errors still surface.
conda_activate() {
    set +u
    source "$CONDA_SH"
    conda activate "$1" > /dev/null
    set -u
}

# conda_deactivate: same nounset relaxation and stdout suppression as
# conda_activate, for deactivate.d hooks that restore saved variables.
conda_deactivate() {
    set +u
    conda deactivate > /dev/null
    set -u
}
export -f conda_activate conda_deactivate

# print_elapsed <start_epoch_seconds>
print_elapsed() {
    local start_time="$1"
    local label="${2:-Script}"
    local end_time
    end_time=$(date +%s)
    local elapsed=$((end_time - start_time))
    local hours=$((elapsed / 3600))
    local minutes=$(((elapsed % 3600) / 60))
    local seconds=$((elapsed % 60))
    printf "%s took %02d:%02d:%02d (HH:MM:SS)\n" "$label" $hours $minutes $seconds
}
export -f print_elapsed

# build_cif_list_from_cluster_reps <dataset> <cluster_reps_csv>
# Prints (to stdout) the comma-separated list of cif_restraints_file paths from every data row
# of cluster_reps_csv, in row order - the i-th LIG residue in final_model.pdb (resid i, 1-based)
# corresponds to the i-th (1-indexed) DATA row of filter2's cluster_reps.csv, and RSR (stage 6b's
# rsr_final and stage 7b's rsr_rotamer) needs one CIF restraints file per LIG residue in that
# same row order. Returns non-zero (with an ERROR [<dataset>]: message on stderr) if the
# cif_restraints_file column is missing, any row's CIF path is missing/doesn't exist, or no data
# rows are found.
build_cif_list_from_cluster_reps() {
    local dataset="$1" cluster_reps_csv="$2"

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
        echo "ERROR [${dataset}]: cif_restraints_file column not found in $cluster_reps_csv" >&2
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
            echo "ERROR [${dataset}]: CIF not found for row ${row_num} of ${cluster_reps_csv}: '${cif_path}'" >&2
            any_cif_missing=1
            continue
        fi

        cif_paths+=("$cif_path")
    done < "$cluster_reps_csv"

    if [ $any_cif_missing -ne 0 ]; then
        echo "ERROR [${dataset}]: One or more CIF restraint files were missing, aborting." >&2
        return 1
    fi

    if [ ${#cif_paths[@]} -eq 0 ]; then
        echo "ERROR [${dataset}]: No data rows found in $cluster_reps_csv" >&2
        return 1
    fi

    (IFS=,; echo "${cif_paths[*]}")
}
export -f build_cif_list_from_cluster_reps

# write_params_txt <output_file> <name=value> [<name=value> ...]
# Records the CLI-configurable parameters actually used for a stage's run,
# into its output directory. An empty value means the corresponding
# program.sh flag wasn't given, so the underlying script's own argparse
# default applied instead.
write_params_txt() {
    local output_file="$1"
    shift
    local kv name value
    {
        for kv in "$@"; do
            name="${kv%%=*}"
            value="${kv#*=}"
            if [ -z "$value" ]; then
                echo "${name}: (not set - script default used)"
            else
                echo "${name}: ${value}"
            fi
        done
    } > "$output_file"
}
export -f write_params_txt

######################################################################
# Stage 0a: convert_ligs
######################################################################
# For every dataset in datasets.txt, looks up its ligand_name/smiles (via
# LOOKUP_FILE/LIG_SMILES_LOOKUP_FILE) and converts that ligand's pdb file(s)
# under LIG_PDB_DIR to mol2 - assign_bond_orders.py (CONDA_ENV_QFIT: rdkit
# assigns bond orders from SMILES onto the pdb's 3D coordinates, writes an
# sdf) then obabel (also CONDA_ENV_QFIT - openbabel/pdb2pqr are installed there
# alongside qfit's own dependencies: sdf -> mol2) - same tool
# pdb_final_geometry's existing per-ligand mol2 files were made with, just
# with bond orders taken from SMILES instead of eLBOW. LIG_PDB_DIR is laid
# out one subdirectory per ligand name, e.g.
# LIG_PDB_DIR/<ligand_name>/<ligand_name>.pdb, with extra directories
# LIG_PDB_DIR/<ligand_name>_<suffix>/<ligand_name>_<suffix>.pdb (e.g.
# "_R"/"_S") for a ligand with multiple stereoisomer variants - every
# matching directory is converted. This is dataset-scoped, not
# run-name-scoped (like calc_apo_rscc below), so it runs once per dataset
# regardless of run_name and is skipped per-pdb whenever that pdb's mol2
# already exists.

convert_ligs_process_dataset() {
    local dataset=$1

    local lookup=$(grep "^${dataset} " "$LOOKUP_FILE")
    if [ -z "$lookup" ]; then
        echo "Warning: No match found for dataset ${dataset}, skipping."
        return 1
    fi
    local fragment_id=$(echo "$lookup" | awk '{print $2}')

    local smiles_lookup=$(grep "^${dataset} " "$LIG_SMILES_LOOKUP_FILE")
    local smiles=$(echo "$smiles_lookup" | awk '{print $2}')
    if [ -z "$smiles" ]; then
        echo "Warning [${dataset}]: no SMILES for ligand_name=${fragment_id}, skipping."
        return 1
    fi

    echo "Processing ${dataset}: ligand_name=${fragment_id}"

    local pdb_dirs=()
    while IFS= read -r -d '' dir; do
        local dir_name=$(basename "$dir")
        if [[ "$dir_name" == "$fragment_id" || "$dir_name" == "${fragment_id}_"* ]]; then
            pdb_dirs+=("$dir")
        fi
    done < <(find "$LIG_PDB_DIR" -maxdepth 1 -mindepth 1 -type d -name "${fragment_id}*" -print0 | sort -z)

    if [[ ${#pdb_dirs[@]} -eq 0 ]]; then
        echo "Warning [${dataset}]: No directories found matching '${fragment_id}' under ${LIG_PDB_DIR}, skipping."
        return 1
    fi

    for pdb_dir in "${pdb_dirs[@]}"; do
        local dir_name=$(basename "$pdb_dir")
        local pdb_file="${pdb_dir}/${dir_name}.pdb"
        local sdf_file="${pdb_dir}/${dir_name}.sdf"
        local mol2_file="${pdb_dir}/${dir_name}.mol2"

        if [[ ! -f "$pdb_file" ]]; then
            echo "  Warning [${dataset}]: Expected PDB file not found: ${pdb_file}, skipping."
            continue
        fi
        if [[ -f "$mol2_file" ]]; then
            echo "  Skipping [${dataset}]: ${mol2_file} already exists."
            continue
        fi

        echo "  Converting ${dir_name}.pdb"

        conda_activate "$CONDA_ENV_QFIT"
        python "$ASSIGN_BOND_ORDERS_PY" "$pdb_file" "$smiles" "$sdf_file"
        local status=$?
        conda_deactivate
        if [[ $status -ne 0 ]]; then
            echo "  ERROR [${dataset}]: assign_bond_orders.py failed on ${pdb_file} with exit code ${status}"
            return 1
        fi

        conda_activate "$CONDA_ENV_QFIT"
        obabel "$sdf_file" -O "$mol2_file"
        status=$?
        conda_deactivate
        if [[ $status -ne 0 ]]; then
            echo "  ERROR [${dataset}]: obabel failed converting ${sdf_file} with exit code ${status}"
            return 1
        fi
    done
}
export -f convert_ligs_process_dataset

do_convert_ligs() {
    echo "Starting run"
    local start_time=$(date +%s)
    printf '%s\n' "${DATASETS[@]}" | parallel -j "$NUM_PARALLEL_DEFAULT" convert_ligs_process_dataset {}
    echo "All jobs completed"
    print_elapsed "$start_time"
}

######################################################################
# Stage 0b: calc_apo_rscc
######################################################################
# Computes the per-residue RSCC of each dataset's baseline
# {dataset}-aligned-structure.pdb (no PLACER/RSR involved) so later analysis
# scripts have an apo baseline to compare backbone/final refined RSCC
# against. This is dataset-scoped, not run-name-scoped, so it runs once per
# dataset regardless of run_name and is skipped per-dataset whenever its
# output csv already exists.

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

    local bfactor_extra_args=()
    [ -n "$bfactor" ] && bfactor_extra_args+=(--bfactor "$bfactor")

    calc_rscc "${structure}" "${event_maps[@]}" "${resolution}" "${output_csv}" "${bfactor_extra_args[@]}"

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
    printf '%s\n' "${DATASETS[@]}" | parallel -j "$NUM_PARALLEL_DEFAULT" calc_apo_rscc_process_dataset {}
    echo "All jobs completed"
    print_elapsed "$start_time"
}

######################################################################
# Stage 0c: calc_ref_set_rscc (only runs when -c is given)
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

    local bfactor_extra_args=()
    [ -n "$bfactor" ] && bfactor_extra_args+=(--bfactor "$bfactor")

    calc_rscc "${structure}" "${event_maps[@]}" "${resolution}" "${output_csv}" "${bfactor_extra_args[@]}"

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
    printf '%s\n' "${DATASETS[@]}" | parallel -j "$NUM_PARALLEL_DEFAULT" calc_ref_set_rscc_process_dataset {}
    echo "All jobs completed"
    print_elapsed "$start_time"
}

######################################################################
# Stage 0d: ref_set_despot (only runs when -c and despot_run_name are given)
######################################################################
# Scores each dataset's reference-set structure (REF_SET/<dataset>/<REF_SET_PDB_PATTERN>)
# with DESPOT, the same way as Stage 7a's own final_model_refined.pdb: symmetry_expand
# into a realistic crystal environment (expand_distance_cutoff), convert the expanded
# protein and split-out ligand to mol2, score with DESPOT's score_complex.py. Unlike
# Stage 7a's input, the reference structure still carries explicit ligand hydrogens, ordered
# waters, and DMSO (resname DMS, a common crystallization cryoprotectant) (e.g. from PanDDA) -
# symmetry_expand's own --strip flag removes all three before anything else runs, since
# DESPOT scoring isn't set up to expect any of them. Every
# output, including the intermediate expanded/mol2 files, is written directly into
# REF_SET/<dataset>/ (one reference structure per dataset, no run-name nesting needed -
# reused as-is across every run_name/despot_run_name combination scored against the same
# reference set), alongside the reference RSCC csv Stage 0c already writes there.

ref_set_despot_process_dataset() {
    local dataset=$1
    local reference_dataset_dir="${REF_SET}/${dataset}"

    if [ ! -d "$reference_dataset_dir" ]; then
        echo "Warning [${dataset}]: reference set folder ${reference_dataset_dir} not found, skipping."
        return 1
    fi

    local pdb_pattern="${REF_SET_PDB_PATTERN//\{dataset\}/${dataset}}"
    local structure="${reference_dataset_dir}/${pdb_pattern}"
    local despot_csv="${reference_dataset_dir}/${dataset}_DESPOT.csv"

    if [ "$overwrite" -ne 1 ] && files_exist "$despot_csv"; then
        echo "Skipping [${dataset}]: ref_set_despot already complete (${despot_csv} exists)."
        return 0
    fi

    if [ ! -f "$structure" ]; then
        echo "Warning [${dataset}]: reference structure not found: ${structure}, skipping."
        return 1
    fi

    local cell_lookup=$(grep "^${dataset} " "$DESPOT_CELL_LOOKUP_FILE")
    if [ -z "$cell_lookup" ]; then
        echo "Warning [${dataset}]: no crystal cell/space group info found in ${CSV_FILE}, skipping."
        return 1
    fi
    local cl_dataset a b c alpha beta gamma space_group
    read -r cl_dataset a b c alpha beta gamma space_group <<< "$cell_lookup"

    local smiles_lookup=$(grep "^${dataset} " "$LIG_SMILES_LOOKUP_FILE")
    local smiles=$(echo "$smiles_lookup" | awk '{print $2}')
    if [ -z "$smiles" ]; then
        echo "Warning [${dataset}]: no SMILES found, skipping."
        return 1
    fi

    echo "Processing ${dataset}: space_group=${space_group}, cell=(${a} ${b} ${c} ${alpha} ${beta} ${gamma})"

    local ref_despot_log="${reference_dataset_dir}/despot_log.txt"
    exec > >(tee "$ref_despot_log") 2>&1

    local dataset_start_time=$(date +%s)

    local expanded_pdb="${reference_dataset_dir}/expanded.pdb"
    local ligs_dir="${reference_dataset_dir}/ligs"
    local ligs_mol2="${reference_dataset_dir}/ligs.mol2"
    local expanded_mol2="${reference_dataset_dir}/expanded.mol2"

    local step_start_time=$(date +%s)
    conda_activate "$CONDA_ENV_QFIT"
    symmetry_expand --strip "$structure" "$expanded_pdb" "$space_group" "$a" "$b" "$c" "$alpha" "$beta" "$gamma" \
        "$expand_distance_cutoff" "$ligs_dir"
    local status=$?
    conda_deactivate
    print_elapsed "$step_start_time" "[${dataset}] symmetry_expand"
    if [ $status -ne 0 ]; then
        echo "ERROR [${dataset}]: symmetry_expand failed with exit code ${status}"
        print_elapsed "$dataset_start_time" "[${dataset}] ref_set_despot"
        return 1
    fi

    # symmetry_expand writes one lig<chain><resi>[-<altloc>].pdb per ligand instance into
    # ligs_dir (split by altloc so a genuinely disordered instance gets its own DESPOT score -
    # see symmetry_expand.py's ligand_output_dir) - pdb_to_mol2.sh/assign_bond_orders.py
    # combine every instance found here into one ligs.sdf/ligs.mol2, one molecule per instance,
    # named after its own lig<label>.pdb basename.
    shopt -s nullglob
    local ligand_pdbs=("${ligs_dir}"/lig*.pdb)
    shopt -u nullglob
    if [ ${#ligand_pdbs[@]} -eq 0 ]; then
        echo "Warning [${dataset}]: no ligand (resname LIG) instance found in ${structure}; skipping."
        print_elapsed "$dataset_start_time" "[${dataset}] ref_set_despot"
        return 1
    fi

    "$PDB_TO_MOL2_SH" "${reference_dataset_dir}/ligs" "$smiles" "$CONDA_SH" "$CONDA_ENV_QFIT" \
        "$CONDA_ENV_QFIT" "$ASSIGN_BOND_ORDERS_PY" "${ligand_pdbs[@]}"
    status=$?
    if [ $status -ne 0 ]; then
        echo "ERROR [${dataset}]: pdb_to_mol2.sh failed on ${ligand_pdbs[*]} with exit code ${status}"
        print_elapsed "$dataset_start_time" "[${dataset}] ref_set_despot"
        return 1
    fi

    step_start_time=$(date +%s)
    "$PROTEIN_TO_MOL2_SH" "$expanded_pdb" "$CONDA_SH" "$CONDA_ENV_QFIT"
    status=$?
    print_elapsed "$step_start_time" "[${dataset}] pdb2pqr"
    if [ $status -ne 0 ]; then
        echo "ERROR [${dataset}]: protein_to_mol2.sh failed on ${expanded_pdb} with exit code ${status}"
        print_elapsed "$dataset_start_time" "[${dataset}] ref_set_despot"
        return 1
    fi

    step_start_time=$(date +%s)
    conda_activate "$CONDA_ENV_DESPOT"
    python "$DESPOT_SCRIPT" -p "$expanded_mol2" -l "$ligs_mol2" -o "$despot_csv" --database "$DESPOT_DATABASE"
    status=$?
    conda_deactivate
    print_elapsed "$step_start_time" "[${dataset}] despot score_complex.py"
    if [ $status -ne 0 ]; then
        echo "ERROR [${dataset}]: DESPOT score_complex.py failed with exit code ${status}"
        print_elapsed "$dataset_start_time" "[${dataset}] ref_set_despot"
        return 1
    fi

    echo "Completed [${dataset}]: ${despot_csv}"
    print_elapsed "$dataset_start_time" "[${dataset}] ref_set_despot"
}
export -f ref_set_despot_process_dataset

do_ref_set_despot() {
    # See do_despot's identical comment: pins BLAS/OpenMP/numba threading in
    # DESPOT's score_complex.py to 1 thread per process, scoped to just this
    # stage, so `parallel`'s fan-out doesn't oversubscribe the machine.
    export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMBA_NUM_THREADS=1

    echo "Starting run"
    local start_time=$(date +%s)
    printf '%s\n' "${DATASETS[@]}" | parallel -j "$NUM_PARALLEL_DEFAULT" --line-buffer ref_set_despot_process_dataset {}
    echo "All jobs completed"
    print_elapsed "$start_time"

    unset OMP_NUM_THREADS OPENBLAS_NUM_THREADS MKL_NUM_THREADS NUMEXPR_NUM_THREADS VECLIB_MAXIMUM_THREADS NUMBA_NUM_THREADS
}

######################################################################
# Stage 1a: fit_ligand
######################################################################

fit_ligand_process_dataset() {
    local dataset=$1

    local manifest_file="${DATASETS_DIR}/${dataset}/${run_name}/fit_ligand_manifest.csv"
    if [ "$overwrite" -ne 1 ] && files_exist "$manifest_file"; then
        echo "Skipping [${dataset}]: fit_ligand already complete (${manifest_file} exists)."
        return 0
    fi

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

    local run_out_dir="${DATASETS_DIR}/${dataset}/${run_name}"
    mkdir -p "${run_out_dir}"
    write_params_txt "${run_out_dir}/fit_ligand_params.txt" \
        "z_threshold=${z_threshold}" \
        "num_peaks=${num_peaks}"

    for pdb_dir in "${pdb_dirs[@]}"; do
        local dir_name=$(basename "$pdb_dir")
        local pdb_file="${pdb_dir}/${dir_name}.pdb"

        if [[ ! -f "$pdb_file" ]]; then
            echo "  Warning: Expected PDB file not found: ${pdb_file}, skipping."
            continue
        fi

        local out_dir="${DATASETS_DIR}/${dataset}/${run_name}"
        mkdir -p "${out_dir}"

        echo "  Running fit_ligand: PDB=${dir_name}, run_name=${run_name}"

        local fit_ligand_extra_args=()
        [ -n "$z_threshold" ] && fit_ligand_extra_args+=(-z "$z_threshold")
        [ -n "$num_peaks" ] && fit_ligand_extra_args+=(-n "$num_peaks")
        [ -n "$fit_ligand_rmsd_cutoff" ] && fit_ligand_extra_args+=(--rmsd_cutoff "$fit_ligand_rmsd_cutoff")

        fit_ligand "${DATASETS_DIR}/${dataset}" \
            "${pdb_file}" \
            -r ${resolution} \
            --run_name ${run_name} \
            "${fit_ligand_extra_args[@]}" \
            > "${out_dir}/ligandfit_${dir_name}.txt" 2>&1

        echo "  Completed: ${dataset} / ${dir_name}"
    done
}
export -f fit_ligand_process_dataset

do_fit_ligand() {
    echo "Starting run"
    local start_time=$(date +%s)
    printf '%s\n' "${DATASETS[@]}" | parallel -j "$NUM_PARALLEL_DEFAULT" fit_ligand_process_dataset {}
    echo "All jobs completed"
    print_elapsed "$start_time"
}

######################################################################
# Stage 1c: centroid_rmsd_all (only runs when -c is given)
######################################################################
# Pooled (cross-dataset) histogram under GRAPHS_DIR/<run_name>/: minimum
# ligand centroid distance from every reference LIG conformation to the
# closest fit_ligand output pose (after CA superposition onto the
# reference), before any PLACER sampling has happened.

centroid_rmsd_all_outputs_exist() {
    files_exist "${GRAPHS_DIR}/${run_name}/centroid_rmsd_all.png"
}

do_centroid_rmsd_all() {
    conda_activate "$CONDA_ENV_EVAL"

    local out_dir="${GRAPHS_DIR}/${run_name}"
    echo "Starting run"
    local start_time=$(date +%s)
    python "$CENTROID_RMSD_ALL_PY" \
        "$run_name" \
        --datasets-dir "$DATASETS_DIR" --datasets-file "$DATASETS_FILE" \
        --ref-set "$REF_SET" --ref-pdb-pattern "$REF_SET_PDB_PATTERN" --graphs-dir "$out_dir"
    echo "All jobs completed"
    print_elapsed "$start_time"
}

######################################################################
# Stage 1b: plot_fit_ligand_counts (always runs, not gated behind -c)
######################################################################
# Pooled (cross-dataset) histogram under GRAPHS_DIR/<run_name>/: number of
# fit_ligand output poses per dataset (one data point per dataset), read
# straight from each dataset's fit_ligand_manifest.csv row count. Doesn't
# touch the reference set, so it runs on every stage-1 invocation.

fit_ligand_counts_outputs_exist() {
    files_exist "${GRAPHS_DIR}/${run_name}/fit_ligand_counts.png"
}

do_plot_fit_ligand_counts() {
    conda_activate "$CONDA_ENV_EVAL"

    local out_dir="${GRAPHS_DIR}/${run_name}"
    echo "Starting run"
    local start_time=$(date +%s)
    python "$PLOT_FIT_LIGAND_COUNTS_PY" \
        "$run_name" \
        --datasets-dir "$DATASETS_DIR" --datasets-file "$DATASETS_FILE" \
        --graphs-dir "$out_dir"
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

    echo "========= Dataset: ${dataset} (GPU ${gpu_id}) ========="

    local dataset_dir="${DATASETS_DIR}/${dataset}"
    local run_dir="${dataset_dir}/${run_name}"
    local manifest_file="${run_dir}/fit_ligand_manifest.csv"

    if [[ ! -f "$manifest_file" ]]; then
        echo "  Warning: manifest not found: ${manifest_file}, skipping."
        return
    fi

    local out_dir="${run_dir}/${placer_run_name}"

    if [ "$overwrite" -ne 1 ] && glob_nonempty "${out_dir}/*_model.pdb"; then
        echo "  Skipping [${dataset}]: placer already complete (${out_dir}/*_model.pdb found)."
        return 0
    fi

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
    for dataset in "${DATASETS[@]}"; do
        local gpu_id=${GPU_IDS_ARR[$((idx % NUM_GPUS))]}
        echo "${dataset} ${gpu_id}"
        idx=$((idx + 1))
    done | parallel -j "$NUM_GPUS" --line-buffer --colsep ' ' placer_process_dataset {1} {2}

    echo "All jobs completed"
    print_elapsed "$start_time"
    conda_deactivate
}

######################################################################
# Stage 2b: rsr_placer
######################################################################

rsr_placer_process_dataset() {
    local dataset=$1
    local dataset_dir="${DATASETS_DIR}/${dataset}"
    local placer_dir="${dataset_dir}/${run_name}/${placer_run_name}"

    if [ "$overwrite" -ne 1 ] && glob_nonempty "${placer_dir}/*_refined.pdb"; then
        echo "Skipping [${dataset}]: rsr_placer already complete (${placer_dir}/*_refined.pdb found)."
        return 0
    fi

    conda_activate "$CONDA_ENV_RSR"

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

        local rsr_extra_args=()
        [ -n "$rsr_n_cycles" ] && rsr_extra_args+=(--n-cycles "$rsr_n_cycles")
        [ -n "$rsr_map_weight" ] && rsr_extra_args+=(--map-weight "$rsr_map_weight")

        python "$RSR_SCRIPT_LIGAND" \
            "$input_pdb" \
            "$map_file" \
            "$output_pdb" \
            --cif-restraints "$cif_path" \
            "${rsr_extra_args[@]}"
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
    printf '%s\n' "${DATASETS[@]}" | parallel -j "$NUM_PARALLEL_DEFAULT" rsr_placer_process_dataset {}
    echo "All jobs completed"
    print_elapsed "$start_time"
}

######################################################################
# Stage 2c: calc_placer_sampling refined/unrefined (only runs when -c is given)
######################################################################
# Pooled (cross-dataset) histograms under GRAPHS_DIR/<run_name>/<placer_run_name>/:
# minimum symmetry-aware RMSD from every reference LIG conformation to the
# closest round-1 PLACER-sampled ligand conformer, scored both after RSR
# (placer_sampling.png) and on PLACER's own raw output (placer_sampling_unrefined.png).

do_placer_sampling_refined_round1() {
    conda_activate "$CONDA_ENV_EVAL"

    local out_dir="${GRAPHS_DIR}/${run_name}/${placer_run_name}"
    echo "Starting run"
    local start_time=$(date +%s)
    python "$CALC_PLACER_SAMPLING_PY" \
        "$run_name" "$placer_run_name" \
        --datasets-dir "$DATASETS_DIR" --datasets-file "$DATASETS_FILE" \
        --ref-set "$REF_SET" --ref-pdb-pattern "$REF_SET_PDB_PATTERN" --graphs-dir "$out_dir"
    echo "All jobs completed"
    print_elapsed "$start_time"
}

do_placer_sampling_unrefined_round1() {
    conda_activate "$CONDA_ENV_EVAL"

    local out_dir="${GRAPHS_DIR}/${run_name}/${placer_run_name}"
    echo "Starting run"
    local start_time=$(date +%s)
    python "$CALC_PLACER_SAMPLING_UNREFINED_PY" \
        "$run_name" "$placer_run_name" \
        --datasets-dir "$DATASETS_DIR" --datasets-file "$DATASETS_FILE" \
        --ref-set "$REF_SET" --ref-pdb-pattern "$REF_SET_PDB_PATTERN" --graphs-dir "$out_dir"
    echo "All jobs completed"
    print_elapsed "$start_time"
}

placer_sampling_round1_outputs_exist() {
    files_exist "${GRAPHS_DIR}/${run_name}/${placer_run_name}/placer_sampling.png" \
                "${GRAPHS_DIR}/${run_name}/${placer_run_name}/placer_sampling_unrefined.png"
}

do_placer_sampling_round1() {
    do_placer_sampling_refined_round1
    do_placer_sampling_unrefined_round1
}

######################################################################
# Stage 3a: filter
######################################################################

filter_process_dataset() {
    local dataset=$1
    local dataset_dir="${DATASETS_DIR}/${dataset}"
    local cluster_reps_csv="${dataset_dir}/${run_name}/${placer_run_name}/${filter_run_name}/cluster_reps.csv"

    if [ "$overwrite" -ne 1 ] && files_exist "$cluster_reps_csv"; then
        echo "Skipping [${dataset}]: filter already complete (${cluster_reps_csv} exists)."
        return 0
    fi

    conda_activate "$CONDA_ENV_QFIT"

    local lookup=$(grep "^${dataset} " "$LOOKUP_FILE")
    if [ -z "$lookup" ]; then
        echo "Warning: No match found for dataset ${dataset}, skipping."
        return 1
    fi

    local fragment_id=$(echo "$lookup" | awk '{print $2}')
    local resolution=$(echo "$lookup" | awk '{print $3}')

    echo "Processing ${dataset}: fragment_id=${fragment_id}, resolution=${resolution}"

    local f1_extra_args=()
    [ -n "$f1_filter_proportion" ] && f1_extra_args+=(--filter_proportion "$f1_filter_proportion")
    [ -n "$f1_min_cluster_proportion" ] && f1_extra_args+=(--min_cluster_proportion "$f1_min_cluster_proportion")
    [ -n "$f1_rscc_cutoff" ] && f1_extra_args+=(--rscc_cutoff "$f1_rscc_cutoff")
    [ -n "$f1_clustering_mode" ] && f1_extra_args+=(--clustering_mode "$f1_clustering_mode")
    [ -n "$f1_clustering_cutoff" ] && f1_extra_args+=(--clustering_cutoff "$f1_clustering_cutoff")
    [ -n "$f1_clash_vdw_scale" ] && f1_extra_args+=(--clash_vdw_scale "$f1_clash_vdw_scale")

    filter ${dataset_dir} \
        "${dataset_dir}/${run_name}/${placer_run_name}/*_refined.pdb" \
        "${dataset_dir}/${run_name}/*.pdb" \
        $run_name/${placer_run_name}/${filter_run_name} \
        -r ${resolution} \
        "${f1_extra_args[@]}"

    local filter_exit=$?
    if [ $filter_exit -ne 0 ]; then
        echo "ERROR [${dataset}]: filter failed with exit code ${filter_exit}"
        return 1
    fi

    write_params_txt "${dataset_dir}/${run_name}/${placer_run_name}/${filter_run_name}/filter_params.txt" \
        "filter_proportion=${f1_filter_proportion}" \
        "min_cluster_proportion=${f1_min_cluster_proportion}" \
        "rscc_cutoff=${f1_rscc_cutoff}" \
        "clustering_mode=${f1_clustering_mode}" \
        "clustering_cutoff=${f1_clustering_cutoff}"

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
    printf '%s\n' "${DATASETS[@]}" | parallel -j "$NUM_PARALLEL_DEFAULT" filter_process_dataset {}
    echo "All jobs completed"
    print_elapsed "$start_time"
}

######################################################################
# Stage 3b: rsr_backbone
######################################################################

rsr_backbone_process_dataset() {
    local dataset=$1
    local dataset_dir="${DATASETS_DIR}/${dataset}"
    local run_dir_check="${dataset_dir}/${run_name}/${placer_run_name}/${filter_run_name}"

    if [ "$overwrite" -ne 1 ] && glob_nonempty "${run_dir_check}/${dataset}_backbone_refined_*.pdb"; then
        echo "Skipping [${dataset}]: rsr_backbone already complete (${run_dir_check}/${dataset}_backbone_refined_*.pdb found)."
        return 0
    fi

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

    local rsr_extra_args=()
    [ -n "$rsr_n_cycles" ] && rsr_extra_args+=(--n-cycles "$rsr_n_cycles")
    [ -n "$rsr_map_weight" ] && rsr_extra_args+=(--map-weight "$rsr_map_weight")
    [ -n "$rsr_backbone_cutoff" ] && rsr_extra_args+=(--cutoff "$rsr_backbone_cutoff")
    [ -n "$rsr_moved_threshold" ] && rsr_extra_args+=(--moved-threshold "$rsr_moved_threshold")

    python "$RSR_SCRIPT_PROTEIN" \
        "$multimodel_pdb" \
        "$apo_pdb" \
        "$map_file" \
        "$output_pdb" \
        --cif-list "$cif_list" \
        "${rsr_extra_args[@]}"
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
    printf '%s\n' "${DATASETS[@]}" | parallel -j "$NUM_PARALLEL_DEFAULT" rsr_backbone_process_dataset {}
    echo "All jobs completed"
    print_elapsed "$start_time"

    unset OMP_NUM_THREADS OPENBLAS_NUM_THREADS MKL_NUM_THREADS NUMEXPR_NUM_THREADS VECLIB_MAXIMUM_THREADS
}

######################################################################
# Stage 3c: calc_backbone_refined_rscc
######################################################################

calc_backbone_refined_rscc_process_dataset() {
    local dataset=$1
    local dataset_dir="${DATASETS_DIR}/${dataset}"
    local run_dir_check="${dataset_dir}/${run_name}/${placer_run_name}/${filter_run_name}"

    if [ "$overwrite" -ne 1 ] && glob_nonempty "${run_dir_check}/${dataset}_backbone_refined_"*"_rscc.csv"; then
        echo "Skipping [${dataset}]: calc_backbone_refined_rscc already complete (${run_dir_check}/${dataset}_backbone_refined_*_rscc.csv found)."
        return 0
    fi

    conda_activate "$CONDA_ENV_QFIT"
    shopt -s nullglob

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
    local bfactor_extra_args=()
    [ -n "$bfactor" ] && bfactor_extra_args+=(--bfactor "$bfactor")
    for structure in "${structures[@]}"; do
        output_csv="${structure%.pdb}_rscc.csv"

        calc_rscc "${structure}" "${event_maps[@]}" "${resolution}" "${output_csv}" "${bfactor_extra_args[@]}"

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
    printf '%s\n' "${DATASETS[@]}" | parallel -j "$NUM_PARALLEL_DEFAULT" calc_backbone_refined_rscc_process_dataset {}
    echo "All jobs completed"
    print_elapsed "$start_time"
}

######################################################################
# Stage 3d: reference-set comparison (only runs when -c is given)
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

filter1_ref_comparison_outputs_exist() {
    local out_dir="${GRAPHS_DIR}/${run_name}/${placer_run_name}/${filter_run_name}"
    files_exist "${out_dir}/lig_vs_reference_rscc.png" \
                "${out_dir}/backbone_refined_vs_reference_rscc.png" \
                "${out_dir}/backbone_refined_vs_reference_rscc_restricted.png"
}

do_filter1_ref_comparison() {
    do_plot_lig_vs_ref_filter1
    do_plot_residues_vs_ref_backbone
}

######################################################################
# Stage 4a: placer2 (round 2)
######################################################################

placer2_process_dataset() {
    local dataset=$1
    local gpu_id=$2

    export CUDA_VISIBLE_DEVICES=$gpu_id

    echo "========= Dataset: ${dataset} (GPU ${gpu_id}) ========="

    local dataset_dir="${DATASETS_DIR}/${dataset}"
    local filter_dir="${dataset_dir}/${run_name}/${placer_run_name}/${filter_run_name}"

    if [ "$overwrite" -ne 1 ] && glob_nonempty "${filter_dir}/${placer2_run_name}/${dataset}_backbone_refined_"*"_model.pdb"; then
        echo "  Skipping [${dataset}]: placer2 already complete (${filter_dir}/${placer2_run_name}/${dataset}_backbone_refined_*_model.pdb found)."
        return 0
    fi

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
    for dataset in "${DATASETS[@]}"; do
        local gpu_id=${GPU_IDS_ARR[$((idx % NUM_GPUS))]}
        echo "${dataset} ${gpu_id}"
        idx=$((idx + 1))
    done | parallel -j "$NUM_GPUS" --line-buffer --colsep ' ' placer2_process_dataset {1} {2}

    echo "All jobs completed"
    print_elapsed "$start_time"
    conda_deactivate
}

######################################################################
# Stage 4b: rsr_placer2
######################################################################

rsr_placer2_process_dataset() {
    local dataset=$1
    local dataset_dir="${DATASETS_DIR}/${dataset}"
    local filter_dir="${dataset_dir}/${run_name}/${placer_run_name}/${filter_run_name}"
    local placer2_dir="${filter_dir}/${placer2_run_name}"

    if [ "$overwrite" -ne 1 ] && glob_nonempty "${placer2_dir}/${dataset}_backbone_refined_"*"_refined.pdb"; then
        echo "Skipping [${dataset}]: rsr_placer2 already complete (${placer2_dir}/${dataset}_backbone_refined_*_refined.pdb found)."
        return 0
    fi

    conda_activate "$CONDA_ENV_RSR"

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

        local rsr_extra_args=()
        [ -n "$rsr_n_cycles" ] && rsr_extra_args+=(--n-cycles "$rsr_n_cycles")
        [ -n "$rsr_map_weight" ] && rsr_extra_args+=(--map-weight "$rsr_map_weight")

        python "$RSR_SCRIPT_LIGAND" \
            "$input_pdb" \
            "$map_file" \
            "$output_pdb" \
            --cif-restraints "$cif_path" \
            "${rsr_extra_args[@]}"
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
    printf '%s\n' "${DATASETS[@]}" | parallel -j "$NUM_PARALLEL_DEFAULT" rsr_placer2_process_dataset {}
    echo "All jobs completed"
    print_elapsed "$start_time"
}

######################################################################
# Stage 4c: calc_placer_sampling refined/unrefined (only runs when -c is given)
######################################################################
# Pooled (cross-dataset) histograms under
# GRAPHS_DIR/<run_name>/<placer_run_name>/<filter_run_name>/<placer2_run_name>/:
# same comparison as stage 2c/2d, but for round-2 PLACER samples.

do_placer_sampling_refined_round2() {
    conda_activate "$CONDA_ENV_EVAL"

    local out_dir="${GRAPHS_DIR}/${run_name}/${placer_run_name}/${filter_run_name}/${placer2_run_name}"
    echo "Starting run"
    local start_time=$(date +%s)
    python "$CALC_PLACER_SAMPLING_PY" \
        "$run_name" "$placer_run_name" "$filter_run_name" "$placer2_run_name" \
        --datasets-dir "$DATASETS_DIR" --datasets-file "$DATASETS_FILE" \
        --ref-set "$REF_SET" --ref-pdb-pattern "$REF_SET_PDB_PATTERN" --graphs-dir "$out_dir"
    echo "All jobs completed"
    print_elapsed "$start_time"
}

do_placer_sampling_unrefined_round2() {
    conda_activate "$CONDA_ENV_EVAL"

    local out_dir="${GRAPHS_DIR}/${run_name}/${placer_run_name}/${filter_run_name}/${placer2_run_name}"
    echo "Starting run"
    local start_time=$(date +%s)
    python "$CALC_PLACER_SAMPLING_UNREFINED_PY" \
        "$run_name" "$placer_run_name" "$filter_run_name" "$placer2_run_name" \
        --datasets-dir "$DATASETS_DIR" --datasets-file "$DATASETS_FILE" \
        --ref-set "$REF_SET" --ref-pdb-pattern "$REF_SET_PDB_PATTERN" --graphs-dir "$out_dir"
    echo "All jobs completed"
    print_elapsed "$start_time"
}

placer_sampling_round2_outputs_exist() {
    local out_dir="${GRAPHS_DIR}/${run_name}/${placer_run_name}/${filter_run_name}/${placer2_run_name}"
    files_exist "${out_dir}/placer_sampling.png" "${out_dir}/placer_sampling_unrefined.png"
}

do_placer_sampling_round2() {
    do_placer_sampling_refined_round2
    do_placer_sampling_unrefined_round2
}

######################################################################
# Stage 5a: filter2
######################################################################

filter2_process_dataset() {
    local dataset=$1
    local dataset_dir="${DATASETS_DIR}/${dataset}"
    local cluster_reps_csv_check="${dataset_dir}/${run_name}/${placer_run_name}/${filter_run_name}/${placer2_run_name}/${filter2_run_name}/cluster_reps.csv"

    if [ "$overwrite" -ne 1 ] && files_exist "$cluster_reps_csv_check"; then
        echo "Skipping [${dataset}]: filter2 already complete (${cluster_reps_csv_check} exists)."
        return 0
    fi

    conda_activate "$CONDA_ENV_QFIT"

    local lookup=$(grep "^${dataset} " "$LOOKUP_FILE")
    if [ -z "$lookup" ]; then
        echo "Warning: No match found for dataset ${dataset}, skipping."
        return 1
    fi

    local fragment_id=$(echo "$lookup" | awk '{print $2}')
    local resolution=$(echo "$lookup" | awk '{print $3}')

    echo "Processing ${dataset}: fragment_id=${fragment_id}, resolution=${resolution}"

    local f2_extra_args=()
    [ -n "$f2_filter_proportion" ] && f2_extra_args+=(--filter_proportion "$f2_filter_proportion")
    [ -n "$f2_min_cluster_proportion" ] && f2_extra_args+=(--min_cluster_proportion "$f2_min_cluster_proportion")
    [ -n "$f2_rscc_cutoff" ] && f2_extra_args+=(--rscc_cutoff "$f2_rscc_cutoff")
    [ -n "$f2_clustering_mode" ] && f2_extra_args+=(--clustering_mode "$f2_clustering_mode")
    [ -n "$f2_clustering_cutoff" ] && f2_extra_args+=(--clustering_cutoff "$f2_clustering_cutoff")
    [ -n "$f2_clash_vdw_scale" ] && f2_extra_args+=(--clash_vdw_scale "$f2_clash_vdw_scale")

    filter "${dataset_dir}" \
        "${dataset_dir}/${run_name}/${placer_run_name}/${filter_run_name}/${placer2_run_name}/*_refined.pdb" \
        "${dataset_dir}/${run_name}/${placer_run_name}/${filter_run_name}/*_refined_*.pdb" \
        ${run_name}/${placer_run_name}/${filter_run_name}/${placer2_run_name}/${filter2_run_name} \
        -r ${resolution} \
        "${f2_extra_args[@]}"

    local filter_exit=$?
    if [ $filter_exit -ne 0 ]; then
        echo "ERROR [${dataset}]: filter failed with exit code ${filter_exit}"
        return 1
    fi

    local filter2_dir="${dataset_dir}/${run_name}/${placer_run_name}/${filter_run_name}/${placer2_run_name}/${filter2_run_name}"
    write_params_txt "${filter2_dir}/filter_params.txt" \
        "filter_proportion=${f2_filter_proportion}" \
        "min_cluster_proportion=${f2_min_cluster_proportion}" \
        "rscc_cutoff=${f2_rscc_cutoff}" \
        "clustering_mode=${f2_clustering_mode}" \
        "clustering_cutoff=${f2_clustering_cutoff}"

    # --- Post-hoc: carry the cif_restraints_file column over from filter_run_name's
    # cluster_reps.csv into filter2_run_name's cluster_reps.csv ---
    local filter_csv="${dataset_dir}/${run_name}/${placer_run_name}/${filter_run_name}/cluster_reps.csv"
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
    printf '%s\n' "${DATASETS[@]}" | parallel -j "$NUM_PARALLEL_DEFAULT" filter2_process_dataset {}
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

filter2_ref_comparison_outputs_exist() {
    files_exist "${GRAPHS_DIR}/${run_name}/${placer_run_name}/${filter_run_name}/${placer2_run_name}/${filter2_run_name}/lig_vs_reference_rscc.png"
}

######################################################################
# Stage 6a: build_final
######################################################################

build_final_process_dataset() {
    local dataset=$1
    local dataset_dir="${DATASETS_DIR}/${dataset}"
    local placer2_dir="${dataset_dir}/${run_name}/${placer_run_name}/${filter_run_name}/${placer2_run_name}"
    local filter2_dir="${placer2_dir}/${filter2_run_name}"
    local final_dir_check="${filter2_dir}/${final_run_name}"

    if [ "$overwrite" -ne 1 ] && files_exist "${final_dir_check}/final_model.pdb" "${final_dir_check}/residues_with_placer_conformers.csv"; then
        echo "Skipping [${dataset}]: build_final already complete (${final_dir_check}/final_model.pdb exists)."
        return 0
    fi

    conda_activate "$CONDA_ENV_QFIT"

    local lookup=$(grep "^${dataset} " "$LOOKUP_FILE")
    if [ -z "$lookup" ]; then
        echo "Warning: No match found for dataset ${dataset}, skipping."
        return 1
    fi

    local fragment_id=$(echo "$lookup" | awk '{print $2}')
    local resolution=$(echo "$lookup" | awk '{print $3}')

    echo "Processing ${dataset}: fragment_id=${fragment_id}, resolution=${resolution}"

    local apo_structure="${dataset_dir}/${dataset}-aligned-structure.pdb"

    if [ ! -f "$apo_structure" ]; then
        echo "ERROR [${dataset}]: apo structure not found: ${apo_structure}"
        return 1
    fi

    local clash_extra_args=()
    [ -n "$clash_vdw_scale" ] && clash_extra_args+=(--clash_vdw_scale "$clash_vdw_scale")
    [ -n "$hbond_clash_vdw_scale" ] && clash_extra_args+=(--hbond_clash_vdw_scale "$hbond_clash_vdw_scale")
    [ -n "$max_clash_group_size" ] && clash_extra_args+=(--max_clash_group_size "$max_clash_group_size")
    [ -n "$max_clash_group_expansions" ] && clash_extra_args+=(--max_clash_group_expansions "$max_clash_group_expansions")
    [ -n "$clash_domain_top_k" ] && clash_extra_args+=(--clash_domain_top_k "$clash_domain_top_k")
    [ -n "$clash_solve_node_budget" ] && clash_extra_args+=(--clash_solve_node_budget "$clash_solve_node_budget")

    build_final_model "${dataset_dir}" \
        "${placer2_dir}/*_refined.pdb" \
        "${filter2_dir}/cluster_rep_models.pdb" \
        "${apo_structure}" \
        ${run_name}/${placer_run_name}/${filter_run_name}/${placer2_run_name}/${filter2_run_name}/${final_run_name} \
        -r ${resolution} \
        "${clash_extra_args[@]}"

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
    printf '%s\n' "${DATASETS[@]}" | parallel -j "$NUM_PARALLEL_DEFAULT" build_final_process_dataset {}
    echo "All jobs completed"
    print_elapsed "$start_time"
}

######################################################################
# Stage 6b: rsr_final
######################################################################

rsr_final_process_dataset() {
    local dataset=$1
    local dataset_dir="${DATASETS_DIR}/${dataset}"
    local filter2_dir="${dataset_dir}/${run_name}/${placer_run_name}/${filter_run_name}/${placer2_run_name}/${filter2_run_name}"
    local final_dir="${filter2_dir}/${final_run_name}"

    if [ "$overwrite" -ne 1 ] && files_exist "${final_dir}/final_model_refined.pdb"; then
        echo "Skipping [${dataset}]: rsr_final already complete (${final_dir}/final_model_refined.pdb exists)."
        return 0
    fi

    conda_activate "$CONDA_ENV_RSR"

    local map_file
    map_file=$(find "${dataset_dir}" -maxdepth 1 -name "${dataset}-event_1*" | head -1)
    if [ -z "$map_file" ]; then
        echo "ERROR [${dataset}]: No event map found matching ${dataset}-event_1* in ${dataset_dir}"
        return 1
    fi

    local cluster_reps_csv="${filter2_dir}/cluster_reps.csv"

    if [ ! -f "$cluster_reps_csv" ]; then
        echo "ERROR [${dataset}]: cluster_reps.csv not found: $cluster_reps_csv"
        return 1
    fi

    local cif_list
    cif_list=$(build_cif_list_from_cluster_reps "$dataset" "$cluster_reps_csv") || return 1

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
    echo "[${dataset}] Using CIF restraints list: $cif_list"

    local rsr_extra_args=()
    [ -n "$rsr_n_cycles" ] && rsr_extra_args+=(--n-cycles "$rsr_n_cycles")
    [ -n "$rsr_map_weight" ] && rsr_extra_args+=(--map-weight "$rsr_map_weight")
    [ -n "$rsr_moved_threshold" ] && rsr_extra_args+=(--moved-threshold "$rsr_moved_threshold")

    python "$RSR_SCRIPT_FINAL" \
        "$final_pdb" \
        "$residues_csv" \
        "$map_file" \
        "$output_pdb" \
        --cif-list "$cif_list" \
        "${rsr_extra_args[@]}"
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
    printf '%s\n' "${DATASETS[@]}" | parallel -j "$NUM_PARALLEL_DEFAULT" rsr_final_process_dataset {}
    echo "All jobs completed"
    print_elapsed "$start_time"

    unset OMP_NUM_THREADS OPENBLAS_NUM_THREADS MKL_NUM_THREADS NUMEXPR_NUM_THREADS VECLIB_MAXIMUM_THREADS
}

######################################################################
# Stage 6c: calc_final_refined_rscc
######################################################################

calc_final_refined_rscc_process_dataset() {
    local dataset=$1
    local dataset_dir="${DATASETS_DIR}/${dataset}"
    local final_dir="${dataset_dir}/${run_name}/${placer_run_name}/${filter_run_name}/${placer2_run_name}/${filter2_run_name}/${final_run_name}"
    local structure="${final_dir}/final_model_refined.pdb"
    local output_csv="${structure%.pdb}_rscc.csv"

    if [ "$overwrite" -ne 1 ] && files_exist "$output_csv"; then
        echo "Skipping [${dataset}]: calc_final_refined_rscc already complete (${output_csv} exists)."
        return 0
    fi

    conda_activate "$CONDA_ENV_QFIT"
    shopt -s nullglob

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

    if [ ! -f "$structure" ]; then
        echo "Warning [${dataset}]: final_model_refined.pdb not found: ${structure}, skipping."
        return 1
    fi

    local bfactor_extra_args=()
    [ -n "$bfactor" ] && bfactor_extra_args+=(--bfactor "$bfactor")

    calc_rscc "${structure}" "${event_maps[@]}" "${resolution}" "${output_csv}" "${bfactor_extra_args[@]}"

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
    printf '%s\n' "${DATASETS[@]}" | parallel -j "$NUM_PARALLEL_DEFAULT" calc_final_refined_rscc_process_dataset {}
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

final_ref_comparison_outputs_exist() {
    local out_dir="${GRAPHS_DIR}/${run_name}/${placer_run_name}/${filter_run_name}/${placer2_run_name}/${filter2_run_name}/${final_run_name}"
    files_exist "${out_dir}/final_refined_vs_reference_rscc.png" \
                "${out_dir}/final_refined_vs_reference_rscc_restricted.png"
}

######################################################################
# Stage 6e: aggregate_clash_groups (no -c needed - just concatenates each
# dataset's own sidechain_clash_groups.csv, already written by build_final in
# Stage 6a; no reference set involved)
######################################################################
# Run-wide concatenation of every dataset's sidechain_clash_groups.csv into
# GRAPHS_DIR/<run>/.../<final_run_name>/sidechain_clash_groups_combined.csv.

do_aggregate_clash_groups() {
    conda_activate "$CONDA_ENV_EVAL"

    local out_dir="${GRAPHS_DIR}/${run_name}/${placer_run_name}/${filter_run_name}/${placer2_run_name}/${filter2_run_name}/${final_run_name}"
    echo "Starting run"
    local start_time=$(date +%s)
    python "$AGGREGATE_CLASH_GROUPS_PY" \
        "$run_name" "$placer_run_name" "$filter_run_name" "$placer2_run_name" "$filter2_run_name" "$final_run_name" \
        --datasets-dir "$DATASETS_DIR" --datasets-file "$DATASETS_FILE" --graphs-dir "$out_dir"
    echo "All jobs completed"
    print_elapsed "$start_time"
}

clash_groups_aggregate_outputs_exist() {
    local out_dir="${GRAPHS_DIR}/${run_name}/${placer_run_name}/${filter_run_name}/${placer2_run_name}/${filter2_run_name}/${final_run_name}"
    files_exist "${out_dir}/sidechain_clash_groups_combined.csv"
}

######################################################################
# Stage 7a: rotamer_optimize
######################################################################
# For each dataset (only runs when rotamer_run_name is given), re-samples chi/aromatic-angle
# rotamers for every residue in final_model.pdb (stage 6a's output, BEFORE rsr_final's
# refinement - NOT final_model_refined.pdb) listed in residues_with_placer_conformers.csv that
# scores below RSCC 0.5 against the dataset's (deduped) event maps, keeping the resampled
# conformer only if it improves RSCC by >= 0.1 over the starting conformer - see
# qfit/command_line/rotamer_optimize.py. residues_with_placer_conformers.csv is read implicitly
# by rotamer_optimize itself, as a sidecar of its model_file argument.

rotamer_optimize_process_dataset() {
    local dataset=$1
    local dataset_dir="${DATASETS_DIR}/${dataset}"
    local final_dir="${dataset_dir}/${run_name}/${placer_run_name}/${filter_run_name}/${placer2_run_name}/${filter2_run_name}/${final_run_name}"
    local rotamer_dir="${final_dir}/${rotamer_run_name}"
    local rotamer_output_folder="${run_name}/${placer_run_name}/${filter_run_name}/${placer2_run_name}/${filter2_run_name}/${final_run_name}/${rotamer_run_name}"

    if [ "$overwrite" -ne 1 ] && files_exist "${rotamer_dir}/rotamer_optimized.pdb"; then
        echo "Skipping [${dataset}]: rotamer_optimize already complete (${rotamer_dir}/rotamer_optimized.pdb exists)."
        return 0
    fi

    local final_pdb="${final_dir}/final_model.pdb"
    local residues_csv="${final_dir}/residues_with_placer_conformers.csv"

    if [ ! -f "$final_pdb" ]; then
        echo "ERROR [${dataset}]: final_pdb not found: ${final_pdb}"
        return 1
    fi
    if [ ! -f "$residues_csv" ]; then
        echo "ERROR [${dataset}]: residues_csv not found: ${residues_csv}"
        return 1
    fi

    conda_activate "$CONDA_ENV_QFIT"

    local lookup=$(grep "^${dataset} " "$LOOKUP_FILE")
    if [ -z "$lookup" ]; then
        echo "Warning: No match found for dataset ${dataset}, skipping."
        return 1
    fi
    local resolution=$(echo "$lookup" | awk '{print $3}')

    echo "[${dataset}] Model: $final_pdb"
    echo "[${dataset}] Residues: $residues_csv"
    echo "[${dataset}] Resolution: $resolution"

    local rotamer_extra_args=()
    [ -n "$rotamer_rscc_threshold" ] && rotamer_extra_args+=(--rscc_threshold "$rotamer_rscc_threshold")
    [ -n "$rotamer_rscc_improvement_threshold" ] && rotamer_extra_args+=(--rscc_improvement_threshold "$rotamer_rscc_improvement_threshold")
    [ -n "$clash_vdw_scale" ] && rotamer_extra_args+=(--clash_vdw_scale "$clash_vdw_scale")
    [ -n "$hbond_clash_vdw_scale" ] && rotamer_extra_args+=(--hbond_clash_vdw_scale "$hbond_clash_vdw_scale")
    [ -n "$max_clash_group_size" ] && rotamer_extra_args+=(--max_clash_group_size "$max_clash_group_size")
    [ -n "$max_clash_group_expansions" ] && rotamer_extra_args+=(--max_clash_group_expansions "$max_clash_group_expansions")
    [ -n "$clash_domain_top_k" ] && rotamer_extra_args+=(--clash_domain_top_k "$clash_domain_top_k")
    [ -n "$clash_solve_node_budget" ] && rotamer_extra_args+=(--clash_solve_node_budget "$clash_solve_node_budget")

    rotamer_optimize "$dataset_dir" "$final_pdb" "$rotamer_output_folder" -r "$resolution" \
        "${rotamer_extra_args[@]}"
    local exit_code=$?
    if [ $exit_code -ne 0 ]; then
        echo "ERROR [${dataset}]: rotamer_optimize failed with exit code $exit_code"
        return 1
    fi

    echo "Completed [${dataset}]: ${final_pdb} -> ${rotamer_dir}/rotamer_optimized.pdb"
}
export -f rotamer_optimize_process_dataset

do_rotamer_optimize() {
    conda_activate "$CONDA_ENV_QFIT"

    echo "Starting run"
    local start_time=$(date +%s)
    printf '%s\n' "${DATASETS[@]}" | parallel -j "$NUM_PARALLEL_DEFAULT" rotamer_optimize_process_dataset {}
    echo "All jobs completed"
    print_elapsed "$start_time"
}

######################################################################
# Stage 7b: rsr_rotamer
######################################################################
# Real-space refines rotamer_optimized.pdb (stage 7a's output), restricted to the same
# residues_with_placer_conformers.csv residue list rotamer_optimize was run against, via
# real_space_refine_final.py (SINGLE-mode coot refinement - same invocation stage 6b's rsr_final
# uses on final_model.pdb).

rsr_rotamer_process_dataset() {
    local dataset=$1
    local dataset_dir="${DATASETS_DIR}/${dataset}"
    local filter2_dir="${dataset_dir}/${run_name}/${placer_run_name}/${filter_run_name}/${placer2_run_name}/${filter2_run_name}"
    local final_dir="${filter2_dir}/${final_run_name}"
    local rotamer_dir="${final_dir}/${rotamer_run_name}"

    if [ "$overwrite" -ne 1 ] && files_exist "${rotamer_dir}/rotamer_refined.pdb"; then
        echo "Skipping [${dataset}]: rsr_rotamer already complete (${rotamer_dir}/rotamer_refined.pdb exists)."
        return 0
    fi

    conda_activate "$CONDA_ENV_RSR"

    local map_file
    map_file=$(find "${dataset_dir}" -maxdepth 1 -name "${dataset}-event_1*" | head -1)
    if [ -z "$map_file" ]; then
        echo "ERROR [${dataset}]: No event map found matching ${dataset}-event_1* in ${dataset_dir}"
        return 1
    fi

    local cluster_reps_csv="${filter2_dir}/cluster_reps.csv"
    if [ ! -f "$cluster_reps_csv" ]; then
        echo "ERROR [${dataset}]: cluster_reps.csv not found: $cluster_reps_csv"
        return 1
    fi

    local cif_list
    cif_list=$(build_cif_list_from_cluster_reps "$dataset" "$cluster_reps_csv") || return 1

    local input_pdb="${rotamer_dir}/rotamer_optimized.pdb"
    local residues_csv="${final_dir}/residues_with_placer_conformers.csv"
    local output_pdb="${rotamer_dir}/rotamer_refined.pdb"

    if [ ! -f "$input_pdb" ]; then
        echo "ERROR [${dataset}]: rotamer_optimized.pdb not found: ${input_pdb}"
        return 1
    fi
    if [ ! -f "$residues_csv" ]; then
        echo "ERROR [${dataset}]: residues_csv not found: ${residues_csv}"
        return 1
    fi

    echo "[${dataset}] Map: $map_file"
    echo "[${dataset}] Using CIF restraints list: $cif_list"

    local rsr_extra_args=()
    [ -n "$rsr_n_cycles" ] && rsr_extra_args+=(--n-cycles "$rsr_n_cycles")
    [ -n "$rsr_map_weight" ] && rsr_extra_args+=(--map-weight "$rsr_map_weight")
    [ -n "$rsr_moved_threshold" ] && rsr_extra_args+=(--moved-threshold "$rsr_moved_threshold")

    python "$RSR_SCRIPT_FINAL" \
        "$input_pdb" \
        "$residues_csv" \
        "$map_file" \
        "$output_pdb" \
        --cif-list "$cif_list" \
        "${rsr_extra_args[@]}"
    local exit_code=$?
    if [ $exit_code -ne 0 ]; then
        echo "ERROR [${dataset}]: Refinement failed with exit code $exit_code"
        return 1
    fi

    echo "Completed: ${dataset}"
}
export -f rsr_rotamer_process_dataset

do_rsr_rotamer() {
    conda_activate "$CONDA_ENV_RSR"
    export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1

    echo "Starting run"
    local start_time=$(date +%s)
    printf '%s\n' "${DATASETS[@]}" | parallel -j "$NUM_PARALLEL_DEFAULT" rsr_rotamer_process_dataset {}
    echo "All jobs completed"
    print_elapsed "$start_time"

    unset OMP_NUM_THREADS OPENBLAS_NUM_THREADS MKL_NUM_THREADS NUMEXPR_NUM_THREADS VECLIB_MAXIMUM_THREADS
}

######################################################################
# Stage 7c: calc_rotamer_refined_rscc
######################################################################
# Per-residue RSCC of rotamer_refined.pdb (stage 7b's output), restricted (via calc_rscc's
# --residues-csv) to residues_with_placer_conformers.csv - the same residues rotamer_optimize
# was run against.

calc_rotamer_refined_rscc_process_dataset() {
    local dataset=$1
    local dataset_dir="${DATASETS_DIR}/${dataset}"
    local final_dir="${dataset_dir}/${run_name}/${placer_run_name}/${filter_run_name}/${placer2_run_name}/${filter2_run_name}/${final_run_name}"
    local rotamer_dir="${final_dir}/${rotamer_run_name}"
    local structure="${rotamer_dir}/rotamer_refined.pdb"
    local output_csv="${rotamer_dir}/rotamer_refined_rscc.csv"

    if [ "$overwrite" -ne 1 ] && files_exist "$output_csv"; then
        echo "Skipping [${dataset}]: calc_rotamer_refined_rscc already complete (${output_csv} exists)."
        return 0
    fi

    conda_activate "$CONDA_ENV_QFIT"
    shopt -s nullglob

    local lookup=$(grep "^${dataset} " "$LOOKUP_FILE")
    if [ -z "$lookup" ]; then
        echo "Warning: No match found for dataset ${dataset}, skipping."
        return 1
    fi
    local resolution=$(echo "$lookup" | awk '{print $3}')

    local event_maps=("${dataset_dir}/${dataset}-event_"*)
    if [ ${#event_maps[@]} -eq 0 ]; then
        echo "Warning [${dataset}]: no event maps found matching ${dataset_dir}/${dataset}-event_*, skipping."
        return 1
    fi

    if [ ! -f "$structure" ]; then
        echo "Warning [${dataset}]: rotamer_refined.pdb not found: ${structure}, skipping."
        return 1
    fi

    local residues_csv="${final_dir}/residues_with_placer_conformers.csv"
    if [ ! -f "$residues_csv" ]; then
        echo "Warning [${dataset}]: residues_csv not found: ${residues_csv}, skipping."
        return 1
    fi

    local bfactor_extra_args=()
    [ -n "$bfactor" ] && bfactor_extra_args+=(--bfactor "$bfactor")

    calc_rscc "${structure}" "${event_maps[@]}" "${resolution}" "${output_csv}" \
        --residues-csv "$residues_csv" "${bfactor_extra_args[@]}"

    local calc_exit=$?
    if [ $calc_exit -ne 0 ]; then
        echo "ERROR [${dataset}]: calc_rscc failed on ${structure} with exit code ${calc_exit}"
        return 1
    fi

    echo "Completed [${dataset}]: ${structure} -> ${output_csv}"
}
export -f calc_rotamer_refined_rscc_process_dataset

do_calc_rotamer_rscc() {
    conda_activate "$CONDA_ENV_QFIT"

    echo "Starting run"
    local start_time=$(date +%s)
    printf '%s\n' "${DATASETS[@]}" | parallel -j "$NUM_PARALLEL_DEFAULT" calc_rotamer_refined_rscc_process_dataset {}
    echo "All jobs completed"
    print_elapsed "$start_time"
}

######################################################################
# Stage 7d: select_optimized_residues
######################################################################
# For each residues_with_placer_conformers.csv residue, compares RSCC in
# final_model_refined_rscc.csv (6c) vs rotamer_refined_rscc.csv (7c) and reverts to
# final_model_refined's conformation unless rotamer wins by >= REVERT_MIN_DIFF (0.1).
#                                        -> .../<final_run_name>/<rotamer_run_name>/
#                                           optimized.pdb + optimized_rscc.csv +
#                                           reverted_residues.csv

select_optimized_residues_process_dataset() {
    local dataset=$1
    local dataset_dir="${DATASETS_DIR}/${dataset}"
    local final_dir="${dataset_dir}/${run_name}/${placer_run_name}/${filter_run_name}/${placer2_run_name}/${filter2_run_name}/${final_run_name}"
    local rotamer_dir="${final_dir}/${rotamer_run_name}"
    local final_model="${final_dir}/final_model_refined.pdb"
    local rotamer_model="${rotamer_dir}/rotamer_refined.pdb"
    local final_rscc_csv="${final_dir}/final_model_refined_rscc.csv"
    local rotamer_rscc_csv="${rotamer_dir}/rotamer_refined_rscc.csv"
    local residues_csv="${final_dir}/residues_with_placer_conformers.csv"
    local output_pdb="${rotamer_dir}/optimized.pdb"
    local output_rscc_csv="${rotamer_dir}/optimized_rscc.csv"
    local reverted_csv="${rotamer_dir}/reverted_residues.csv"

    if [ "$overwrite" -ne 1 ] && files_exist "$output_pdb" "$output_rscc_csv" "$reverted_csv"; then
        echo "Skipping [${dataset}]: select_optimized_residues already complete (${output_pdb} exists)."
        return 0
    fi

    for f in "$final_model" "$rotamer_model" "$final_rscc_csv" "$rotamer_rscc_csv" "$residues_csv"; do
        if [ ! -f "$f" ]; then
            echo "Warning [${dataset}]: required file not found: ${f}, skipping."
            return 1
        fi
    done

    local select_extra_args=()
    [ -n "$revert_min_diff" ] && select_extra_args+=(--revert_min_diff "$revert_min_diff")

    conda_activate "$CONDA_ENV_QFIT"
    select_optimized_residues "$final_model" "$rotamer_model" "$final_rscc_csv" "$rotamer_rscc_csv" \
        "$residues_csv" "$output_pdb" "${select_extra_args[@]}"
    local exit_code=$?
    if [ $exit_code -ne 0 ]; then
        echo "ERROR [${dataset}]: select_optimized_residues failed with exit code $exit_code"
        return 1
    fi

    echo "Completed [${dataset}]: ${output_pdb}, ${output_rscc_csv}, ${reverted_csv}"
}
export -f select_optimized_residues_process_dataset

do_select_optimized_residues() {
    conda_activate "$CONDA_ENV_QFIT"

    echo "Starting run"
    local start_time=$(date +%s)
    printf '%s\n' "${DATASETS[@]}" | parallel -j "$NUM_PARALLEL_DEFAULT" select_optimized_residues_process_dataset {}
    echo "All jobs completed"
    print_elapsed "$start_time"
}

######################################################################
# Stage 7e: plot_residues_vs_ref_rotamer (only runs when -c is given)
######################################################################
# Per-residue RSCC of optimized.pdb (7d) vs reference, restricted to
# residues_with_placer_conformers.csv, pooled across datasets. Also writes
# rotamer_refined_vs_reference_rscc_outliers.csv (residues >= 0.1 worse than reference).

do_plot_residues_vs_ref_rotamer() {
    conda_activate "$CONDA_ENV_EVAL"

    local out_dir="${GRAPHS_DIR}/${run_name}/${placer_run_name}/${filter_run_name}/${placer2_run_name}/${filter2_run_name}/${final_run_name}/${rotamer_run_name}"
    echo "Starting run"
    local start_time=$(date +%s)
    python "$PLOT_RESIDUES_VS_REF_ROTAMER_PY" \
        "$run_name" "$placer_run_name" "$filter_run_name" "$placer2_run_name" "$filter2_run_name" "$final_run_name" "$rotamer_run_name" \
        --datasets-dir "$DATASETS_DIR" --datasets-file "$DATASETS_FILE" \
        --ref-set "$REF_SET" --ref-pdb-pattern "$REF_SET_PDB_PATTERN" --graphs-dir "$out_dir"
    echo "All jobs completed"
    print_elapsed "$start_time"
}

rotamer_ref_comparison_outputs_exist() {
    local out_dir="${GRAPHS_DIR}/${run_name}/${placer_run_name}/${filter_run_name}/${placer2_run_name}/${filter2_run_name}/${final_run_name}/${rotamer_run_name}"
    files_exist "${out_dir}/rotamer_refined_vs_reference_rscc_restricted.png"
}

######################################################################
# Stage 7f: aggregate_rotamer_worse_residues (only runs when -c is given)
######################################################################
# Pooled csv (no plot) of residues_with_placer_conformers.csv residues whose optimized RSCC
# (7d) is more than 0.1 worse than either the reference residue's RSCC or final_model_refined's.

do_aggregate_rotamer_worse_residues() {
    conda_activate "$CONDA_ENV_EVAL"

    local out_dir="${GRAPHS_DIR}/${run_name}/${placer_run_name}/${filter_run_name}/${placer2_run_name}/${filter2_run_name}/${final_run_name}/${rotamer_run_name}"
    echo "Starting run"
    local start_time=$(date +%s)
    python "$AGGREGATE_ROTAMER_WORSE_RESIDUES_PY" \
        "$run_name" "$placer_run_name" "$filter_run_name" "$placer2_run_name" "$filter2_run_name" "$final_run_name" "$rotamer_run_name" \
        --datasets-dir "$DATASETS_DIR" --datasets-file "$DATASETS_FILE" \
        --ref-set "$REF_SET" --ref-pdb-pattern "$REF_SET_PDB_PATTERN" --graphs-dir "$out_dir"
    echo "All jobs completed"
    print_elapsed "$start_time"
}

rotamer_worse_residues_outputs_exist() {
    local out_dir="${GRAPHS_DIR}/${run_name}/${placer_run_name}/${filter_run_name}/${placer2_run_name}/${filter2_run_name}/${final_run_name}/${rotamer_run_name}"
    files_exist "${out_dir}/rotamer_refined_worse_residues.csv"
}

######################################################################
# Stage 8a: despot
######################################################################
# For each dataset (only with rotamer_run_name and despot_run_name), pools every placer2
# conformer via extract_ligand_conformers, expands optimized.pdb's protein around them
# (symmetry_expand --ligand-conformers-pdb), converts protein + ligands to mol2, and scores every
# conformer with DESPOT's score_complex.py. despot_filter.py then reselects, per filter2 cluster,
# the pose maximizing RSCC - --despot_rscc_weight*normalized_DESPOT, kept only if it clears
# --despot_rscc_threshold and --despot_threshold.

despot_process_dataset() {
    local dataset=$1
    local dataset_dir="${DATASETS_DIR}/${dataset}"
    local placer2_dir="${dataset_dir}/${run_name}/${placer_run_name}/${filter_run_name}/${placer2_run_name}"
    local filter2_dir="${placer2_dir}/${filter2_run_name}"
    local final_dir="${filter2_dir}/${final_run_name}"
    local rotamer_dir="${final_dir}/${rotamer_run_name}"
    local final_model="${rotamer_dir}/optimized.pdb"
    local apo_structure="${dataset_dir}/${dataset}-aligned-structure.pdb"
    local despot_dir="${rotamer_dir}/${despot_run_name}"
    local despot_csv_check="${despot_dir}/${dataset}_DESPOT.csv"
    local conformer_map_check="${despot_dir}/conformer_map.csv"
    local despot_filtered_check="${despot_dir}/despot_filtered.pdb"

    if [ "$overwrite" -ne 1 ] && files_exist "$despot_csv_check" "$conformer_map_check" "$despot_filtered_check"; then
        echo "Skipping [${dataset}]: despot already complete (${despot_filtered_check} exists)."
        return 0
    fi

    if [ ! -f "$final_model" ]; then
        echo "Warning [${dataset}]: optimized.pdb not found: ${final_model}, skipping."
        return 1
    fi
    if [ ! -f "$apo_structure" ]; then
        echo "Warning [${dataset}]: apo structure not found: ${apo_structure}, skipping."
        return 1
    fi

    local cell_lookup=$(grep "^${dataset} " "$DESPOT_CELL_LOOKUP_FILE")
    if [ -z "$cell_lookup" ]; then
        echo "Warning [${dataset}]: no crystal cell/space group info found in ${CSV_FILE}, skipping."
        return 1
    fi
    local cl_dataset a b c alpha beta gamma space_group
    read -r cl_dataset a b c alpha beta gamma space_group <<< "$cell_lookup"

    local smiles_lookup=$(grep "^${dataset} " "$LIG_SMILES_LOOKUP_FILE")
    local smiles=$(echo "$smiles_lookup" | awk '{print $2}')
    if [ -z "$smiles" ]; then
        echo "Warning [${dataset}]: no SMILES found, skipping."
        return 1
    fi

    local lookup=$(grep "^${dataset} " "$LOOKUP_FILE")
    if [ -z "$lookup" ]; then
        echo "Warning [${dataset}]: no resolution found in ${LOOKUP_FILE}, skipping."
        return 1
    fi
    local resolution=$(echo "$lookup" | awk '{print $3}')

    echo "Processing ${dataset}: space_group=${space_group}, cell=(${a} ${b} ${c} ${alpha} ${beta} ${gamma}), resolution=${resolution}"

    mkdir -p "$despot_dir"

    local despot_log="${despot_dir}/log.txt"
    exec > >(tee "$despot_log") 2>&1

    local dataset_start_time=$(date +%s)

    local ligs_pdb="${despot_dir}/ligs.pdb"
    local conformer_map_csv="${despot_dir}/conformer_map.csv"
    local expanded_pdb="${despot_dir}/expanded.pdb"
    local original_ligand_dir="${despot_dir}/original_ligand"
    local ligs_mol2="${despot_dir}/ligs.mol2"
    local expanded_mol2="${despot_dir}/expanded.mol2"
    local despot_csv="${despot_dir}/${dataset}_DESPOT.csv"

    local step_start_time=$(date +%s)
    conda_activate "$CONDA_ENV_QFIT"
    extract_ligand_conformers "$placer2_dir" "$dataset" "$ligs_pdb" "$conformer_map_csv"
    local status=$?
    conda_deactivate
    print_elapsed "$step_start_time" "[${dataset}] extract_ligand_conformers"
    if [ $status -ne 0 ]; then
        echo "ERROR [${dataset}]: extract_ligand_conformers failed with exit code ${status}"
        print_elapsed "$dataset_start_time" "[${dataset}] despot"
        return 1
    fi

    step_start_time=$(date +%s)
    conda_activate "$CONDA_ENV_QFIT"
    # original_ligand_dir captures final_model's own ligand instance for the record only
    # (never read downstream) - the scored ligands are $ligs_pdb via --ligand-conformers-pdb.
    symmetry_expand "$final_model" "$expanded_pdb" "$space_group" "$a" "$b" "$c" "$alpha" "$beta" "$gamma" \
        "$expand_distance_cutoff" "$original_ligand_dir" --ligand-conformers-pdb "$ligs_pdb"
    status=$?
    conda_deactivate
    print_elapsed "$step_start_time" "[${dataset}] symmetry_expand"
    if [ $status -ne 0 ]; then
        echo "ERROR [${dataset}]: symmetry_expand failed with exit code ${status}"
        print_elapsed "$dataset_start_time" "[${dataset}] despot"
        return 1
    fi

    "$PDB_TO_MOL2_SH" "${despot_dir}/ligs" "$smiles" "$CONDA_SH" "$CONDA_ENV_QFIT" "$CONDA_ENV_QFIT" \
        "$ASSIGN_BOND_ORDERS_PY" "$ligs_pdb"
    status=$?
    if [ $status -ne 0 ]; then
        echo "ERROR [${dataset}]: pdb_to_mol2.sh failed on ${ligs_pdb} with exit code ${status}"
        print_elapsed "$dataset_start_time" "[${dataset}] despot"
        return 1
    fi

    step_start_time=$(date +%s)
    "$PROTEIN_TO_MOL2_SH" "$expanded_pdb" "$CONDA_SH" "$CONDA_ENV_QFIT"
    status=$?
    print_elapsed "$step_start_time" "[${dataset}] pdb2pqr"
    if [ $status -ne 0 ]; then
        echo "ERROR [${dataset}]: protein_to_mol2.sh failed on ${expanded_pdb} with exit code ${status}"
        print_elapsed "$dataset_start_time" "[${dataset}] despot"
        return 1
    fi

    step_start_time=$(date +%s)
    conda_activate "$CONDA_ENV_DESPOT"
    python "$DESPOT_SCRIPT" -p "$expanded_mol2" -l "$ligs_mol2" -o "$despot_csv" --database "$DESPOT_DATABASE"
    status=$?
    conda_deactivate
    print_elapsed "$step_start_time" "[${dataset}] despot score_complex.py"
    if [ $status -ne 0 ]; then
        echo "ERROR [${dataset}]: DESPOT score_complex.py failed with exit code ${status}"
        print_elapsed "$dataset_start_time" "[${dataset}] despot"
        return 1
    fi

    shopt -s nullglob
    local event_maps=("${dataset_dir}/${dataset}-event_"*)
    shopt -u nullglob
    if [ ${#event_maps[@]} -eq 0 ]; then
        echo "ERROR [${dataset}]: no event maps found matching ${dataset_dir}/${dataset}-event_*"
        print_elapsed "$dataset_start_time" "[${dataset}] despot"
        return 1
    fi

    local despot_filtered_pdb="${despot_dir}/despot_filtered.pdb"
    local despot_filter_args=()
    [ -n "$despot_threshold" ] && despot_filter_args+=(--despot-threshold "$despot_threshold")
    [ -n "$despot_rscc_threshold" ] && despot_filter_args+=(--rscc-threshold "$despot_rscc_threshold")
    [ -n "$despot_rscc_weight" ] && despot_filter_args+=(--rscc-weight "$despot_rscc_weight")
    [ -n "$bfactor" ] && despot_filter_args+=(--bfactor "$bfactor")

    step_start_time=$(date +%s)
    conda_activate "$CONDA_ENV_QFIT"
    despot_filter "$final_model" "$apo_structure" "$filter2_dir" "$despot_dir" "${event_maps[@]}" "$resolution" \
        "$despot_filtered_pdb" "${despot_filter_args[@]}"
    status=$?
    conda_deactivate
    print_elapsed "$step_start_time" "[${dataset}] despot_filter"
    if [ $status -ne 0 ]; then
        echo "ERROR [${dataset}]: despot_filter failed with exit code ${status}"
        print_elapsed "$dataset_start_time" "[${dataset}] despot"
        return 1
    fi

    echo "Completed [${dataset}]: ${despot_csv}, ${despot_filtered_pdb}"
    print_elapsed "$dataset_start_time" "[${dataset}] despot"
}
export -f despot_process_dataset

do_despot() {
    # Caps BLAS/OpenMP/numba threading in DESPOT's score_complex.py to 1 thread each, scoped to
    # this stage only.
    export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMBA_NUM_THREADS=1

    echo "Starting run"
    local start_time=$(date +%s)
    printf '%s\n' "${DATASETS[@]}" | parallel -j "$NUM_PARALLEL_DEFAULT" --line-buffer despot_process_dataset {}
    echo "All jobs completed"
    print_elapsed "$start_time"

    unset OMP_NUM_THREADS OPENBLAS_NUM_THREADS MKL_NUM_THREADS NUMEXPR_NUM_THREADS VECLIB_MAXIMUM_THREADS NUMBA_NUM_THREADS
}

######################################################################
# Stage 8b: plot_lig_vs_ref_despot (only with -c)
######################################################################
# Pooled RSCC of despot_filter-surviving ligand poses vs the reference set, matched by centroid
# distance like stages 3d/5b.

do_plot_lig_vs_ref_despot() {
    conda_activate "$CONDA_ENV_EVAL"

    local out_dir="${GRAPHS_DIR}/${run_name}/${placer_run_name}/${filter_run_name}/${placer2_run_name}/${filter2_run_name}/${final_run_name}/${rotamer_run_name}/${despot_run_name}"
    echo "Starting run"
    local start_time=$(date +%s)
    python "$PLOT_LIG_VS_REF_DESPOT_PY" \
        "$run_name" "$placer_run_name" "$filter_run_name" \
        "$placer2_run_name" "$filter2_run_name" "$final_run_name" "${rotamer_run_name}/${despot_run_name}" \
        --datasets-dir "$DATASETS_DIR" --datasets-file "$DATASETS_FILE" \
        --ref-set "$REF_SET" --ref-pdb-pattern "$REF_SET_PDB_PATTERN" --graphs-dir "$out_dir"
    echo "All jobs completed"
    print_elapsed "$start_time"
}

despot_lig_vs_ref_outputs_exist() {
    files_exist "${GRAPHS_DIR}/${run_name}/${placer_run_name}/${filter_run_name}/${placer2_run_name}/${filter2_run_name}/${final_run_name}/${rotamer_run_name}/${despot_run_name}/lig_vs_reference_rscc.png"
}

######################################################################
# Stage 8c: plot_despot_vs_ref (only with -c)
######################################################################
# Pooled scatter of each dataset's reference-set DESPOT score (0d) vs the matched pipeline
# ligand's DESPOT score, both normalized, matched like stages 3d/5b/8b.

do_plot_despot_vs_ref() {
    conda_activate "$CONDA_ENV_EVAL"

    local out_dir="${GRAPHS_DIR}/${run_name}/${placer_run_name}/${filter_run_name}/${placer2_run_name}/${filter2_run_name}/${final_run_name}/${rotamer_run_name}/${despot_run_name}"
    echo "Starting run"
    local start_time=$(date +%s)
    python "$PLOT_DESPOT_VS_REF_PY" \
        "$run_name" "$placer_run_name" "$filter_run_name" \
        "$placer2_run_name" "$filter2_run_name" "$final_run_name" "${rotamer_run_name}/${despot_run_name}" \
        --datasets-dir "$DATASETS_DIR" --datasets-file "$DATASETS_FILE" \
        --ref-set "$REF_SET" --ref-pdb-pattern "$REF_SET_PDB_PATTERN" --graphs-dir "$out_dir"
    echo "All jobs completed"
    print_elapsed "$start_time"
}

despot_vs_ref_outputs_exist() {
    files_exist "${GRAPHS_DIR}/${run_name}/${placer_run_name}/${filter_run_name}/${placer2_run_name}/${filter2_run_name}/${final_run_name}/${rotamer_run_name}/${despot_run_name}/despot_vs_reference.png"
}

######################################################################
# Stage 8d: plot_rscc_despot_tradeoff (only with -c)
######################################################################
# Pooled scatter of despot_filter survivors: y = pipeline RSCC - reference RSCC, x = reference
# DESPOT - pipeline DESPOT (both normalized).

do_plot_rscc_despot_tradeoff() {
    conda_activate "$CONDA_ENV_EVAL"

    local out_dir="${GRAPHS_DIR}/${run_name}/${placer_run_name}/${filter_run_name}/${placer2_run_name}/${filter2_run_name}/${final_run_name}/${rotamer_run_name}/${despot_run_name}"
    echo "Starting run"
    local start_time=$(date +%s)
    python "$PLOT_RSCC_DESPOT_TRADEOFF_PY" \
        "$run_name" "$placer_run_name" "$filter_run_name" \
        "$placer2_run_name" "$filter2_run_name" "$final_run_name" "${rotamer_run_name}/${despot_run_name}" \
        --datasets-dir "$DATASETS_DIR" --datasets-file "$DATASETS_FILE" \
        --ref-set "$REF_SET" --ref-pdb-pattern "$REF_SET_PDB_PATTERN" --graphs-dir "$out_dir"
    echo "All jobs completed"
    print_elapsed "$start_time"
}

rscc_despot_tradeoff_outputs_exist() {
    files_exist "${GRAPHS_DIR}/${run_name}/${placer_run_name}/${filter_run_name}/${placer2_run_name}/${filter2_run_name}/${final_run_name}/${rotamer_run_name}/${despot_run_name}/rscc_despot_tradeoff_vs_reference.png"
}

######################################################################
# Stage 8e: plot_residues_vs_ref_despot (only with -c)
######################################################################
# Per-residue RSCC of optimized vs reference, restricted to despot_run_name/modified_residues.csv
# (7e's residues minus whatever reset_protein_to_apo_where_unbacked reset to apo).

do_plot_residues_vs_ref_despot() {
    conda_activate "$CONDA_ENV_EVAL"

    local out_dir="${GRAPHS_DIR}/${run_name}/${placer_run_name}/${filter_run_name}/${placer2_run_name}/${filter2_run_name}/${final_run_name}/${rotamer_run_name}/${despot_run_name}"
    echo "Starting run"
    local start_time=$(date +%s)
    python "$PLOT_RESIDUES_VS_REF_DESPOT_PY" \
        "$run_name" "$placer_run_name" "$filter_run_name" "$placer2_run_name" "$filter2_run_name" \
        "$final_run_name" "$rotamer_run_name" "$despot_run_name" \
        --datasets-dir "$DATASETS_DIR" --datasets-file "$DATASETS_FILE" \
        --ref-set "$REF_SET" --ref-pdb-pattern "$REF_SET_PDB_PATTERN" --graphs-dir "$out_dir"
    echo "All jobs completed"
    print_elapsed "$start_time"
}

despot_residues_vs_ref_outputs_exist() {
    files_exist "${GRAPHS_DIR}/${run_name}/${placer_run_name}/${filter_run_name}/${placer2_run_name}/${filter2_run_name}/${final_run_name}/${rotamer_run_name}/${despot_run_name}/rotamer_refined_despot_vs_reference_rscc_restricted.png"
}

######################################################################
# Stage 9: analysis plots (collapsed into one idempotent unit)
######################################################################
# Every plot below (per-dataset and pooled) is checked/skipped together - see
# stage9_outputs_exist() and stage9_plots() below.

# plot_cluster_reps_rscc: pooled histograms of the cluster-rep RSCC values
# already written into cluster_reps.csv by filter/filter2 - no RSCC values
# are computed here.
do_plot_cluster_reps_rscc() {
    conda_activate "$CONDA_ENV_EVAL"

    echo "Starting run"
    local start_time=$(date +%s)
    python "$PLOT_CLUSTER_REPS_PY" \
        "$run_name" "$placer_run_name" "$filter_run_name" \
        "$placer2_run_name" "$filter2_run_name" "$final_run_name" \
        --datasets-dir "$DATASETS_DIR" --datasets-file "$DATASETS_FILE" \
        --rotamer-run-name "$rotamer_run_name" --despot-run-name "$despot_run_name"
    echo "All jobs completed"
    print_elapsed "$start_time"
}

# aggregate_protein_rscc: scatter plots comparing every protein residue's
# RSCC (apo vs backbone vs final), pooling the per-residue csvs already
# written by calc_apo_rscc, calc_backbone_refined_rscc, and
# calc_final_refined_rscc.
do_aggregate_protein_rscc() {
    conda_activate "$CONDA_ENV_EVAL"

    echo "Starting run"
    local start_time=$(date +%s)
    python "$AGGREGATE_PROTEIN_RSCC_PY" \
        "$run_name" "$placer_run_name" "$filter_run_name" \
        "$placer2_run_name" "$filter2_run_name" "$final_run_name" \
        --datasets-dir "$DATASETS_DIR" --datasets-file "$DATASETS_FILE" \
        --rotamer-run-name "$rotamer_run_name" --despot-run-name "$despot_run_name"
    echo "All jobs completed"
    print_elapsed "$start_time"
}

# aggregate_lig_rscc: filter_2-vs-filter_1 ligand RSCC scatter (the only
# ligand RSCC comparison that makes sense - apo has no ligand).
do_aggregate_lig_rscc() {
    conda_activate "$CONDA_ENV_EVAL"

    echo "Starting run"
    local start_time=$(date +%s)
    python "$AGGREGATE_LIG_RSCC_PY" \
        "$run_name" "$placer_run_name" "$filter_run_name" \
        "$placer2_run_name" "$filter2_run_name" "$final_run_name" \
        --datasets-dir "$DATASETS_DIR" --datasets-file "$DATASETS_FILE" \
        --rotamer-run-name "$rotamer_run_name" --despot-run-name "$despot_run_name"
    echo "All jobs completed"
    print_elapsed "$start_time"
}

# Pooled (cross-dataset) counterparts of the plots above, into stage9_graphs_dir. Scatter plots
# are colored by point density; histograms are not.

do_plot_cluster_reps_rscc_pooled() {
    conda_activate "$CONDA_ENV_EVAL"

    local out_dir
    out_dir=$(stage9_graphs_dir)
    echo "Starting run"
    local start_time=$(date +%s)
    python "$PLOT_CLUSTER_REPS_POOLED_PY" \
        "$run_name" "$placer_run_name" "$filter_run_name" \
        "$placer2_run_name" "$filter2_run_name" "$final_run_name" \
        --datasets-dir "$DATASETS_DIR" --datasets-file "$DATASETS_FILE" --graphs-dir "$out_dir"
    echo "All jobs completed"
    print_elapsed "$start_time"
}

do_plot_protein_rscc_pooled() {
    conda_activate "$CONDA_ENV_EVAL"

    local out_dir
    out_dir=$(stage9_graphs_dir)
    echo "Starting run"
    local start_time=$(date +%s)
    python "$PLOT_PROTEIN_RSCC_POOLED_PY" \
        "$run_name" "$placer_run_name" "$filter_run_name" \
        "$placer2_run_name" "$filter2_run_name" "$final_run_name" \
        --datasets-dir "$DATASETS_DIR" --datasets-file "$DATASETS_FILE" --graphs-dir "$out_dir"
    echo "All jobs completed"
    print_elapsed "$start_time"
}

######################################################################
# Stage 9 member: plot_rotamer_vs_pipeline (moved from the old Stage 7f - runs whenever
# rotamer_run_name is given, not gated behind -c)
######################################################################
# Per-residue RSCC of optimized.pdb (7d) vs final_model_refined and backbone_refined, restricted
# to residues_with_placer_conformers.csv, pooled across datasets, into stage9_graphs_dir.

do_plot_rotamer_vs_pipeline() {
    conda_activate "$CONDA_ENV_EVAL"

    local out_dir
    out_dir=$(stage9_graphs_dir)
    echo "Starting run"
    local start_time=$(date +%s)
    python "$PLOT_ROTAMER_VS_PIPELINE_PY" \
        "$run_name" "$placer_run_name" "$filter_run_name" "$placer2_run_name" "$filter2_run_name" "$final_run_name" "$rotamer_run_name" \
        --datasets-dir "$DATASETS_DIR" --datasets-file "$DATASETS_FILE" --graphs-dir "$out_dir"
    echo "All jobs completed"
    print_elapsed "$start_time"
}

rotamer_vs_pipeline_outputs_exist() {
    local out_dir
    out_dir=$(stage9_graphs_dir)
    files_exist "${out_dir}/rotamer_refined_vs_final_refined_rscc_restricted.png" \
                "${out_dir}/rotamer_refined_vs_backbone_refined_rscc_restricted.png"
}

######################################################################
# Stage 9 member: plot_despot_energies + plot_despot_energies_pooled (moved from the old Stage
# 8b - only runs when despot_run_name is given)
######################################################################
# Per-dataset histogram of normalized DESPOT scores (despot_filtered_scores.csv), written via
# dataset_graphs_dir, plus the pooled cross-dataset histogram into stage9_graphs_dir/ligand_energies.png.

do_plot_despot_energies() {
    conda_activate "$CONDA_ENV_EVAL"

    echo "Starting run"
    local start_time=$(date +%s)
    python "$PLOT_DESPOT_ENERGIES_PY" \
        "$run_name" "$placer_run_name" "$filter_run_name" \
        "$placer2_run_name" "$filter2_run_name" "$final_run_name" "${rotamer_run_name}/${despot_run_name}" \
        --datasets-dir "$DATASETS_DIR" --datasets-file "$DATASETS_FILE"
    echo "All jobs completed"
    print_elapsed "$start_time"
}

do_plot_despot_energies_pooled() {
    conda_activate "$CONDA_ENV_EVAL"

    local out_dir
    out_dir=$(stage9_graphs_dir)
    echo "Starting run"
    local start_time=$(date +%s)
    python "$PLOT_DESPOT_ENERGIES_POOLED_PY" \
        "$run_name" "$placer_run_name" "$filter_run_name" \
        "$placer2_run_name" "$filter2_run_name" "$final_run_name" "${rotamer_run_name}/${despot_run_name}" \
        --datasets-dir "$DATASETS_DIR" --datasets-file "$DATASETS_FILE" --graphs-dir "$out_dir"
    echo "All jobs completed"
    print_elapsed "$start_time"
}

do_despot_plots() {
    do_plot_despot_energies
    do_plot_despot_energies_pooled
}

# despot_plots_outputs_exist: pooled ligand_energies.png exists, and every dataset with a
# DESPOT csv also has its per-dataset despot_energies.png.
despot_plots_outputs_exist() {
    local out_dir
    out_dir=$(stage9_graphs_dir)
    files_exist "${out_dir}/ligand_energies.png" || return 1

    local dataset
    for dataset in "${DATASETS[@]}"; do
        local final_dir="${DATASETS_DIR}/${dataset}/${run_name}/${placer_run_name}/${filter_run_name}/${placer2_run_name}/${filter2_run_name}/${final_run_name}"
        local despot_csv="${final_dir}/${rotamer_run_name}/${despot_run_name}/${dataset}_DESPOT.csv"
        local dataset_graphs
        dataset_graphs=$(dataset_stage9_graphs_dir "$dataset")
        if [ -f "$despot_csv" ] && [ ! -f "${dataset_graphs}/despot_energies.png" ]; then
            return 1
        fi
    done
    return 0
}

######################################################################
# Stage 9 member: plot_despot_ligand_summary + plot_despot_ligand_summary_single (moved from
# the old Stage 8d - only runs when despot_run_name is given)
######################################################################
# Pooled scatter of despot_filter-surviving ligands' normalized DESPOT score (x) vs
# cluster_reps.csv RSCC (y), into stage9_graphs_dir. Plus the per-dataset counterpart, each point
# labeled by chain+resi, written via dataset_graphs_dir.

do_plot_despot_ligand_summary() {
    conda_activate "$CONDA_ENV_EVAL"

    local out_dir
    out_dir=$(stage9_graphs_dir)
    echo "Starting run"
    local start_time=$(date +%s)
    python "$PLOT_DESPOT_LIGAND_SUMMARY_PY" \
        "$run_name" "$placer_run_name" "$filter_run_name" \
        "$placer2_run_name" "$filter2_run_name" "$final_run_name" "${rotamer_run_name}/${despot_run_name}" \
        --datasets-dir "$DATASETS_DIR" --datasets-file "$DATASETS_FILE" --graphs-dir "$out_dir"
    echo "All jobs completed"
    print_elapsed "$start_time"
}

do_plot_despot_ligand_summary_single() {
    conda_activate "$CONDA_ENV_EVAL"

    echo "Starting run"
    local start_time=$(date +%s)
    python "$PLOT_DESPOT_LIGAND_SUMMARY_SINGLE_PY" \
        "$run_name" "$placer_run_name" "$filter_run_name" \
        "$placer2_run_name" "$filter2_run_name" "$final_run_name" "${rotamer_run_name}/${despot_run_name}" \
        --datasets-dir "$DATASETS_DIR" --datasets-file "$DATASETS_FILE"
    echo "All jobs completed"
    print_elapsed "$start_time"
}

do_despot_ligand_summary_plots() {
    do_plot_despot_ligand_summary_single
    do_plot_despot_ligand_summary
}

# despot_ligand_summary_outputs_exist: pooled ligand_summary.png exists, and every dataset with
# a despot_filter-kept ligand also has its per-dataset ligand_summary.png.
despot_ligand_summary_outputs_exist() {
    local out_dir
    out_dir=$(stage9_graphs_dir)
    files_exist "${out_dir}/ligand_summary.png" || return 1

    local dataset
    for dataset in "${DATASETS[@]}"; do
        local despot_dir="${DATASETS_DIR}/${dataset}/${run_name}/${placer_run_name}/${filter_run_name}/${placer2_run_name}/${filter2_run_name}/${final_run_name}/${rotamer_run_name}/${despot_run_name}"
        local scores_csv="${despot_dir}/despot_filtered_scores.csv"
        local dataset_graphs
        dataset_graphs=$(dataset_stage9_graphs_dir "$dataset")
        if [ -f "$scores_csv" ] && grep -q ',True$' "$scores_csv" && [ ! -f "${dataset_graphs}/ligand_summary.png" ]; then
            return 1
        fi
    done
    return 0
}

do_stage9_plots() {
    do_plot_cluster_reps_rscc
    do_aggregate_protein_rscc
    do_aggregate_lig_rscc
    do_plot_cluster_reps_rscc_pooled
    do_plot_protein_rscc_pooled
    if [ -n "$rotamer_run_name" ]; then
        do_plot_rotamer_vs_pipeline
        if [ -n "$despot_run_name" ]; then
            do_despot_plots
            do_despot_ligand_summary_plots
        fi
    fi
}

# stage9_outputs_exist: true only if every pooled output exists (plus rotamer/despot pooled
# outputs when those run names are given), and every dataset with a final_model_refined_rscc.csv
# has its per-dataset outputs too.
stage9_outputs_exist() {
    local pooled_dir
    pooled_dir=$(stage9_graphs_dir)
    files_exist \
        "${pooled_dir}/cluster_reps_1_pooled.png" "${pooled_dir}/cluster_reps_2_pooled.png" \
        "${pooled_dir}/protein_backbone_vs_apo_rscc_placer_conformers_pooled.png" \
        "${pooled_dir}/protein_final_vs_apo_rscc_placer_conformers_pooled.png" \
        "${pooled_dir}/protein_final_vs_backbone_rscc_placer_conformers_pooled.png" || return 1

    if [ -n "$rotamer_run_name" ]; then
        rotamer_vs_pipeline_outputs_exist || return 1
        if [ -n "$despot_run_name" ]; then
            despot_plots_outputs_exist || return 1
            despot_ligand_summary_outputs_exist || return 1
        fi
    fi

    local dataset
    for dataset in "${DATASETS[@]}"; do
        local final_dir="${DATASETS_DIR}/${dataset}/${run_name}/${placer_run_name}/${filter_run_name}/${placer2_run_name}/${filter2_run_name}/${final_run_name}"
        [ -f "${final_dir}/final_model_refined_rscc.csv" ] || continue

        local graphs_dir
        graphs_dir=$(dataset_stage9_graphs_dir "$dataset")
        files_exist \
            "${graphs_dir}/cluster_reps_1.png" "${graphs_dir}/cluster_reps_2.png" \
            "${graphs_dir}/protein_backbone_vs_apo_rscc_placer_conformers.png" \
            "${graphs_dir}/protein_final_vs_apo_rscc_placer_conformers.png" \
            "${graphs_dir}/protein_final_vs_backbone_rscc_placer_conformers.png" \
            "${graphs_dir}/lig_filter2_vs_filter1_rscc.png" || return 1
    done
    return 0
}

######################################################################
# Stage orchestration (unchanged shape: check -> run_step -> label)
######################################################################

stage0_apo_rscc() {
    run_step "Stage 0a: convert_ligs" do_convert_ligs
    run_step "Stage 0b: calc_apo_rscc" do_calc_apo_rscc
    if [ "$compare_ref_set" -eq 1 ]; then
        run_step "Stage 0c: calc_ref_set_rscc" do_calc_ref_set_rscc
        if [ -n "$despot_run_name" ]; then
            run_step "Stage 0d: ref_set_despot" do_ref_set_despot
        fi
    fi
}

stage1_run() {
    run_step "Stage 1a: fit_ligand (${run_name})" do_fit_ligand
    run_step_pooled_replot "Stage 1b: plot_fit_ligand_counts (${run_name})" \
        fit_ligand_counts_outputs_exist do_plot_fit_ligand_counts
    if [ "$compare_ref_set" -eq 1 ]; then
        run_step_pooled_replot "Stage 1c: centroid_rmsd_all (${run_name})" \
            centroid_rmsd_all_outputs_exist do_centroid_rmsd_all
    fi
}

stage2_placer() {
    run_step "Stage 2a: placer (${placer_run_name})" do_placer
    run_step "Stage 2b: rsr_placer (${placer_run_name})" do_rsr_placer
    if [ "$compare_ref_set" -eq 1 ]; then
        run_step_pooled_replot "Stage 2c: calc_placer_sampling (${placer_run_name})" \
            placer_sampling_round1_outputs_exist do_placer_sampling_round1
    fi
}

stage3_filter() {
    run_step "Stage 3a: filter (${filter_run_name})" do_filter
    run_step "Stage 3b: rsr_backbone (${filter_run_name})" do_rsr_backbone
    run_step "Stage 3c: calc_backbone_refined_rscc (${filter_run_name})" do_calc_backbone_rscc
    if [ "$compare_ref_set" -eq 1 ]; then
        run_step_pooled_replot "Stage 3d: plot_lig_vs_ref_filter1 + plot_residues_vs_ref_backbone (${filter_run_name})" \
            filter1_ref_comparison_outputs_exist do_filter1_ref_comparison
    fi
}

stage4_placer2() {
    run_step "Stage 4a: placer2 (${placer2_run_name})" do_placer2
    run_step "Stage 4b: rsr_placer2 (${placer2_run_name})" do_rsr_placer2
    if [ "$compare_ref_set" -eq 1 ]; then
        run_step_pooled_replot "Stage 4c: calc_placer_sampling (${placer2_run_name})" \
            placer_sampling_round2_outputs_exist do_placer_sampling_round2
    fi
}

stage5_filter2() {
    run_step "Stage 5a: filter2 (${filter2_run_name})" do_filter2
    if [ "$compare_ref_set" -eq 1 ]; then
        run_step_pooled_replot "Stage 5b: plot_lig_vs_ref_filter2 (${filter2_run_name})" \
            filter2_ref_comparison_outputs_exist do_plot_lig_vs_ref_filter2
    fi
}

stage6_final() {
    run_step "Stage 6a: build_final (${final_run_name})" do_build_final
    run_step "Stage 6b: rsr_final (${final_run_name})" do_rsr_final
    run_step "Stage 6c: calc_final_refined_rscc (${final_run_name})" do_calc_final_rscc
    if [ "$compare_ref_set" -eq 1 ]; then
        run_step_pooled_replot "Stage 6d: plot_residues_vs_ref_final (${final_run_name})" \
            final_ref_comparison_outputs_exist do_plot_residues_vs_ref_final
    fi
    run_step_pooled_replot "Stage 6e: aggregate_clash_groups (${final_run_name})" \
        clash_groups_aggregate_outputs_exist do_aggregate_clash_groups
}

stage7_rotamer() {
    run_step "Stage 7a: rotamer_optimize (${rotamer_run_name})" do_rotamer_optimize
    run_step "Stage 7b: rsr_rotamer (${rotamer_run_name})" do_rsr_rotamer
    run_step "Stage 7c: calc_rotamer_refined_rscc (${rotamer_run_name})" do_calc_rotamer_rscc
    run_step "Stage 7d: select_optimized_residues (${rotamer_run_name})" do_select_optimized_residues
    if [ "$compare_ref_set" -eq 1 ]; then
        run_step_pooled_replot "Stage 7e: plot_residues_vs_ref_rotamer (${rotamer_run_name})" \
            rotamer_ref_comparison_outputs_exist do_plot_residues_vs_ref_rotamer
        run_step_pooled_replot "Stage 7f: aggregate_rotamer_worse_residues (${rotamer_run_name})" \
            rotamer_worse_residues_outputs_exist do_aggregate_rotamer_worse_residues
    fi
}

stage8_despot() {
    run_step "Stage 8a: despot (${despot_run_name})" do_despot
    if [ "$compare_ref_set" -eq 1 ]; then
        run_step_pooled_replot "Stage 8b: plot_lig_vs_ref_despot (${despot_run_name})" \
            despot_lig_vs_ref_outputs_exist do_plot_lig_vs_ref_despot
    fi
    if [ "$compare_ref_set" -eq 1 ]; then
        run_step_pooled_replot "Stage 8c: plot_despot_vs_ref (${despot_run_name})" \
            despot_vs_ref_outputs_exist do_plot_despot_vs_ref
    fi
    if [ "$compare_ref_set" -eq 1 ]; then
        run_step_pooled_replot "Stage 8d: plot_rscc_despot_tradeoff (${despot_run_name})" \
            rscc_despot_tradeoff_outputs_exist do_plot_rscc_despot_tradeoff
    fi
    if [ "$compare_ref_set" -eq 1 ]; then
        run_step_pooled_replot "Stage 8e: plot_residues_vs_ref_despot (${despot_run_name})" \
            despot_residues_vs_ref_outputs_exist do_plot_residues_vs_ref_despot
    fi
}

stage9_plots() {
    run_step_pooled_replot "Stage 9: analysis plots (${final_run_name})" \
        stage9_outputs_exist do_stage9_plots
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
    if [ -n "$rotamer_run_name" ]; then
        stage7_rotamer
        if [ -n "$despot_run_name" ]; then
            stage8_despot
        fi
    fi
    stage9_plots
fi

overall_end=$(date +%s)
elapsed=$((overall_end - overall_start))
hours=$((elapsed / 3600))
minutes=$(((elapsed % 3600) / 60))
seconds=$((elapsed % 60))
echo "========= program.sh complete ========="
printf "Total time: %02d:%02d:%02d (HH:MM:SS)\n" $hours $minutes $seconds

# --- test_case sanity check: confirm despot's output actually landed everywhere it
# should (every dataset's own despot_filtered.pdb, plus the two pooled -c graphs) ---
if [ -n "$despot_run_name" ]; then
    despot_ok=1
    for dataset in "${DATASETS[@]}"; do
        despot_filtered_pdb="${DATASETS_DIR}/${dataset}/${run_name}/${placer_run_name}/${filter_run_name}/${placer2_run_name}/${filter2_run_name}/${final_run_name}/${rotamer_run_name}/${despot_run_name}/despot_filtered.pdb"
        if [ ! -f "$despot_filtered_pdb" ]; then
            despot_ok=0
        fi
    done
    despot_graphs_dir="${GRAPHS_DIR}/${run_name}/${placer_run_name}/${filter_run_name}/${placer2_run_name}/${filter2_run_name}/${final_run_name}/${rotamer_run_name}/${despot_run_name}"
    if [ ! -f "${despot_graphs_dir}/lig_vs_reference_rscc.png" ] || [ ! -f "${despot_graphs_dir}/despot_vs_reference.png" ]; then
        despot_ok=0
    fi
    if [ "$despot_ok" -eq 1 ]; then
        echo "${run_name} test_successful"
    fi
fi

