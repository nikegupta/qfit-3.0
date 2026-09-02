#!/bin/bash
#
# program.sh - combined driver for the full ligand-fitting/PLACER/RSR pipeline.
#takes in 7 positional args corresponding to the stages of the pipeline:
#run_name, placer_run_name, filter_run_name, placer2_run_name, filter2_run_name, final_run_name, despot_run_name
#
# Runs, in order:
#   0a. convert_ligs                    -> LIG_PDB_DIR/<ligand_name>*/<ligand_name>*.mol2
#   0b. calc_apo_rscc                   -> <dataset>/<dataset>-aligned-structure_rscc.csv
#       + calc_apo_z                    -> <dataset>/<dataset>-aligned-structure_z.csv
#   0c. calc_ref_set_rscc (only with -c) -> REF_SET/<dataset>/<REF_SET_PDB_PATTERN%.pdb>_rscc.csv
#   0d. ref_set_despot (only with -c and <despot_run_name>): symmetry_expand
#       (--strip, dropping ligand hydrogens + HOH waters + DMS residues first) the reference
#       structure, convert to mol2, score with DESPOT's score_complex.py -
#       same workflow as Stage 7a, run once per dataset (not nested under any
#       run_name)                        -> REF_SET/<dataset>/<dataset>_DESPOT.csv
#   1a. fit_ligand                      -> <run_name>/
#   1b. plot_fit_ligand_counts (always) -> GRAPHS_DIR/<run_name>/
#   1c. centroid_rmsd_all (only with -c) -> GRAPHS_DIR/<run_name>/
#   2a. placer                          -> <run_name>/<placer_run_name>/
#   2b. rsr_placer                      -> <run_name>/<placer_run_name>/
#   2c. calc_placer_sampling (refined + unrefined, only with -c)
#                                        -> GRAPHS_DIR/<run_name>/<placer_run_name>/
#   3a. filter                          -> .../<filter_run_name>/
#   3b. rsr_backbone                    -> .../<filter_run_name>/
#   3c. calc_backbone_refined_rscc      -> .../<filter_run_name>/
#   3d. plot_lig_vs_ref_filter1, plot_residues_vs_ref_backbone (only with -c)
#                                        -> GRAPHS_DIR/<run_name>/.../<filter_run_name>/
#   4a. placer2                         -> .../<placer2_run_name>/
#   4b. rsr_placer2                     -> .../<placer2_run_name>/
#   4c. calc_placer_sampling (refined + unrefined, only with -c)
#                                        -> GRAPHS_DIR/<run_name>/.../<placer2_run_name>/
#   5a. filter2 (runs the same `filter` script as stage 3a, not `filter_all`)
#                                        -> .../<filter2_run_name>/
#   5b. plot_lig_vs_ref_filter2 (only with -c)
#                                        -> GRAPHS_DIR/<run_name>/.../<filter2_run_name>/
#   6a. build_final                     -> .../<final_run_name>/
#   6b. rsr_final                       -> .../<final_run_name>/
#   6c. calc_final_refined_rscc         -> .../<final_run_name>/
#       + calc_final_refined_z          -> .../<final_run_name>/final_model_refined_z.csv
#       + calc_final_refined_rscc_b     -> .../<final_run_name>/final_model_refined_rscc_b.csv
#         (restricted to residues_with_placer_conformers.csv)
#   6d. plot_residues_vs_ref_final (only with -c)
#                                        -> GRAPHS_DIR/<run_name>/.../<final_run_name>/
#   6e. aggregate_clash_groups (always - no reference set needed): concatenates every
#       dataset's own sidechain_clash_groups.csv (written by build_final in Stage 6a) into
#       one run-wide csv
#                                        -> GRAPHS_DIR/<run_name>/.../<final_run_name>/
#                                           sidechain_clash_groups_combined.csv
#   7a. despot (only runs when <despot_run_name> is given)
#       extract_ligand_conformers pools every placer2 round-2 conformer (not just filter2's
#       selected representative) into one ligs.pdb; symmetry_expand --ligand-conformers-pdb
#       expands final_model_refined.pdb's protein around all of them; convert the expanded
#       protein and ligand conformers to mol2 (lig_scripts/pdb_to_mol2.sh,
#       lig_scripts/protein_to_mol2.sh); score every conformer with DESPOT's score_complex.py
#                                        -> .../<final_run_name>/<despot_run_name>/
#                                           <dataset>_DESPOT.csv + conformer_map.csv + ligs.pdb
#       + despot_filter reselects, per filter2 cluster, the pose maximizing
#         RSCC - --despot_rscc_weight*normalized_DESPOT among that cluster's MSE-vs-DESPOT
#         Pareto front (RSCC computed internally via qfit's transformer, no external calc_rscc
#         needed), keeping it only if it clears both --despot_rscc_threshold and
#         --despot_threshold (both unset by default: despot_filter's own defaults apply) -
#         see despot_filter.py's own docstring
#                                        -> .../<final_run_name>/<despot_run_name>/despot_filtered.pdb
#                                           + despot_filtered_scores.csv (unchanged shape - now
#                                             describing the reselected winner)
#                                           + cluster_reps.csv (filter2's cluster_reps.csv, same
#                                             row order, plus the reselected winner's own info -
#                                             see despot_filter.py)
#   7b. plot_despot_energies (per-dataset histogram, heavy-atom-normalized - now reflects every
#       placer2 conformer's DESPOT score, not just the final poses)
#                                        -> .../<final_run_name>/graphs/
#       + plot_despot_energies_pooled   -> GRAPHS_DIR/<run_name>/.../<final_run_name>/<despot_run_name>/
#   7c. plot_lig_vs_ref_despot (only with -c): despot_filtered.pdb's
#       surviving ligands' RSCC (despot_run_name/cluster_reps.csv's despot_rscc - the
#       reselected winner's own value) vs the reference set, matched the same way as
#       stages 3d/5b - see rscc_common.py's alive_rows
#                                        -> GRAPHS_DIR/<run_name>/.../<final_run_name>/<despot_run_name>/
#   7d. plot_despot_ligand_summary: every surviving ligand's normalized
#       DESPOT score vs its cluster_reps.csv RSCC, no -c needed
#                                        -> GRAPHS_DIR/<run_name>/.../<final_run_name>/<despot_run_name>/
#       + plot_despot_ligand_summary_single (per-dataset, chain+resi-labeled)
#                                        -> .../<final_run_name>/<despot_run_name>/
#   7e. plot_despot_vs_ref (only with -c): each dataset's reference-set
#       DESPOT score (Stage 0d) vs the matched pipeline ligand's DESPOT
#       score, both normalized per heavy atom, matched the same way as
#       stages 3d/5b/7c
#                                        -> GRAPHS_DIR/<run_name>/.../<final_run_name>/<despot_run_name>/
#   7f. plot_rscc_despot_tradeoff (only with -c): per despot_filter-surviving, reference-matched
#       ligand, pipeline RSCC - reference RSCC (y) vs reference DESPOT - pipeline DESPOT (x),
#       reusing 7c/7e's own matching - see rscc_common.py
#                                        -> GRAPHS_DIR/<run_name>/.../<final_run_name>/<despot_run_name>/
#   8.  analysis_scripts/*.py           -> .../<final_run_name>/graphs/
#       (cluster-rep and per-residue RSCC plots, filter_2-vs-filter_1 ligand
#       RSCC, final-vs-apo Z-map statistics, per-dataset final-ligand Z-map
#       histograms, and per-dataset bfactor-sensitivity plots, from stage 6c's
#       csvs) + pooled (cross-dataset) counterparts (density-colored
#       scatter/hist), once <final_run_name> is given and stage 6's output
#       exists for every dataset - collapsed into one idempotent unit (see
#       "Idempotency" below), not sub-lettered like the other stages.
#                                        -> GRAPHS_DIR/<run_name>/.../<final_run_name>/
#
#
#
# Modularity: pass only as many of the seven run-name arguments as you want
# to run through (e.g. just <run_name> <placer_run_name> <filter_run_name>
# stops after stage 3). Each stage's directory tree is nested under the
# previous stage's, so re-running with a new name at any point (e.g. a new
# filter_run_name under an existing run_name/placer_run_name) naturally
# branches off the old results without touching them. Stage 0 is
# dataset-scoped rather than run-name-scoped, so it always runs (skipping
# per-dataset once that dataset's apo RSCC csv exists) regardless of which
# run-name arguments are given. Stage 7 (despot) is optional even once
# <final_run_name> is given - it only runs when <despot_run_name> is also
# given - while stage 8 (analysis) always runs once <final_run_name> is
# given, independent of stage 7.
#
# Idempotency: every step checks whether its own actual output file(s)
# already exist - for a given dataset (main pipeline steps 0b-0d, 7a) or for the
# whole run (graphing steps 1b, 1c, 2c, 3d, 4c, 5b, 6d, 7b, 7c, 7d, 7e, 8) - and skips
# just that piece of work if so, so previous runs are never overwritten.
# Main pipeline steps with a variable number of outputs per dataset (PLACER
# and RSR rounds) use a loose "at least one matching output exists" check -
# a partially-failed dataset is treated as done and needs --overwrite to
# resume. Pass --overwrite to force every step to re-run in place regardless
# of existing output (including the graphing steps), or --replot to force
# just the graphing steps (1b, 1c, 2c, 3d, 4c, 5b, 6d, 7b, 7c, 7d, 7e, 8) to redo.
# Stage 8 has no readiness precondition of its own - it runs whenever
# <final_run_name> is given, and its own idempotency check (like every other
# graphing step) simply skips any dataset that doesn't yet have the output
# it needs (e.g. one with no final_model_refined_rscc.csv - see
# stage8_outputs_exist), rather than blocking the whole stage on it.
#
# Dataset scoping: by default every stage runs over every dataset listed in
# DATASETS_FILE (datasets.txt). Pass --dataset <id[,id...]> to restrict the
# entire invocation (all stages) to just the given dataset(s) instead -
# DATASETS_FILE is repointed at a generated temp file listing only those
# datasets before any stage runs. Every graphing step (pooled or not) is
# skipped entirely when --dataset is given, since it would otherwise
# silently overwrite the full-run plot with a partial one.

set -uo pipefail

usage() {
    cat <<EOF
Usage: $0 <run_name> [placer_run_name [filter_run_name [placer2_run_name [filter2_run_name [final_run_name [despot_run_name]]]]]]
           [-n <num_placer_confs>] [-n2 <num_placer2_confs>] [-g <gpu_ids>] [-p <num_parallel>] [-c] [--overwrite] [--replot]
           [--dataset <id[,id...]>] [--bfactors <list>]
           [--z_threshold <float>] [--num_peaks <int>]
           [--f1_filter_proportion <float>] [--f1_min_cluster_proportion <float>]
           [--f1_rscc_cutoff <float>] [--f1_clustering_mode <all-atom|centroid>]
           [--f1_clustering_cutoff <float>]
           [--f2_filter_proportion <float>] [--f2_min_cluster_proportion <float>]
           [--f2_rscc_cutoff <float>] [--f2_clustering_mode <all-atom|centroid>]
           [--f2_clustering_cutoff <float>] [--despot_threshold <float>]
           [--despot_rscc_threshold <float>] [--despot_rscc_weight <float>]

Only <run_name> is required. Supplying fewer than all seven names runs only
that many stages of the pipeline (see header comment for the stage list).
<despot_run_name> is optional even when <final_run_name> is given - stage 7
(DESPOT energy scoring) only runs when it's also supplied; stage 8 (analysis)
always runs once <final_run_name> is given, independent of stage 7.

Options:
  -n <num_placer_confs>    Number of PLACER conformers for round 1 (placer -n). Default: 1000
  -n2 <num_placer2_confs>  Number of PLACER conformers for round 2 (placer2 -n). Default: 1000
  -g <gpu_ids>             Comma-separated GPU ids for both PLACER rounds. Default: 0
  -p <num_parallel>        CPU parallelism for every non-PLACER stage (calc_apo_rscc, calc_apo_z,
                            fit_ligand, rsr_placer, filter, rsr_backbone, calc_backbone_refined_rscc,
                            rsr_placer2, filter2, build_final, rsr_final, calc_final_refined_rscc,
                            calc_final_refined_z, despot, ref_set_despot). Default: 1
  -c                       Also compare results to the reference set (REF_SET). Runs
                            calc_ref_set_rscc as stage 0c: per dataset, computes RSCC of
                            REF_SET/<dataset>/<REF_SET_PDB_PATTERN>, skipping any dataset whose
                            output csv already exists. When <despot_run_name> is also given, also
                            runs ref_set_despot as stage 0d: per dataset, DESPOT-scores
                            REF_SET/<dataset>/<REF_SET_PDB_PATTERN> (symmetry_expand --strip,
                            mol2 conversion, score_complex.py) into
                            REF_SET/<dataset>/<dataset>_DESPOT.csv. Also runs pooled
                            (cross-dataset) ligand/residue comparison plots into GRAPHS_DIR: stage
                            1c (centroid_rmsd_all), 2c and 4c (calc_placer_sampling, refined
                            + unrefined), 3d (plot_lig_vs_ref_filter1, plot_residues_vs_ref_backbone),
                            5b (plot_lig_vs_ref_filter2), 6d (plot_residues_vs_ref_final), and (only
                            when <despot_run_name> is also given) 7c (plot_lig_vs_ref_despot) and 7e
                            (plot_despot_vs_ref).
  --overwrite              Force every requested step to re-run in place, even if its output
                            already exists (normally such a step is skipped - see "Idempotency"
                            in the header comment). Applies to every stage, including the
                            graphing steps. Does not affect stage 8's precondition that stage 6
                            already be complete for every dataset.
  --replot                 Force just the graphing steps (1b, 1c, 2c, 3d, 4c, 5b, 6d, 7b, 7c, 7d, 7e, 8) to
                            re-run in place, even if their output already exists. Does not
                            affect the main pipeline steps (use --overwrite for those too).
  --dataset <id[,id...]>   Run only on this dataset, or comma-separated list of datasets
                            (e.g. x00001-1 or x00001-1,x00002-1), instead of every dataset
                            listed in DATASETS_FILE (datasets.txt). Every dataset given must
                            already have a directory under DATASETS_DIR. Applies to every
                            stage (0-8) for the whole invocation.
  --bfactors <list>        B-factor(s) passed to calc_rscc_b (stage 6c): a single value
                            (e.g. "20") or a comma-separated list (e.g. "20,40,60,80,100").
                            calc_rscc_b is only run on final_model_refined.pdb, restricted to
                            the residues listed in residues_with_placer_conformers.csv, and
                            scores every (event map, bfactor) combination separately - a
                            residue with 4 event maps and 5 bfactors gets 20 rows in
                            final_model_refined_rscc_b.csv. Passing fewer than 2 bfactors
                            makes that csv's spearmans_rho column always empty (a rank
                            correlation needs >= 2 points). Default: "20,40,60,80,100".
  --z_threshold <float>            fit_ligand -z/--z_threshold: Z-score threshold for peak
                                    detection (stage 1a). Default (unset): fit_ligand's own
                                    default (4).
  --num_peaks <int>                fit_ligand -n/--num_peaks: number of peaks to find (stage 1a).
                                    Default (unset): fit_ligand's own default (100).
  --f1_filter_proportion <float>       filter --filter_proportion for stage 3a (filter_run_name).
  --f1_min_cluster_proportion <float>  filter --min_cluster_proportion for stage 3a.
  --f1_rscc_cutoff <float>             filter --rscc_cutoff for stage 3a.
  --f1_clustering_mode <all-atom|centroid>  filter --clustering_mode for stage 3a.
  --f1_clustering_cutoff <float>       filter --clustering_cutoff for stage 3a.
  --f2_filter_proportion <float>       filter --filter_proportion for stage 5a (filter2_run_name).
  --f2_min_cluster_proportion <float>  filter --min_cluster_proportion for stage 5a.
  --f2_rscc_cutoff <float>             filter --rscc_cutoff for stage 5a.
  --f2_clustering_mode <all-atom|centroid>  filter --clustering_mode for stage 5a.
  --f2_clustering_cutoff <float>       filter --clustering_cutoff for stage 5a.
                                    All f1_*/f2_* options are left unset by default, so
                                    filter's own argparse defaults apply. Stage 5a (filter2_run_name)
                                    now runs the same "filter" script as stage 3a (filter_run_name)
                                    instead of "filter_all" - see header comment.
  --despot_threshold <float>       despot_filter --despot-threshold for stage 7a: the reselected
                                    winning pose's per-heavy-atom-normalized DESPOT score must be
                                    <= this to survive. Left unset by default, so despot_filter's
                                    own argparse default (-1.0) applies.
  --despot_rscc_threshold <float>  despot_filter --rscc-threshold for stage 7a: the reselected
                                    winning pose's RSCC must be >= this to survive (both this and
                                    --despot_threshold must pass). Left unset by default, so
                                    despot_filter's own argparse default (0.6) applies.
  --despot_rscc_weight <float>     despot_filter --rscc-weight for stage 7a: per filter2 cluster,
                                    the Pareto-front (MSE vs normalized DESPOT) candidate
                                    maximizing RSCC - despot_rscc_weight*normalized_DESPOT is
                                    selected as that cluster's pose. Left unset by default, so
                                    despot_filter's own argparse default (0.05) applies.

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
  $0 run_1 placer_1 filter_1 placer2_1 filter2_1 final_1 despot_1
EOF
    exit 1
}

# --- User-specified configuration: edit these for your environment ---
BASE_DIR="/home/ngupta/main/program_exp"
CSV_FILE="${BASE_DIR}/pxr_fragments.csv"
LIG_PDB_DIR="${BASE_DIR}/pdb_final_geometry"
CONDA_SH="/home/ngupta/miniconda3/etc/profile.d/conda.sh"
CONDA_ENV_QFIT="nikhils_program_exp"
CONDA_ENV_PLACER="placer_env"
CONDA_ENV_RSR="nikhils_program_exp"
CONDA_ENV_EVAL="nikhils_program_exp"
CONDA_ENV_OBABEL="openbabel"
CONDA_ENV_DESPOT="DESPOT"
RUN_PLACER_PY="/home/ngupta/PLACER/PLACER/run_PLACER.py"
DESPOT_SCRIPT="/home/ngupta/DESPOT/scripts/score_complex.py"
DESPOT_DATABASE="CROWN"
EXPAND_DISTANCE_CUTOFF=10
DATASETS_DIR="${BASE_DIR}/datasets"
DATASETS_FILE="${BASE_DIR}/datasets.txt"
RSR_SCRIPTS_DIR="${BASE_DIR}/qfit-3.0/rsr_scripts"
ANALYSIS_SCRIPTS_DIR="${BASE_DIR}/qfit-3.0/analysis_scripts"
LIG_SCRIPTS_DIR="${BASE_DIR}/qfit-3.0/lig_scripts"
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
AGGREGATE_CLASH_GROUPS_PY="${ANALYSIS_SCRIPTS_DIR}/aggregate_clash_groups.py"
CENTROID_RMSD_ALL_PY="${ANALYSIS_SCRIPTS_DIR}/centroid_rmsd_all.py"
CALC_PLACER_SAMPLING_PY="${ANALYSIS_SCRIPTS_DIR}/calc_placer_sampling.py"
CALC_PLACER_SAMPLING_UNREFINED_PY="${ANALYSIS_SCRIPTS_DIR}/calc_placer_sampling_unrefined.py"
PLOT_FIT_LIGAND_COUNTS_PY="${ANALYSIS_SCRIPTS_DIR}/plot_fit_ligand_counts.py"
PLOT_FINAL_VS_APO_Z_PY="${ANALYSIS_SCRIPTS_DIR}/plot_final_vs_apo_z.py"
PLOT_FINAL_LIG_Z_PY="${ANALYSIS_SCRIPTS_DIR}/plot_final_lig_z.py"
PLOT_BFACTOR_SENSITIVITY_PY="${ANALYSIS_SCRIPTS_DIR}/plot_bfactor_sensitivity.py"
PLOT_CLUSTER_REPS_POOLED_PY="${ANALYSIS_SCRIPTS_DIR}/plot_cluster_reps_rscc_pooled.py"
PLOT_PROTEIN_RSCC_POOLED_PY="${ANALYSIS_SCRIPTS_DIR}/plot_protein_rscc_pooled.py"
PLOT_Z_POOLED_PY="${ANALYSIS_SCRIPTS_DIR}/plot_z_pooled.py"
PLOT_BFACTOR_RHO_POOLED_PY="${ANALYSIS_SCRIPTS_DIR}/plot_bfactor_rho_pooled.py"
PLOT_DESPOT_ENERGIES_PY="${ANALYSIS_SCRIPTS_DIR}/plot_despot_energies.py"
PLOT_DESPOT_ENERGIES_POOLED_PY="${ANALYSIS_SCRIPTS_DIR}/plot_despot_energies_pooled.py"
PLOT_LIG_VS_REF_DESPOT_PY="${ANALYSIS_SCRIPTS_DIR}/plot_lig_vs_ref_despot.py"
PLOT_DESPOT_LIGAND_SUMMARY_PY="${ANALYSIS_SCRIPTS_DIR}/plot_despot_ligand_summary.py"
PLOT_DESPOT_LIGAND_SUMMARY_SINGLE_PY="${ANALYSIS_SCRIPTS_DIR}/plot_despot_ligand_summary_single.py"
PLOT_DESPOT_VS_REF_PY="${ANALYSIS_SCRIPTS_DIR}/plot_despot_vs_ref.py"
PLOT_RSCC_DESPOT_TRADEOFF_PY="${ANALYSIS_SCRIPTS_DIR}/plot_rscc_despot_tradeoff.py"
ASSIGN_BOND_ORDERS_PY="${LIG_SCRIPTS_DIR}/assign_bond_orders.py"
PDB_TO_MOL2_SH="${LIG_SCRIPTS_DIR}/pdb_to_mol2.sh"
PROTEIN_TO_MOL2_SH="${LIG_SCRIPTS_DIR}/protein_to_mol2.sh"

for f in "$DATASETS_FILE" "$CSV_FILE" "$RSR_SCRIPT_LIGAND" "$RSR_SCRIPT_PROTEIN" "$RSR_SCRIPT_FINAL" \
         "$RUN_PLACER_PY" "$PLOT_CLUSTER_REPS_PY" "$AGGREGATE_PROTEIN_RSCC_PY" "$AGGREGATE_LIG_RSCC_PY" \
         "$PLOT_LIG_VS_REF_FILTER1_PY" "$PLOT_LIG_VS_REF_FILTER2_PY" \
         "$PLOT_RESIDUES_VS_REF_BACKBONE_PY" "$PLOT_RESIDUES_VS_REF_FINAL_PY" \
         "$CENTROID_RMSD_ALL_PY" "$CALC_PLACER_SAMPLING_PY" "$CALC_PLACER_SAMPLING_UNREFINED_PY" \
         "$PLOT_FIT_LIGAND_COUNTS_PY" "$PLOT_FINAL_VS_APO_Z_PY" "$PLOT_FINAL_LIG_Z_PY" \
         "$PLOT_BFACTOR_SENSITIVITY_PY" "$ASSIGN_BOND_ORDERS_PY" "$PLOT_CLUSTER_REPS_POOLED_PY" \
         "$PLOT_PROTEIN_RSCC_POOLED_PY" "$PLOT_Z_POOLED_PY" \
         "$PLOT_BFACTOR_RHO_POOLED_PY" "$PLOT_DESPOT_ENERGIES_PY" "$PLOT_DESPOT_ENERGIES_POOLED_PY" \
         "$PLOT_LIG_VS_REF_DESPOT_PY" "$PLOT_DESPOT_LIGAND_SUMMARY_PY" "$PLOT_DESPOT_LIGAND_SUMMARY_SINGLE_PY" \
         "$PLOT_DESPOT_VS_REF_PY" "$PLOT_RSCC_DESPOT_TRADEOFF_PY" \
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
despot_run_name=""

num_placer_confs=100
num_placer2_confs=100
gpu_ids="0"
num_parallel=""
compare_ref_set=0
overwrite=0
replot=0

# B-factor(s) passed to calc_rscc_b (stage 6c, final_model_refined.pdb
# only) - always passed explicitly, since calc_rscc_b's own single-bfactor
# default would make its spearmans_rho column always empty (needs >=2
# distinct bfactors per residue/event-map group).
bfactors="20,40,60,80,100"

# --dataset <id[,id...]>: run only on this subset of datasets instead of
# reading DATASETS_FILE. dataset_arg holds the raw CLI value; if set, it's
# expanded into DATASET_OVERRIDE_FILE (a generated temp file, one dataset
# per line) which DATASETS_FILE is then repointed to - see below.
dataset_arg=""
DATASET_OVERRIDE_FILE=""

# fit_ligand tunables (stage 1a). Left empty by default so fit_ligand's own
# argparse defaults (-z/--z_threshold=4, -n/--num_peaks=100) apply; only
# passed through when explicitly set here.
z_threshold=""
num_peaks=""

# filter tunables (stage 3a, filter_run_name), left empty by default so
# filter's own argparse defaults apply.
f1_filter_proportion=""
f1_min_cluster_proportion=""
f1_rscc_cutoff=""
f1_clustering_mode=""
f1_clustering_cutoff=""

# filter tunables (stage 5a, filter2_run_name) - same underlying `filter`
# script as f1_*, set independently.
f2_filter_proportion=""
f2_min_cluster_proportion=""
f2_rscc_cutoff=""
f2_clustering_mode=""
f2_clustering_cutoff=""

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
        --bfactors)
            bfactors="$2"
            shift 2
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
                     "$filter2_run_name" "$final_run_name" "$despot_run_name"; do
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
# that enumerates datasets directly in this shell (stage8_outputs_exist,
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
export run_name placer_run_name filter_run_name placer2_run_name filter2_run_name final_run_name despot_run_name
export num_placer_confs num_placer2_confs compare_ref_set overwrite replot bfactors
export z_threshold num_peaks
export f1_filter_proportion f1_min_cluster_proportion f1_rscc_cutoff \
       f1_clustering_mode f1_clustering_cutoff
export f2_filter_proportion f2_min_cluster_proportion f2_rscc_cutoff \
       f2_clustering_mode f2_clustering_cutoff
export despot_threshold despot_rscc_threshold despot_rscc_weight
export BASE_DIR DATASETS_DIR DATASETS_FILE CSV_FILE LIG_PDB_DIR ASSIGN_BOND_ORDERS_PY
export RSR_SCRIPT_LIGAND RSR_SCRIPT_PROTEIN RSR_SCRIPT_FINAL
export ANALYSIS_SCRIPTS_DIR PLOT_CLUSTER_REPS_PY AGGREGATE_PROTEIN_RSCC_PY AGGREGATE_LIG_RSCC_PY
export PLOT_LIG_VS_REF_FILTER1_PY PLOT_LIG_VS_REF_FILTER2_PY
export PLOT_RESIDUES_VS_REF_BACKBONE_PY PLOT_RESIDUES_VS_REF_FINAL_PY GRAPHS_DIR
export AGGREGATE_CLASH_GROUPS_PY
export CENTROID_RMSD_ALL_PY CALC_PLACER_SAMPLING_PY CALC_PLACER_SAMPLING_UNREFINED_PY
export PLOT_FIT_LIGAND_COUNTS_PY
export PLOT_FINAL_VS_APO_Z_PY
export PLOT_FINAL_LIG_Z_PY
export PLOT_BFACTOR_SENSITIVITY_PY
export PLOT_CLUSTER_REPS_POOLED_PY PLOT_PROTEIN_RSCC_POOLED_PY
export PLOT_Z_POOLED_PY PLOT_BFACTOR_RHO_POOLED_PY
export PLOT_DESPOT_ENERGIES_PY PLOT_DESPOT_ENERGIES_POOLED_PY PLOT_LIG_VS_REF_DESPOT_PY PLOT_DESPOT_LIGAND_SUMMARY_PY
export PLOT_DESPOT_LIGAND_SUMMARY_SINGLE_PY
export PLOT_DESPOT_VS_REF_PY
export PLOT_RSCC_DESPOT_TRADEOFF_PY
export PDB_TO_MOL2_SH PROTEIN_TO_MOL2_SH DESPOT_SCRIPT DESPOT_DATABASE EXPAND_DISTANCE_CUTOFF
export REF_SET REF_SET_PDB_PATTERN
export CONDA_SH CONDA_ENV_QFIT CONDA_ENV_RSR CONDA_ENV_PLACER CONDA_ENV_EVAL CONDA_ENV_OBABEL CONDA_ENV_DESPOT
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
# For graphing steps (1b, 1c, 2c, 3d, 4c, 5b, 6d, 7b, 7c, 7d, 7e, 8): skips run_fn (via
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
# sdf) then obabel (CONDA_ENV_OBABEL: sdf -> mol2) - same tool
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

        conda_activate "$CONDA_ENV_OBABEL"
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
    printf '%s\n' "${DATASETS[@]}" | parallel -j "$NUM_PARALLEL_DEFAULT" calc_apo_rscc_process_dataset {}
    echo "All jobs completed"
    print_elapsed "$start_time"
}

######################################################################
# Stage 0b: calc_apo_z
######################################################################
# Computes the per-residue Z-map statistics (max/min/average Z-score) of
# each dataset's baseline {dataset}-aligned-structure.pdb against its own
# Z-map ({dataset}-z_map.native.ccp4 - the same file fit_ligand.py's
# LigandPlacer uses to find PLACER peaks), so later analysis can compare the
# final-refined structure's Z-map statistics against this apo baseline.
# Dataset-scoped like calc_apo_rscc, and skipped per-dataset whenever its
# output csv already exists.

calc_apo_z_process_dataset() {
    conda_activate "$CONDA_ENV_QFIT"

    local dataset=$1
    local dataset_dir="${DATASETS_DIR}/${dataset}"

    local structure="${dataset_dir}/${dataset}-aligned-structure.pdb"
    local zmap="${dataset_dir}/${dataset}-z_map.native.ccp4"
    local output_csv="${dataset_dir}/${dataset}-aligned-structure_z.csv"

    if [ -f "$output_csv" ]; then
        echo "Skipping [${dataset}]: ${output_csv} already exists."
        return 0
    fi

    if [ ! -f "$structure" ]; then
        echo "Warning [${dataset}]: aligned structure not found: ${structure}, skipping."
        return 1
    fi
    if [ ! -f "$zmap" ]; then
        echo "Warning [${dataset}]: Z-map not found: ${zmap}, skipping."
        return 1
    fi

    local lookup=$(grep "^${dataset} " "$LOOKUP_FILE")
    if [ -z "$lookup" ]; then
        echo "Warning: No match found for dataset ${dataset}, skipping."
        return 1
    fi
    local resolution=$(echo "$lookup" | awk '{print $3}')

    echo "Processing ${dataset}: resolution=${resolution}"

    calc_z "${structure}" "${zmap}" "${resolution}" "${output_csv}"

    local calc_exit=$?
    if [ $calc_exit -ne 0 ]; then
        echo "ERROR [${dataset}]: calc_z failed on ${structure} with exit code ${calc_exit}"
        return 1
    fi

    echo "Completed [${dataset}]: ${structure} -> ${output_csv}"
}
export -f calc_apo_z_process_dataset

do_calc_apo_z() {
    conda_activate "$CONDA_ENV_QFIT"

    echo "Starting run"
    local start_time=$(date +%s)
    printf '%s\n' "${DATASETS[@]}" | parallel -j "$NUM_PARALLEL_DEFAULT" calc_apo_z_process_dataset {}
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
    printf '%s\n' "${DATASETS[@]}" | parallel -j "$NUM_PARALLEL_DEFAULT" calc_ref_set_rscc_process_dataset {}
    echo "All jobs completed"
    print_elapsed "$start_time"
}

######################################################################
# Stage 0d: ref_set_despot (only runs when -c and despot_run_name are given)
######################################################################
# Scores each dataset's reference-set structure (REF_SET/<dataset>/<REF_SET_PDB_PATTERN>)
# with DESPOT, the same way as Stage 7a's own final_model_refined.pdb: symmetry_expand
# into a realistic crystal environment (EXPAND_DISTANCE_CUTOFF), convert the expanded
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
        "$EXPAND_DISTANCE_CUTOFF" "$ligs_dir"
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
        "$CONDA_ENV_OBABEL" "$ASSIGN_BOND_ORDERS_PY" "${ligand_pdbs[@]}"
    status=$?
    if [ $status -ne 0 ]; then
        echo "ERROR [${dataset}]: pdb_to_mol2.sh failed on ${ligand_pdbs[*]} with exit code ${status}"
        print_elapsed "$dataset_start_time" "[${dataset}] ref_set_despot"
        return 1
    fi

    step_start_time=$(date +%s)
    "$PROTEIN_TO_MOL2_SH" "$expanded_pdb" "$CONDA_SH" "$CONDA_ENV_OBABEL"
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

        echo "  Running fit_ligand: PDB=${dir_name}, sampling=${run_name}"

        local fit_ligand_extra_args=()
        [ -n "$z_threshold" ] && fit_ligand_extra_args+=(-z "$z_threshold")
        [ -n "$num_peaks" ] && fit_ligand_extra_args+=(-n "$num_peaks")

        fit_ligand "${DATASETS_DIR}/${dataset}" \
            "${pdb_file}" \
            -r ${resolution} \
            --sampling ${run_name} \
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
    printf '%s\n' "${DATASETS[@]}" | parallel -j "$NUM_PARALLEL_DEFAULT" calc_final_refined_rscc_process_dataset {}
    echo "All jobs completed"
    print_elapsed "$start_time"
}

######################################################################
# Stage 6c: calc_final_refined_z
######################################################################
# Computes final_model_refined.pdb's per-residue Z-map statistics
# (max/min/average Z-score) against the dataset's Z-map, so
# plot_final_vs_apo_z (stage 8) can compare them to the apo baseline
# written by calc_apo_z (stage 0c).

calc_final_refined_z_process_dataset() {
    local dataset=$1
    local dataset_dir="${DATASETS_DIR}/${dataset}"
    local final_dir="${dataset_dir}/${run_name}/${placer_run_name}/${filter_run_name}/${placer2_run_name}/${filter2_run_name}/${final_run_name}"
    local structure="${final_dir}/final_model_refined.pdb"
    local output_csv="${structure%.pdb}_z.csv"

    if [ "$overwrite" -ne 1 ] && files_exist "$output_csv"; then
        echo "Skipping [${dataset}]: calc_final_refined_z already complete (${output_csv} exists)."
        return 0
    fi

    conda_activate "$CONDA_ENV_QFIT"

    local lookup=$(grep "^${dataset} " "$LOOKUP_FILE")
    if [ -z "$lookup" ]; then
        echo "Warning: No match found for dataset ${dataset}, skipping."
        return 1
    fi

    local resolution=$(echo "$lookup" | awk '{print $3}')

    echo "Processing ${dataset}: resolution=${resolution}"

    local zmap="${dataset_dir}/${dataset}-z_map.native.ccp4"
    if [ ! -f "$zmap" ]; then
        echo "Warning [${dataset}]: Z-map not found: ${zmap}, skipping."
        return 1
    fi

    if [ ! -f "$structure" ]; then
        echo "Warning [${dataset}]: final_model_refined.pdb not found: ${structure}, skipping."
        return 1
    fi

    calc_z "${structure}" "${zmap}" "${resolution}" "${output_csv}"

    local calc_exit=$?
    if [ $calc_exit -ne 0 ]; then
        echo "ERROR [${dataset}]: calc_z failed on ${structure} with exit code ${calc_exit}"
        return 1
    fi

    echo "Completed [${dataset}]: ${structure} -> ${output_csv}"
}
export -f calc_final_refined_z_process_dataset

do_calc_final_z() {
    conda_activate "$CONDA_ENV_QFIT"

    echo "Starting run"
    local start_time=$(date +%s)
    printf '%s\n' "${DATASETS[@]}" | parallel -j "$NUM_PARALLEL_DEFAULT" calc_final_refined_z_process_dataset {}
    echo "All jobs completed"
    print_elapsed "$start_time"
}

######################################################################
# Stage 6c: calc_final_refined_rscc_b
######################################################################
# Computes final_model_refined.pdb's per-residue, per-event-map, per-bfactor
# RSCC (plus a spearmans_rho column, computed by calc_rscc_b itself),
# restricted to the residues listed in residues_with_placer_conformers.csv
# (this sweep is expensive - every residue x every event map x every
# bfactor - so it's never run over every residue in the structure).
# final_model_refined_rscc_b.csv feeds the bfactor-sensitivity line/histogram
# plots (stage 8).

calc_final_refined_rscc_b_process_dataset() {
    local dataset=$1
    local dataset_dir="${DATASETS_DIR}/${dataset}"
    local final_dir="${dataset_dir}/${run_name}/${placer_run_name}/${filter_run_name}/${placer2_run_name}/${filter2_run_name}/${final_run_name}"
    local structure="${final_dir}/final_model_refined.pdb"
    local residues_file="${final_dir}/residues_with_placer_conformers.csv"
    local output_csv="${structure%.pdb}_rscc_b.csv"

    if [ "$overwrite" -ne 1 ] && files_exist "$output_csv"; then
        echo "Skipping [${dataset}]: calc_final_refined_rscc_b already complete (${output_csv} exists)."
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
    if [ ! -s "$residues_file" ]; then
        echo "Warning [${dataset}]: residues_with_placer_conformers.csv not found or empty: ${residues_file}, skipping."
        return 1
    fi

    calc_rscc_b "${structure}" "${event_maps[@]}" "${resolution}" "${output_csv}" "${residues_file}" \
        --bfactors "$bfactors"

    local calc_exit=$?
    if [ $calc_exit -ne 0 ]; then
        echo "ERROR [${dataset}]: calc_rscc_b failed on ${structure} with exit code ${calc_exit}"
        return 1
    fi

    echo "Completed [${dataset}]: ${structure} -> ${output_csv}"
}
export -f calc_final_refined_rscc_b_process_dataset

do_calc_final_rscc_b() {
    conda_activate "$CONDA_ENV_QFIT"

    echo "Starting run"
    local start_time=$(date +%s)
    printf '%s\n' "${DATASETS[@]}" | parallel -j "$NUM_PARALLEL_DEFAULT" calc_final_refined_rscc_b_process_dataset {}
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
# Stage 7a: despot
######################################################################
# For each dataset (only runs when despot_run_name is given), scores every placer2 round-2
# conformer (not just the one pose filter2/build_final_model happened to select) against DESPOT:
# extract_ligand_conformers pools every conformer into one ligs.pdb, symmetry_expand's
# --ligand-conformers-pdb mode expands final_model_refined.pdb's protein into a crystal
# environment realistic for all of them (EXPAND_DISTANCE_CUTOFF, default 10 A), the expanded
# protein and ligand conformers are converted to mol2 (lig_scripts/pdb_to_mol2.sh,
# lig_scripts/protein_to_mol2.sh), and DESPOT's score_complex.py scores every conformer at once.
# despot_filter.py then reselects, per filter2 cluster, the pose that best trades off RSCC
# against DESPOT (Pareto front over MSE/DESPOT, real RSCC computed internally via qfit's
# transformer, winner = argmax(RSCC - rscc_weight*normalized_DESPOT), kept only if it clears both
# --despot_rscc_threshold and --despot_threshold) - see despot_filter.py's own docstring. Every
# output is written into .../<final_run_name>/<despot_run_name>/, following the same nested
# run-name convention as every other stage.
#
# Note: since <dataset>_DESPOT.csv now scores every placer2 conformer rather than just the 4-ish
# final poses, Stage 7b's per-conformer score histograms (plot_despot_energies.py) reflect that
# larger, differently-shaped population - an expected consequence of scoring before selection.

despot_process_dataset() {
    local dataset=$1
    local dataset_dir="${DATASETS_DIR}/${dataset}"
    local placer2_dir="${dataset_dir}/${run_name}/${placer_run_name}/${filter_run_name}/${placer2_run_name}"
    local filter2_dir="${placer2_dir}/${filter2_run_name}"
    local final_dir="${filter2_dir}/${final_run_name}"
    local final_model="${final_dir}/final_model_refined.pdb"
    local despot_dir="${final_dir}/${despot_run_name}"
    local despot_csv_check="${despot_dir}/${dataset}_DESPOT.csv"
    local conformer_map_check="${despot_dir}/conformer_map.csv"
    local despot_filtered_check="${despot_dir}/despot_filtered.pdb"

    if [ "$overwrite" -ne 1 ] && files_exist "$despot_csv_check" "$conformer_map_check" "$despot_filtered_check"; then
        echo "Skipping [${dataset}]: despot already complete (${despot_filtered_check} exists)."
        return 0
    fi

    if [ ! -f "$final_model" ]; then
        echo "Warning [${dataset}]: final_model_refined.pdb not found: ${final_model}, skipping."
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
    # original_ligand_dir just captures final_model's own single ligand instance for the
    # record (never read downstream) - the actual scored ligand(s) are $ligs_pdb, from
    # extract_ligand_conformers above (every placer2 conformer, chain L - never carries
    # altloc), targeted via --ligand-conformers-pdb below.
    symmetry_expand "$final_model" "$expanded_pdb" "$space_group" "$a" "$b" "$c" "$alpha" "$beta" "$gamma" \
        "$EXPAND_DISTANCE_CUTOFF" "$original_ligand_dir" --ligand-conformers-pdb "$ligs_pdb"
    status=$?
    conda_deactivate
    print_elapsed "$step_start_time" "[${dataset}] symmetry_expand"
    if [ $status -ne 0 ]; then
        echo "ERROR [${dataset}]: symmetry_expand failed with exit code ${status}"
        print_elapsed "$dataset_start_time" "[${dataset}] despot"
        return 1
    fi

    "$PDB_TO_MOL2_SH" "${despot_dir}/ligs" "$smiles" "$CONDA_SH" "$CONDA_ENV_QFIT" "$CONDA_ENV_OBABEL" \
        "$ASSIGN_BOND_ORDERS_PY" "$ligs_pdb"
    status=$?
    if [ $status -ne 0 ]; then
        echo "ERROR [${dataset}]: pdb_to_mol2.sh failed on ${ligs_pdb} with exit code ${status}"
        print_elapsed "$dataset_start_time" "[${dataset}] despot"
        return 1
    fi

    step_start_time=$(date +%s)
    "$PROTEIN_TO_MOL2_SH" "$expanded_pdb" "$CONDA_SH" "$CONDA_ENV_OBABEL"
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

    step_start_time=$(date +%s)
    conda_activate "$CONDA_ENV_QFIT"
    despot_filter "$final_model" "$filter2_dir" "$despot_dir" "${event_maps[@]}" "$resolution" \
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
    # Prevent native libraries underneath pandas/numpy/scipy (BLAS, OpenMP)
    # in DESPOT's score_complex.py from each spawning one thread per core on
    # the machine. NUMBA_NUM_THREADS covers score_complex.py's own numba
    # threading layer separately - it isn't governed by the OMP/BLAS vars
    # above (confirmed: with only those set, a single process still peaked
    # at ~68 threads). Scoped to just this stage (exported here, unset
    # below) so it doesn't affect other stages' parallelism.
    export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMBA_NUM_THREADS=1

    echo "Starting run"
    local start_time=$(date +%s)
    printf '%s\n' "${DATASETS[@]}" | parallel -j "$NUM_PARALLEL_DEFAULT" --line-buffer despot_process_dataset {}
    echo "All jobs completed"
    print_elapsed "$start_time"

    unset OMP_NUM_THREADS OPENBLAS_NUM_THREADS MKL_NUM_THREADS NUMEXPR_NUM_THREADS VECLIB_MAXIMUM_THREADS NUMBA_NUM_THREADS
}

######################################################################
# Stage 7b: plot_despot_energies + plot_despot_energies_pooled
######################################################################
# Per-dataset histogram of heavy-atom-normalized DESPOT ligand binding-energy
# scores, read from .../<final_run_name>/<despot_run_name>/despot_filtered_scores.csv
# (despot_filter.py's own per-instance normalized_score column - not the raw,
# un-normalized <dataset>_DESPOT.csv score, which isn't comparable across
# differently-sized ligands), written into that dataset's existing
# .../<final_run_name>/graphs/ folder (plot data csv saved alongside it there
# too) - the same per-dataset location every other stage-8 plot uses (not
# nested under despot_run_name) - plus the pooled (cross-dataset) counterpart:
# every dataset's normalized DESPOT scores combined into one histogram,
# GRAPHS_DIR/<run>/.../<final_run_name>/<despot_run_name>/ligand_energies.png
# (nested under despot_run_name, unlike the other pooled plots, since the
# scores are specific to one despot_run_name). The pooled half is still
# guarded by run_step_pooled_replot's --dataset check, since it writes into
# one shared location regardless of which datasets were actually run.

do_plot_despot_energies() {
    conda_activate "$CONDA_ENV_EVAL"

    echo "Starting run"
    local start_time=$(date +%s)
    python "$PLOT_DESPOT_ENERGIES_PY" \
        "$run_name" "$placer_run_name" "$filter_run_name" \
        "$placer2_run_name" "$filter2_run_name" "$final_run_name" "$despot_run_name" \
        --datasets-dir "$DATASETS_DIR" --datasets-file "$DATASETS_FILE"
    echo "All jobs completed"
    print_elapsed "$start_time"
}

do_plot_despot_energies_pooled() {
    conda_activate "$CONDA_ENV_EVAL"

    local out_dir="${GRAPHS_DIR}/${run_name}/${placer_run_name}/${filter_run_name}/${placer2_run_name}/${filter2_run_name}/${final_run_name}/${despot_run_name}"
    echo "Starting run"
    local start_time=$(date +%s)
    python "$PLOT_DESPOT_ENERGIES_POOLED_PY" \
        "$run_name" "$placer_run_name" "$filter_run_name" \
        "$placer2_run_name" "$filter2_run_name" "$final_run_name" "$despot_run_name" \
        --datasets-dir "$DATASETS_DIR" --datasets-file "$DATASETS_FILE" --graphs-dir "$out_dir"
    echo "All jobs completed"
    print_elapsed "$start_time"
}

do_despot_plots() {
    do_plot_despot_energies
    do_plot_despot_energies_pooled
}

# despot_plots_outputs_exist: pooled ligand_energies.png exists, AND every
# dataset that actually has a DESPOT csv also has its per-dataset
# despot_energies.png (datasets with no DESPOT csv - e.g. no SMILES - can
# never get a plot, so they're not required).
despot_plots_outputs_exist() {
    local out_dir="${GRAPHS_DIR}/${run_name}/${placer_run_name}/${filter_run_name}/${placer2_run_name}/${filter2_run_name}/${final_run_name}/${despot_run_name}"
    files_exist "${out_dir}/ligand_energies.png" || return 1

    local dataset
    for dataset in "${DATASETS[@]}"; do
        local final_dir="${DATASETS_DIR}/${dataset}/${run_name}/${placer_run_name}/${filter_run_name}/${placer2_run_name}/${filter2_run_name}/${final_run_name}"
        local despot_csv="${final_dir}/${despot_run_name}/${dataset}_DESPOT.csv"
        if [ -f "$despot_csv" ] && [ ! -f "${final_dir}/graphs/despot_energies.png" ]; then
            return 1
        fi
    done
    return 0
}

######################################################################
# Stage 7c: plot_lig_vs_ref_despot (only with -c)
######################################################################
# Pooled plot restricted to the ligand poses that survived despot_filter
# (despot_run_name/despot_filtered.pdb): filter2_run_name/cluster_reps.csv's
# RSCC for just those surviving instances vs the reference set, matched by
# centroid distance the same way stages 3d/5b do - see plot_lig_vs_ref_despot.py.
# Nested under despot_run_name, like Stage 7b's pooled half, since which
# ligands survive is specific to one despot_run_name/--despot_threshold.

do_plot_lig_vs_ref_despot() {
    conda_activate "$CONDA_ENV_EVAL"

    local out_dir="${GRAPHS_DIR}/${run_name}/${placer_run_name}/${filter_run_name}/${placer2_run_name}/${filter2_run_name}/${final_run_name}/${despot_run_name}"
    echo "Starting run"
    local start_time=$(date +%s)
    python "$PLOT_LIG_VS_REF_DESPOT_PY" \
        "$run_name" "$placer_run_name" "$filter_run_name" \
        "$placer2_run_name" "$filter2_run_name" "$final_run_name" "$despot_run_name" \
        --datasets-dir "$DATASETS_DIR" --datasets-file "$DATASETS_FILE" \
        --ref-set "$REF_SET" --ref-pdb-pattern "$REF_SET_PDB_PATTERN" --graphs-dir "$out_dir"
    echo "All jobs completed"
    print_elapsed "$start_time"
}

despot_lig_vs_ref_outputs_exist() {
    files_exist "${GRAPHS_DIR}/${run_name}/${placer_run_name}/${filter_run_name}/${placer2_run_name}/${filter2_run_name}/${final_run_name}/${despot_run_name}/lig_vs_reference_rscc.png"
}

######################################################################
# Stage 7d: plot_despot_ligand_summary + plot_despot_ligand_summary_single
######################################################################
# Pooled scatter of every surviving (despot_filter-kept) ligand's heavy-atom-
# normalized DESPOT score (x) against its filter2_run_name/cluster_reps.csv
# RSCC (y) - see plot_despot_ligand_summary.py. Doesn't need -c: RSCC here
# comes from cluster_reps.csv, not the reference set. Nested under
# despot_run_name, like Stage 7b/7c, since the surviving ligands are specific
# to one despot_run_name/--despot_threshold. Plus the per-dataset counterpart:
# each dataset's own surviving ligands only, written directly into that
# dataset's own .../<final_run_name>/<despot_run_name>/ directory (not
# graphs_dir) with each point labeled (chain+resi, e.g. 'C1') since a single
# dataset typically has few enough surviving ligands for that to stay
# readable - see plot_despot_ligand_summary_single.py.

do_plot_despot_ligand_summary() {
    conda_activate "$CONDA_ENV_EVAL"

    local out_dir="${GRAPHS_DIR}/${run_name}/${placer_run_name}/${filter_run_name}/${placer2_run_name}/${filter2_run_name}/${final_run_name}/${despot_run_name}"
    echo "Starting run"
    local start_time=$(date +%s)
    python "$PLOT_DESPOT_LIGAND_SUMMARY_PY" \
        "$run_name" "$placer_run_name" "$filter_run_name" \
        "$placer2_run_name" "$filter2_run_name" "$final_run_name" "$despot_run_name" \
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
        "$placer2_run_name" "$filter2_run_name" "$final_run_name" "$despot_run_name" \
        --datasets-dir "$DATASETS_DIR" --datasets-file "$DATASETS_FILE"
    echo "All jobs completed"
    print_elapsed "$start_time"
}

do_despot_ligand_summary_plots() {
    do_plot_despot_ligand_summary_single
    do_plot_despot_ligand_summary
}

# despot_ligand_summary_outputs_exist: pooled ligand_summary.png exists, AND
# every dataset that actually has at least one despot_filter-kept ligand
# (despot_filtered_scores.csv has a 'kept'=True row) also has its own
# per-dataset ligand_summary.png (datasets with no kept ligand never get a
# plot - single-dataset or pooled - so they're not required).
despot_ligand_summary_outputs_exist() {
    local out_dir="${GRAPHS_DIR}/${run_name}/${placer_run_name}/${filter_run_name}/${placer2_run_name}/${filter2_run_name}/${final_run_name}/${despot_run_name}"
    files_exist "${out_dir}/ligand_summary.png" || return 1

    local dataset
    for dataset in "${DATASETS[@]}"; do
        local despot_dir="${DATASETS_DIR}/${dataset}/${run_name}/${placer_run_name}/${filter_run_name}/${placer2_run_name}/${filter2_run_name}/${final_run_name}/${despot_run_name}"
        local scores_csv="${despot_dir}/despot_filtered_scores.csv"
        if [ -f "$scores_csv" ] && grep -q ',True$' "$scores_csv" && [ ! -f "${despot_dir}/ligand_summary.png" ]; then
            return 1
        fi
    done
    return 0
}

######################################################################
# Stage 7e: plot_despot_vs_ref (only with -c)
######################################################################
# Pooled scatter of each dataset's reference-set DESPOT score (Stage 0d's
# REF_SET/<dataset>/<dataset>_DESPOT.csv) against the matched pipeline
# ligand's DESPOT score (despot_run_name/despot_filtered_scores.csv), both
# heavy-atom-normalized, matched by centroid distance the same way stages
# 3d/5b/7c do - see plot_despot_vs_ref.py. Nested under despot_run_name,
# like Stage 7b/7c/7d, since the pipeline-side scores are specific to one
# despot_run_name/--despot_threshold.

do_plot_despot_vs_ref() {
    conda_activate "$CONDA_ENV_EVAL"

    local out_dir="${GRAPHS_DIR}/${run_name}/${placer_run_name}/${filter_run_name}/${placer2_run_name}/${filter2_run_name}/${final_run_name}/${despot_run_name}"
    echo "Starting run"
    local start_time=$(date +%s)
    python "$PLOT_DESPOT_VS_REF_PY" \
        "$run_name" "$placer_run_name" "$filter_run_name" \
        "$placer2_run_name" "$filter2_run_name" "$final_run_name" "$despot_run_name" \
        --datasets-dir "$DATASETS_DIR" --datasets-file "$DATASETS_FILE" \
        --ref-set "$REF_SET" --ref-pdb-pattern "$REF_SET_PDB_PATTERN" --graphs-dir "$out_dir"
    echo "All jobs completed"
    print_elapsed "$start_time"
}

despot_vs_ref_outputs_exist() {
    files_exist "${GRAPHS_DIR}/${run_name}/${placer_run_name}/${filter_run_name}/${placer2_run_name}/${filter2_run_name}/${final_run_name}/${despot_run_name}/despot_vs_reference.png"
}

######################################################################
# Stage 7f: plot_rscc_despot_tradeoff (only with -c)
######################################################################
# Pooled scatter of despot_filter.py's RSCC/DESPOT reselection tradeoff relative to the
# reference structure, restricted to despot_filter survivors (same alive_rows as 7c/7e): y =
# pipeline RSCC - reference RSCC, x = reference DESPOT - pipeline DESPOT (both normalized) - see
# plot_rscc_despot_tradeoff.py. Nested under despot_run_name, like Stage 7b/7c/7d/7e, since the
# pipeline-side scores are specific to one despot_run_name/--despot_threshold/
# --despot_rscc_threshold/--despot_rscc_weight.

do_plot_rscc_despot_tradeoff() {
    conda_activate "$CONDA_ENV_EVAL"

    local out_dir="${GRAPHS_DIR}/${run_name}/${placer_run_name}/${filter_run_name}/${placer2_run_name}/${filter2_run_name}/${final_run_name}/${despot_run_name}"
    echo "Starting run"
    local start_time=$(date +%s)
    python "$PLOT_RSCC_DESPOT_TRADEOFF_PY" \
        "$run_name" "$placer_run_name" "$filter_run_name" \
        "$placer2_run_name" "$filter2_run_name" "$final_run_name" "$despot_run_name" \
        --datasets-dir "$DATASETS_DIR" --datasets-file "$DATASETS_FILE" \
        --ref-set "$REF_SET" --ref-pdb-pattern "$REF_SET_PDB_PATTERN" --graphs-dir "$out_dir"
    echo "All jobs completed"
    print_elapsed "$start_time"
}

rscc_despot_tradeoff_outputs_exist() {
    files_exist "${GRAPHS_DIR}/${run_name}/${placer_run_name}/${filter_run_name}/${placer2_run_name}/${filter2_run_name}/${final_run_name}/${despot_run_name}/rscc_despot_tradeoff_vs_reference.png"
}

######################################################################
# Stage 8: analysis plots (collapsed into one idempotent unit)
######################################################################
# Every plot below (per-dataset and pooled) is checked/skipped together as
# one stage - see stage8_outputs_exist() and stage8_plots() at the end of
# this section. Individual do_* functions are unchanged; only the
# orchestration is collapsed.

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
        --datasets-dir "$DATASETS_DIR" --datasets-file "$DATASETS_FILE"
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
        --datasets-dir "$DATASETS_DIR" --datasets-file "$DATASETS_FILE"
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
        --datasets-dir "$DATASETS_DIR" --datasets-file "$DATASETS_FILE"
    echo "All jobs completed"
    print_elapsed "$start_time"
}

# plot_final_vs_apo_z: scatter plots comparing every residue's Z-map
# statistics (max/min/average Z-score) between final_model_refined and the
# apo baseline, pooling the per-residue csvs already written by calc_apo_z
# (stage 0b) and calc_final_refined_z (stage 6c).
do_plot_final_vs_apo_z() {
    conda_activate "$CONDA_ENV_EVAL"

    echo "Starting run"
    local start_time=$(date +%s)
    python "$PLOT_FINAL_VS_APO_Z_PY" \
        "$run_name" "$placer_run_name" "$filter_run_name" \
        "$placer2_run_name" "$filter2_run_name" "$final_run_name" \
        --datasets-dir "$DATASETS_DIR" --datasets-file "$DATASETS_FILE"
    echo "All jobs completed"
    print_elapsed "$start_time"
}

# plot_final_lig_z: histograms of every LIG residue's Z-map statistics
# (max/min/average Z-score) in final_model_refined.pdb - one set of 3
# histograms per dataset, analogous to plot_cluster_reps_rscc.py's
# cluster_reps_1/2 histograms. Reads final_model_refined_z.csv
# (calc_final_refined_z, stage 6c), scoped to the LIG residues found by
# scanning final_model_refined.pdb itself.
do_plot_final_lig_z() {
    conda_activate "$CONDA_ENV_EVAL"

    echo "Starting run"
    local start_time=$(date +%s)
    python "$PLOT_FINAL_LIG_Z_PY" \
        "$run_name" "$placer_run_name" "$filter_run_name" \
        "$placer2_run_name" "$filter2_run_name" "$final_run_name" \
        --datasets-dir "$DATASETS_DIR" --datasets-file "$DATASETS_FILE"
    echo "All jobs completed"
    print_elapsed "$start_time"
}

# plot_bfactor_sensitivity: per-dataset RSCC-vs-bfactor line plots (raw and
# normalized) plus a spearmans_rho histogram, built from that dataset's own
# final_model_refined_rscc_b.csv (stage 6c), into .../<final_run_name>/graphs/
# with each plot's underlying data saved alongside it there too.
do_plot_bfactor_sensitivity() {
    conda_activate "$CONDA_ENV_EVAL"

    echo "Starting run"
    local start_time=$(date +%s)
    python "$PLOT_BFACTOR_SENSITIVITY_PY" \
        "$run_name" "$placer_run_name" "$filter_run_name" \
        "$placer2_run_name" "$filter2_run_name" "$final_run_name" \
        --datasets-dir "$DATASETS_DIR" --datasets-file "$DATASETS_FILE"
    echo "All jobs completed"
    print_elapsed "$start_time"
}

# Pooled (cross-dataset) counterparts of the plots above: same underlying
# data (no RSCC/Z/rho values computed here), but combined across every
# dataset in datasets.txt into a single plot per comparison instead of one
# per dataset, into GRAPHS_DIR/<run>/.../<final_run_name>/ (same pooled-plot
# location the -c reference-set comparisons use) rather than each dataset's
# own .../<final_run_name>/graphs/. The RSCC/Z scatter plots are colored by
# point density, since pooling makes overplotting far worse than any single
# dataset's plot has; the histograms are not - a histogram's bar heights
# already are the density.

do_plot_cluster_reps_rscc_pooled() {
    conda_activate "$CONDA_ENV_EVAL"

    local out_dir="${GRAPHS_DIR}/${run_name}/${placer_run_name}/${filter_run_name}/${placer2_run_name}/${filter2_run_name}/${final_run_name}"
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

    local out_dir="${GRAPHS_DIR}/${run_name}/${placer_run_name}/${filter_run_name}/${placer2_run_name}/${filter2_run_name}/${final_run_name}"
    echo "Starting run"
    local start_time=$(date +%s)
    python "$PLOT_PROTEIN_RSCC_POOLED_PY" \
        "$run_name" "$placer_run_name" "$filter_run_name" \
        "$placer2_run_name" "$filter2_run_name" "$final_run_name" \
        --datasets-dir "$DATASETS_DIR" --datasets-file "$DATASETS_FILE" --graphs-dir "$out_dir"
    echo "All jobs completed"
    print_elapsed "$start_time"
}

do_plot_z_pooled() {
    conda_activate "$CONDA_ENV_EVAL"

    local out_dir="${GRAPHS_DIR}/${run_name}/${placer_run_name}/${filter_run_name}/${placer2_run_name}/${filter2_run_name}/${final_run_name}"
    echo "Starting run"
    local start_time=$(date +%s)
    python "$PLOT_Z_POOLED_PY" \
        "$run_name" "$placer_run_name" "$filter_run_name" \
        "$placer2_run_name" "$filter2_run_name" "$final_run_name" \
        --datasets-dir "$DATASETS_DIR" --datasets-file "$DATASETS_FILE" --graphs-dir "$out_dir"
    echo "All jobs completed"
    print_elapsed "$start_time"
}

do_plot_bfactor_rho_pooled() {
    conda_activate "$CONDA_ENV_EVAL"

    local out_dir="${GRAPHS_DIR}/${run_name}/${placer_run_name}/${filter_run_name}/${placer2_run_name}/${filter2_run_name}/${final_run_name}"
    echo "Starting run"
    local start_time=$(date +%s)
    python "$PLOT_BFACTOR_RHO_POOLED_PY" \
        "$run_name" "$placer_run_name" "$filter_run_name" \
        "$placer2_run_name" "$filter2_run_name" "$final_run_name" \
        --datasets-dir "$DATASETS_DIR" --datasets-file "$DATASETS_FILE" --graphs-dir "$out_dir"
    echo "All jobs completed"
    print_elapsed "$start_time"
}

do_stage8_plots() {
    do_plot_cluster_reps_rscc
    do_aggregate_protein_rscc
    do_aggregate_lig_rscc
    do_plot_final_vs_apo_z
    do_plot_final_lig_z
    do_plot_bfactor_sensitivity
    do_plot_cluster_reps_rscc_pooled
    do_plot_protein_rscc_pooled
    do_plot_z_pooled
    do_plot_bfactor_rho_pooled
}

# stage8_outputs_exist: true only if every pooled output already exists,
# and, for every dataset that actually has a final_model_refined_rscc.csv
# (some legitimately never will - e.g. filter/filter2 rejected every
# candidate for that dataset, so build_final_model never produced a
# final_model.pdb to refine in the first place - see build_final_model.py),
# every one of stage 8's per-dataset outputs already exists. A dataset with
# no final_model_refined_rscc.csv can never produce those per-dataset
# outputs, so it's skipped here rather than permanently blocking stage 8
# from ever being considered complete.
stage8_outputs_exist() {
    local pooled_dir="${GRAPHS_DIR}/${run_name}/${placer_run_name}/${filter_run_name}/${placer2_run_name}/${filter2_run_name}/${final_run_name}"
    files_exist \
        "${pooled_dir}/cluster_reps_1_pooled.png" "${pooled_dir}/cluster_reps_2_pooled.png" \
        "${pooled_dir}/protein_backbone_vs_apo_rscc_placer_conformers_pooled.png" \
        "${pooled_dir}/protein_final_vs_apo_rscc_placer_conformers_pooled.png" \
        "${pooled_dir}/protein_final_vs_backbone_rscc_placer_conformers_pooled.png" \
        "${pooled_dir}/final_vs_apo_max_z_placer_conformers_pooled.png" \
        "${pooled_dir}/final_vs_apo_min_z_placer_conformers_pooled.png" \
        "${pooled_dir}/final_vs_apo_average_z_placer_conformers_pooled.png" \
        "${pooled_dir}/bfactor_sensitivity_spearman_rho_hist_pooled.png" || return 1

    local dataset
    for dataset in "${DATASETS[@]}"; do
        local final_dir="${DATASETS_DIR}/${dataset}/${run_name}/${placer_run_name}/${filter_run_name}/${placer2_run_name}/${filter2_run_name}/${final_run_name}"
        [ -f "${final_dir}/final_model_refined_rscc.csv" ] || continue

        local graphs_dir="${final_dir}/graphs"
        files_exist \
            "${graphs_dir}/cluster_reps_1.png" "${graphs_dir}/cluster_reps_2.png" \
            "${graphs_dir}/protein_backbone_vs_apo_rscc_placer_conformers.png" \
            "${graphs_dir}/protein_final_vs_apo_rscc_placer_conformers.png" \
            "${graphs_dir}/protein_final_vs_backbone_rscc_placer_conformers.png" \
            "${graphs_dir}/lig_filter2_vs_filter1_rscc.png" \
            "${graphs_dir}/final_vs_apo_max_z_placer_conformers.png" \
            "${graphs_dir}/final_vs_apo_min_z_placer_conformers.png" \
            "${graphs_dir}/final_vs_apo_average_z_placer_conformers.png" \
            "${graphs_dir}/final_lig_max_z.png" "${graphs_dir}/final_lig_min_z.png" \
            "${graphs_dir}/final_lig_average_z.png" \
            "${graphs_dir}/bfactor_sensitivity_lines.png" \
            "${graphs_dir}/bfactor_sensitivity_lines_normalized.png" \
            "${graphs_dir}/bfactor_sensitivity_spearman_rho_hist.png" || return 1
    done
    return 0
}

######################################################################
# Stage orchestration (unchanged shape: check -> run_step -> label)
######################################################################

stage0_apo_rscc() {
    run_step "Stage 0a: convert_ligs" do_convert_ligs
    run_step "Stage 0b: calc_apo_rscc" do_calc_apo_rscc
    # run_step "Stage 0b: calc_apo_z" do_calc_apo_z
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
    # temporarily disabled - not deleted, just not run right now
    # run_step "Stage 6c: calc_final_refined_z (${final_run_name})" do_calc_final_z
    # run_step "Stage 6c: calc_final_refined_rscc_b (${final_run_name})" do_calc_final_rscc_b
    if [ "$compare_ref_set" -eq 1 ]; then
        run_step_pooled_replot "Stage 6d: plot_residues_vs_ref_final (${final_run_name})" \
            final_ref_comparison_outputs_exist do_plot_residues_vs_ref_final
    fi
    run_step_pooled_replot "Stage 6e: aggregate_clash_groups (${final_run_name})" \
        clash_groups_aggregate_outputs_exist do_aggregate_clash_groups
}

stage7_despot() {
    run_step "Stage 7a: despot (${despot_run_name})" do_despot
    run_step_pooled_replot "Stage 7b: plot_despot_energies + plot_despot_energies_pooled (${despot_run_name})" \
        despot_plots_outputs_exist do_despot_plots
    if [ "$compare_ref_set" -eq 1 ]; then
        run_step_pooled_replot "Stage 7c: plot_lig_vs_ref_despot (${despot_run_name})" \
            despot_lig_vs_ref_outputs_exist do_plot_lig_vs_ref_despot
    fi
    run_step_pooled_replot "Stage 7d: plot_despot_ligand_summary + plot_despot_ligand_summary_single (${despot_run_name})" \
        despot_ligand_summary_outputs_exist do_despot_ligand_summary_plots
    if [ "$compare_ref_set" -eq 1 ]; then
        run_step_pooled_replot "Stage 7e: plot_despot_vs_ref (${despot_run_name})" \
            despot_vs_ref_outputs_exist do_plot_despot_vs_ref
    fi
    if [ "$compare_ref_set" -eq 1 ]; then
        run_step_pooled_replot "Stage 7f: plot_rscc_despot_tradeoff (${despot_run_name})" \
            rscc_despot_tradeoff_outputs_exist do_plot_rscc_despot_tradeoff
    fi
}

stage8_plots() {
    run_step_pooled_replot "Stage 8: analysis plots (${final_run_name})" \
        stage8_outputs_exist do_stage8_plots
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
    if [ -n "$despot_run_name" ]; then
        stage7_despot
    fi
    stage8_plots
fi

overall_end=$(date +%s)
elapsed=$((overall_end - overall_start))
hours=$((elapsed / 3600))
minutes=$(((elapsed % 3600) / 60))
seconds=$((elapsed % 60))
echo "========= program.sh complete ========="
printf "Total time: %02d:%02d:%02d (HH:MM:SS)\n" $hours $minutes $seconds

