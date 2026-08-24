#!/usr/bin/env python3
"""
Pooled (across every dataset in datasets.txt) ligand summary: each surviving
ligand's (i.e. present in despot_run_name/despot_filtered.pdb - see
despot_filtered_scores.csv's 'kept' column, written by despot_filter.py)
heavy-atom-normalized DESPOT score (x) against its filter2_run_name/
cluster_reps.csv RSCC (y) - no RSCC recomputed here, matched by the same
resi-as-row-index convention plot_lig_vs_ref_despot.py uses.

Its --graphs-dir is pointed at GRAPHS_DIR/.../<final_run_name>/<despot_run_name>/
(nested under despot_run_name, like plot_despot_energies_pooled.py), since the
scores/survivors are specific to one despot_run_name/--despot_threshold:
  ligand_summary.png

Run at the end of stage 7, whenever <despot_run_name> is given - doesn't need
-c/the reference set (RSCC here comes from cluster_reps.csv, not a reference
structure).

Usage:
  plot_despot_ligand_summary.py <run_name> <placer_run_name> <filter_run_name> \\
      <placer2_run_name> <filter2_run_name> <final_run_name> <despot_run_name> \\
      --graphs-dir <dir> [options]
"""
from rscc_common import build_despot_pooled_argparser, run_despot_ligand_summary


def main():
    args = build_despot_pooled_argparser(__doc__).parse_args()
    run_despot_ligand_summary(args)


if __name__ == '__main__':
    main()
