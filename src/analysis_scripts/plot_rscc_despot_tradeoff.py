#!/usr/bin/env python3
"""
Pooled (across every dataset in datasets.txt) scatter of despot_filter.py's RSCC/DESPOT
reselection tradeoff relative to the reference structure, restricted to the ligand poses that
survived despot_filter.py's physical-plausibility filter (despot_run_name/despot_filtered.pdb) -
the same restriction plot_lig_vs_ref_despot.py/plot_despot_vs_ref.py use, so this plot's matched
pairs agree with theirs. Reuses those two plots' own centroid-matching (_dataset_lig_vs_ref/
_dataset_despot_vs_ref via _dataset_rscc_despot_tradeoff) rather than a third matching pass - see
rscc_common.py.

For every matched ligand:
  y = pipeline RSCC - reference RSCC (despot_filter.py's reselected winner's own, individually
      computed RSCC - despot_run_name/cluster_reps.csv's despot_rscc column - NOT filter2's
      possibly-superseded original-representative RSCC)
  x = reference DESPOT - pipeline DESPOT (both heavy-atom-normalized)

--graphs-dir is pointed at GRAPHS_DIR/.../<final_run_name>/<despot_run_name>/ (nested under
despot_run_name, like Stage 7b/7c/7d/7e), since which ligands survive - and their reselected
RSCC/DESPOT - is specific to one despot_run_name/--despot_threshold/--despot_rscc_threshold/
--despot_rscc_weight:
  rscc_despot_tradeoff_vs_reference.png

Run at the end of stage 7, only when both -c (compare to reference set) and <despot_run_name>
are given.

Usage:
  plot_rscc_despot_tradeoff.py <run_name> <placer_run_name> <filter_run_name> \\
      <placer2_run_name> <filter2_run_name> <final_run_name> <despot_run_name> \\
      --ref-set <dir> --graphs-dir <dir> [options]
"""
from pathlib import Path

from rscc_common import (
    build_ref_argparser, plot_rscc_despot_tradeoff, dataset_final_dir, find_all_lig_residues,
)


def main():
    args = build_ref_argparser(
        __doc__,
        ['run_name', 'placer_run_name', 'filter_run_name', 'placer2_run_name',
         'filter2_run_name', 'final_run_name', 'despot_run_name'],
    ).parse_args()

    def run_dir_for_dataset(dataset):
        return (Path(args.datasets_dir) / dataset / args.run_name / args.placer_run_name /
                args.filter_run_name / args.placer2_run_name / args.filter2_run_name)

    def alive_rows_for_dataset(dataset):
        despot_filtered_pdb = (dataset_final_dir(args.datasets_dir, dataset, args) /
                                args.despot_run_name / 'despot_filtered.pdb')
        return {resnum for _, resnum in find_all_lig_residues(despot_filtered_pdb)}

    def cluster_csv_override_for_dataset(dataset):
        return (dataset_final_dir(args.datasets_dir, dataset, args) / args.despot_run_name
                / 'cluster_reps.csv')

    plot_rscc_despot_tradeoff(
        args, run_dir_for_dataset,
        title='RSCC/DESPOT reselection tradeoff vs Reference, DESPOT-filtered',
        out_name='rscc_despot_tradeoff_vs_reference.png',
        alive_rows_for_dataset=alive_rows_for_dataset,
        cluster_csv_override_for_dataset=cluster_csv_override_for_dataset,
    )


if __name__ == '__main__':
    main()
