#!/usr/bin/env python3
"""
Pooled (across every dataset in datasets.txt) heavy-atom-normalized DESPOT
score comparison: each dataset's reference-set structure's own DESPOT score
(REF_SET/<dataset>/<dataset>_DESPOT.csv, written by program.sh's Stage 0d)
vs the matched pipeline ligand's DESPOT score (despot_run_name/
despot_filtered_scores.csv), restricted to the ligand poses that survived
despot_filter.py's physical-plausibility filter (despot_run_name/
despot_filtered.pdb) - matched by the same centroid-distance strategy
plot_lig_vs_ref_despot.py uses for RSCC (see rscc_common.py's
_dataset_despot_vs_ref/alive_rows), so this plot's matched pairs - and its
unmatched-reference/excess-pipeline counts - agree exactly with
lig_vs_reference_rscc.png in the same despot_run_name folder. Both sides are
normalized by their own ligand's heavy-atom count before plotting, so the
scores are directly comparable to each other.

--graphs-dir is pointed at GRAPHS_DIR/.../<final_run_name>/<despot_run_name>/
(nested under despot_run_name, like plot_despot_energies_pooled.py/
plot_lig_vs_ref_despot.py), since the pipeline-side scores are specific to
one despot_run_name:
  despot_vs_reference.png

Run at the end of stage 7, only when both -c (compare to reference set) and
<despot_run_name> are given.

Usage:
  plot_despot_vs_ref.py <run_name> <placer_run_name> <filter_run_name> \\
      <placer2_run_name> <filter2_run_name> <final_run_name> <despot_run_name> \\
      --ref-set <dir> --graphs-dir <dir> [options]
"""
from pathlib import Path

from rscc_common import build_ref_argparser, plot_despot_vs_ref, dataset_final_dir, find_all_lig_residues


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

    plot_despot_vs_ref(
        args, run_dir_for_dataset,
        title='DESPOT Score vs Reference, DESPOT-filtered',
        out_name='despot_vs_reference.png',
        alive_rows_for_dataset=alive_rows_for_dataset,
    )


if __name__ == '__main__':
    main()
