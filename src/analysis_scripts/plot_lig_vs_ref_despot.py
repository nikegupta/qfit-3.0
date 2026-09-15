#!/usr/bin/env python3
"""
Pooled (across every dataset in datasets.txt) ligand RSCC comparison,
restricted to the ligand poses that survived despot_filter.py's physical-
plausibility filter (despot_run_name/despot_filtered.pdb): despot_run_name/
cluster_reps.csv's despot_rscc column (despot_filter.py's own reselected-winner
RSCC, individually computed - NOT filter2_run_name/cluster_reps.csv's rscc,
which is the original, possibly-superseded representative's value) for just the
surviving instances, vs the reference set's matched ligand RSCC - same
centroid-distance matching plot_lig_vs_ref_filter1.py/plot_lig_vs_ref_filter2.py
use.

Despot filtering runs on final_model_refined.pdb, built from filter2_run_name/
cluster_rep_models.pdb - a despot_filtered.pdb ligand instance's residue
number is the exact same 1-indexed position as its originating
cluster_rep_models.pdb MODEL / cluster_reps.csv data row (build_final_model.py
numbers each ligand instance by its source model's position and never
renumbers them; despot_filter.py only removes some instances, it doesn't
renumber the survivors either) - see find_all_lig_residues.

Unlike plot_lig_vs_ref_filter1.py/_filter2.py, its --graphs-dir is pointed at
GRAPHS_DIR/.../<final_run_name>/<despot_run_name>/ (nested under
despot_run_name, like plot_despot_energies_pooled.py), since which ligands
survive is specific to one despot_run_name/threshold:
  lig_vs_reference_rscc.png

Run at the end of stage 7, only when both -c (compare to reference set) and
<despot_run_name> are given.

Usage:
  plot_lig_vs_ref_despot.py <run_name> <placer_run_name> <filter_run_name> \\
      <placer2_run_name> <filter2_run_name> <final_run_name> <despot_run_name> \\
      --ref-set <dir> --graphs-dir <dir> [options]
"""
from pathlib import Path

from rscc_common import build_ref_argparser, plot_lig_vs_ref, dataset_final_dir, find_all_lig_residues


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

    plot_lig_vs_ref(
        args, run_dir_for_dataset,
        title='Ligand RSCC vs Reference, DESPOT-filtered',
        out_name='lig_vs_reference_rscc.png',
        alive_rows_for_dataset=alive_rows_for_dataset,
        resi_col_name='despot_filtered_resi', chain_col_name='despot_filtered_chain',
        cluster_csv_override_for_dataset=cluster_csv_override_for_dataset,
        rscc_column='despot_rscc',
    )


if __name__ == '__main__':
    main()
