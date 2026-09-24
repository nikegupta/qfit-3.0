#!/usr/bin/env python3
"""
Pooled (across every dataset in datasets.txt) per-residue RSCC comparison:
optimized's structure RSCC (rotamer_run_name/optimized_rscc.csv - the same
values plot_residues_vs_ref_rotamer.py uses) vs the reference set,
restricted to despot_run_name/modified_residues.csv - the residues that kept
a non-apo, PLACER-derived conformation after despot_filter.py's
reset_protein_to_apo_where_unbacked (i.e. residues_with_placer_conformers.csv
minus whatever despot_filter reset back to apo because their only backing
placer file's ligand was rejected). despot_filter.py's own input is now
optimized.pdb (not rotamer_refined.pdb), so this plot's RSCC source matches
despot_filtered.pdb's actual protein conformations. This restricted set is
always <= the one plot_residues_vs_ref_rotamer.py restricts to, so this
plot's point count is never larger, and is exactly equal when despot_filter
rejected no ligands.

No RSCC is computed here - both sides are read from the calc_rscc csvs
already on disk (select_optimized_residues and calc_ref_set_rscc).

Produces rotamer_refined_despot_vs_reference_rscc_restricted.png, nested
under despot_run_name (like every other despot-stage plot), since which
residues are modified is specific to one despot_run_name/--despot_threshold/
--despot_rscc_threshold/--despot_rscc_weight. Also writes
rotamer_refined_despot_vs_reference_rscc_outliers.csv: every restricted
residue where ref_rscc - structure_rscc >= OUTLIER_MIN_DIFF (candidate cases
where the pipeline picked a worse-fitting rotamer than the reference has -
same tally/threshold as plot_residues_vs_ref_final.py's and
plot_residues_vs_ref_rotamer.py's outliers csvs).

Run at the end of stage 8, only when both -c (compare to reference set) and
<despot_run_name> are given.

Usage:
  plot_residues_vs_ref_despot.py <run_name> <placer_run_name> <filter_run_name> \\
      <placer2_run_name> <filter2_run_name> <final_run_name> <rotamer_run_name> \\
      <despot_run_name> --ref-set <dir> --graphs-dir <dir> [options]
"""
from pathlib import Path

from rscc_common import (
    build_ref_argparser, plot_residues_vs_ref_restricted, read_calc_rscc_csv,
    read_residue_conformer_list,
)

# minimum ref_rscc - structure_rscc to be written to
# rotamer_refined_despot_vs_reference_rscc_outliers.csv (see plot_residues_vs_ref_restricted)
OUTLIER_MIN_DIFF = 0.1


def main():
    args = build_ref_argparser(
        __doc__,
        ['run_name', 'placer_run_name', 'filter_run_name', 'placer2_run_name',
         'filter2_run_name', 'final_run_name', 'rotamer_run_name', 'despot_run_name'],
    ).parse_args()

    def rotamer_dir(dataset):
        return (Path(args.datasets_dir) / dataset / args.run_name / args.placer_run_name /
                args.filter_run_name / args.placer2_run_name / args.filter2_run_name /
                args.final_run_name / args.rotamer_run_name)

    def despot_dir(dataset):
        return rotamer_dir(dataset) / args.despot_run_name

    def collect_structure_rscc(dataset):
        df = read_calc_rscc_csv(rotamer_dir(dataset) / 'optimized_rscc.csv')
        return dict(zip(df['residue'], df['rscc']))

    def collect_restrict_labels(dataset):
        return read_residue_conformer_list(despot_dir(dataset) / 'modified_residues.csv')

    plot_residues_vs_ref_restricted(
        args, collect_structure_rscc, collect_restrict_labels,
        out_dir=args.graphs_dir, out_prefix='rotamer_refined_despot',
        structure_label='Modified',
        outlier_min_diff=OUTLIER_MIN_DIFF,
    )


if __name__ == '__main__':
    main()
