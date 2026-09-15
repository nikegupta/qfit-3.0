#!/usr/bin/env python3
"""
Pooled (across every dataset in datasets.txt) per-residue RSCC comparison:
rotamer_run_name's OPTIMIZED structure (select_optimized_residues.py's
optimized.pdb/optimized_rscc.csv - per residue, whichever of
final_model_refined/rotamer_refined scored higher, NOT the raw
rotamer_refined_rscc.csv) vs the reference set, restricted to
residues_with_placer_conformers.csv - the only residues optimized.pdb ever
touches (rotamer_optimize.py only resamples those, and
calc_rotamer_refined_rscc/select_optimized_residues restrict to them), so
there's no separate "all residues" plot the way plot_residues_vs_ref_final.py
has one (it would just duplicate this one). No RSCC is computed here - both
sides are read from the calc_rscc csvs already on disk
(select_optimized_residues and calc_ref_set_rscc).

Produces rotamer_refined_vs_reference_rscc_restricted.png. Also writes
rotamer_refined_vs_reference_rscc_outliers.csv: every restricted residue
where ref_rscc - structure_rscc >= OUTLIER_MIN_DIFF (candidate cases where
the pipeline picked a worse-fitting rotamer than the reference has - same
tally as plot_residues_vs_ref_final.py's outliers csv).

Run at the end of stage 7, after select_optimized_residues.py, only when -c
(compare to reference set) is given.

Usage:
  plot_residues_vs_ref_rotamer.py <run_name> <placer_run_name> <filter_run_name> \\
      <placer2_run_name> <filter2_run_name> <final_run_name> <rotamer_run_name> \\
      --ref-set <dir> --graphs-dir <dir> [options]
"""
from pathlib import Path

from rscc_common import (
    build_ref_argparser, plot_residues_vs_ref_restricted, read_calc_rscc_csv,
    read_residue_conformer_list,
)

# minimum ref_rscc - structure_rscc to be written to
# rotamer_refined_vs_reference_rscc_outliers.csv (see plot_residues_vs_ref_restricted)
OUTLIER_MIN_DIFF = 0.1


def main():
    args = build_ref_argparser(
        __doc__,
        ['run_name', 'placer_run_name', 'filter_run_name', 'placer2_run_name',
         'filter2_run_name', 'final_run_name', 'rotamer_run_name'],
    ).parse_args()

    def final_dir(dataset):
        return (Path(args.datasets_dir) / dataset / args.run_name / args.placer_run_name /
                args.filter_run_name / args.placer2_run_name / args.filter2_run_name /
                args.final_run_name)

    def rotamer_dir(dataset):
        return final_dir(dataset) / args.rotamer_run_name

    def collect_structure_rscc(dataset):
        df = read_calc_rscc_csv(rotamer_dir(dataset) / 'optimized_rscc.csv')
        return dict(zip(df['residue'], df['rscc']))

    def collect_restrict_labels(dataset):
        return read_residue_conformer_list(final_dir(dataset) / 'residues_with_placer_conformers.csv')

    plot_residues_vs_ref_restricted(
        args, collect_structure_rscc, collect_restrict_labels,
        out_dir=args.graphs_dir, out_prefix='rotamer_refined', structure_label='Rotamer-Refined',
        outlier_min_diff=OUTLIER_MIN_DIFF,
    )


if __name__ == '__main__':
    main()
