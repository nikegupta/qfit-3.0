#!/usr/bin/env python3
"""
Pooled (across every dataset in datasets.txt) per-residue RSCC comparison:
final_run_name's final_model_refined structure vs the reference set,
matched by residue label (chain+resnum). No RSCC is computed here - both
sides are read from the calc_rscc csvs already on disk
(calc_final_refined_rscc and calc_ref_set_rscc).

Produces two plots: one over every matched residue, and one restricted to
the residues listed in final_run_name/residues_with_placer_conformers.csv.
Also writes final_refined_vs_reference_rscc_outliers.csv: every restricted
residue where ref_rscc - structure_rscc >= OUTLIER_MIN_DIFF (candidate cases
where the pipeline picked a worse-fitting rotamer than the reference has).

Run at the end of stage 6, only when -c (compare to reference set) is given.

Usage:
  plot_residues_vs_ref_final.py <run_name> <placer_run_name> <filter_run_name> \\
      <placer2_run_name> <filter2_run_name> <final_run_name> \\
      --ref-set <dir> --graphs-dir <dir> [options]
"""
from pathlib import Path

from rscc_common import (
    build_ref_argparser, plot_residues_vs_ref, read_calc_rscc_csv, read_residue_conformer_list,
)

# minimum ref_rscc - structure_rscc to be written to
# final_refined_vs_reference_rscc_outliers.csv (see plot_residues_vs_ref)
OUTLIER_MIN_DIFF = 0.1


def main():
    args = build_ref_argparser(
        __doc__,
        ['run_name', 'placer_run_name', 'filter_run_name', 'placer2_run_name',
         'filter2_run_name', 'final_run_name'],
    ).parse_args()

    def final_dir(dataset):
        return (Path(args.datasets_dir) / dataset / args.run_name / args.placer_run_name /
                args.filter_run_name / args.placer2_run_name / args.filter2_run_name /
                args.final_run_name)

    def collect_structure_rscc(dataset):
        df = read_calc_rscc_csv(final_dir(dataset) / 'final_model_refined_rscc.csv')
        return dict(zip(df['residue'], df['rscc']))

    def collect_restrict_labels(dataset):
        return read_residue_conformer_list(final_dir(dataset) / 'residues_with_placer_conformers.csv')

    plot_residues_vs_ref(
        args, collect_structure_rscc, collect_restrict_labels,
        out_dir=args.graphs_dir, out_prefix='final_refined', structure_label='Final-Refined',
        outlier_min_diff=OUTLIER_MIN_DIFF,
    )


if __name__ == '__main__':
    main()
