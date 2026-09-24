#!/usr/bin/env python3
"""
Pooled (across every dataset in datasets.txt) scatter of calc_placer_sampling.py's minimum
ligand RMSD, round-2 PLACER samples (y) vs round-1 PLACER samples (x), for every reference
ligand present in both rounds' placer_sampling.csv (round1_csv, round2_csv) - see
plot_placer2_vs_placer1_rmsd's own docstring in rscc_common.py.

Only compares the refined placer_sampling.csv (not placer_sampling_unrefined.csv).

When --filter1-csv (Stage 3d's pooled lig_vs_reference_rscc.csv) is given, the compared pairs are
further restricted to reference ligands that found a match in filter (Stage 3a) - see
rscc_common.restrict_to_filter_matched for why unmatched reference ligands otherwise pollute this
plot with large, meaningless RMSDs.

Run at the end of stage 4c (calc_placer_sampling, round 2), only when -c is given (both input
csvs are themselves only written with -c).

Usage:
  plot_placer2_vs_placer1_rmsd.py <round1_csv> <round2_csv> --out-dir <dir> [--filter1-csv <path>]

Output:
  <out_dir>/placer2_vs_placer1_rmsd.png
"""
import argparse

from rscc_common import plot_placer2_vs_placer1_rmsd


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('round1_csv', help="Stage 2c's placer_sampling.csv (round-1 PLACER samples)")
    p.add_argument('round2_csv', help="Stage 4c's placer_sampling.csv (round-2 PLACER samples)")
    p.add_argument('--out-dir', required=True,
                   help='Output directory for the plot (by convention, round2_csv\'s own directory)')
    p.add_argument('--filter1-csv', default=None,
                   help="Stage 3d's pooled lig_vs_reference_rscc.csv - when given, restricts to "
                        "reference ligands that found a match in filter (Stage 3a)")
    args = p.parse_args()

    plot_placer2_vs_placer1_rmsd(args.round1_csv, args.round2_csv, args.out_dir,
                                  filter1_csv=args.filter1_csv)


if __name__ == '__main__':
    main()
