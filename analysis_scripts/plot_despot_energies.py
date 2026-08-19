#!/usr/bin/env python3
"""
For every dataset independently (no pooling across datasets), plots a
histogram of DESPOT ligand binding-energy scores already written by
program.sh's despot stage (DESPOT's score_complex.py, via its own -o
argument) into .../<final_run_name>/<despot_run_name>/<dataset>_DESPOT.csv.
No scores are computed here.

Written into that dataset's existing .../<final_run_name>/graphs/ and csvs/
folders - the same per-dataset location every other analysis_scripts plot in
this stage uses, not nested under despot_run_name:
  despot_energies.png

Usage:
  plot_despot_energies.py <run_name> <placer_run_name> <filter_run_name> \\
      <placer2_run_name> <filter2_run_name> <final_run_name> <despot_run_name> [options]
"""
from rscc_common import build_despot_argparser, run_despot_energies_single


def main():
    args = build_despot_argparser(__doc__).parse_args()
    run_despot_energies_single(args)


if __name__ == '__main__':
    main()
