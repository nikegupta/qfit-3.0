#!/usr/bin/env python3
"""
For every dataset independently (no pooling across datasets), plots a
histogram of heavy-atom-normalized DESPOT ligand binding-energy scores
already written by despot_filter.py into .../<final_run_name>/<despot_run_name>/
despot_filtered_scores.csv (its own normalized_score column - the raw,
un-normalized <dataset>_DESPOT.csv score isn't plotted anywhere in this
pipeline, since it isn't comparable across differently-sized ligands).
No scores are computed or normalized here.

Written into that dataset's existing .../<final_run_name>/graphs/ folder -
the same per-dataset location every other analysis_scripts plot in this
stage uses, not nested under despot_run_name:
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
