#!/usr/bin/env python3
"""
Pooled (across every dataset in datasets.txt) counterpart of
plot_despot_energies.py: every dataset's heavy-atom-normalized DESPOT ligand
binding-energy scores (despot_filtered_scores.csv's normalized_score column)
combined into a single histogram.

Unlike the other stage 7/8 pooled plots, this one's --graphs-dir is pointed
at GRAPHS_DIR/.../<final_run_name>/<despot_run_name>/ (nested under
despot_run_name), since the DESPOT scores themselves are specific to one
despot_run_name:
  ligand_energies.png

Usage:
  plot_despot_energies_pooled.py <run_name> <placer_run_name> <filter_run_name> \\
      <placer2_run_name> <filter2_run_name> <final_run_name> <despot_run_name> \\
      --graphs-dir <dir> [options]
"""
from rscc_common import build_despot_pooled_argparser, run_despot_energies_pooled


def main():
    args = build_despot_pooled_argparser(__doc__).parse_args()
    run_despot_energies_pooled(args)


if __name__ == '__main__':
    main()
