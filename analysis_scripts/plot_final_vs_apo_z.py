#!/usr/bin/env python3
"""
Final-vs-apo Z-map statistics aggregator.

Reads the per-residue Z-map statistics csvs already written by calc_apo_z and
calc_final_refined_z (no Z-map values are computed here), and produces
scatter plots comparing final_model_refined's per-residue Z-map statistics
against the apo structure's own (same Z-map, different structure):

  max_z     final-refined vs apo
  min_z     final-refined vs apo
  average_z final-refined vs apo

Like the RSCC aggregators, each comparison is restricted to the residues
listed in final_run_name's residues_with_placer_conformers.csv.

Usage:
  plot_final_vs_apo_z.py <run_name> <placer_run_name> <filter_run_name> \\
      <placer2_run_name> <filter2_run_name> <final_run_name> [options]
"""
from rscc_common import build_common_argparser, run_z_aggregator


def main():
    args = build_common_argparser(__doc__).parse_args()
    run_z_aggregator(args)


if __name__ == '__main__':
    main()
