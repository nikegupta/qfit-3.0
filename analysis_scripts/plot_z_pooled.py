#!/usr/bin/env python3
"""
Pooled (across every dataset in datasets.txt) counterpart of
plot_final_vs_apo_z.py: the same final-vs-apo Z-map statistic comparisons
(max/min/average Z-score, restricted to each dataset's
residues_with_placer_conformers.csv), combined into a single scatter plot
per statistic instead of one per dataset, colored by point density
(pooling every dataset's residues creates far more overplotting than any
single dataset's plot has).

Doesn't compare against the reference set, so it runs unconditionally
alongside plot_final_vs_apo_z.py - not gated behind -c.

Usage:
  plot_z_pooled.py <run_name> <placer_run_name> <filter_run_name> \\
      <placer2_run_name> <filter2_run_name> <final_run_name> --graphs-dir <dir> [options]
"""
from rscc_common import build_pooled_argparser, run_z_aggregator_pooled


def main():
    args = build_pooled_argparser(__doc__).parse_args()
    run_z_aggregator_pooled(args)


if __name__ == '__main__':
    main()
