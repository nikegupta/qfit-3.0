#!/usr/bin/env python3
"""
Pooled (across every dataset in datasets.txt) counterpart of
aggregate_lig_rscc.py: the same fitted-LIG-residue RSCC comparisons
(backbone-vs-apo, final-vs-apo), combined into a single scatter plot per
comparison instead of one per dataset, colored by point density (pooling
every dataset's ligand creates far more overplotting than any single
dataset's plot has). final-vs-backbone is skipped here too, same as
aggregate_lig_rscc.py - plot_filter2_vs_filter1_lig_rscc already covers
that comparison.

Doesn't compare against the reference set, so it runs unconditionally
alongside aggregate_lig_rscc.py - not gated behind -c.

Usage:
  plot_lig_rscc_pooled.py <run_name> <placer_run_name> <filter_run_name> \\
      <placer2_run_name> <filter2_run_name> <final_run_name> --graphs-dir <dir> [options]
"""
from rscc_common import build_pooled_argparser, run_rscc_aggregator_pooled


def main():
    args = build_pooled_argparser(__doc__).parse_args()
    run_rscc_aggregator_pooled(args, mode='lig')


if __name__ == '__main__':
    main()
