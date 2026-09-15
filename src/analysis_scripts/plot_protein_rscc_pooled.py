#!/usr/bin/env python3
"""
Pooled (across every dataset in datasets.txt) counterpart of
aggregate_protein_rscc.py: the same per-residue protein RSCC comparisons
(backbone-vs-apo, final-vs-apo, final-vs-backbone; restricted to each
dataset's residues_with_placer_conformers.csv), combined into a single
scatter plot per comparison instead of one per dataset, colored by point
density (pooling every dataset's residues creates far more overplotting
than any single dataset's plot has).

Doesn't compare against the reference set, so it runs unconditionally
alongside aggregate_protein_rscc.py - not gated behind -c.

Usage:
  plot_protein_rscc_pooled.py <run_name> <placer_run_name> <filter_run_name> \\
      <placer2_run_name> <filter2_run_name> <final_run_name> --graphs-dir <dir> [options]
"""
from rscc_common import build_pooled_argparser, run_rscc_aggregator_pooled


def main():
    args = build_pooled_argparser(__doc__).parse_args()
    run_rscc_aggregator_pooled(args)


if __name__ == '__main__':
    main()
