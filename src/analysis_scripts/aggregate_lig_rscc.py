#!/usr/bin/env python3
"""
Ligand-residue RSCC aggregator.

The apo structure has no ligand, so a ligand-vs-apo RSCC comparison can never
have data - that comparison type is protein-only (see
aggregate_protein_rscc.py). The only ligand RSCC comparison that makes sense
is filter_2 vs filter_1: each filter2_run_name cluster rep is matched back to
the filter_run_name cluster rep it originated from (via the round-1
backbone-refined model index embedded in filter2's placer_file column), and
their 'rscc' values (already written by filter.py/filter2's clustering, not
recomputed here) are plotted against each other, with a count of how many
filter_run_name cluster reps never made it into filter2_run_name (i.e. were
lost between rounds). ("Histogram of ligand RSCC after filter1/filter2" is
already covered by plot_cluster_reps_rscc.py's cluster_reps_1/2 histograms,
which read the same cluster_reps.csv 'rscc' column.)

Usage:
  aggregate_lig_rscc.py <run_name> <placer_run_name> <filter_run_name> \\
      <placer2_run_name> <filter2_run_name> <final_run_name> [options]
"""
from rscc_common import build_common_argparser, plot_filter2_vs_filter1_lig_rscc


def main():
    args = build_common_argparser(__doc__).parse_args()
    plot_filter2_vs_filter1_lig_rscc(args)


if __name__ == '__main__':
    main()
