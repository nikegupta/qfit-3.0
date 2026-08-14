#!/usr/bin/env python3
"""
Pooled (across every dataset in datasets.txt) counterpart of
plot_cluster_reps_rscc.py: combines every dataset's cluster-representative
RSCC values (cluster_reps.csv's 'rscc' column, already written by
filter/filter2 - no RSCC is computed here) into a single histogram per
stage instead of one per dataset:

  cluster_reps_1_pooled.png : filter_run_name's cluster_reps.csv   (post-round-1 filter)
  cluster_reps_2_pooled.png : filter2_run_name's cluster_reps.csv  (post-round-2 filter)

Doesn't compare against the reference set, so it runs unconditionally
alongside plot_cluster_reps_rscc.py - not gated behind -c.

Usage:
  plot_cluster_reps_rscc_pooled.py <run_name> <placer_run_name> <filter_run_name> \\
      <placer2_run_name> <filter2_run_name> <final_run_name> --graphs-dir <dir> [options]
"""
from rscc_common import build_pooled_argparser, run_cluster_reps_pooled


def main():
    args = build_pooled_argparser(__doc__).parse_args()
    run_cluster_reps_pooled(args)


if __name__ == '__main__':
    main()
