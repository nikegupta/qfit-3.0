#!/usr/bin/env python3
"""
Pooled (across every dataset in datasets.txt) counterpart of
plot_bfactor_sensitivity.py's spearman-rho histogram only - not the two
RSCC-vs-bfactor line plots, which don't pool meaningfully since each line
is already one residue's full bfactor sweep. Combines every dataset's
canonical per-residue spearmans_rho (from final_model_refined_rscc_b.csv)
into one histogram.

Doesn't compare against the reference set, so it runs unconditionally
alongside plot_bfactor_sensitivity.py - not gated behind -c.

Usage:
  plot_bfactor_rho_pooled.py <run_name> <placer_run_name> <filter_run_name> \\
      <placer2_run_name> <filter2_run_name> <final_run_name> --graphs-dir <dir> [options]
"""
from rscc_common import build_pooled_argparser, run_bfactor_rho_pooled


def main():
    args = build_pooled_argparser(__doc__).parse_args()
    run_bfactor_rho_pooled(args)


if __name__ == '__main__':
    main()
