#!/usr/bin/env python3
"""
For every dataset independently (no pooling across datasets), scatters that
dataset's own surviving (despot_filter-kept) ligands' heavy-atom-normalized
DESPOT score (x) against their filter2_run_name/cluster_reps.csv RSCC (y) -
same matching plot_despot_ligand_summary.py (the pooled counterpart) uses,
just restricted to one dataset.

Written directly into that dataset's own .../<final_run_name>/<despot_run_name>/
directory (not graphs_dir - despot_run_name is already a per-dataset
location):
  ligand_summary.png
with a matching csv saved alongside it. A single dataset typically has only
a handful of surviving ligands, so each point is labeled in red with its
chain+resi (e.g. 'C1' for chain C, residue 1) - unlike the pooled plot,
which would be unreadable with a label per point across every dataset.

Doesn't need -c: RSCC here comes from cluster_reps.csv, not the reference
set.

Usage:
  plot_despot_ligand_summary_single.py <run_name> <placer_run_name> <filter_run_name> \\
      <placer2_run_name> <filter2_run_name> <final_run_name> <despot_run_name> [options]
"""
from rscc_common import build_despot_argparser, run_despot_ligand_summary_single


def main():
    args = build_despot_argparser(__doc__).parse_args()
    run_despot_ligand_summary_single(args)


if __name__ == '__main__':
    main()
