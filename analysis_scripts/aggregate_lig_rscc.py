#!/usr/bin/env python3
"""
Ligand-residue RSCC aggregator.

Reads the per-residue RSCC csvs already written by calc_apo_rscc,
calc_backbone_refined_rscc, and calc_final_refined_rscc (no RSCC values are
computed here), and produces scatter plots in the style of qfit's
compare_lig_rscc, comparing the fitted LIG residue's RSCC across pipeline
stages:

  backbone-refined vs apo
  final-refined     vs apo

The apo structure has no ligand, so both "vs apo" comparisons will have no
data points and are skipped with a message - this is expected, not an
error. Each comparison is also plotted restricted to the residues listed in
final_run_name's residues_with_placer_conformers.csv (which never includes
the ligand, so those plots are expected to be empty too); both variants are
produced for structural symmetry with aggregate_protein_rscc.py.

final-refined vs backbone-refined is intentionally omitted here (unlike
aggregate_protein_rscc.py) since it's redundant with the filter_2-vs-filter_1
plot below, which compares the same two stages' ligand RSCC more directly.

Also plots filter_2 vs filter_1 ligand RSCC: each filter2_run_name cluster
rep is matched back to the filter_run_name cluster rep it originated from
(via the round-1 backbone-refined model index embedded in filter2's
placer_file column), and their 'rscc' values (already written by
filter.py/filter2's clustering, not recomputed here) are plotted against
each other, with a count of how many filter_run_name cluster reps never
made it into filter2_run_name (i.e. were lost between rounds).

Usage:
  aggregate_lig_rscc.py <run_name> <placer_run_name> <filter_run_name> \\
      <placer2_run_name> <filter2_run_name> <final_run_name> [options]
"""
from rscc_common import build_common_argparser, run_rscc_aggregator, plot_filter2_vs_filter1_lig_rscc


def main():
    args = build_common_argparser(__doc__).parse_args()
    run_rscc_aggregator(args, mode='lig')
    plot_filter2_vs_filter1_lig_rscc(args)


if __name__ == '__main__':
    main()
