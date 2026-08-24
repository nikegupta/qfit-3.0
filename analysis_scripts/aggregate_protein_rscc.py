#!/usr/bin/env python3
"""
Protein-residue RSCC aggregator.

Reads the per-residue RSCC csvs already written by calc_apo_rscc,
calc_backbone_refined_rscc, and calc_final_refined_rscc (no RSCC values are
computed here), and produces scatter plots in the style of qfit's
compare_lig_rscc, comparing every protein residue's RSCC (i.e. every residue
except the fitted LIG) across pipeline stages:

  backbone-refined vs apo
  final-refined     vs apo
  final-refined     vs backbone-refined (best RSCC across its cluster reps)

Each comparison is restricted to the residues listed in final_run_name's
residues_with_placer_conformers.csv - the unrestricted (all-residues)
variant was dropped since it carried no information beyond this one.

Usage:
  aggregate_protein_rscc.py <run_name> <placer_run_name> <filter_run_name> \\
      <placer2_run_name> <filter2_run_name> <final_run_name> [options]
"""
from rscc_common import build_common_argparser, run_rscc_aggregator


def main():
    args = build_common_argparser(__doc__).parse_args()
    run_rscc_aggregator(args)


if __name__ == '__main__':
    main()
