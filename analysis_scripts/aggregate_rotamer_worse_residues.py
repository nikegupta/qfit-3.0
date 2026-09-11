#!/usr/bin/env python3
"""
Pooled (across every dataset in datasets.txt) CSV of every residue - restricted to
residues_with_placer_conformers.csv, the only residues rotamer_refined.pdb ever touches -
whose rotamer_refined RSCC is more than 0.1 worse than either the matched reference-structure
residue's RSCC, or that dataset's pre-rotamer-optimization final_model_refined RSCC. A residue
can appear twice (once per comparison) if it clears the threshold against both baselines. No
RSCC is computed here - every value is read from calc_rscc csvs already on disk.

Produces rotamer_refined_worse_residues.csv.

Run at the end of stage 7, only when -c (compare to reference set) is given.

Usage:
  aggregate_rotamer_worse_residues.py <run_name> <placer_run_name> <filter_run_name> \\
      <placer2_run_name> <filter2_run_name> <final_run_name> <rotamer_run_name> \\
      --ref-set <dir> --graphs-dir <dir> [options]
"""
from rscc_common import build_ref_argparser, run_rotamer_worse_residues


def main():
    args = build_ref_argparser(
        __doc__,
        ['run_name', 'placer_run_name', 'filter_run_name', 'placer2_run_name',
         'filter2_run_name', 'final_run_name', 'rotamer_run_name'],
    ).parse_args()

    run_rotamer_worse_residues(args)


if __name__ == '__main__':
    main()
