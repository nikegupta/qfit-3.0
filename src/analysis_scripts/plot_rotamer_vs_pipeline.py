#!/usr/bin/env python3
"""
Pooled (across every dataset in datasets.txt) per-residue RSCC comparison of
rotamer_run_name's OPTIMIZED structure (select_optimized_residues.py's
optimized_rscc.csv, not the raw rotamer_refined_rscc.csv) against two other
already-scored pipeline structures, restricted to
residues_with_placer_conformers.csv - the only residues optimized.pdb ever
touches:
  - final_model_refined (stage 6, pre-rotamer-optimization) ->
    rotamer_refined_vs_final_refined_rscc_restricted.png
  - backbone_refined (stage 3, best across cluster reps - the same 'apo set'
    baseline plot_protein_rscc_pooled.py's backbone-vs-apo/final-vs-apo
    plots use) -> rotamer_refined_vs_backbone_refined_rscc_restricted.png

No RSCC is computed here - all three sides are read from the calc_rscc csvs
already on disk. Doesn't compare against the reference set, so - like
plot_protein_rscc_pooled.py - it runs unconditionally at the end of stage 7,
after select_optimized_residues.py, not gated behind -c.

Usage:
  plot_rotamer_vs_pipeline.py <run_name> <placer_run_name> <filter_run_name> \\
      <placer2_run_name> <filter2_run_name> <final_run_name> <rotamer_run_name> \\
      --graphs-dir <dir> [options]
"""
from rscc_common import build_pooled_argparser, run_rotamer_vs_pipeline_pooled


def main():
    args = build_pooled_argparser(
        __doc__,
        ['run_name', 'placer_run_name', 'filter_run_name', 'placer2_run_name',
         'filter2_run_name', 'final_run_name', 'rotamer_run_name'],
    ).parse_args()
    run_rotamer_vs_pipeline_pooled(args)


if __name__ == '__main__':
    main()
