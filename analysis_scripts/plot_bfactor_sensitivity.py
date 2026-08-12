#!/usr/bin/env python3
"""
For every dataset independently (no pooling across datasets), builds
bfactor-sensitivity plots from the per-residue, per-event-map, per-bfactor
RSCC csv already written by calc_final_refined_rscc_b
(final_model_refined_rscc_b.csv, one row per (event_map, bfactor)
combination, restricted to the residues in residues_with_placer_conformers.csv).
No RSCC or spearman's rho is computed here - calc_rscc_b already wrote both.

For each residue, the "canonical" event map is whichever one contains that
residue's single highest RSCC value across all of its (event_map, bfactor)
rows. Three plots are produced per dataset, into that dataset's own
.../<final_run_name>/graphs/ folder, plus a matching csv of each plot's
underlying data into the sibling .../<final_run_name>/csvs/ folder:

  bfactor_sensitivity_lines.png             RSCC (y) vs bfactor (x), one
                                             line per residue, using its
                                             canonical event map's full
                                             bfactor sweep
  bfactor_sensitivity_lines_normalized.png  same, but each line is shifted
                                             so its lowest-bfactor RSCC is 0
  bfactor_sensitivity_spearman_rho_hist.png histogram of each residue's
                                             canonical spearmans_rho (already
                                             written by calc_rscc_b for that
                                             residue/event-map group), one
                                             value per residue

A single dataset can have up to ~200 residues in residues_with_placer_conformers.csv,
so the two line plots have no per-residue legend - lines are colored by
their spearman rho instead (diverging colormap + colorbar), so color reads
as "how bfactor-sensitive is this residue" rather than "which residue is
this".

Run at the end of stage 7, alongside the other analysis scripts - not gated
behind -c, since it doesn't compare against the reference set.

Usage:
  plot_bfactor_sensitivity.py <run_name> <placer_run_name> <filter_run_name> \\
      <placer2_run_name> <filter2_run_name> <final_run_name> [options]
"""
from rscc_common import build_common_argparser, run_bfactor_sensitivity_plots


def main():
    args = build_common_argparser(__doc__).parse_args()
    run_bfactor_sensitivity_plots(args)


if __name__ == '__main__':
    main()
