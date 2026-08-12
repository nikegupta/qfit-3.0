#!/usr/bin/env python3
"""
For every dataset independently (no pooling across datasets), plots
histograms of the Z-map statistics (max_z, min_z, average_z) already written
into final_model_refined_z.csv by calc_final_refined_z, restricted to the
LIG residues actually present in final_model_refined.pdb - i.e. every
surviving ligand pose in the final model. Which residues to include is
determined by scanning final_model_refined.pdb directly for residues named
'LIG' (not inferred from any other csv). No Z-map values are computed here.

Analogous to plot_cluster_reps_rscc.py's cluster_reps_1/2 histograms: each
dataset gets its own graphs/ folder inside its own final_run_name directory:
  final_lig_max_z.png
  final_lig_min_z.png
  final_lig_average_z.png

The values behind each histogram are also written to a matching csv (same
basename, .csv instead of .png) in the sibling csvs/ folder, indexed by
resnum - each LIG residue's own residue number in final_model_refined.pdb
(the same position-based indexing convention as filter2_run_name's
cluster_reps.csv rows - see cluster_reps_2.csv).

Usage:
  plot_final_lig_z.py <run_name> <placer_run_name> <filter_run_name> \\
      <placer2_run_name> <filter2_run_name> <final_run_name> [options]
"""
from rscc_common import build_common_argparser, run_final_lig_z_histograms


def main():
    args = build_common_argparser(__doc__).parse_args()
    run_final_lig_z_histograms(args)


if __name__ == '__main__':
    main()
