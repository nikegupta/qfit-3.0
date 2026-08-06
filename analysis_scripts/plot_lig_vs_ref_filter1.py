#!/usr/bin/env python3
"""
Pooled (across every dataset in datasets.txt) ligand RSCC comparison:
filter_run_name/cluster_reps.csv's cluster-rep ligands vs the reference
set's ligands, matched by centroid distance the way qfit's
compare_lig_rscc.py matches reference/multimodel LIG instances - except no
RSCC is computed here. Both sides' RSCC values are already on disk (the
cluster_reps.csv 'rscc' column written by filter.py, and the
calc_ref_set_rscc csv for the reference structure), only geometry (LIG atom
coordinates, for centroid matching) is read fresh.

Run at the end of stage 3, only when -c (compare to reference set) is given.

Usage:
  plot_lig_vs_ref_filter1.py <run_name> <placer_run_name> <filter_run_name> \\
      --ref-set <dir> --graphs-dir <dir> [options]
"""
from pathlib import Path

from rscc_common import build_ref_argparser, plot_lig_vs_ref


def main():
    args = build_ref_argparser(
        __doc__, ['run_name', 'placer_run_name', 'filter_run_name']
    ).parse_args()

    def run_dir_for_dataset(dataset):
        return (Path(args.datasets_dir) / dataset / args.run_name /
                args.placer_run_name / args.filter_run_name)

    plot_lig_vs_ref(
        args, run_dir_for_dataset,
        title=f'Ligand RSCC vs Reference ({args.run_name}/{args.placer_run_name}/{args.filter_run_name})',
        out_name='lig_vs_reference_rscc.png',
    )


if __name__ == '__main__':
    main()
