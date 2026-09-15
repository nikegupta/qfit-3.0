#!/usr/bin/env python3
"""
Same comparison as plot_lig_vs_ref_filter1.py, one stage later: pooled
ligand RSCC of filter2_run_name/cluster_reps.csv's cluster-rep ligands vs
the reference set, matched by centroid distance (no RSCC computed here).

Run at the end of stage 5, only when -c (compare to reference set) is given.

Usage:
  plot_lig_vs_ref_filter2.py <run_name> <placer_run_name> <filter_run_name> \\
      <placer2_run_name> <filter2_run_name> --ref-set <dir> --graphs-dir <dir> [options]
"""
from pathlib import Path

from rscc_common import build_ref_argparser, plot_lig_vs_ref


def main():
    args = build_ref_argparser(
        __doc__,
        ['run_name', 'placer_run_name', 'filter_run_name', 'placer2_run_name', 'filter2_run_name'],
    ).parse_args()

    def run_dir_for_dataset(dataset):
        return (Path(args.datasets_dir) / dataset / args.run_name / args.placer_run_name /
                args.filter_run_name / args.placer2_run_name / args.filter2_run_name)

    plot_lig_vs_ref(
        args, run_dir_for_dataset,
        title='Ligand RSCC vs Reference, Filter 2',
        out_name='lig_vs_reference_rscc.png',
    )


if __name__ == '__main__':
    main()
