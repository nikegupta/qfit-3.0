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

When --cell-lookup-file is given, ALSO writes a crystal-symmetry-aware twin
(lig_vs_reference_rscc_sa.png/.csv/etc.) - see rscc_common.plot_lig_vs_ref's cell_lookup
argument.

Usage:
  plot_lig_vs_ref_filter1.py <run_name> <placer_run_name> <filter_run_name> \\
      --ref-set <dir> --graphs-dir <dir> [--cell-lookup-file <path>] [options]
"""
from pathlib import Path

from rscc_common import build_ref_argparser, plot_lig_vs_ref, read_cell_lookup


def main():
    parser = build_ref_argparser(
        __doc__, ['run_name', 'placer_run_name', 'filter_run_name']
    )
    parser.add_argument(
        '--cell-lookup-file', default=None,
        help='Optional path to a "dataset a b c alpha beta gamma space_group" lookup file '
             '(the same format/file program.sh\'s DESPOT_CELL_LOOKUP_FILE already builds from '
             'CSV_FILE). When given, also writes a crystal-symmetry-aware twin of every output.',
    )
    args = parser.parse_args()

    def run_dir_for_dataset(dataset):
        return (Path(args.datasets_dir) / dataset / args.run_name /
                args.placer_run_name / args.filter_run_name)

    cell_lookup = read_cell_lookup(args.cell_lookup_file) if args.cell_lookup_file else None

    plot_lig_vs_ref(
        args, run_dir_for_dataset,
        title='Filter 1 vs Reference RSCC',
        out_name='lig_vs_reference_rscc.png',
        cell_lookup=cell_lookup,
    )


if __name__ == '__main__':
    main()
