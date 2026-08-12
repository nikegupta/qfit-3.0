#!/usr/bin/env python3
"""
For every dataset independently (no pooling across datasets), plots
histograms of the cluster-representative RSCC values already written into
the 'rscc' column of that dataset's cluster_reps.csv by the filter/filter2
pipeline stages. No RSCC values are computed here - this only reads what
filter.sh/filter2.sh already wrote.

Each dataset gets its own graphs/ folder inside its own final_run_name
directory:
  cluster_reps_1.png : filter_run_name's cluster_reps.csv   (post-round-1 filter)
  cluster_reps_2.png : filter2_run_name's cluster_reps.csv  (post-round-2 filter)

The rscc values behind each histogram are also written to a matching csv
(same basename, .csv instead of .png) in the sibling csvs/ folder, indexed
by cluster_rep_index - each row's 1-based position in that cluster_reps.csv
(same position-based indexing convention used throughout the pipeline).
For cluster_reps_2.csv specifically, this index equals the LIG residue
number that row ends up as in final_model.pdb.

Usage:
  plot_cluster_reps_rscc.py <run_name> <placer_run_name> <filter_run_name> \\
      <placer2_run_name> <filter2_run_name> <final_run_name> [options]
"""
from pathlib import Path

import pandas as pd

from rscc_common import (
    build_common_argparser, dataset_graphs_dir, dataset_csvs_dir, read_datasets,
    plot_rscc_histogram, write_plot_csv,
)


def cluster_rep_rscc_values(csv_path):
    """Returns a DataFrame with columns ['cluster_rep_index', 'rscc'] - one
    row per data row of csv_path's cluster_reps.csv, with cluster_rep_index
    set to that row's 1-based position in the file (dropped rows, e.g. a
    missing rscc, keep the position numbering of the rows that remain)."""
    empty = pd.DataFrame(columns=['cluster_rep_index', 'rscc'])
    if not csv_path.exists():
        print(f'  Warning: cluster_reps.csv not found: {csv_path}')
        return empty
    df = pd.read_csv(csv_path)
    if 'rscc' not in df.columns:
        print(f'  Warning: no rscc column in {csv_path}')
        return empty
    df = df.reset_index(drop=True)
    df['cluster_rep_index'] = df.index + 1
    return df[['cluster_rep_index', 'rscc']].dropna(subset=['rscc'])


def main():
    args = build_common_argparser(__doc__).parse_args()
    datasets = read_datasets(args.datasets_file)

    for dataset in datasets:
        dataset_dir = Path(args.datasets_dir) / dataset
        graphs_dir = dataset_graphs_dir(args.datasets_dir, dataset, args)
        csvs_dir = dataset_csvs_dir(args.datasets_dir, dataset, args)

        print(f'=== {dataset}: cluster_reps_1 (filter_run_name) ===')
        csv1 = (dataset_dir / args.run_name / args.placer_run_name /
                args.filter_run_name / 'cluster_reps.csv')
        values1_df = cluster_rep_rscc_values(csv1)
        plot_rscc_histogram(
            values1_df['rscc'],
            title=f'Cluster-Rep RSCC ({dataset}, {args.filter_run_name})',
            xlabel='RSCC',
            out_path=graphs_dir / 'cluster_reps_1.png',
        )
        write_plot_csv(csvs_dir, 'cluster_reps_1.png', values1_df)

        print(f'=== {dataset}: cluster_reps_2 (filter2_run_name) ===')
        csv2 = (dataset_dir / args.run_name / args.placer_run_name / args.filter_run_name /
                args.placer2_run_name / args.filter2_run_name / 'cluster_reps.csv')
        values2_df = cluster_rep_rscc_values(csv2)
        plot_rscc_histogram(
            values2_df['rscc'],
            title=f'Cluster-Rep RSCC ({dataset}, {args.filter2_run_name})',
            xlabel='RSCC',
            out_path=graphs_dir / 'cluster_reps_2.png',
        )
        write_plot_csv(csvs_dir, 'cluster_reps_2.png', values2_df)


if __name__ == '__main__':
    main()
