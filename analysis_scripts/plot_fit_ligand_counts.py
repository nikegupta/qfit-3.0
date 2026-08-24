#!/usr/bin/env python3
"""
Histogram of the number of fit_ligand output poses per dataset: one data
point per dataset in datasets.txt, counting the data rows (one per
output_pdb) in that dataset's <run_name>/fit_ligand_manifest.csv.

Doesn't compare against the reference set, so it runs unconditionally at
the end of stage 1 (fit_ligand) - not gated behind -c.

Usage:
  plot_fit_ligand_counts.py <run_name> --graphs-dir <dir> [options]

Output:
  <graphs-dir>/fit_ligand_counts.png
"""
import argparse
import csv
from pathlib import Path

import pandas as pd

from rscc_common import (
    DEFAULT_DATASETS_DIR, DEFAULT_DATASETS_FILE, read_datasets, plot_count_histogram,
    write_plot_csv,
)


def count_manifest_rows(manifest_path):
    with open(manifest_path, newline='') as f:
        reader = csv.reader(f)
        next(reader, None)  # header
        return sum(1 for row in reader if row)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_name')
    parser.add_argument('--datasets-dir', default=DEFAULT_DATASETS_DIR,
                         help='Root directory containing per-dataset folders')
    parser.add_argument('--datasets-file', default=DEFAULT_DATASETS_FILE,
                         help='Path to newline-delimited list of dataset names')
    parser.add_argument('--graphs-dir', required=True,
                         help='Output directory for the pooled (cross-dataset) plot')
    args = parser.parse_args()

    datasets = read_datasets(args.datasets_file)
    rows = []
    for dataset in datasets:
        manifest_path = Path(args.datasets_dir) / dataset / args.run_name / 'fit_ligand_manifest.csv'
        if not manifest_path.exists():
            print(f'  Warning: manifest not found for {dataset}: {manifest_path}')
            continue
        n = count_manifest_rows(manifest_path)
        print(f'  {dataset}: {n} fit_ligand output pose(s)')
        rows.append({'dataset': dataset, 'count': n})

    out_dir = Path(args.graphs_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_name = 'fit_ligand_counts.png'
    plot_count_histogram(
        [row['count'] for row in rows],
        title=f'Fit-Ligand Output Poses per Dataset ({args.run_name})',
        xlabel='Number of fit_ligand Output Poses',
        out_path=out_dir / out_name,
    )
    if rows:
        write_plot_csv(out_dir, out_name, pd.DataFrame(rows))


if __name__ == '__main__':
    main()
