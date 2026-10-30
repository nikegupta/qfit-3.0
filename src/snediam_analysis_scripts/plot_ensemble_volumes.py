#!/usr/bin/env python3
"""
Histogram of ensemble mask volumes from generate_null.sh stage 1 (generate_conformers): one data
point per chain pocket ensemble, pooled across every pocket folder (chain<X>) of every dataset in
datasets.txt. The volume is ensemble_mask.volume_A3 in each pocket's conformer_summary.json: the
union of the vdW volumes of every kept placement, read from
<dataset>/<conformer_run_name>/chain<X>/.

Usage:
  plot_ensemble_volumes.py <conformer_run_name> --datasets-dir <dir> --datasets-file <file>
                           --graphs-dir <dir>

Output:
  <graphs-dir>/ensemble_volumes.png
  <graphs-dir>/ensemble_volumes.csv   (one row per pocket ensemble - the plotted data)
"""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Shared chart style (also used by plot_conformer_acceptance.py).
SURFACE = '#fcfcfb'
TEXT_PRIMARY = '#0b0b0b'
TEXT_SECONDARY = '#52514e'
GRID = '#e4e3df'
SERIES = '#2a78d6'

CSV_FIELDS = ['dataset', 'pocket', 'n_heavy_atoms', 'conformers_kept', 'complete',
              'volume_A3']


def read_datasets(datasets_file):
    with open(datasets_file) as f:
        return [line.strip() for line in f if line.strip()]


def collect_summaries(datasets_dir, datasets, run_name):
    """[(dataset, pocket folder name, summary dict)] for every pocket ensemble (chain<X>) of
    every dataset."""
    out = []
    for dataset in datasets:
        paths = sorted((Path(datasets_dir) / dataset / run_name).glob(
            '*/conformer_summary.json'))
        if not paths:
            print(f'  Warning: no {run_name} conformer summaries found for {dataset}')
        for path in paths:
            out.append((dataset, path.parent.name, json.loads(path.read_text())))
    return out


def style_axes(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(axis='y', color=GRID, linewidth=0.8, zorder=0)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=8, length=0)
    for side in ('top', 'right', 'left'):
        ax.spines[side].set_visible(False)
    ax.spines['bottom'].set_color(GRID)


def add_titles(fig, title, subtitle):
    fig.suptitle(title, x=0.01, y=0.99, ha='left', fontsize=13, color=TEXT_PRIMARY)
    fig.text(0.01, 0.90, subtitle, ha='left', fontsize=9, color=TEXT_SECONDARY)
    fig.tight_layout(rect=(0, 0, 1, 0.88))


def write_csv(path, rows, fields):
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def plot(rows, run_name, out_png):
    volumes = np.array([r['volume_A3'] for r in rows])
    median = float(np.median(volumes))
    n_datasets = len({r['dataset'] for r in rows})

    fig, ax = plt.subplots(figsize=(7, 4.4), facecolor=SURFACE)
    style_axes(ax)
    ax.hist(volumes, bins='auto', color=SERIES, edgecolor=SURFACE, linewidth=1.5, zorder=2)
    ax.axvline(median, color=TEXT_SECONDARY, linewidth=1, linestyle='--', zorder=3)
    ax.annotate(f'median {median:,.0f} Å³', xy=(median, 1), xycoords=('data', 'axes fraction'),
                xytext=(4, -2), textcoords='offset points', ha='left', va='top', fontsize=8,
                color=TEXT_SECONDARY)
    ax.set_xlabel('Ensemble mask volume (Å³)', fontsize=9, color=TEXT_SECONDARY)
    ax.set_ylabel('Pocket ensembles', fontsize=9, color=TEXT_SECONDARY)
    ax.yaxis.get_major_locator().set_params(integer=True)
    add_titles(fig, f'Volume sampled by the null ensemble ({run_name})',
               f'{len(rows)} pocket ensembles across {n_datasets} datasets; range '
               f'{volumes.min():,.0f}–{volumes.max():,.0f} Å³')
    fig.savefig(out_png, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('conformer_run_name')
    parser.add_argument('--datasets-dir', required=True,
                        help='Root directory containing per-dataset folders')
    parser.add_argument('--datasets-file', required=True,
                        help='Path to newline-delimited list of dataset names')
    parser.add_argument('--graphs-dir', required=True,
                        help='Output directory for the pooled (cross-dataset) plot')
    args = parser.parse_args()

    rows = []
    for dataset, pocket, s in collect_summaries(args.datasets_dir,
                                                read_datasets(args.datasets_file),
                                                args.conformer_run_name):
        mask = s.get('ensemble_mask')
        if not mask:
            print(f'  Warning: {dataset} {pocket} has no ensemble mask (no conformers kept)')
            continue
        rows.append({'dataset': dataset, 'pocket': pocket,
                     'n_heavy_atoms': s['ligand']['n_heavy_atoms'],
                     'conformers_kept': s['conformers_kept'], 'complete': s['complete'],
                     'volume_A3': mask['volume_A3']})
    if not rows:
        raise SystemExit(f'No ensemble masks found for {args.conformer_run_name}')

    out_dir = Path(args.graphs_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / 'ensemble_volumes.csv', rows, CSV_FIELDS)
    plot(rows, args.conformer_run_name, out_dir / 'ensemble_volumes.png')
    print(f'Wrote {out_dir / "ensemble_volumes.png"} and ensemble_volumes.csv '
          f'({len(rows)} pocket ensembles)')


if __name__ == '__main__':
    main()
