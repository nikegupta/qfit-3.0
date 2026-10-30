#!/usr/bin/env python3
"""
Scatter plot of conformer acceptance rate against ligand heavy-atom count from generate_null.sh
stage 1 (generate_conformers): one point per chain pocket ensemble, pooled across every pocket
folder (chain<X>) of every dataset in datasets.txt. Acceptance is the share of candidate
placements that didn't clash (conformers_kept / candidates_tried in each pocket's
conformer_summary.json, read from <dataset>/<conformer_run_name>/chain<X>/); heavy atoms are the
template ligand's. Ensembles that stopped at --max-candidates before reaching --n-conformers are
drawn as open circles.

Usage:
  plot_conformer_acceptance.py <conformer_run_name> --datasets-dir <dir> --datasets-file <file>
                               --graphs-dir <dir>

Output:
  <graphs-dir>/conformer_acceptance.png
  <graphs-dir>/conformer_acceptance.csv   (one row per pocket ensemble - the plotted data)
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from plot_ensemble_volumes import (GRID, SERIES, SURFACE, TEXT_SECONDARY, add_titles,
                                   collect_summaries, read_datasets, style_axes, write_csv)

CSV_FIELDS = ['dataset', 'pocket', 'n_heavy_atoms', 'candidates_tried', 'conformers_kept',
              'acceptance', 'complete']
# Horizontal spread (heavy atoms) so ensembles with the same atom count don't sit on one line.
JITTER = 0.15


def plot(rows, run_name, out_png):
    rng = np.random.default_rng(0)  # fixed, so replots look the same
    atoms = np.array([r['n_heavy_atoms'] for r in rows], dtype=float)
    acceptance = np.array([r['acceptance'] for r in rows])
    complete = np.array([r['complete'] for r in rows])
    x = atoms + rng.uniform(-JITTER, JITTER, size=len(rows))
    n_datasets = len({r['dataset'] for r in rows})

    fig, ax = plt.subplots(figsize=(7, 4.4), facecolor=SURFACE)
    style_axes(ax)
    ax.grid(axis='x', color=GRID, linewidth=0.8, zorder=0)
    ax.scatter(x[complete], 100 * acceptance[complete], s=40, color=SERIES,
               edgecolor=SURFACE, linewidth=1.2, zorder=3, label='Reached --n-conformers')
    if (~complete).any():
        ax.scatter(x[~complete], 100 * acceptance[~complete], s=40, facecolor=SURFACE,
                   edgecolor=SERIES, linewidth=1.5, zorder=3,
                   label='Stopped at --max-candidates')
        ax.legend(frameon=False, fontsize=8, labelcolor=TEXT_SECONDARY, loc='upper right')
    ax.set_yscale('log')
    ax.set_xticks(np.arange(int(atoms.min()), int(atoms.max()) + 1))
    ax.set_xlabel('Ligand heavy atoms', fontsize=9, color=TEXT_SECONDARY)
    ax.set_ylabel('Acceptance (% of candidates kept)', fontsize=9, color=TEXT_SECONDARY)
    median = 100 * float(np.median(acceptance))
    add_titles(fig, f'Null conformer acceptance by ligand size ({run_name})',
               f'{len(rows)} pocket ensembles across {n_datasets} datasets; median acceptance '
               f'{median:.3g}%')
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
        if not s['candidates_tried']:
            continue
        rows.append({'dataset': dataset, 'pocket': pocket,
                     'n_heavy_atoms': s['ligand']['n_heavy_atoms'],
                     'candidates_tried': s['candidates_tried'],
                     'conformers_kept': s['conformers_kept'],
                     'acceptance': s['acceptance'], 'complete': s['complete']})
    # Log axis: an ensemble with nothing kept can't be drawn; it's still in the CSV.
    plotted = [r for r in rows if r['acceptance'] > 0]
    for r in rows:
        if r['acceptance'] == 0:
            print(f'  Warning: {r["dataset"]} {r["pocket"]} kept no conformers; not plotted')
    if not plotted:
        raise SystemExit(f'No conformer summaries with accepted conformers for '
                         f'{args.conformer_run_name}')

    out_dir = Path(args.graphs_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / 'conformer_acceptance.csv', rows, CSV_FIELDS)
    plot(plotted, args.conformer_run_name, out_dir / 'conformer_acceptance.png')
    print(f'Wrote {out_dir / "conformer_acceptance.png"} and conformer_acceptance.csv '
          f'({len(plotted)} pocket ensembles plotted)')


if __name__ == '__main__':
    main()
