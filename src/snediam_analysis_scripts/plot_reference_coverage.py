#!/usr/bin/env python3
"""
Bar chart of how much of each modeled (reference) ligand lies inside its chain's null ensemble
volume, from generate_null.sh stage 1 (generate_conformers): one data point per reference ligand
(every LIG residue of the chain), pooled across every chain pocket (chain<X>) of every dataset in
datasets.txt.

Coverage is reference_ligands[].percent_inside in each pocket's conformer_summary.json (read from
<dataset>/<conformer_run_name>/chain<X>/): the share of the reference ligand's vdW volume,
sampled on the mask's grid, that lies inside ensemble_mask.ccp4.

Usage:
  plot_reference_coverage.py <conformer_run_name> --datasets-dir <dir> --datasets-file <file>
                             --graphs-dir <dir>

Output:
  <graphs-dir>/reference_coverage.png
  <graphs-dir>/reference_coverage.csv   (one row per reference ligand - the plotted data)
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from plot_ensemble_volumes import (SERIES, SURFACE, TEXT_SECONDARY, add_titles,
                                   collect_summaries, read_datasets, style_axes, write_csv)

# Coverage bins: (label, lower bound inclusive, upper bound exclusive). A whole ligand inside the
# volume gets its own bar.
BINS = [
    ('<50%', 0, 50),
    ('50–59%', 50, 60),
    ('60–69%', 60, 70),
    ('70–79%', 70, 80),
    ('80–89%', 80, 90),
    ('90–99%', 90, 100),
    ('100%', 100, np.inf),
]
CSV_FIELDS = ['dataset', 'pocket', 'chain', 'resi', 'n_heavy_atoms', 'centroid_to_center_A',
              'conformers_kept', 'volume_A3', 'percent_inside']


def plot(rows, run_name, out_png):
    percents = np.array([r['percent_inside'] for r in rows])
    labels = [label for label, _, _ in BINS]
    counts = [int(((percents >= lo) & (percents < hi)).sum()) for _, lo, hi in BINS]
    n_datasets = len({r['dataset'] for r in rows})

    fig, ax = plt.subplots(figsize=(7, 4.4), facecolor=SURFACE)
    style_axes(ax)
    x = np.arange(len(BINS))
    ax.bar(x, counts, width=0.8, color=SERIES, edgecolor=SURFACE, linewidth=1.5, zorder=2)
    for xi, c in zip(x, counts):
        if c:
            ax.annotate(str(c), xy=(xi, c), xytext=(0, 3), textcoords='offset points',
                        ha='center', va='bottom', fontsize=8, color=TEXT_SECONDARY)
    ax.set_xticks(x, labels)
    ax.set_xlabel('Reference ligand volume inside the ensemble volume', fontsize=9,
                  color=TEXT_SECONDARY)
    ax.set_ylabel('Reference ligands', fontsize=9, color=TEXT_SECONDARY)
    ax.yaxis.get_major_locator().set_params(integer=True)
    add_titles(fig, f'Reference ligand inside its null ensemble volume ({run_name})',
               f'{counts[-1]} of {len(rows)} reference ligands across {n_datasets} datasets are '
               f'fully inside; lowest {percents.min():.1f}%')
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
        for ref in s['reference_ligands']:
            rows.append({'dataset': dataset, 'pocket': pocket, 'chain': s['chain'],
                         'resi': ref['resi'], 'n_heavy_atoms': ref['n_heavy_atoms'],
                         'centroid_to_center_A': ref['centroid_to_center_A'],
                         'conformers_kept': s['conformers_kept'],
                         'volume_A3': mask['volume_A3'],
                         'percent_inside': ref['percent_inside']})
    if not rows:
        raise SystemExit(f'No ensemble masks found for {args.conformer_run_name}')

    out_dir = Path(args.graphs_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / 'reference_coverage.csv', rows, CSV_FIELDS)
    plot(rows, args.conformer_run_name, out_dir / 'reference_coverage.png')
    print(f'Wrote {out_dir / "reference_coverage.png"} and reference_coverage.csv '
          f'({len(rows)} reference ligands)')


if __name__ == '__main__':
    main()
