#!/usr/bin/env python3
"""
Histogram of an EDIA score of every null conformer of one chain pocket (generate_null.sh stage
2b), from <dataset_dir>/<conformer_run_name>/chain<X>/edia_scores.csv (run_edia): by default
mean_atom_edia (the plain mean of the conformer's per-atom EDIA), or any other numeric column
(--metric, e.g. EDIAm), in 0.01 bins, conformer count per bin, x axis from the lowest value -
0.01 to the highest + 0.01. Conformers without a value (failed EDIA calls) are left out and
counted in the subtitle.

Usage:
  plot_edia_histogram.py <dataset_dir> <conformer_run_name> <chain> [--metric <column>]

Output:
  <dataset_dir>/<conformer_run_name>/chain<X>/<metric>_histogram.png (metric lower-cased, e.g.
  mean_atom_edia_histogram.png, ediam_histogram.png)
"""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from plot_ensemble_volumes import SERIES, SURFACE, TEXT_SECONDARY, add_titles, style_axes

EDIA_BIN_WIDTH = 0.01
EDIA_PAD = 0.01
DEFAULT_METRIC = 'mean_atom_edia'
METRIC_LABELS = {'mean_atom_edia': 'mean atom EDIA'}


def metric_label(metric, capital=False):
    """Display name of an edia_scores.csv column, e.g. 'mean atom EDIA', 'EDIAm'."""
    label = METRIC_LABELS.get(metric, metric)
    return label[0].upper() + label[1:] if capital else label


def read_metric(path, metric):
    """(rows, values, n_missing): the rows of an EDIA score CSV, the metric column's finite
    values as floats, and the number of rows without one (empty, e.g. a failed call, or not a
    number). Raises KeyError if the CSV has no such column."""
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        if metric not in (reader.fieldnames or []):
            raise KeyError(metric)
        rows = list(reader)
    values = []
    for r in rows:
        try:
            v = float(r[metric])
        except (TypeError, ValueError):
            continue
        if np.isfinite(v):
            values.append(v)
    return rows, np.array(values), len(rows) - len(values)


def padded_bins(values, width, pad):
    """(bin edges, x limits) for values: the axis runs from min - pad to max + pad, and the bins
    are centred on multiples of width (EDIA is reported to 0.01, so each value sits in the
    middle of its bin rather than on an edge), covering the whole axis."""
    lo, hi = float(np.min(values)) - pad, float(np.max(values)) + pad
    first = np.floor(lo / width + 0.5)
    last = np.ceil(hi / width - 0.5)
    # Rounded, so a value exactly on an edge (e.g. a percentile ending in .x5) is binned
    # consistently (upper bin) rather than by floating-point noise.
    edges = np.round((np.arange(first, last + 2) - 0.5) * width, 10)
    return edges, (lo, hi)


def plot(values, title, subtitle, out_png, ylabel='Conformers', bin_width=EDIA_BIN_WIDTH,
         pad=EDIA_PAD, xlabel='Mean atom EDIA', log_y=False):
    fig, ax = plt.subplots(figsize=(7, 4.4), facecolor=SURFACE)
    style_axes(ax)
    bins, xlim = padded_bins(values, bin_width, pad)
    ax.hist(values, bins=bins, color=SERIES, edgecolor=SURFACE, linewidth=0.5, zorder=2,
            log=log_y)
    median = float(np.median(values))
    ax.axvline(median, color=TEXT_SECONDARY, linewidth=1, linestyle='--', zorder=3)
    ax.annotate(f'median {median:.2f}', xy=(median, 1), xycoords=('data', 'axes fraction'),
                xytext=(4, -2), textcoords='offset points', ha='left', va='top', fontsize=8,
                color=TEXT_SECONDARY)
    ax.set_xlim(*xlim)
    ax.set_xlabel(xlabel, fontsize=9, color=TEXT_SECONDARY)
    ax.set_ylabel(ylabel, fontsize=9, color=TEXT_SECONDARY)
    if not log_y:
        ax.yaxis.get_major_locator().set_params(integer=True)
    add_titles(fig, title, subtitle)
    fig.savefig(out_png, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('dataset_dir')
    p.add_argument('conformer_run_name')
    p.add_argument('chain')
    p.add_argument('--metric', default=DEFAULT_METRIC,
                   help=f'edia_scores.csv column to plot. Default: {DEFAULT_METRIC}.')
    args = p.parse_args()

    dataset = Path(args.dataset_dir).name
    pocket = Path(args.dataset_dir) / args.conformer_run_name / f'chain{args.chain}'
    try:
        _, values, n_missing = read_metric(pocket / 'edia_scores.csv', args.metric)
    except KeyError:
        raise SystemExit(f'No {args.metric} column in {pocket}/edia_scores.csv (scored by an '
                         f'earlier run_edia? rerun stage 2a)')
    if not len(values):
        raise SystemExit(f'No {args.metric} values in {pocket}/edia_scores.csv')
    params_file = pocket / 'edia_params.json'
    map_name = (Path(json.loads(params_file.read_text())['map']).name
                if params_file.exists() else 'unknown map')
    subtitle = (f'{len(values)} null conformers'
                + (f' ({n_missing} without a score left out)' if n_missing else '')
                + f'; median {np.median(values):.2f}; map {map_name}; {EDIA_BIN_WIDTH:g} bins')
    out = pocket / f'{args.metric.lower()}_histogram.png'
    plot(values, f'Null conformer {metric_label(args.metric)}: {dataset} chain {args.chain} '
                 f'({args.conformer_run_name})', subtitle, out,
         xlabel=metric_label(args.metric, capital=True))
    print(f'Wrote {out}')


if __name__ == '__main__':
    main()
