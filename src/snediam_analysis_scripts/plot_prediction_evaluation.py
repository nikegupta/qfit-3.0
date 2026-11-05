#!/usr/bin/env python3
"""
Percentile of every predicted structure's ligand in its matched reference ligand's null
distribution (evaluate_structures.sh stage 3), pooled across every dataset in the datasets
file.

For each prediction folder <test-dir>/<dataset>/<distribution-run>/<run_name>/<prediction>/:
  match.txt, match.json             the matched reference ligand (match_ligand, stage 1)
  prediction_residue_scores.csv     the prediction ligand's scores (run_edia, stage 2)
and the matched null, <distribution-dir>/<dataset>/<distribution-run>/<match>/
(prep_distribution.sh):
  distribution_residue_scores.csv   the null conformers' scores
  reference_residue_scores.csv      the reference ligand's scores
The score ranked is --metric (default mean_atom_edia, the plain mean of the per-atom EDIA):
percentile = 100 * (n_below + 0.5 * n_equal) / n over the null conformers with a score, as in
plot_edia_evaluation.py. The reference ligand's percentile in the same null is reported
alongside.

Outputs, in <graphs-dir>:
  evaluation.csv   one row per prediction (dataset, prediction, match, centroid distance, CA
                   fit, the prediction's and the reference's score and percentile, the null's
                   size and median, status). A prediction not matched or not scored is listed
                   with its status (no_match, no_scores, failed, no_null) rather than dropped.
  prediction_<m>_percentile.png, .csv   histogram of the scored predictions' percentiles
                   (1-point bins, log count scale), <m> the metric lower-cased.
  prediction_vs_reference_<m>.png, .csv   scatter: each scored prediction's score (y) against
                   its matched reference ligand's (x; reference_residue_scores.csv of the
                   matched null), with the y = x line.
  prediction_<m>_by_reference.png, .csv   one category per matched reference ligand
                   (<dataset> / <match>, those matched by at least one scored prediction): a
                   violin of that ligand's null distribution, its predictions' scores as dots
                   and the reference ligand's own score as a diamond. The CSV has one row per
                   plotted value (kind null, prediction or reference).

Usage:
  plot_prediction_evaluation.py <run_name> --test-dir <dir> --distribution-dir <dir>
                                --distribution-run <name> --datasets-file <file>
                                --graphs-dir <dir> [--metric <column>]
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from plot_edia_evaluation import PERCENTILE_BIN_WIDTH, PERCENTILE_PAD, percentile_of
from plot_edia_histogram import DEFAULT_METRIC, metric_label, plot, read_metric
from plot_ensemble_volumes import (GRID, SERIES, SURFACE, TEXT_PRIMARY, TEXT_SECONDARY,
                                   add_titles, read_datasets, style_axes, write_csv)

PREDICTION_CSV = 'prediction_residue_scores.csv'
# Null violins: a recessive neutral fill, so the predictions (SERIES) and the reference ligand
# (TEXT_PRIMARY, diamond) read on top of it.
NULL_FILL = '#d9d7d1'
NULL_EDGE = '#a9a7a1'
MARKER_SIZE = 40  # points^2: ~8 px markers


def plot_vs_reference(rows, metric, run, out_png):
    """Scatter of prediction score (y) against matched reference score (x), with y = x."""
    label = metric_label(metric, capital=True)
    x = np.array([r[f'reference_{metric}'] for r in rows], dtype=float)
    y = np.array([r[f'prediction_{metric}'] for r in rows], dtype=float)
    lo = float(min(x.min(), y.min())) - 0.03
    hi = float(max(x.max(), y.max())) + 0.03
    fig, ax = plt.subplots(figsize=(7.5, 7.2), facecolor=SURFACE)
    style_axes(ax)
    ax.grid(axis='x', color=GRID, linewidth=0.8, zorder=0)
    ax.plot([lo, hi], [lo, hi], color=TEXT_SECONDARY, linewidth=1, linestyle='--', zorder=1)
    ax.annotate('y = x', xy=(hi, hi), xytext=(-14, -4), textcoords='offset points',
                ha='right', va='top', fontsize=8, color=TEXT_SECONDARY)
    ax.scatter(x, y, s=MARKER_SIZE, color=SERIES, edgecolor=SURFACE, linewidth=1.5, zorder=3)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect('equal')
    ax.set_xlabel(f'Matched reference ligand {metric_label(metric)}', fontsize=9,
                  color=TEXT_SECONDARY)
    ax.set_ylabel(f'Predicted ligand {metric_label(metric)}', fontsize=9, color=TEXT_SECONDARY)
    n_above = int((y > x).sum())
    add_titles(fig, f'Predicted vs matched reference {metric_label(metric)} ({run})',
               f'{len(rows)} predictions across {len({r["dataset"] for r in rows})} datasets; '
               f'{n_above} above the reference, {len(rows) - n_above} at or below')
    fig.savefig(out_png, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_by_reference(categories, metric, run, out_png):
    """Per matched reference ligand: violin of its null, its predictions (dots) and the
    reference's own score (diamond). categories: [(dataset, match, null values, prediction
    values, reference value or None)]."""
    label = metric_label(metric)
    n = len(categories)
    fig, ax = plt.subplots(figsize=(max(7, 1.1 * n + 2), 5), facecolor=SURFACE)
    style_axes(ax)
    positions = np.arange(1, n + 1)
    nulls = [np.asarray(c[2], dtype=float) for c in categories]
    # A violin needs spread (its density estimate fails on identical values); those nulls get
    # only their median line.
    spread = [i for i, v in enumerate(nulls) if len(v) > 1 and np.ptp(v) > 0]
    parts = (ax.violinplot([nulls[i] for i in spread], positions=positions[spread], widths=0.8,
                           showextrema=False) if spread else {'bodies': []})
    for body in parts['bodies']:
        body.set_facecolor(NULL_FILL)
        body.set_edgecolor(NULL_EDGE)
        body.set_linewidth(0.8)
        body.set_alpha(1)
        body.set_zorder(1)
    for pos, null in zip(positions, nulls):
        median = float(np.median(null))
        ax.plot([pos - 0.18, pos + 0.18], [median, median], color=NULL_EDGE, linewidth=2,
                solid_capstyle='round', zorder=2)
    for pos, (_, _, _, preds, ref) in zip(positions, categories):
        preds = np.asarray(preds, dtype=float)
        # Deterministic spread so equal-ish scores stay visible.
        offsets = (np.linspace(-0.12, 0.12, len(preds)) if len(preds) > 1 else np.zeros(1))
        ax.scatter(pos + offsets, preds, s=MARKER_SIZE, color=SERIES, edgecolor=SURFACE,
                   linewidth=1.5, zorder=4)
        if ref is not None:
            ax.scatter([pos + 0.28], [ref], s=MARKER_SIZE + 10, marker='D', color=TEXT_PRIMARY,
                       edgecolor=SURFACE, linewidth=1.5, zorder=4)
    ax.set_xticks(positions)
    ax.set_xticklabels([f'{d}\n{m}' for d, m, *_ in categories], fontsize=8,
                       color=TEXT_SECONDARY, rotation=0 if n <= 8 else 45,
                       ha='center' if n <= 8 else 'right')
    ax.set_xlim(0.4, n + 0.6)
    ax.set_ylabel(f'{label[0].upper()}{label[1:]}', fontsize=9, color=TEXT_SECONDARY)
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=NULL_FILL, edgecolor=NULL_EDGE,
                             label='Null distribution (line: median)'),
               plt.Line2D([], [], marker='o', linestyle='', color=SERIES,
                          markeredgecolor=SURFACE, markersize=7, label='Predictions'),
               plt.Line2D([], [], marker='D', linestyle='', color=TEXT_PRIMARY,
                          markeredgecolor=SURFACE, markersize=6, label='Reference ligand')]
    legend = ax.legend(handles=handles, loc='upper left', bbox_to_anchor=(0, 1.02), ncol=3,
                       frameon=False, fontsize=8, handletextpad=0.4, columnspacing=1.2)
    for text in legend.get_texts():
        text.set_color(TEXT_SECONDARY)
    n_preds = sum(len(c[3]) for c in categories)
    add_titles(fig, f'Predicted ligand {label} by matched reference ligand ({run})',
               f'{n} reference ligands across {len({c[0] for c in categories})} datasets; '
               f'{n_preds} predictions; violins: each ligand\'s null conformers')
    fig.savefig(out_png, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def first_score(path, metric):
    """(score or None, status) of the first row of a residue score CSV."""
    if not path.exists():
        return None, 'no_scores'
    try:
        rows, values, _ = read_metric(path, metric)
    except KeyError:
        return None, f'no_{metric}_column'
    if not rows:
        return None, 'no_scores'
    if not len(values):
        return None, rows[0].get('status') or 'failed'
    return float(values[0]), 'ok'


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('run_name')
    p.add_argument('--test-dir', required=True,
                   help='Folder of per-dataset test structure folders '
                        '(<dataset>/<distribution-run>/<run_name>/).')
    p.add_argument('--distribution-dir', required=True,
                   help='prep_distribution.sh output: <dataset>/<run>/<match>/ null CSVs.')
    p.add_argument('--distribution-run', required=True,
                   help='The generate_null.sh run collected there (prep_distribution.sh).')
    p.add_argument('--datasets-file', required=True)
    p.add_argument('--graphs-dir', required=True)
    p.add_argument('--metric', default=DEFAULT_METRIC,
                   help=f'Score column ranked. Default: {DEFAULT_METRIC}.')
    args = p.parse_args()
    run, metric = args.run_name, args.metric
    label, slug = metric_label(metric), metric.lower()

    rows, nulls = [], {}
    for dataset in read_datasets(args.datasets_file):
        run_dir = Path(args.test_dir) / dataset / args.distribution_run / run
        folders = sorted(d for d in run_dir.iterdir() if d.is_dir()) if run_dir.is_dir() else []
        if not folders:
            print(f'  Warning: no {run} prediction folders for {dataset} ({run_dir})')
        for folder in folders:
            row = {'dataset': dataset, 'prediction': folder.name, 'match': '',
                   'centroid_distance_A': '', 'n_ca': '', 'ca_rmsd_A': '',
                   f'prediction_{metric}': '', 'percentile': '', 'n_conformers': '',
                   f'null_median_{metric}': '', f'reference_{metric}': '',
                   'reference_percentile': '', 'status': ''}
            rows.append(row)
            if not (folder / 'match.txt').exists():
                row['status'] = 'no_match'
                continue
            row['match'] = (folder / 'match.txt').read_text().strip()
            if (folder / 'match.json').exists():
                best = json.loads((folder / 'match.json').read_text())['match']
                row.update({k: best[k] for k in ('centroid_distance_A', 'n_ca', 'ca_rmsd_A')})
            score, status = first_score(folder / PREDICTION_CSV, metric)
            null_dir = (Path(args.distribution_dir) / dataset / args.distribution_run
                        / row['match'])
            key = (dataset, row['match'])
            if key not in nulls:
                null_csv = null_dir / 'distribution_residue_scores.csv'
                try:
                    nulls[key] = read_metric(null_csv, metric)[1] if null_csv.exists() else None
                except KeyError:
                    nulls[key] = None
            null = nulls[key]
            if null is None or not len(null):
                row['status'] = 'no_null'
                continue
            row['n_conformers'] = len(null)
            row[f'null_median_{metric}'] = round(float(np.median(null)), 4)
            ref, _ = first_score(null_dir / 'reference_residue_scores.csv', metric)
            if ref is not None:
                row[f'reference_{metric}'] = ref
                row['reference_percentile'] = round(percentile_of(ref, null), 2)
            row['status'] = status
            if score is None:
                continue
            row[f'prediction_{metric}'] = score
            row['percentile'] = round(percentile_of(score, null), 2)

    if not rows:
        raise SystemExit(f'No {run} prediction folders found')
    out = Path(args.graphs_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / 'evaluation.csv', rows, list(rows[0]))
    scored = [r for r in rows if r['percentile'] != '']
    counts = {}
    for r in rows:
        counts[r['status']] = counts.get(r['status'], 0) + 1
    print(f'Wrote {out / "evaluation.csv"}: {len(rows)} predictions ({counts})')
    if not scored:
        print('  No scored predictions; no histogram written')
        return
    values = np.array([r['percentile'] for r in scored])
    name = f'prediction_{slug}_percentile'
    write_csv(out / f'{name}.csv', scored,
              ['dataset', 'prediction', 'match', f'prediction_{metric}', 'percentile',
               'n_conformers', f'null_median_{metric}'])
    n_ds = len({r['dataset'] for r in scored})
    plot(values,
         f'Predicted ligand {label} percentile in the matched null distribution ({run})',
         f'{len(values)} predictions across {n_ds} datasets; median {np.median(values):.1f}; '
         f'{int((values >= 95).sum())} at or above the 95th percentile; '
         f'{PERCENTILE_BIN_WIDTH:g}-point bins; log count scale',
         out / f'{name}.png', ylabel='Predictions', bin_width=PERCENTILE_BIN_WIDTH,
         pad=PERCENTILE_PAD, xlabel=f'Percentile of the null conformer {label}', log_y=True)
    print(f'Wrote {name}.png (and CSV) to {out}')

    # Prediction vs matched reference ligand.
    paired = [r for r in scored if r[f'reference_{metric}'] != '']
    if paired:
        name = f'prediction_vs_reference_{slug}'
        write_csv(out / f'{name}.csv', paired,
                  ['dataset', 'prediction', 'match', f'reference_{metric}',
                   f'prediction_{metric}'])
        plot_vs_reference(paired, metric, run, out / f'{name}.png')
        print(f'Wrote {name}.png (and CSV) to {out}')
    else:
        print('  No prediction with a reference ligand score; no prediction vs reference plot')

    # By matched reference ligand: its null (violin), its predictions, the reference.
    groups = {}
    for r in scored:
        groups.setdefault((r['dataset'], r['match']), []).append(r)
    categories, plotted = [], []
    for (dataset, match_label), members in sorted(groups.items()):
        null = nulls[(dataset, match_label)]
        ref = members[0][f'reference_{metric}']
        ref = None if ref == '' else float(ref)
        preds = [m[f'prediction_{metric}'] for m in members]
        categories.append((dataset, match_label, null, preds, ref))
        base = {'dataset': dataset, 'match': match_label}
        plotted += [{**base, 'kind': 'null', 'prediction': '', metric: round(float(v), 4)}
                    for v in null]
        plotted += [{**base, 'kind': 'prediction', 'prediction': m['prediction'],
                     metric: m[f'prediction_{metric}']} for m in members]
        if ref is not None:
            plotted.append({**base, 'kind': 'reference', 'prediction': '', metric: ref})
    name = f'prediction_{slug}_by_reference'
    write_csv(out / f'{name}.csv', plotted, ['dataset', 'match', 'kind', 'prediction', metric])
    plot_by_reference(categories, metric, run, out / f'{name}.png')
    print(f'Wrote {name}.png (and CSV) to {out}')


if __name__ == '__main__':
    main()
