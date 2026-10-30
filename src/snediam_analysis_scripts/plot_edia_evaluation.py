#!/usr/bin/env python3
"""
Evaluation of the deposited ligands against their null distributions (generate_null.sh stage 3),
pooled across every dataset in datasets.txt. Four separate histograms, each with the plotted
data as a CSV. Everything is read from the run's pockets, <dataset>/<conformer_run_name>/
chain<X>/ (run_edia, stage 2a), where the reference ligands were scored with the same map and
settings as the pocket's null conformers. The score plotted is --metric, a column of
edia_scores.csv and reference_edia.csv: by default mean_atom_edia (the plain mean of a
molecule's per-atom EDIA), or e.g. EDIAm. With <m> the metric lower-cased:

  structure_<m>             The score of every reference ligand (each LIG residue of the
                            dataset's structure, one per altloc): reference_edia.csv. One point
                            per reference.
  null_median_<m>           Median score of the null conformers of each chain pocket:
                            edia_scores.csv (references not included). One point per pocket.
  null_mean_<m>             The same, mean score.
  structure_<m>_percentile  Percentile of each reference's score in the null distribution of
                            its pocket: the percentage of null conformers scoring below it, ties
                            counted as half (100 * (n_below + 0.5 * n_equal) / n). One point per
                            reference.

Score histograms use 0.01 bins on an x axis from the lowest value - 0.01 to the highest + 0.01
(as the per-pocket histograms, plot_edia_histogram.py); the percentile histogram uses 1-point
bins from the lowest percentile - 1 to the highest + 1, with a log count axis. Molecules
without a score (failed EDIA calls) are left out and counted; a pocket without the metric's
column (scored by an earlier run_edia) is warned about and left out, and a pocket without
reference_edia.csv contributes only to the null plots.

Usage:
  plot_edia_evaluation.py <conformer_run_name> --datasets-dir <dir> --datasets-file <file>
                          --graphs-dir <dir> [--metric <column>]

Output (in <graphs-dir>), e.g. for the default metric:
  structure_mean_atom_edia.png, structure_mean_atom_edia.csv
  null_median_mean_atom_edia.png, null_median_mean_atom_edia.csv
  null_mean_mean_atom_edia.png, null_mean_mean_atom_edia.csv
  structure_mean_atom_edia_percentile.png, structure_mean_atom_edia_percentile.csv
"""
import argparse
import json
from pathlib import Path

import numpy as np

from plot_edia_histogram import DEFAULT_METRIC, EDIA_BIN_WIDTH, metric_label, plot, read_metric
from plot_ensemble_volumes import read_datasets, write_csv

PERCENTILE_BIN_WIDTH = 1.0
PERCENTILE_PAD = 1.0


def map_name(params_file):
    if not params_file.exists():
        return ''
    return Path(json.loads(params_file.read_text()).get('map', '')).name


def percentile_of(value, null):
    return 100 * ((null < value).sum() + 0.5 * (null == value).sum()) / len(null)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('conformer_run_name')
    parser.add_argument('--datasets-dir', required=True,
                        help='Root directory containing per-dataset folders')
    parser.add_argument('--datasets-file', required=True,
                        help='Path to newline-delimited list of dataset names')
    parser.add_argument('--graphs-dir', required=True,
                        help='Output directory for the pooled (cross-dataset) plots')
    parser.add_argument('--metric', default=DEFAULT_METRIC,
                        help=f'edia_scores.csv / reference_edia.csv column to evaluate. '
                             f'Default: {DEFAULT_METRIC}.')
    args = parser.parse_args()
    run, metric = args.conformer_run_name, args.metric
    label, slug = metric_label(metric), metric.lower()

    structure_rows, pocket_rows, percentile_rows = [], [], []
    n_null_missing = n_reference_missing = 0
    for dataset in read_datasets(args.datasets_file):
        pockets = sorted(p.parent for p in
                         (Path(args.datasets_dir) / dataset / run).glob('chain*/edia_scores.csv'))
        if not pockets:
            print(f'  Warning: no {run} chain*/edia_scores.csv found for {dataset}')
        for pocket in pockets:
            chain = pocket.name[len('chain'):]
            density = map_name(pocket / 'edia_params.json')

            # Null distribution.
            try:
                _, null, missing = read_metric(pocket / 'edia_scores.csv', metric)
            except KeyError:
                print(f'  Warning: no {metric} column in {pocket}/edia_scores.csv (rerun '
                      f'stage 2a); left out')
                continue
            n_null_missing += missing
            if not len(null):
                print(f'  Warning: no {metric} values in {pocket}/edia_scores.csv')
                continue
            pocket_rows.append({'dataset': dataset, 'pocket': pocket.name, 'chain': chain,
                                'n_conformers': len(null), 'n_without_score': missing,
                                f'median_{metric}': round(float(np.median(null)), 4),
                                f'mean_{metric}': round(float(np.mean(null)), 4),
                                'map': density})

            # Reference ligands, scored with the same map and settings.
            if not (pocket / 'reference_edia.csv').exists():
                print(f'  Warning: no reference_edia.csv in {pocket} (rerun stage 2a)')
                continue
            try:
                rows, _, _ = read_metric(pocket / 'reference_edia.csv', metric)
            except KeyError:
                print(f'  Warning: no {metric} column in {pocket}/reference_edia.csv (rerun '
                      f'stage 2a)')
                continue
            for r in rows:
                ligand = f'{dataset} LIG {chain}{r.get("resi")}' + (
                    f' alt {r["altloc"]}' if r.get('altloc') else '')
                try:
                    score = float(r[metric])
                except (TypeError, ValueError):
                    score = np.nan
                if not np.isfinite(score):
                    print(f'  Warning: no {metric} for {ligand}'
                          + (f' (EDIA call {r["status"]})' if r.get('status') else ''))
                    n_reference_missing += 1
                    continue
                row = {'dataset': dataset, 'pocket': pocket.name, 'chain': chain,
                       'resi': r['resi'], 'altloc': r.get('altloc', ''), metric: score,
                       'map': density}
                structure_rows.append(row)
                percentile_rows.append({**row, 'n_conformers': len(null),
                                        'n_conformers_below': int((null < score).sum()),
                                        'n_conformers_equal': int((null == score).sum()),
                                        f'null_median_{metric}': round(float(np.median(null)), 4),
                                        'percentile': round(percentile_of(score, null), 2)})

    if not structure_rows and not pocket_rows:
        raise SystemExit(f'No {run} {metric} scores found')
    if n_null_missing or n_reference_missing:
        print(f'  Left out for having no {metric}: {n_null_missing} null conformers, '
              f'{n_reference_missing} references')
    out_dir = Path(args.graphs_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []

    if structure_rows:
        values = np.array([r[metric] for r in structure_rows])
        n_ds = len({r['dataset'] for r in structure_rows})
        name = f'structure_{slug}'
        write_csv(out_dir / f'{name}.csv', structure_rows,
                  ['dataset', 'pocket', 'chain', 'resi', 'altloc', metric, 'map'])
        plot(values, f'{metric_label(metric, capital=True)} of the reference ligands ({run})',
             f'{len(values)} references (LIG residues, per altloc) across {n_ds} datasets; '
             f'median {np.median(values):.2f}; {EDIA_BIN_WIDTH:g} bins',
             out_dir / f'{name}.png', ylabel='Ligands', xlabel=metric_label(metric, capital=True))
        written.append(name)

    if pocket_rows:
        n_ds = len({r['dataset'] for r in pocket_rows})
        for stat in ('median', 'mean'):
            values = np.array([r[f'{stat}_{metric}'] for r in pocket_rows])
            name = f'null_{stat}_{slug}'
            write_csv(out_dir / f'{name}.csv', pocket_rows,
                      ['dataset', 'pocket', 'chain', 'n_conformers', 'n_without_score',
                       f'{stat}_{metric}', 'map'])
            plot(values, f'{stat.capitalize()} null conformer {label} per pocket ({run})',
                 f'{len(values)} pockets across {n_ds} datasets; median '
                 f'{np.median(values):.2f}; {EDIA_BIN_WIDTH:g} bins',
                 out_dir / f'{name}.png', ylabel='Pockets',
                 xlabel=f'{stat.capitalize()} null conformer {label}')
            written.append(name)

    if percentile_rows:
        values = np.array([r['percentile'] for r in percentile_rows])
        n_ds = len({r['dataset'] for r in percentile_rows})
        name = f'structure_{slug}_percentile'
        write_csv(out_dir / f'{name}.csv', percentile_rows,
                  ['dataset', 'pocket', 'chain', 'resi', 'altloc', metric, 'n_conformers',
                   'n_conformers_below', 'n_conformers_equal', f'null_median_{metric}',
                   'percentile', 'map'])
        plot(values,
             f'Reference ligand {label} percentile in its pocket\'s null distribution ({run})',
             f'{len(values)} references (per altloc) across {n_ds} datasets; '
             f'{int((values >= 95).sum())} at or above the 95th percentile; '
             f'{PERCENTILE_BIN_WIDTH:g}-point bins; log count scale',
             out_dir / f'{name}.png', ylabel='Ligands',
             bin_width=PERCENTILE_BIN_WIDTH, pad=PERCENTILE_PAD,
             xlabel=f'Percentile of the null conformer {label}', log_y=True)
        written.append(name)

    print(f'Wrote {", ".join(f"{w}.png" for w in written)} (and CSVs) to {out_dir}')


if __name__ == '__main__':
    main()
