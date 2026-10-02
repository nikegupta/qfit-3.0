#!/usr/bin/env python3
"""
Pooled (across every dataset in datasets.txt) summary of build_final_model.py's and
rotamer_optimize.py's new ligand-clash filtering (qfit.command_line.ligand_clash) - how many
residues, per dataset, had every one of their sampled candidate conformers excluded for
clashing with a surviving ligand instance, and therefore fell back to their pre-sampling
conformation (apo, for build_final_model; the input model_file conformation, for
rotamer_optimize).

Reusable at both call sites (run right after Stage 6a build_final and right after Stage 7a
rotamer_optimize) via a single --run-subpath argument: the path from
<datasets_dir>/<dataset>/ down to the folder containing that stage's own
ligand_clash_filtered.csv (e.g. ".../filter2_1/final_1" for build_final,
".../filter2_1/final_1/rotamer_1" for rotamer_optimize - whatever output_folder that stage's
own program.sh invocation used).

Not gated behind -c: this is purely a summary of each run's own output, no reference-set
comparison involved.

Usage:
  plot_ligand_clash_filtered.py --run-subpath <path> --datasets-dir <dir> \\
      --datasets-file <path> --graphs-dir <dir>

Output:
  <graphs-dir>/ligand_clash_filtered.png - histogram of residues reset-to-apo/input per
    dataset (mean/median/total annotated).
  <graphs-dir>/ligand_clash_filtered.csv - the exact values behind it: one row per dataset,
    dataset/n_residues_tracked/n_residues_with_any_exclusion/n_residues_reset/
    total_candidates_sampled/total_candidates_excluded.
"""
import argparse
from pathlib import Path

import pandas as pd

from rscc_common import plot_count_histogram, read_datasets, write_plot_csv


def build_argparser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        '--run-subpath', required=True,
        help='Path from <datasets-dir>/<dataset>/ to the folder containing that stage\'s own '
             'ligand_clash_filtered.csv (e.g. the same output_folder build_final_model.py or '
             'rotamer_optimize.py was just given).',
    )
    p.add_argument('--datasets-dir', required=True, type=Path,
                    help='Root directory containing per-dataset folders')
    p.add_argument('--datasets-file', required=True, type=Path,
                    help='Path to datasets.txt (one dataset id per line)')
    p.add_argument('--graphs-dir', required=True, type=Path,
                    help='Output directory for the pooled plot/csv')
    return p


def process_dataset(dataset, args):
    csv_path = args.datasets_dir / dataset / args.run_subpath / 'ligand_clash_filtered.csv'
    if not csv_path.is_file():
        return None
    df = pd.read_csv(csv_path)
    if df.empty:
        return {'dataset': dataset, 'n_residues_tracked': 0, 'n_residues_with_any_exclusion': 0,
                'n_residues_reset': 0, 'total_candidates_sampled': 0,
                'total_candidates_excluded': 0}
    return {
        'dataset': dataset,
        'n_residues_tracked': len(df),
        'n_residues_with_any_exclusion': int((df['n_excluded_for_ligand_clash'] > 0).sum()),
        'n_residues_reset': int((df['reset_to_apo_due_to_clash'] == 'yes').sum()),
        'total_candidates_sampled': int(df['n_candidates_sampled'].sum()),
        'total_candidates_excluded': int(df['n_excluded_for_ligand_clash'].sum()),
    }


def main():
    args = build_argparser().parse_args()
    datasets = read_datasets(args.datasets_file)

    rows = []
    for dataset in datasets:
        result = process_dataset(dataset, args)
        if result is not None:
            rows.append(result)

    if not rows:
        print('No dataset(s) with a ligand_clash_filtered.csv found; nothing to write.')
        return

    out_df = pd.DataFrame(rows)
    out_dir = args.graphs_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_name = 'ligand_clash_filtered.png'

    total_reset = int(out_df['n_residues_reset'].sum())
    total_excluded = int(out_df['total_candidates_excluded'].sum())
    total_sampled = int(out_df['total_candidates_sampled'].sum())
    pct = 100 * total_excluded / total_sampled if total_sampled > 0 else 0.0

    plot_count_histogram(
        list(out_df['n_residues_reset']),
        title='Residues Reset to Apo/Input Due to Ligand Clash per Dataset',
        xlabel='Number of Residues Reset',
        out_path=out_dir / out_name,
        show_total=True,
        extra_stats={'Candidates excluded': f'{total_excluded}/{total_sampled} ({pct:.2f}%)'},
    )
    write_plot_csv(out_dir, out_name, out_df)
    print(f'{len(out_df)} dataset(s) pooled; {total_reset} residue(s) reset to apo/input total '
          f'across every dataset.')


if __name__ == '__main__':
    main()
