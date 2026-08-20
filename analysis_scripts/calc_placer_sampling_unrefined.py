#!/usr/bin/env python3
"""
Same comparison as calc_placer_sampling.py, but scores PLACER's raw sampled
models directly - before that round's own RSR ligand refinement has been
applied. Pairs with calc_placer_sampling.py to show how much RSR moves the
sampled ligand poses towards (or away from) the reference.

Supports the same two input modes, chosen by how many positional run-name
arguments are given:

  MODE A (2 args: run_name placer_run_name)
    Round-1 PLACER's own raw output, before rsr_placer: scores every
    <placer_run_name>/*_model.pdb.

  MODE B (4 args: run_name placer_run_name filter_run_name placer2_run_name)
    Round-2 PLACER's own raw output, before rsr_placer2: scores every
    <placer_run_name>/<filter_run_name>/<placer2_run_name>/{dataset}_backbone_refined_*_model.pdb.

Run at the end of stage 2 (rsr_placer) and stage 4 (rsr_placer2) - after the
round's raw *_model.pdb files exist - only when -c (compare to reference
set) is given.

Usage:
  calc_placer_sampling_unrefined.py <run_name> <placer_run_name> \\
      --ref-set <dir> --graphs-dir <dir> [options]
  calc_placer_sampling_unrefined.py <run_name> <placer_run_name> <filter_run_name> <placer2_run_name> \\
      --ref-set <dir> --graphs-dir <dir> [options]

Output:
  <graphs-dir>/placer_sampling_unrefined.png
"""
from pathlib import Path

import pandas as pd

from rscc_common import (
    build_placer_sampling_argparser, resolve_placer_sampling_mode, read_datasets,
    process_placer_sampling_dataset, plot_distance_histogram, ref_pdb_path, write_plot_csv,
)


def main():
    args = build_placer_sampling_argparser(__doc__).parse_args()
    mode_b, run_tag = resolve_placer_sampling_mode(args)

    datasets = read_datasets(args.datasets_file)
    all_rows = []
    for dataset in datasets:
        base = Path(args.datasets_dir) / dataset / args.run_name / args.placer_run_name
        if mode_b:
            model_dir = base / args.filter_run_name / args.placer2_run_name
            file_pattern = f'{dataset}_backbone_refined_*_model.pdb'
        else:
            model_dir = base
            file_pattern = '*_model.pdb'

        rows = process_placer_sampling_dataset(
            model_dir, ref_pdb_path(args, dataset), file_pattern,
            args.model_chain, args.model_resi,
        )
        print(f'  {dataset}: {len(rows)} ref LIG conformation(s) matched')
        for row in rows:
            row['dataset'] = dataset
        all_rows.extend(rows)

    out_dir = Path(args.graphs_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_name = 'placer_sampling_unrefined.png'
    plot_distance_histogram(
        [row['rmsd'] for row in all_rows],
        title=f'Placer Sampling (Unrefined): Reference Ligand to Nearest Sampled Model\n{run_tag}',
        xlabel='Minimum Ligand RMSD to Closest Raw Sampled Model (Å)',
        out_path=out_dir / out_name,
        bin_width=0.25,
    )
    if all_rows:
        write_plot_csv(out_dir, out_name,
                        pd.DataFrame(all_rows)[['dataset', 'ref_chain', 'ref_resi', 'ref_altloc', 'rmsd']])


if __name__ == '__main__':
    main()
