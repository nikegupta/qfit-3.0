#!/usr/bin/env python3
"""
Pooled (across every dataset in datasets.txt) analysis of how well
PLACER-sampled models, after their own RSR ligand refinement, sample each
reference ligand pose. No RSCC or fresh RMSD-vs-cached-value is read here -
the RMSD between sampled and reference ligand geometry is computed directly
(there's no cached value to reuse before RSR has run at either round).

Supports two input modes, chosen by how many positional run-name arguments
are given:

  MODE A (2 args: run_name placer_run_name)
    Round-1 PLACER samples, after rsr_placer: scores every
    <placer_run_name>/*_refined.pdb.

  MODE B (4 args: run_name placer_run_name filter_run_name placer2_run_name)
    Round-2 PLACER samples, after rsr_placer2: scores every
    <placer_run_name>/<filter_run_name>/<placer2_run_name>/{dataset}_backbone_refined_*_refined.pdb.

Either mode's matched files may contain a single model or multiple models
(MODEL/ENDMDL blocks) - both are handled uniformly.

For each reference LIG conformation (altloc-aware), keeps the minimum
symmetry-aware RMSD to the closest sampled+refined ligand conformer across
every matched file for that dataset, pooled across all datasets into a
single histogram.

Run at the end of stage 2 (rsr_placer) and stage 4 (rsr_placer2), only when
-c (compare to reference set) is given.

Usage:
  calc_placer_sampling.py <run_name> <placer_run_name> \\
      --ref-set <dir> --graphs-dir <dir> [options]
  calc_placer_sampling.py <run_name> <placer_run_name> <filter_run_name> <placer2_run_name> \\
      --ref-set <dir> --graphs-dir <dir> [options]

Output:
  <graphs-dir>/placer_sampling.png
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
            file_pattern = f'{dataset}_backbone_refined_*_refined.pdb'
        else:
            model_dir = base
            file_pattern = '*_refined.pdb'

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
    out_name = 'placer_sampling.png'
    plot_distance_histogram(
        [row['rmsd'] for row in all_rows],
        title=f'Placer Sampling (Refined): Reference Ligand to Nearest Sampled Model\n{run_tag}',
        xlabel='Minimum Ligand RMSD to Closest Sampled+Refined Model (Å)',
        out_path=out_dir / out_name,
        bin_width=0.25,
    )
    if all_rows:
        write_plot_csv(out_dir, out_name,
                        pd.DataFrame(all_rows)[['dataset', 'ref_chain', 'ref_resi', 'ref_altloc', 'rmsd']])


if __name__ == '__main__':
    main()
