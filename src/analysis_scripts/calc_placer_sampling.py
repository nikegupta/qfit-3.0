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

In mode B, when --filter1-csv (Stage 3d's pooled lig_vs_reference_rscc.csv) is given, also
writes a placer_sampling_restricted.png/.csv restricted to reference ligands that found a match
in filter (Stage 3a) - see rscc_common.restrict_to_filter_matched for why unmatched reference
ligands otherwise pollute this histogram with large, meaningless RMSDs.

When --cell-lookup-file is given, ALSO writes a crystallographic-symmetry-aware twin of every
output (placer_sampling_sa.png/.csv, and placer_sampling_sa_restricted.png/.csv in mode B with
--filter1-csv) - see process_placer_sampling_dataset's symmetry argument: each sampled
conformer is matched by centroid to its single nearest crystal-symmetry image of the reference
ligand (not every image), then RMSD is computed only against that one image. This is a
DIFFERENT symmetry than compute_rmsd_symmetric's own atom-permutation matching, which both the
plain and _sa outputs already use.

Usage:
  calc_placer_sampling.py <run_name> <placer_run_name> \\
      --ref-set <dir> --graphs-dir <dir> [options]
  calc_placer_sampling.py <run_name> <placer_run_name> <filter_run_name> <placer2_run_name> \\
      --ref-set <dir> --graphs-dir <dir> [--filter1-csv <path>] [--cell-lookup-file <path>] [options]

Output:
  <graphs-dir>/placer_sampling.png
  <graphs-dir>/placer_sampling_restricted.png (mode B with --filter1-csv only)
  <graphs-dir>/placer_sampling_sa.png (--cell-lookup-file only)
  <graphs-dir>/placer_sampling_sa_restricted.png (mode B with --filter1-csv and
    --cell-lookup-file only)
"""
from pathlib import Path

import pandas as pd

from rscc_common import (
    build_placer_sampling_argparser, resolve_placer_sampling_mode, read_datasets,
    process_placer_sampling_dataset, plot_distance_histogram, ref_pdb_path, write_plot_csv,
    restrict_to_filter_matched, read_cell_lookup, CrystalSymmetry,
)


def run_pass(args, mode_b, model_dir_for, file_pattern, symmetry_lookup, out_name, title):
    """One full pooled pass (direct or symmetry-aware, per symmetry_lookup) across every
    dataset, writing out_name's histogram/csv and, in mode B with --filter1-csv, its
    filter-matched-restricted twin. symmetry_lookup: None for the plain/direct pass, or
    {dataset: CrystalSymmetry} for the symmetry-aware pass (see process_placer_sampling_dataset's
    own symmetry argument) - a dataset missing from it falls back to direct comparison."""
    datasets = read_datasets(args.datasets_file)
    all_rows = []
    for dataset in datasets:
        model_dir, file_pat = model_dir_for(dataset), file_pattern(dataset)
        symmetry = symmetry_lookup.get(dataset) if symmetry_lookup is not None else None
        rows = process_placer_sampling_dataset(
            model_dir, ref_pdb_path(args, dataset), file_pat,
            args.model_chain, args.model_resi, symmetry=symmetry,
        )
        print(f'  {dataset}: {len(rows)} ref LIG conformation(s) matched')
        for row in rows:
            row['dataset'] = dataset
        all_rows.extend(rows)

    out_dir = Path(args.graphs_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_distance_histogram(
        [row['rmsd'] for row in all_rows],
        title=title,
        xlabel='Minimum Ligand RMSD to Closest Sampled+Refined Model (Å)',
        out_path=out_dir / out_name,
        bin_width=0.25,
    )
    rows_df = None
    if all_rows:
        rows_df = pd.DataFrame(all_rows)[['dataset', 'ref_chain', 'ref_resi', 'ref_altloc', 'rmsd',
                                           'placer_file', 'model_idx']]
        write_plot_csv(out_dir, out_name, rows_df)

    if mode_b and args.filter1_csv:
        restricted_name = f'{Path(out_name).stem}_restricted{Path(out_name).suffix}'
        if rows_df is None:
            print(f'  No PLACER sampling rows; skipping {restricted_name}.')
        else:
            restricted_df, filter1_existed = restrict_to_filter_matched(rows_df, args.filter1_csv)
            if not filter1_existed:
                print(f'  filter1 lig_vs_reference_rscc.csv not found at {args.filter1_csv}; '
                      f'skipping {restricted_name}.')
            else:
                plot_distance_histogram(
                    list(restricted_df['rmsd']),
                    title=f'{title}, Filter-Matched',
                    xlabel='Minimum Ligand RMSD to Closest Sampled+Refined Model (Å)',
                    out_path=out_dir / restricted_name,
                    bin_width=0.25,
                )
                if not restricted_df.empty:
                    write_plot_csv(out_dir, restricted_name, restricted_df)


def main():
    args = build_placer_sampling_argparser(__doc__).parse_args()
    mode_b, _ = resolve_placer_sampling_mode(args)

    def model_dir_for(dataset):
        base = Path(args.datasets_dir) / dataset / args.run_name / args.placer_run_name
        return base / args.filter_run_name / args.placer2_run_name if mode_b else base

    def file_pattern(dataset):
        return f'{dataset}_backbone_refined_*_refined.pdb' if mode_b else '*_refined.pdb'

    run_pass(args, mode_b, model_dir_for, file_pattern, None,
              'placer_sampling.png', 'Placer Sampling RMSD (Refined)')

    if args.cell_lookup_file:
        cells = read_cell_lookup(args.cell_lookup_file)
        symmetry_lookup = {dataset: CrystalSymmetry(cell) for dataset, cell in cells.items()}
        run_pass(args, mode_b, model_dir_for, file_pattern, symmetry_lookup,
                  'placer_sampling_sa.png', 'Placer Sampling RMSD (Refined, Symmetry Aware)')


if __name__ == '__main__':
    main()
