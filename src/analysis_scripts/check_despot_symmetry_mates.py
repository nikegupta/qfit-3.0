#!/usr/bin/env python3
"""
Pooled (across every dataset in datasets.txt) check of how many of despot_filter's own
surviving ligand instances (despot_run_name/despot_filtered.pdb) are crystallographic
symmetry mates of ANOTHER surviving instance in the SAME despot_filtered.pdb - i.e. redundant
placements of one site at a symmetry-equivalent position. Same grouping logic as
check_fit_ligand_symmetry_mates.py's own symmetry_groups: two instances are mates if a
non-identity crystal-symmetry image of one's centroid lands within --centroid-cutoff (default
2.0 A - same default/meaning as build_ref_argparser's own option, used throughout stages
3/5/6/8) of the other's, while the two aren't already within the cutoff directly (that's a
plain duplicate, not a symmetry mate). Mates are chained, so every set of symmetry-equivalent
instances forms one group. Just applied here to despot_filter's own surviving instances
instead of fit_ligand's raw output poses.

Unlike check_fit_ligand_symmetry_mates.py, despot_filtered.pdb already shares the reference's
crystallographic frame (same trusted-frame convention check_excess_symmetry_mates.py/
plot_lig_vs_ref_despot.py/plot_despot_vs_ref.py rely on at this pipeline stage), so no
CA-superposition step is needed here - every ligand instance's centroid is read directly from
despot_filtered.pdb's own coordinates.

Crystal cell/space group come from --cell-lookup-file: "dataset a b c alpha beta gamma
space_group" lines (the same lookup file program.sh already builds for despot/symmetry_expand
as DESPOT_CELL_LOOKUP_FILE); reference PDBs' CRYST1 records can't be trusted for this.

Run at the end of stage 8, only when both -c (compare to reference set) and <despot_run_name>
are given.

Usage:
  check_despot_symmetry_mates.py <run_name> <placer_run_name> <filter_run_name> \\
      <placer2_run_name> <filter2_run_name> <final_run_name> <rotamer_run_name> \\
      <despot_run_name> --ref-set <dir> --graphs-dir <dir> --cell-lookup-file <path> [options]

Output:
  <graphs-dir>/despot_symmetry_mates.png - histogram of the number of redundant symmetry-mate
    instances per dataset (instances beyond the first in each symmetry-equivalent group), with
    mean/median/total annotated.
  <graphs-dir>/despot_symmetry_mates.csv - the exact values behind it: one row per dataset,
    dataset/n_poses_checked/n_poses_with_mate/n_symmetry_groups/n_redundant_symmetry_mates.

Run inside CONDA_ENV_QFIT (needs cctbx for the space-group operators).
"""
from pathlib import Path

import numpy as np
import pandas as pd

from rscc_common import (
    CrystalSymmetry, build_ref_argparser, read_cell_lookup, read_datasets, read_pdb_raw_atoms,
    lig_conformations_filtered, plot_count_histogram, write_plot_csv,
)


def despot_dir(dataset, args):
    """<datasets_dir>/<dataset>/<run>/<placer>/<filter>/<placer2>/<filter2>/<final>/<rotamer>/
    <despot> - same positional-argument convention check_excess_symmetry_mates.py's own
    despot_dir uses (rotamer_run_name and despot_run_name are separate positionals, not
    combined into one "rotamer/despot" string - duplicated rather than imported, these are
    independent, self-contained analysis scripts)."""
    return (Path(args.datasets_dir) / dataset / args.run_name / args.placer_run_name /
            args.filter_run_name / args.placer2_run_name / args.filter2_run_name /
            args.final_run_name / args.rotamer_run_name / args.despot_run_name)


def symmetry_groups(centroids, symmetry, cutoff):
    """Union-find over instance pairs that are symmetry mates (non-identity image within
    cutoff, not directly within cutoff) - identical to check_fit_ligand_symmetry_mates.py's
    own symmetry_groups (duplicated rather than imported). Returns the groups of instance
    indices with more than one member."""
    parent = list(range(len(centroids)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(centroids)):
        for j in range(i + 1, len(centroids)):
            if np.linalg.norm(centroids[i] - centroids[j]) <= cutoff:
                continue
            if symmetry.distance(centroids[i], centroids[j], include_identity=False) <= cutoff:
                parent[find(j)] = find(i)
    groups = {}
    for i in range(len(centroids)):
        groups.setdefault(find(i), []).append(i)
    return [members for members in groups.values() if len(members) > 1]


def process_dataset(dataset, args, cells):
    despot_filtered_pdb = despot_dir(dataset, args) / 'despot_filtered.pdb'
    empty = {'dataset': dataset, 'n_poses_checked': 0, 'n_poses_with_mate': 0,
             'n_symmetry_groups': 0, 'n_redundant_symmetry_mates': 0}
    if not despot_filtered_pdb.is_file():
        return None
    if dataset not in cells:
        print(f'  {dataset}: no cell/space-group info in --cell-lookup-file; skipping.')
        return None
    symmetry = CrystalSymmetry(cells[dataset])

    atoms = read_pdb_raw_atoms(despot_filtered_pdb)
    confs = lig_conformations_filtered(atoms)
    if not confs:
        print(f'  {dataset}: no LIG atoms in {despot_filtered_pdb}.')
        return empty

    centroids = list(confs.values())
    groups = symmetry_groups(centroids, symmetry, args.centroid_cutoff)
    n_with_mate = sum(len(members) for members in groups)
    n_redundant = sum(len(members) - 1 for members in groups)
    print(f'  {dataset}: {n_redundant}/{len(centroids)} despot_filtered instance(s) are '
          f'redundant symmetry mates ({len(groups)} symmetry-equivalent group(s), '
          f'{n_with_mate} instances).')
    return {'dataset': dataset, 'n_poses_checked': len(centroids),
            'n_poses_with_mate': n_with_mate, 'n_symmetry_groups': len(groups),
            'n_redundant_symmetry_mates': n_redundant}


def main():
    parser = build_ref_argparser(
        __doc__,
        ['run_name', 'placer_run_name', 'filter_run_name', 'placer2_run_name',
         'filter2_run_name', 'final_run_name', 'rotamer_run_name', 'despot_run_name'],
    )
    parser.add_argument(
        '--cell-lookup-file', required=True,
        help='Path to a "dataset a b c alpha beta gamma space_group" lookup file (the same '
             'format/file program.sh\'s DESPOT_CELL_LOOKUP_FILE already builds from CSV_FILE)',
    )
    args = parser.parse_args()

    cells = read_cell_lookup(args.cell_lookup_file)
    datasets = read_datasets(args.datasets_file)

    rows = []
    for dataset in datasets:
        result = process_dataset(dataset, args, cells)
        if result is not None:
            rows.append(result)

    if not rows:
        print('No dataset(s) checked; nothing to write.')
        return

    out_df = pd.DataFrame(rows)
    out_dir = Path(args.graphs_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_name = 'despot_symmetry_mates.png'

    total_redundant = int(out_df['n_redundant_symmetry_mates'].sum())
    total_poses = int(out_df['n_poses_checked'].sum())
    percentage = 100 * total_redundant / total_poses if total_poses > 0 else 0.0

    plot_count_histogram(
        list(out_df['n_redundant_symmetry_mates']),
        title=f'DESPOT-Filtered Redundant Symmetry-Mate Instances per Dataset ({args.despot_run_name})',
        xlabel=f'Instances That Are Symmetry Mates of Another (≤ {args.centroid_cutoff} Å)',
        out_path=out_dir / out_name,
        show_total=True,
        extra_stats={'Percentage': f'{percentage:.1f}%'},
    )
    write_plot_csv(out_dir, out_name, out_df)


if __name__ == '__main__':
    main()
