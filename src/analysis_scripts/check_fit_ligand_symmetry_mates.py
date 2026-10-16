#!/usr/bin/env python3
"""
Pooled (across every dataset in datasets.txt) check of how many of fit_ligand's own output
poses (run_name/fit_ligand_manifest.csv) are crystallographic symmetry mates of another pose
placed for the same dataset - i.e. redundant placements of one site at a symmetry-equivalent
position. Two poses are symmetry mates if a non-identity crystal-symmetry image of one
(space-group operators and lattice translations) puts its ligand centroid within
--centroid-cutoff (default 2.0 A - same default/meaning as build_ref_argparser's own option,
used throughout stages 3/5/6/8) of the other's, while the two are not already within the
cutoff of each other directly (that is a plain duplicate, not a symmetry mate). Mates are
chained, so every set of symmetry-equivalent poses forms one group.

fit_ligand's output structures aren't guaranteed to share the reference's crystallographic
frame (see centroid_rmsd_all.py's own docstring), so each pose is first superimposed onto the
reference by protein CA atoms (matched by chain_id/res_id, via biotite) - the same alignment
step centroid_rmsd_all.py does - and every pose is then compared in that one frame.

Crystal cell/space group come from --cell-lookup-file: "dataset a b c alpha beta gamma
space_group" lines (the same lookup file program.sh already builds for despot/symmetry_expand
as DESPOT_CELL_LOOKUP_FILE); reference PDBs' CRYST1 records can't be trusted for this.

Run at the end of stage 1 (fit_ligand), only when -c (compare to reference set) is given.

Usage:
  check_fit_ligand_symmetry_mates.py <run_name> --ref-set <dir> --graphs-dir <dir> \\
      --cell-lookup-file <path> [options]

Output:
  <graphs-dir>/fit_ligand_symmetry_mates.png - histogram of the number of redundant
    symmetry-mate poses per dataset (poses beyond the first in each symmetry-equivalent
    group), with mean/median/total annotated.
  <graphs-dir>/fit_ligand_symmetry_mates.csv - the exact values behind it: one row per
    dataset, dataset/n_poses_checked/n_poses_with_mate/n_symmetry_groups/
    n_redundant_symmetry_mates.

Run inside CONDA_ENV_QFIT (needs cctbx for the space-group operators).
"""
from pathlib import Path

import numpy as np
import pandas as pd
import biotite.structure as struc
import biotite.structure.io.pdb as pdb

from rscc_common import (
    CrystalSymmetry, build_ref_argparser, read_cell_lookup, read_datasets, read_pdb_raw_atoms,
    lig_conformations_filtered, plot_count_histogram, ref_pdb_path, write_plot_csv,
)


def read_pdb_biotite(path):
    pdb_file = pdb.PDBFile.read(str(path))
    return pdb_file.get_structure(model=1)


def get_protein_ca(structure):
    ca = structure[struc.filter_amino_acids(structure)]
    return ca[ca.atom_name == 'CA']


def align_to_reference(mobile_struct, target_struct):
    """Superimposes mobile onto target using CA atoms matched by (chain_id, res_id) - same
    technique as centroid_rmsd_all.py's own align_to_reference (duplicated rather than
    imported - these are independent, self-contained analysis scripts). Raises ValueError if
    no residues are in common."""
    mobile_ca = get_protein_ca(mobile_struct)
    target_ca = get_protein_ca(target_struct)

    mobile_res = set(zip(mobile_ca.chain_id, mobile_ca.res_id))
    target_res = set(zip(target_ca.chain_id, target_ca.res_id))
    common_res = mobile_res & target_res
    if not common_res:
        raise ValueError('No CA residues in common between model and reference.')

    def select_sorted(ca_atoms):
        mask = np.array([(ch, ri) in common_res
                          for ch, ri in zip(ca_atoms.chain_id, ca_atoms.res_id)])
        subset = ca_atoms[mask]
        order = np.argsort([f'{ch}_{ri:06d}' for ch, ri in zip(subset.chain_id, subset.res_id)])
        return subset[order]

    _, transformation = struc.superimpose(select_sorted(mobile_ca), select_sorted(target_ca))
    return transformation


def symmetry_groups(centroids, symmetry, cutoff):
    """Union-find over pose pairs that are symmetry mates (non-identity image within cutoff,
    not directly within cutoff). Returns the groups of pose indices with more than one
    member."""
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
    run_dir = Path(args.datasets_dir) / dataset / args.run_name
    manifest_csv = run_dir / 'fit_ligand_manifest.csv'
    ref_pdb = ref_pdb_path(args, dataset)
    if not manifest_csv.is_file():
        return None
    if not ref_pdb.exists():
        print(f'  {dataset}: reference structure not found: {ref_pdb}; skipping.')
        return None
    if dataset not in cells:
        print(f'  {dataset}: no cell/space-group info in --cell-lookup-file; skipping.')
        return None
    symmetry = CrystalSymmetry(cells[dataset])

    manifest = pd.read_csv(manifest_csv)
    empty = {'dataset': dataset, 'n_poses_checked': 0, 'n_poses_with_mate': 0,
             'n_symmetry_groups': 0, 'n_redundant_symmetry_mates': 0}
    if manifest.empty:
        print(f'  {dataset}: fit_ligand_manifest.csv has no rows.')
        return empty

    ref_struct_bio = read_pdb_biotite(ref_pdb)
    centroids = []
    for _, row in manifest.iterrows():
        output_pdb = Path(row['output_pdb'])
        if not output_pdb.is_file():
            continue
        try:
            transformation = align_to_reference(read_pdb_biotite(output_pdb), ref_struct_bio)
        except Exception as e:
            print(f'    Warning: could not align {output_pdb.name} ({dataset}): {e}')
            continue
        model_atoms = read_pdb_raw_atoms(output_pdb)
        model_confs = lig_conformations_filtered(model_atoms, chain_id='C', res_id=1)
        if not model_confs:
            model_confs = lig_conformations_filtered(model_atoms)
            if not model_confs:
                continue
        # one centroid per pose: the mean over its ligand conformation(s)
        centroid = np.mean(list(model_confs.values()), axis=0)
        centroids.append(transformation.apply(centroid.reshape(1, 3))[0])

    if not centroids:
        return empty
    groups = symmetry_groups(centroids, symmetry, args.centroid_cutoff)
    n_with_mate = sum(len(members) for members in groups)
    n_redundant = sum(len(members) - 1 for members in groups)
    print(f'  {dataset}: {n_redundant}/{len(centroids)} fit_ligand pose(s) are redundant '
          f'symmetry mates ({len(groups)} symmetry-equivalent group(s), {n_with_mate} poses).')
    return {'dataset': dataset, 'n_poses_checked': len(centroids),
            'n_poses_with_mate': n_with_mate, 'n_symmetry_groups': len(groups),
            'n_redundant_symmetry_mates': n_redundant}


def main():
    parser = build_ref_argparser(__doc__, ['run_name'])
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
    out_name = 'fit_ligand_symmetry_mates.png'

    total_redundant = int(out_df['n_redundant_symmetry_mates'].sum())
    total_poses = int(out_df['n_poses_checked'].sum())
    percentage = 100 * total_redundant / total_poses if total_poses > 0 else 0.0

    plot_count_histogram(
        list(out_df['n_redundant_symmetry_mates']),
        title=f'fit_ligand Redundant Symmetry-Mate Poses per Dataset ({args.run_name})',
        xlabel=f'Poses That Are Symmetry Mates of Another Pose (≤ {args.centroid_cutoff} Å)',
        out_path=out_dir / out_name,
        show_total=True,
        extra_stats={'Percentage': f'{percentage:.1f}%'},
    )
    write_plot_csv(out_dir, out_name, out_df)


if __name__ == '__main__':
    main()
