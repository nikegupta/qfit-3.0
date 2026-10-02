#!/usr/bin/env python3
"""
Pooled (across every dataset in datasets.txt) check of how many of fit_ligand's own output
poses (run_name/fit_ligand_manifest.csv) are crystallographic symmetry mates of a reference
ligand, using the same centroid-distance matching convention as the rest of this pipeline's
reference comparisons (--centroid-cutoff, default 2.0 A - same default/meaning as
build_ref_argparser's own option, used throughout stages 3/5/6/8).

fit_ligand's output structures aren't guaranteed to share the reference's crystallographic
frame yet (see centroid_rmsd_all.py's own docstring - unlike the later-stage reference
comparisons, which trust a shared frame): each pose is first superimposed onto the reference
by protein CA atoms (matched by chain_id/res_id, via biotite) - the exact same alignment step
centroid_rmsd_all.py already does - before its ligand centroid is checked against every
non-identity crystallographic symmetry-equivalent copy of every reference LIG residue.
Symmetry operations are generated with qfit's own UnitCell.iter_struct_orth_symops (the same
broad-phase search symmetry_expand.py uses for protein-mate expansion, and
check_excess_symmetry_mates.py uses for despot's excess ligands) - here checked against a
single already-aligned centroid point rather than a full structure, via the minimal
_CentroidTarget stand-in below (iter_struct_orth_symops only ever reads target.coor).

Crystal cell/space group come from --cell-lookup-file: "dataset a b c alpha beta gamma
space_group" lines (the same lookup file program.sh already builds for despot/symmetry_expand
as DESPOT_CELL_LOOKUP_FILE).

Run at the end of stage 1 (fit_ligand), only when -c (compare to reference set) is given.

Usage:
  check_fit_ligand_symmetry_mates.py <run_name> --ref-set <dir> --graphs-dir <dir> \\
      --cell-lookup-file <path> [options]

Output:
  <graphs-dir>/fit_ligand_symmetry_mates.png - histogram of the number of symmetry-mate
    matches per dataset, with mean/median/total annotated.
  <graphs-dir>/fit_ligand_symmetry_mates.csv - the exact values behind it: one row per
    dataset, dataset/n_poses_checked/n_symmetry_mate_matches.

Run inside CONDA_ENV_QFIT (needs qfit + cctbx, like symmetry_expand.py/despot_filter.py/
check_excess_symmetry_mates.py - CONDA_ENV_EVAL, used by most other analysis scripts, doesn't
have these).
"""
from pathlib import Path

import numpy as np
import pandas as pd
import biotite.structure as struc
import biotite.structure.io.pdb as pdb
from cctbx import crystal

from qfit import Structure as QfitStructure

from rscc_common import (
    build_ref_argparser, read_datasets, read_pdb_raw_atoms, lig_conformations_filtered,
    plot_count_histogram, ref_pdb_path, write_plot_csv,
)


def read_cell_lookup(path):
    """Parses '--cell-lookup-file' lines "dataset a b c alpha beta gamma space_group" (the
    same format/file program.sh's DESPOT_CELL_LOOKUP_FILE already builds from CSV_FILE) into
    {dataset: (a, b, c, alpha, beta, gamma, space_group)}."""
    cells = {}
    with open(path) as f:
        for line in f:
            parts = line.split()
            if len(parts) < 8:
                continue
            dataset = parts[0]
            a, b, c, alpha, beta, gamma = (float(x) for x in parts[1:7])
            space_group = parts[7]
            cells[dataset] = (a, b, c, alpha, beta, gamma, space_group)
    return cells


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


class _CentroidTarget:
    """Minimal .coor-only stand-in for a qfit Structure, for
    UnitCell.iter_struct_orth_symops' target= argument - that method only ever reads
    target.coor (see qfit/xtal/unitcell.py), so a single already-aligned centroid point (a
    fit_ligand pose's ligand centroid, after CA-based superposition onto the reference frame)
    can stand in for a full structure without building one."""
    def __init__(self, centroid):
        self.coor = np.asarray(centroid, dtype=float).reshape(1, 3)


def ref_ligand_sites(ref_structure):
    """Distinct (chain, resi) LIG sites in ref_structure, altloc-agnostic (every altloc of a
    residue grouped together - this check is about crystallographic positional equivalence,
    not exact conformer identity - same convention as check_excess_symmetry_mates.py)."""
    lig = ref_structure.extract('resname LIG')
    if lig.natoms == 0:
        return []
    return sorted(set(zip(lig.chain, lig.resi)))


def closest_symmetry_mate_distance_to_point(ref_structure, ref_lig, point, cushion):
    """Minimum distance, across every non-identity crystallographic symmetry operation,
    between a symmetry-transformed copy of ref_lig's centroid and a single point already
    expressed in ref_structure's crystallographic frame. ref_structure is whichever qfit
    Structure iter_struct_orth_symops is called on (must have had set_crystal_symmetry
    applied); ref_lig is a qfit Structure for one reference ligand site, mutated and restored
    in place. Returns None if no non-identity symop was even considered (ref_lig/point too far
    apart for any mate to plausibly reach within cushion, per iter_struct_orth_symops' own
    broad-phase cushion)."""
    baseline_coor = ref_lig.coor.copy()
    target = _CentroidTarget(point)
    best = None
    for symop in ref_structure.unit_cell.iter_struct_orth_symops(ref_lig, target=target,
                                                                   cushion=cushion):
        if symop.is_identity():
            continue
        ref_lig.rotate(symop.R)
        ref_lig.translate(symop.t)
        dist = float(np.linalg.norm(ref_lig.coor.mean(axis=0) - point))
        if best is None or dist < best:
            best = dist
        ref_lig.coor = baseline_coor
    return best


def process_dataset(dataset, args, cells):
    run_dir = Path(args.datasets_dir) / dataset / args.run_name
    manifest_csv = run_dir / 'fit_ligand_manifest.csv'
    ref_pdb = ref_pdb_path(args, dataset)
    if not manifest_csv.is_file():
        return None
    if not ref_pdb.exists():
        print(f'  {dataset}: reference structure not found: {ref_pdb}; skipping.')
        return None

    cell = cells.get(dataset)
    if cell is None:
        print(f'  {dataset}: no cell/space-group info in --cell-lookup-file; skipping.')
        return None
    a, b, c, alpha, beta, gamma, space_group = cell

    manifest = pd.read_csv(manifest_csv)
    if manifest.empty:
        print(f'  {dataset}: fit_ligand_manifest.csv has no rows.')
        return {'dataset': dataset, 'n_poses_checked': 0, 'n_symmetry_mate_matches': 0}

    ref_struct_bio = read_pdb_biotite(ref_pdb)

    qfit_ref_structure = QfitStructure.fromfile(str(ref_pdb))
    crystal_symmetry = crystal.symmetry(
        unit_cell=(a, b, c, alpha, beta, gamma), space_group_symbol=space_group,
    )
    qfit_ref_structure.set_crystal_symmetry(crystal_symmetry)
    qfit_ref_structure._kwargs['crystal_symmetry'] = crystal_symmetry  # pylint: disable=protected-access

    sites = ref_ligand_sites(qfit_ref_structure)
    if not sites:
        print(f'  {dataset}: no LIG residue found in reference {ref_pdb}; skipping.')
        return None

    n_poses_checked = 0
    n_matches = 0
    for _, row in manifest.iterrows():
        output_pdb = Path(row['output_pdb'])
        if not output_pdb.is_file():
            continue
        try:
            mobile_struct = read_pdb_biotite(output_pdb)
            transformation = align_to_reference(mobile_struct, ref_struct_bio)
        except Exception as e:
            print(f'    Warning: could not align {output_pdb.name} ({dataset}): {e}')
            continue

        model_atoms = read_pdb_raw_atoms(output_pdb)
        model_confs = lig_conformations_filtered(model_atoms, chain_id='C', res_id=1)
        if not model_confs:
            model_confs = lig_conformations_filtered(model_atoms)
            if not model_confs:
                continue

        n_poses_checked += 1
        is_match = False
        for centroid in model_confs.values():
            aligned_centroid = transformation.apply(centroid.reshape(1, 3))[0]
            for ref_chain, ref_resi in sites:
                ref_lig = qfit_ref_structure.extract(
                    f'chain {ref_chain} and resi {ref_resi} and resname LIG')
                if ref_lig.natoms == 0:
                    continue
                dist = closest_symmetry_mate_distance_to_point(
                    qfit_ref_structure, ref_lig, aligned_centroid, cushion=args.centroid_cutoff)
                if dist is not None and dist <= args.centroid_cutoff:
                    is_match = True
                    break
            if is_match:
                break
        if is_match:
            n_matches += 1

    print(f'  {dataset}: {n_matches}/{n_poses_checked} fit_ligand pose(s) are a symmetry mate '
          f'of a reference ligand.')
    return {'dataset': dataset, 'n_poses_checked': n_poses_checked,
            'n_symmetry_mate_matches': n_matches}


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

    total_matches = int(out_df['n_symmetry_mate_matches'].sum())
    total_poses = int(out_df['n_poses_checked'].sum())
    percentage = 100 * total_matches / total_poses if total_poses > 0 else 0.0

    plot_count_histogram(
        list(out_df['n_symmetry_mate_matches']),
        title=f'fit_ligand Symmetry-Mate Matches per Dataset ({args.run_name})',
        xlabel='Number of Symmetry-Mate Matches',
        out_path=out_dir / out_name,
        show_total=True,
        extra_stats={'Percentage': f'{percentage:.1f}%'},
    )
    write_plot_csv(out_dir, out_name, out_df)


if __name__ == '__main__':
    main()
