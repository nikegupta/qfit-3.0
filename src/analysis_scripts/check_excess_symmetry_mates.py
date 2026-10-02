#!/usr/bin/env python3
"""
Pooled (across every dataset in datasets.txt) check of whether each "excess" despot_filtered
ligand instance - a pipeline pose that survived despot_filter.py but was never any reference
ligand's nearest match within centroid_cutoff, see plot_lig_vs_ref_despot.py's own
lig_vs_reference_rscc_excess_pipeline.csv - is actually sitting on a crystallographic symmetry
mate of some reference ligand, rather than a genuinely spurious/unrelated pose. A real ligand
site viewed through a different symmetry operation of the crystal would otherwise look
"excess" purely because the working-cell (identity-operation) comparison in _dataset_lig_vs_ref
never considers anything but the identity operation.

Reads <graphs_dir>/lig_vs_reference_rscc_excess_pipeline.csv (written by
plot_lig_vs_ref_despot.py at Stage 8b - this script must run after it, and only checks
whatever rows are already in that csv) and, for every row, compares that dataset's
despot_filtered.pdb ligand instance (despot_filtered_chain/despot_filtered_resi) against every
crystallographic symmetry-equivalent copy of every LIG residue in the reference structure
(ref_pdb_path), using the same centroid-distance matching convention as plot_lig_vs_ref*.py
(--centroid-cutoff, default 2.0 A) - not just the identity/working-cell comparison already
ruled out by the excess classification.

Symmetry operations are generated with qfit's own UnitCell.iter_struct_orth_symops (the same
broad-phase search symmetry_expand.py uses for protein-mate expansion, applied here to a
reference ligand instead of the whole protein), using each dataset's own unit cell/space group
(read from --cell-lookup-file: "dataset a b c alpha beta gamma space_group" lines, the same
lookup file program.sh already builds for despot/symmetry_expand as DESPOT_CELL_LOOKUP_FILE).
Reference LIG residues are grouped by (chain, resi) only (altloc-agnostic - this check is
about crystallographic positional equivalence, not exact conformer identity).

Run at the end of stage 8, only when both -c (compare to reference set) and <despot_run_name>
are given, after plot_lig_vs_ref_despot.py.

Usage:
  check_excess_symmetry_mates.py <run_name> <placer_run_name> <filter_run_name> \\
      <placer2_run_name> <filter2_run_name> <final_run_name> <rotamer_run_name> \\
      <despot_run_name> --ref-set <dir> --graphs-dir <dir> --cell-lookup-file <path> [options]

Output:
  <graphs-dir>/excess_symmetry_mates.csv - one row per excess ligand instance: dataset,
    despot_filtered_chain, despot_filtered_resi, is_symmetry_mate_of_reference,
    closest_ref_chain, closest_ref_resi, closest_distance_A.
  <graphs-dir>/excess_symmetry_mates.png - bar chart of the percentage that are/aren't.

Run inside CONDA_ENV_QFIT (needs qfit + cctbx, like symmetry_expand.py/despot_filter.py -
CONDA_ENV_EVAL, used by the other despot-stage plotting scripts, doesn't have these).
"""
from pathlib import Path

import numpy as np
import pandas as pd
from cctbx import crystal

from qfit import Structure

from rscc_common import build_ref_argparser, ref_pdb_path, plot_percentage_bar_chart


def read_cell_lookup(path):
    """Parses '--cell-lookup-file' lines "dataset a b c alpha beta gamma space_group" (the
    same format program.sh's DESPOT_CELL_LOOKUP_FILE already builds from CSV_FILE) into
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


def despot_dir(dataset, args):
    return (Path(args.datasets_dir) / dataset / args.run_name / args.placer_run_name /
            args.filter_run_name / args.placer2_run_name / args.filter2_run_name /
            args.final_run_name / args.rotamer_run_name / args.despot_run_name)


def ref_ligand_sites(ref_structure):
    """Distinct (chain, resi) LIG sites in ref_structure, altloc-agnostic (every altloc of a
    residue grouped together - see module docstring for why)."""
    lig = ref_structure.extract('resname LIG')
    if lig.natoms == 0:
        return []
    return sorted(set(zip(lig.chain, lig.resi)))


def closest_symmetry_mate_distance(structure, ref_lig, excess_lig, cushion):
    """Minimum centroid distance, across every non-identity crystallographic symmetry
    operation, between a symmetry-transformed copy of ref_lig and excess_lig (both already-
    extracted Structure objects). structure is whichever Structure iter_struct_orth_symops is
    called on (must have had set_crystal_symmetry applied) - only its .unit_cell matters here.
    Returns None if no non-identity symop was even considered (ref_lig/excess_lig too far
    apart for any mate to plausibly reach within cushion, per iter_struct_orth_symops' own
    broad-phase cushion)."""
    baseline_coor = ref_lig.coor.copy()
    excess_centroid = excess_lig.coor.mean(axis=0)
    best = None
    for symop in structure.unit_cell.iter_struct_orth_symops(ref_lig, target=excess_lig,
                                                               cushion=cushion):
        if symop.is_identity():
            continue
        ref_lig.rotate(symop.R)
        ref_lig.translate(symop.t)
        dist = float(np.linalg.norm(ref_lig.coor.mean(axis=0) - excess_centroid))
        if best is None or dist < best:
            best = dist
        ref_lig.coor = baseline_coor
    return best


def check_one_excess_ligand(row, args, cells):
    dataset = row['dataset']
    chain, resi = row['despot_filtered_chain'], int(row['despot_filtered_resi'])
    label = f'{dataset}_{chain}{resi}'

    cell = cells.get(dataset)
    if cell is None:
        print(f'  {label}: no cell/space-group info in --cell-lookup-file; skipping.')
        return None
    a, b, c, alpha, beta, gamma, space_group = cell

    despot_filtered_pdb = despot_dir(dataset, args) / 'despot_filtered.pdb'
    ref_pdb = ref_pdb_path(args, dataset)
    if not despot_filtered_pdb.is_file():
        print(f'  {label}: despot_filtered.pdb not found: {despot_filtered_pdb}; skipping.')
        return None
    if not ref_pdb.exists():
        print(f'  {label}: reference structure not found: {ref_pdb}; skipping.')
        return None

    despot_structure = Structure.fromfile(str(despot_filtered_pdb))
    excess_lig = despot_structure.extract(f'chain {chain} and resi {resi} and resname LIG')
    if excess_lig.natoms == 0:
        print(f'  {label}: no LIG atoms at chain {chain} resi {resi} in {despot_filtered_pdb}; '
              f'skipping.')
        return None

    ref_structure = Structure.fromfile(str(ref_pdb))
    crystal_symmetry = crystal.symmetry(
        unit_cell=(a, b, c, alpha, beta, gamma), space_group_symbol=space_group,
    )
    ref_structure.set_crystal_symmetry(crystal_symmetry)
    ref_structure._kwargs['crystal_symmetry'] = crystal_symmetry  # pylint: disable=protected-access

    sites = ref_ligand_sites(ref_structure)
    if not sites:
        print(f'  {label}: no LIG residue found in reference {ref_pdb}; skipping.')
        return None

    best_dist, best_site = None, None
    for ref_chain, ref_resi in sites:
        ref_lig = ref_structure.extract(f'chain {ref_chain} and resi {ref_resi} and resname LIG')
        if ref_lig.natoms == 0:
            continue
        dist = closest_symmetry_mate_distance(ref_structure, ref_lig, excess_lig,
                                               cushion=args.centroid_cutoff)
        if dist is not None and (best_dist is None or dist < best_dist):
            best_dist, best_site = dist, (ref_chain, ref_resi)

    is_mate = best_dist is not None and best_dist <= args.centroid_cutoff
    if best_dist is None:
        print(f'  {label}: no symmetry-mate candidate found near any reference ligand.')
    else:
        print(f'  {label}: closest symmetry-mate distance = {best_dist:.2f} A '
              f'(vs reference {best_site[0]}{best_site[1]}) -> '
              f'{"IS" if is_mate else "is NOT"} a symmetry mate.')

    return {
        'dataset': dataset, 'despot_filtered_chain': chain, 'despot_filtered_resi': resi,
        'is_symmetry_mate_of_reference': is_mate,
        'closest_ref_chain': best_site[0] if best_site else '',
        'closest_ref_resi': best_site[1] if best_site else '',
        'closest_distance_A': round(best_dist, 3) if best_dist is not None else '',
    }


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

    out_dir = Path(args.graphs_dir)
    excess_csv = out_dir / 'lig_vs_reference_rscc_excess_pipeline.csv'
    if not excess_csv.is_file():
        print(f'{excess_csv} not found - run plot_lig_vs_ref_despot.py first; nothing to check.')
        return
    excess_df = pd.read_csv(excess_csv)
    if excess_df.empty:
        print('No excess pipeline ligand(s); nothing to check.')
        return

    cells = read_cell_lookup(args.cell_lookup_file)

    rows = []
    for _, row in excess_df.iterrows():
        result = check_one_excess_ligand(row, args, cells)
        if result is not None:
            rows.append(result)

    if not rows:
        print('No excess ligand(s) could be checked; nothing to write.')
        return

    out_df = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / 'excess_symmetry_mates.csv'
    out_df.to_csv(out_csv, index=False)
    print(f'\n{len(out_df)} row(s) written to {out_csv}')

    n_mate = int(out_df['is_symmetry_mate_of_reference'].sum())
    n_total = len(out_df)
    n_not = n_total - n_mate
    plot_percentage_bar_chart(
        ['Symmetry mate\nof reference', 'Not a symmetry\nmate'],
        [100 * n_mate / n_total, 100 * n_not / n_total],
        [n_mate, n_not],
        title='Excess Ligands: Symmetry Mate of Reference?',
        out_path=out_dir / 'excess_symmetry_mates.png',
    )


if __name__ == '__main__':
    main()
