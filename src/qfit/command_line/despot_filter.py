import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from qfit import Structure


def build_argparser():
    p = argparse.ArgumentParser(
        description="Filters out physically implausible ligand poses from a structure, using "
                    "DESPOT binding-energy scores (DESPOT's score_complex.py) normalized per "
                    "heavy atom. Every ligand (resname LIG) instance in protein_structure is "
                    "the same molecule (just a different pose), so it always has the same heavy "
                    "atom count - normalizing by that count makes the per-instance scores "
                    "comparable to a single fixed threshold regardless of ligand size. An "
                    "instance whose normalized score is above threshold is dropped; everything "
                    "else (the protein, and every ligand instance that passes) is written "
                    "unchanged to output_pdb."
    )
    p.add_argument(
        'protein_structure',
        type=Path,
        help='Path to a structure containing one or more ligand (resname LIG) instances, e.g. '
             'final_model_refined.pdb.'
    )
    p.add_argument(
        'ligand_energy_csv',
        type=Path,
        help="Path to a DESPOT score_complex.py output csv (columns: ligand,score). Each "
             "'ligand' value is expected in score_complex.py's own 'lig<chain><resnum>' naming "
             "(e.g. 'ligC1') - see assign_bond_orders.py, which is how that name was assigned "
             "in the first place - so it can be matched directly back to protein_structure's "
             "(chain, resnum) ligand instances."
    )
    p.add_argument(
        'output_pdb',
        type=Path,
        help='Path to write the filtered structure to (e.g. despot_filtered.pdb).'
    )
    p.add_argument(
        '--threshold',
        default=-1.0,
        metavar='<float>',
        type=float,
        help='Per-heavy-atom-normalized DESPOT score above which a ligand instance is '
             'considered physically implausible and removed (default: 0.0)'
    )
    return p


def _instance_key(chain_id, resi, icode):
    """(chain, resi, icode) -> the same 'lig<chain><resnum><icode>' label
    assign_bond_orders.py gives that instance earlier in the pipeline (ligs.pdb
    -> ligs.sdf -> ligs.mol2 -> DESPOT's score_complex.py), i.e. the exact
    string ligand_energy_csv's 'ligand' column holds for it."""
    return f'lig{chain_id}{resi}{icode}'


def filter_ligands(structure, scores_by_label, threshold):
    """Returns (kept_mask, n_kept, n_removed, n_missing_score): a boolean mask
    over structure.atoms (True = keep) that drops every atom belonging to a
    ligand (resname LIG) instance whose DESPOT score, normalized by heavy-atom
    count, is above threshold. Every non-ligand atom is always kept.

    Every distinct (chain, resi, icode) LIG instance is treated as one pose of
    the same ligand molecule - final_model_refined.pdb's ligand instances are
    numbered sequentially starting at 1 (build_final_model.py), the same
    position-based indexing convention as filter2_run_name/cluster_reps.csv's
    rows, so instance resi i corresponds to cluster_reps.csv's i-th data row -
    but that correspondence isn't needed here, since ligand_energy_csv already
    identifies each instance by its own (chain, resnum) label directly.
    """
    chain_arr = structure.chain
    resi_arr = structure.resi
    icode_arr = structure.icode
    is_lig = structure.resn == 'LIG'

    ligand = structure.extract('resname LIG')
    if ligand.natoms == 0:
        return np.ones(structure.natoms, dtype=bool), 0, 0, 0

    n_heavy_atoms = _heavy_atom_count(ligand, chain_arr[is_lig][0], resi_arr[is_lig][0], icode_arr[is_lig][0])
    print(f'{n_heavy_atoms} heavy atom(s) per ligand instance (normalization divisor)')

    instances = []
    seen = set()
    for chain_id, resi, icode in zip(chain_arr[is_lig], resi_arr[is_lig], icode_arr[is_lig]):
        key = (chain_id, resi, icode)
        if key not in seen:
            seen.add(key)
            instances.append(key)
    instances.sort(key=lambda k: (k[1], k[0], k[2]))

    kept_mask = np.ones(structure.natoms, dtype=bool)
    n_kept, n_removed, n_missing_score = 0, 0, 0

    for chain_id, resi, icode in instances:
        label = _instance_key(chain_id, resi, icode)
        if label not in scores_by_label:
            print(f'  WARNING: no DESPOT score found for {label} in ligand_energy_csv; '
                  f'keeping it unfiltered.')
            n_missing_score += 1
            continue

        raw_score = scores_by_label[label]
        normalized_score = raw_score / n_heavy_atoms
        instance_mask = is_lig & (chain_arr == chain_id) & (resi_arr == resi) & (icode_arr == icode)

        if normalized_score > threshold:
            kept_mask[instance_mask] = False
            n_removed += 1
            print(f'  Removing {label}: score={raw_score:.3f}, normalized={normalized_score:.4f} '
                  f'> threshold {threshold}')
        else:
            n_kept += 1
            print(f'  Keeping {label}: score={raw_score:.3f}, normalized={normalized_score:.4f} '
                  f'<= threshold {threshold}')

    return kept_mask, n_kept, n_removed, n_missing_score


def _heavy_atom_count(ligand, chain_id, resi, icode):
    """Number of heavy (non-hydrogen) atoms in one ligand instance - the same
    for every LIG instance in one protein_structure, since they're all poses
    of the same ligand molecule, so any single instance's count works as the
    normalization divisor for all of them."""
    instance = ligand.extract(
        (ligand.chain == chain_id) & (ligand.resi == resi) & (ligand.icode == icode)
    )
    return int(np.sum(instance.e != 'H'))


def main():
    args = build_argparser().parse_args()

    structure = Structure.fromfile(str(args.protein_structure))

    scores_df = pd.read_csv(args.ligand_energy_csv)
    scores_by_label = dict(zip(scores_df['ligand'], scores_df['score']))

    kept_mask, n_kept, n_removed, n_missing_score = filter_ligands(
        structure, scores_by_label, args.threshold
    )

    filtered = structure.extract(kept_mask).copy()
    args.output_pdb.parent.mkdir(parents=True, exist_ok=True)
    filtered.tofile(str(args.output_pdb))

    print(f'Kept {n_kept} ligand instance(s), removed {n_removed} (normalized DESPOT score > '
          f'{args.threshold}), {n_missing_score} with no DESPOT score (kept unfiltered). '
          f'Wrote {args.output_pdb}.')


if __name__ == '__main__':
    main()
