#!/usr/bin/env python3
"""
Pooled (across every dataset in datasets.txt) ligand centroid-distance
comparison: every reference-set LIG conformation vs the ligand poses already
present in fit_ligand's own output structures (run_name/*.pdb) - before any
PLACER sampling or refinement has touched them.

Unlike the later-stage reference comparisons in this pipeline (stages 3/5/6,
which trust that both sides already share the reference's crystallographic
frame), fit_ligand's output structures aren't guaranteed to - so each
candidate structure is first superimposed onto the reference by protein CA
atoms (matched by chain_id/res_id) before ligand centroid distances are
computed.

For each reference LIG conformation (altloc-aware), keeps the minimum
centroid distance to any ligand pose across every fit_ligand PDB for that
dataset, pooled across all datasets into a single histogram.

Run at the end of stage 1 (fit_ligand), only when -c (compare to reference
set) is given.

Usage:
  centroid_rmsd_all.py <run_name> --ref-set <dir> --graphs-dir <dir> [options]
"""
from pathlib import Path

import numpy as np
import pandas as pd
import biotite.structure as struc
import biotite.structure.io.pdb as pdb

from rscc_common import (
    build_ref_argparser, read_datasets, read_pdb_raw_atoms, lig_conformations_filtered,
    plot_distance_histogram, ref_pdb_path, write_plot_csv,
)


def read_pdb_biotite(path):
    pdb_file = pdb.PDBFile.read(str(path))
    return pdb_file.get_structure(model=1)


def get_protein_ca(structure):
    ca = structure[struc.filter_amino_acids(structure)]
    return ca[ca.atom_name == 'CA']


def align_to_reference(mobile_struct, target_struct):
    """Superimposes mobile onto target using CA atoms matched by
    (chain_id, res_id). Raises ValueError if no residues are in common."""
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


def process_dataset(dataset, run_dir, ref_path, model_chain, model_resi):
    """Returns (matched_rows, not_found_rows):
      matched_rows - list of {ref_chain, ref_resi, ref_altloc, dist} dicts,
        one per reference LIG conformation matched against the closest
        ligand pose (after CA superposition) across every *.pdb file
        directly in run_dir.
      not_found_rows - list of {ref_chain, ref_resi, ref_altloc} dicts (no
        dist - there's nothing to measure a distance to) for every
        reference LIG conformation in a dataset whose fit_ligand run
        produced zero output pdbs (run_dir exists and has a reference
        structure, but its *.pdb glob is empty). Distinct from run_dir not
        existing at all, which means fit_ligand simply hasn't been run for
        this dataset yet - not a placement failure - and contributes
        nothing to either list.
    """
    if not run_dir.exists() or not ref_path.exists():
        return [], []

    ref_atoms = read_pdb_raw_atoms(ref_path)
    ref_struct = read_pdb_biotite(ref_path)
    ref_confs = lig_conformations_filtered(ref_atoms)
    if not ref_confs:
        return [], []

    pdb_files = sorted(run_dir.glob('*.pdb'))
    if not pdb_files:
        not_found_rows = [
            {'ref_chain': key[0], 'ref_resi': key[1], 'ref_altloc': key[2]}
            for key in ref_confs
        ]
        return [], not_found_rows

    min_dists = {key: np.inf for key in ref_confs}

    for pdb_file in pdb_files:
        try:
            model_struct = read_pdb_biotite(pdb_file)
            transformation = align_to_reference(model_struct, ref_struct)
        except Exception as e:
            print(f'    Warning: could not align {pdb_file.name} ({dataset}): {e}')
            continue

        model_atoms = read_pdb_raw_atoms(pdb_file)
        model_confs = lig_conformations_filtered(model_atoms, chain_id=model_chain, res_id=model_resi)
        if not model_confs:
            model_confs = lig_conformations_filtered(model_atoms)
            if not model_confs:
                continue

        for ref_key, ref_centroid in ref_confs.items():
            for m_centroid in model_confs.values():
                aligned = transformation.apply(m_centroid.reshape(1, 3))[0]
                dist = float(np.linalg.norm(ref_centroid - aligned))
                if dist < min_dists[ref_key]:
                    min_dists[ref_key] = dist

    matched_rows = [
        {'ref_chain': key[0], 'ref_resi': key[1], 'ref_altloc': key[2], 'dist': dist}
        for key, dist in min_dists.items() if np.isfinite(dist)
    ]
    return matched_rows, []


def main():
    args = build_ref_argparser(__doc__, ['run_name']).parse_args()

    datasets = read_datasets(args.datasets_file)
    all_rows = []
    all_not_found_rows = []
    for dataset in datasets:
        run_dir = Path(args.datasets_dir) / dataset / args.run_name
        ref_path = ref_pdb_path(args, dataset)
        rows, not_found_rows = process_dataset(dataset, run_dir, ref_path, model_chain='C', model_resi=1)
        print(f'  {dataset}: {len(rows)} ref LIG conformation(s) matched'
              + (f', {len(not_found_rows)} not found (fit_ligand placed 0 ligands)'
                 if not_found_rows else ''))
        for row in rows:
            row['dataset'] = dataset
        for row in not_found_rows:
            row['dataset'] = dataset
        all_rows.extend(rows)
        all_not_found_rows.extend(not_found_rows)

    out_dir = Path(args.graphs_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_name = 'centroid_rmsd_all.png'
    plot_distance_histogram(
        [row['dist'] for row in all_rows],
        title=f'Ligand Centroid Distance: fit_ligand Structures vs Reference ({args.run_name})',
        xlabel='Minimum Centroid Distance to Closest fit_ligand Pose (Å)',
        out_path=out_dir / out_name,
        bin_width=1.0,
        not_found_count=len(all_not_found_rows),
    )
    if all_rows or all_not_found_rows:
        combined_rows = all_rows + [{**row, 'dist': None} for row in all_not_found_rows]
        write_plot_csv(out_dir, out_name,
                        pd.DataFrame(combined_rows)[['dataset', 'ref_chain', 'ref_resi', 'ref_altloc', 'dist']])


if __name__ == '__main__':
    main()
