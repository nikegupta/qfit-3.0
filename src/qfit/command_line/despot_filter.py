import argparse
import csv
import sys
from pathlib import Path

import iotbx.pdb
import numpy as np
import pandas as pd

from qfit import Structure
from qfit import XMap
from qfit.xtal.transformer import get_transformer
from qfit.command_line.calc_rscc import parse_bdc

DEFAULT_BFACTOR = 20


def build_argparser():
    p = argparse.ArgumentParser(
        description="Reselects, per filter2 cluster, which placer2 ligand conformer becomes "
                    "that cluster's final pose - instead of always keeping filter2's own "
                    "MSE-selected representative - by trading off DESPOT binding-energy score "
                    "against a real, internally-computed RSCC (via qfit's own transformer - no "
                    "external calc_rscc/dummy-pdb shimmy needed). Per cluster: takes the "
                    "MSE-vs-normalized-DESPOT Pareto front over that cluster's members "
                    "(filter2_dir/cluster_members.csv), computes each front member's RSCC "
                    "against every given map (max across maps), and keeps whichever member "
                    "maximizes RSCC - rscc_weight*normalized_DESPOT - but only if that winner's "
                    "RSCC and normalized DESPOT score both clear their thresholds, otherwise "
                    "that cluster's ligand is dropped entirely."
    )
    p.add_argument(
        'final_model_pdb', type=Path,
        help='Path to final_model_refined.pdb - supplies the protein atoms (unchanged) and '
             'each cluster\'s current ligand instance (chain/resi/icode - only its coordinates '
             'may be replaced by a different conformer; its identity/slot in the output never '
             'changes).'
    )
    p.add_argument(
        'filter2_dir', type=Path,
        help='Directory containing filter2\'s cluster_reps.csv and cluster_members.csv.'
    )
    p.add_argument(
        'despot_run_dir', type=Path,
        help='Directory containing this dataset\'s Stage 7a outputs: <dataset>_DESPOT.csv '
             '(exactly one file matching *_DESPOT.csv is expected), conformer_map.csv, and '
             'ligs.pdb (every placer2 conformer\'s coordinates, chain L, resnum = instance id - '
             'see extract_ligand_conformers).'
    )
    p.add_argument(
        'map_files', type=Path, nargs='+',
        help='Path(s) to one or more density map files (e.g. .ccp4 event maps) to score each '
             'Pareto-front candidate against. A candidate\'s RSCC is the max across all maps '
             'given. Maps whose filename embeds a duplicate 1-BDC_<value>_ (same '
             'background-subtraction fraction as an already-loaded map) are skipped, since they '
             'are identical to that map - see calc_rscc.py\'s parse_bdc.'
    )
    p.add_argument(
        'resolution', type=float,
        help='Resolution (Å) of the map(s), used both for loading the XMap(s) and for the mask '
             'radius.'
    )
    p.add_argument(
        'output_pdb', type=Path,
        help='Path to write the filtered/reselected structure to (e.g. despot_filtered.pdb). '
             'despot_filtered_scores.csv and cluster_reps.csv are written alongside it, in the '
             'same directory.'
    )
    p.add_argument(
        '--despot-threshold', dest='despot_threshold', type=float, default=-1.0, metavar='<float>',
        help='A cluster\'s winning pose must have a per-heavy-atom-normalized DESPOT score <= '
             'this to survive (default: -1.0).'
    )
    p.add_argument(
        '--rscc-threshold', dest='rscc_threshold', type=float, default=0.6, metavar='<float>',
        help='A cluster\'s winning pose must have an RSCC >= this to survive (default: 0.6).'
    )
    p.add_argument(
        '--rscc-weight', dest='rscc_weight', type=float, default=0.05, metavar='<float>',
        help='Weight applied to normalized DESPOT score when picking the winner: '
             'argmax(RSCC - rscc_weight*normalized_DESPOT) (default: 0.05).'
    )
    return p


def pareto_front(mses, scores):
    """Returns a bool list, True for every index i whose (mse, score) is non-dominated: no other
    index j has mses[j] <= mses[i] and scores[j] <= scores[i] with at least one strictly lower
    (both lower is better for MSE and normalized DESPOT score alike). O(n^2), fine at
    per-cluster sizes. Identical to program_exp/test/extract_nondominated_candidates.py's
    pareto_front(), the exploratory workflow this reselection is promoted from."""
    n = len(mses)
    non_dominated = []
    for i in range(n):
        dominated = any(
            j != i and mses[j] <= mses[i] and scores[j] <= scores[i]
            and (mses[j] < mses[i] or scores[j] < scores[i])
            for j in range(n)
        )
        non_dominated.append(not dominated)
    return non_dominated


def find_final_model_lig_instances(structure):
    """Returns [(chain_id, resi, icode), ...] for every distinct LIG instance in structure,
    sorted by resi - build_final_model.py numbers ligand instances 1..N in exactly this order,
    the same order as filter2_dir/cluster_reps.csv's rows, so the i-th instance here
    corresponds to cluster_reps.csv's i-th data row."""
    chain_arr = structure.chain
    resi_arr = structure.resi
    icode_arr = structure.icode
    is_lig = structure.resn == 'LIG'

    instances = []
    seen = set()
    for chain_id, resi, icode in zip(chain_arr[is_lig], resi_arr[is_lig], icode_arr[is_lig]):
        key = (chain_id, resi, icode)
        if key not in seen:
            seen.add(key)
            instances.append(key)
    instances.sort(key=lambda k: (k[1], k[0], k[2]))
    return instances


def load_maps(map_files, resolution):
    """Loads every map in map_files into an {name: XMap} dict, along with a matching
    {name: zeroed-template XMap} dict used to build each candidate's model density. Event maps
    sharing a BDC value are identical - only the first map seen for a given BDC is kept -
    identical deduplication to calc_rscc.py's ResidueRSCCCalculator._load_maps."""
    maps = {}
    map_models = {}
    seen_bdcs = set()
    for map_file in map_files:
        bdc = parse_bdc(map_file.name)
        if bdc is not None:
            if bdc in seen_bdcs:
                print(f'Skipping map {map_file}: duplicate BDC={bdc} '
                      f'(another event map with this BDC was already loaded).')
                continue
            seen_bdcs.add(bdc)
        name = map_file.name
        print(f'Loading map {map_file} at resolution {resolution}')
        maps[name] = XMap.fromfile(str(map_file), resolution=resolution)
        map_model = maps[name].zeros_like(maps[name])
        map_model.set_space_group("P1")
        map_models[name] = map_model
    return maps, map_models


def score_rscc(residue_structure, coor, maps, map_models, rmask, bfactor=DEFAULT_BFACTOR):
    """Max RSCC across every given map for one ligand conformer - same transformer recipe as
    calc_rscc.py's ResidueRSCCCalculator._score_residue (get_transformer, get_conformers_mask,
    get_conformers_densities, np.corrcoef), done directly in-process rather than shelling out to
    calc_rscc against a dummy single-ligand pdb."""
    scaled_bulk_solvent = 0
    coor_set = [coor]
    bfactor_array = [bfactor]

    rsccs = []
    for name in maps:
        transformer = get_transformer("qfit", residue_structure, map_models[name])
        mask = transformer.get_conformers_mask(coor_set, rmask)
        target = maps[name].array[mask]
        for density in transformer.get_conformers_densities(coor_set, bfactor_array):
            model_density = density[mask]
            np.maximum(model_density, scaled_bulk_solvent, out=model_density)
            correlation_matrix = np.corrcoef(model_density, target)
            rsccs.append(correlation_matrix[0, 1])
    return max(rsccs)


def _set_resi(structure, resi):
    """Sets the residue number of every atom in structure to resi - identical to
    build_final_model.py's _set_resi (Structure.resi is a derived, read-only property; the
    residue number has to be changed at the source, each atom's residue_group.resseq)."""
    resseq = iotbx.pdb.resseq_encode(resi)
    seen = set()
    for atom in structure.atoms:
        residue_group = atom.parent().parent()
        if id(residue_group) in seen:
            continue
        residue_group.resseq = resseq
        seen.add(id(residue_group))


def _set_chain(structure, chain_id):
    """Sets the chain id of every atom in structure to chain_id - same technique as
    symmetry_expand.py's _reassign_chain_ids (direct atom.chain().id assignment)."""
    for atom in structure.atoms:
        atom.chain().id = chain_id


CLUSTER_REPS_ORIGINAL_COLUMNS = [
    'placer_file', 'index', 'mse', 'cluster', 'rscc', 'num_members', 'cif_restraints_file',
]
CLUSTER_REPS_DESPOT_COLUMNS = [
    'despot_placer_file', 'despot_index', 'despot_mse', 'despot_normalized_score',
    'despot_rscc', 'despot_tradeoff_score', 'despot_passed',
]


def main():
    args = build_argparser().parse_args()

    despot_csvs = sorted(args.despot_run_dir.glob('*_DESPOT.csv'))
    if len(despot_csvs) != 1:
        sys.exit(f'Error: expected exactly one *_DESPOT.csv in {args.despot_run_dir}, found '
                  f'{len(despot_csvs)}')
    despot_csv = despot_csvs[0]

    conformer_map_csv = args.despot_run_dir / 'conformer_map.csv'
    ligs_pdb = args.despot_run_dir / 'ligs.pdb'
    cluster_reps_csv = args.filter2_dir / 'cluster_reps.csv'
    cluster_members_csv = args.filter2_dir / 'cluster_members.csv'
    for p in (conformer_map_csv, ligs_pdb, cluster_reps_csv, cluster_members_csv,
              args.final_model_pdb):
        if not p.is_file():
            sys.exit(f'Error: required file not found: {p}')

    # --- Join every placer2 conformer to its normalized DESPOT score, by ligand NAME (never
    # position - score_complex.py's own output order isn't reliable) ---
    despot_scores = pd.read_csv(despot_csv)  # columns: ligand, score
    despot_by_name = dict(zip(despot_scores['ligand'], despot_scores['score']))

    conformer_map = pd.read_csv(conformer_map_csv)
    # (source file basename, 0-based index) -> {resnum, raw_score, normalized_score}
    despot_lookup = {}
    for _, row in conformer_map.iterrows():
        raw_score = despot_by_name.get(row['ligand_name'])
        if raw_score is None:
            continue
        key = (Path(row['source_file']).name, int(row['model_number']) - 1)
        despot_lookup[key] = {
            'resnum': int(row['resnum']),
            'raw_score': raw_score,
            'normalized_score': raw_score / row['n_atoms'],
        }

    cluster_reps = pd.read_csv(cluster_reps_csv)
    cluster_members = pd.read_csv(cluster_members_csv)
    accepted_ids = set(cluster_reps['cluster'])
    cluster_members = cluster_members[cluster_members['cluster'].isin(accepted_ids)]

    structure = Structure.fromfile(str(args.final_model_pdb))
    lig_instances = find_final_model_lig_instances(structure)
    if len(lig_instances) != len(cluster_reps):
        print(f'Warning: {len(lig_instances)} LIG instance(s) in {args.final_model_pdb} but '
              f'{len(cluster_reps)} row(s) in {cluster_reps_csv} - using the first '
              f'{min(len(lig_instances), len(cluster_reps))}.')
    n_clusters = min(len(lig_instances), len(cluster_reps))

    ligs_structure = Structure.fromfile(str(ligs_pdb))
    maps, map_models = load_maps(args.map_files, args.resolution)
    rmask = 0.5 + args.resolution / 3.0

    chain_arr = structure.chain
    resi_arr = structure.resi
    icode_arr = structure.icode
    is_lig = structure.resn == 'LIG'

    protein_output = structure.extract('not resname LIG')
    pieces = []
    scores_rows = []
    cluster_rows = []

    for i in range(n_clusters):
        chain_id, resi, icode = lig_instances[i]
        label = f'lig{chain_id}{resi}{icode}'
        rep = cluster_reps.iloc[i]
        cluster_id = rep['cluster']

        cluster_row = {col: rep[col] for col in CLUSTER_REPS_ORIGINAL_COLUMNS}
        cluster_row.update({col: None for col in CLUSTER_REPS_DESPOT_COLUMNS})
        cluster_row['despot_passed'] = False

        members = cluster_members[cluster_members['cluster'] == cluster_id]
        member_infos = []
        for _, member in members.iterrows():
            key = (Path(member['placer_file']).name, int(member['index']))
            info = despot_lookup.get(key)
            if info is None:
                continue
            member_infos.append({
                'placer_file': member['placer_file'], 'index': member['index'],
                'mse': member['mse'], 'resnum': info['resnum'],
                'raw_score': info['raw_score'], 'normalized_score': info['normalized_score'],
            })

        def drop(reason):
            print(f'  Dropping cluster {cluster_id} ({label}): {reason}')
            scores_rows.append({'ligand': label, 'chain': chain_id, 'resi': resi,
                                 'icode': icode, 'raw_score': None, 'normalized_score': None,
                                 'kept': False})
            cluster_rows.append(cluster_row)

        if not member_infos:
            drop('no member has a DESPOT score')
            continue

        nondominated = pareto_front([m['mse'] for m in member_infos],
                                     [m['normalized_score'] for m in member_infos])

        best, best_tradeoff, best_rscc, best_structure = None, None, None, None
        for member, keep in zip(member_infos, nondominated):
            if not keep:
                continue
            candidate_structure = ligs_structure.extract(f'chain L and resi {member["resnum"]}')
            if candidate_structure.natoms == 0:
                print(f'  WARNING: no atoms found in {ligs_pdb} for resnum {member["resnum"]} '
                      f'(cluster {cluster_id}) - skipping this candidate.')
                continue
            coor = candidate_structure.coor.copy()
            rscc = score_rscc(candidate_structure, coor, maps, map_models, rmask)
            tradeoff = rscc - args.rscc_weight * member['normalized_score']
            if best_tradeoff is None or tradeoff > best_tradeoff:
                best, best_tradeoff, best_rscc, best_structure = (
                    member, tradeoff, rscc, candidate_structure)

        if best is None:
            drop('no Pareto-front candidate had coordinates in ligs.pdb')
            continue

        despot_passed = (best_rscc >= args.rscc_threshold
                          and best['normalized_score'] <= args.despot_threshold)
        cluster_row.update({
            'despot_placer_file': best['placer_file'], 'despot_index': best['index'],
            'despot_mse': best['mse'], 'despot_normalized_score': best['normalized_score'],
            'despot_rscc': best_rscc, 'despot_tradeoff_score': best_tradeoff,
            'despot_passed': despot_passed,
        })
        cluster_rows.append(cluster_row)

        print(f'  Cluster {cluster_id} ({label}): winner {best["placer_file"]}[{best["index"]}] '
              f'rscc={best_rscc:.4f} normalized_despot={best["normalized_score"]:.4f} '
              f'tradeoff={best_tradeoff:.4f} passed={despot_passed}')

        if not despot_passed:
            scores_rows.append({'ligand': label, 'chain': chain_id, 'resi': resi,
                                 'icode': icode, 'raw_score': best['raw_score'],
                                 'normalized_score': best['normalized_score'], 'kept': False})
            continue

        orig_natoms = int(np.sum(
            is_lig & (chain_arr == chain_id) & (resi_arr == resi) & (icode_arr == icode)
        ))
        if best_structure.natoms != orig_natoms:
            print(f'  WARNING: atom count mismatch for cluster {cluster_id} ({label}): '
                  f'original has {orig_natoms}, winning candidate has {best_structure.natoms} - '
                  f'dropping.')
            cluster_row['despot_passed'] = False
            scores_rows.append({'ligand': label, 'chain': chain_id, 'resi': resi,
                                 'icode': icode, 'raw_score': best['raw_score'],
                                 'normalized_score': best['normalized_score'], 'kept': False})
            continue

        winner = best_structure.copy()
        _set_resi(winner, int(resi))
        _set_chain(winner, chain_id)
        pieces.append((chain_id, resi, winner))
        scores_rows.append({'ligand': label, 'chain': chain_id, 'resi': resi, 'icode': icode,
                             'raw_score': best['raw_score'],
                             'normalized_score': best['normalized_score'], 'kept': True})

    pieces.sort(key=lambda piece: (piece[0], piece[1]))
    output_structure = protein_output
    for _, _, piece in pieces:
        output_structure = output_structure.combine(piece)

    args.output_pdb.parent.mkdir(parents=True, exist_ok=True)
    output_structure.tofile(str(args.output_pdb))

    scores_csv = args.output_pdb.parent / 'despot_filtered_scores.csv'
    pd.DataFrame(scores_rows, columns=['ligand', 'chain', 'resi', 'icode', 'raw_score',
                                        'normalized_score', 'kept']).to_csv(scores_csv, index=False)

    despot_cluster_reps_csv = args.output_pdb.parent / 'cluster_reps.csv'
    pd.DataFrame(
        cluster_rows, columns=CLUSTER_REPS_ORIGINAL_COLUMNS + CLUSTER_REPS_DESPOT_COLUMNS
    ).to_csv(despot_cluster_reps_csv, index=False)

    n_kept = sum(1 for r in scores_rows if r['kept'])
    print(f'Kept {n_kept}/{len(scores_rows)} ligand instance(s). Wrote {args.output_pdb}, '
          f'{scores_csv}, {despot_cluster_reps_csv}.')


if __name__ == '__main__':
    main()
