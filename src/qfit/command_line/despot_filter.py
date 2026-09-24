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
from qfit.command_line.calc_rscc import parse_bdc, DEFAULT_BFACTOR

# Default margin (in standard deviations of that axis, across a cluster's own candidates) a
# candidate must beat another by, in BOTH mse and normalized DESPOT score, to count as
# dominating it for the Pareto front - see pareto_front()'s own docstring for why this isn't 0
# (the original, unmargined comparison).
PARETO_MARGIN_STD = 0.05


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
                    "that cluster's ligand is dropped entirely. Also enforces that every non-apo "
                    "protein conformation in the output is backed by a placer file whose ligand "
                    "survived this filtering: any residue whose only PLACER-derived conformer(s) "
                    "came from rejected-ligand placer file(s) is reset to its apo_structure "
                    "coordinates (see reset_protein_to_apo_where_unbacked), and the surviving "
                    "restricted residue list is written to modified_residues.csv."
    )
    p.add_argument(
        'final_model_pdb', type=Path,
        help='Path to the current protein+ligand structure (e.g. optimized.pdb, stage 7d\'s '
             'output - never worse per-residue than rotamer_refined.pdb by construction, see '
             'select_optimized_residues.py) - supplies the protein atoms and each cluster\'s '
             'current ligand instance (chain/resi/icode - only its coordinates may be replaced by '
             'a different conformer; its identity/slot in the output never changes). Protein '
             'residues belonging only to a rejected ligand\'s placer file (see apo_structure) are '
             'reset rather than carried through unchanged.'
    )
    p.add_argument(
        'apo_structure', type=Path,
        help='Path to the apo (ligand-free) PANDDA structure for this dataset (the same file '
             'build_final_model.py uses as its own apo fallback). Any protein residue whose only '
             'PLACER-derived conformation came from a placer file backing a ligand that this run '
             'rejects - and that is not ALSO part of a placer file backing a ligand that survives '
             '- is reset to its coordinates here, since this pipeline only wants non-apo protein '
             'conformations near ligands it actually kept.'
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
    p.add_argument(
        '--pareto-margin-std', dest='pareto_margin_std', type=float,
        default=PARETO_MARGIN_STD, metavar='<float>',
        help='How many standard deviations of margin (per axis: MSE, normalized DESPOT score) a '
             'candidate must beat another by to count as dominating it for the Pareto front - '
             f'see pareto_front()\'s docstring (default: {PARETO_MARGIN_STD}). 0 recovers the '
             'original, unmargined comparison.'
    )
    p.add_argument(
        '--bfactor', dest='bfactor', type=float, default=DEFAULT_BFACTOR, metavar='<float>',
        help='B-factor used when generating each Pareto-front candidate\'s model density for '
             f'its own internal RSCC scoring (default: {DEFAULT_BFACTOR}) - same variable/'
             'default as calc_rscc.py\'s own --bfactor.'
    )
    p.add_argument(
        '--residues-with-placer-conformers-csv', dest='residues_with_placer_conformers_csv',
        type=Path, default=None, metavar='<path>',
        help='Path to build_final_model.py\'s residues_with_placer_conformers.csv for this '
             'dataset/run. Defaults to final_model_pdb.parent.parent/'
             'residues_with_placer_conformers.csv (i.e. final_model_pdb is '
             '.../<final_run_name>/<rotamer_run_name>/optimized.pdb and this file lives in '
             '<final_run_name>/, the normal pipeline layout). Filtered down to '
             'modified_residues.csv: the residues that keep a non-apo conformation after this '
             'run\'s reset_protein_to_apo_where_unbacked.'
    )
    return p


def pareto_front(mses, scores, margin_std=PARETO_MARGIN_STD):
    """Returns a bool list, True for every index i whose (mse, score) is non-dominated. Point j
    dominates point i only if it beats i by MORE than margin_std standard deviations (of that
    axis, computed across every point passed in here) in BOTH mse and normalized DESPOT score -
    not the plain "less-or-equal in both, strictly lower in at least one" comparison this
    started as. That plain comparison is highly sensitive to noise-level differences between
    near-identical candidates: a candidate can get pruned off the front - and so never have its
    RSCC computed at all (RSCC is only computed for front survivors - see main()) - just because
    some other candidate edged it out by an amount too small to be a real difference. Requiring
    a real margin (default: 0.25 standard deviations) in both dimensions before counting as
    dominated keeps more plausible candidates in play, at the cost of computing RSCC for a
    larger front. margin_std=0 recovers the original, strict comparison exactly (a 0-point
    margin can never exceed 0, so it degrades to the plain <=/< comparison above's intent, aside
    from the "at least one strictly lower" nuance no longer being separately required once both
    dimensions must already each be strictly margin-better). O(n^2), fine at per-cluster sizes."""
    mses = np.asarray(mses, dtype=float)
    scores = np.asarray(scores, dtype=float)
    n = len(mses)
    mse_margin = margin_std * mses.std()
    score_margin = margin_std * scores.std()
    non_dominated = []
    for i in range(n):
        dominated = any(
            j != i and mses[j] <= mses[i] - mse_margin and scores[j] <= scores[i] - score_margin
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


def _placer_file_residues(placer_file, cache):
    """Returns the {(chain_id, res_num), ...} set of every non-LIG residue found in ANY model
    of placer_file - i.e. every protein residue that placer file could have contributed a
    conformer for, the same per-model union _gatherResidueConformers effectively pools over in
    build_final_model.py. Memoized in `cache` ({str(path): set}), since the same placer_file can
    back more than one cluster/ligand instance. Returns an empty set (after warning) for a
    missing/unreadable file - conservatively, so a broken path never causes a residue to be
    reset that shouldn't be."""
    key = str(placer_file)
    if key in cache:
        return cache[key]

    path = Path(placer_file)
    if not path.is_file():
        print(f'  WARNING: placer file not found, cannot determine its residues: {path}')
        cache[key] = set()
        return cache[key]

    residues = set()
    for model in Structure.fromfile(str(path)).split_models():
        protein = model.extract('not resname LIG')
        for chain_id, res_num in zip(protein.chain, protein.resi):
            residues.add((chain_id, int(res_num)))
    cache[key] = residues
    return residues


def _full_atom_mask(struct, subset_mask):
    """Translates subset_mask - a boolean array aligned to struct's own (possibly
    already-filtered) atom order, i.e. len(subset_mask) == struct.natoms - into a boolean mask
    aligned to struct's full, underlying (pre-selection) atom array.

    Structure.extract() applies a raw (non-string) selection array directly against the object's
    full underlying atom array, not against the object's own current selection - so handing it a
    mask built in struct's own atom order silently selects the wrong atoms wherever that order
    has a gap relative to the full array. This matters here specifically because
    reset_protein_to_apo_where_unbacked's output_structure is itself already a 'not resname LIG'
    selection before replace_residue_in_place below ever touches it. Identical to
    symmetry_expand.py's own _full_atom_mask (see that module's docstring for the full
    explanation) - duplicated here rather than imported since these are independent,
    self-contained command-line scripts."""
    full_mask = np.zeros(struct.total_length, dtype=bool)
    if struct.selection is None:
        full_mask[:] = subset_mask
    else:
        full_indices = np.array(list(struct.selection))
        full_mask[full_indices[subset_mask]] = True
    return full_mask


def replace_residue_in_place(target_structure, chain_id, res_num, replacement_structure, label):
    """Overwrites target_structure's atoms for (chain_id, res_num) with replacement_structure's
    same-named atoms' coordinates/B-factor/occupancy, IN PLACE - i.e. at the same position in
    target_structure's own underlying atom array - rather than deleting the residue and
    appending replacement_structure's copy of it at the end. The remove-and-append approach
    relocates the residue to wherever it falls in the appended tail of the eventual output pdb,
    which silently breaks any downstream tool that infers chain connectivity from sequential
    atom order rather than real 3D distances - pdb2pqr30 (run by protein_to_mol2.sh ahead of
    DESPOT scoring, i.e. right after this function runs) is exactly such a tool: it can see a
    residue relocated away from its real neighbors as a chain break, and cap the "orphaned"
    residue with a fake N/C-terminus (an extra OXT atom carrying real partial charge) even
    though the residue's actual 3D geometry is perfectly bonded to both neighbors. Same fix as
    select_optimized_residues.py's identically-named function, duplicated here rather than
    imported since these are two independent, self-contained command-line scripts.

    Relies on Structure.extract() returning a view over the SAME underlying atom storage as
    target_structure (confirmed empirically - mutating an extracted view's .coor/.b/.q mutates
    target_structure itself), so no separate re-assembly/combine() step is needed afterward. Both
    input masks are translated via _full_atom_mask before being handed to extract() - safe even
    though target_structure here is already a filtered ('not resname LIG') selection, not the
    freshly-loaded structure.

    Returns True on success. Returns False (target_structure left untouched for this residue) if
    either structure doesn't have this residue at all, or the two don't have exactly the same
    atom-name set for it - an identity mismatch can't be resolved by a like-for-like in-place
    swap, so the caller should fall back to the old remove-and-append behavior for just this
    residue in that case.
    """
    target_subset_mask = (target_structure.chain == chain_id) & (target_structure.resi == res_num)
    if not np.any(target_subset_mask):
        return False
    repl_subset_mask = (replacement_structure.chain == chain_id) & (replacement_structure.resi == res_num)
    if not np.any(repl_subset_mask):
        return False

    target_view = target_structure.extract(_full_atom_mask(target_structure, target_subset_mask))
    repl_view = replacement_structure.extract(_full_atom_mask(replacement_structure, repl_subset_mask))

    target_names = list(target_view.name)
    repl_names = list(repl_view.name)
    if sorted(target_names) != sorted(repl_names):
        print(f'  WARNING: {label} has a different atom set in the replacement structure '
              f'({sorted(repl_names)}) than in the target structure ({sorted(target_names)}) - '
              f'cannot do an in-place swap; falling back to append (this residue may end up out '
              f'of sequential order in the output pdb).')
        return False

    repl_coor_by_name = dict(zip(repl_names, repl_view.coor))
    repl_b_by_name = dict(zip(repl_names, repl_view.b))
    repl_q_by_name = dict(zip(repl_names, repl_view.q))

    target_view.coor = np.array([repl_coor_by_name[name] for name in target_names])
    target_view.b = np.array([repl_b_by_name[name] for name in target_names])
    target_view.q = np.array([repl_q_by_name[name] for name in target_names])
    return True


def reset_protein_to_apo_where_unbacked(structure, apo_structure, cluster_rows):
    """Enforces that every non-apo protein conformation in `structure` is backed by a placer
    file whose ligand survived DESPOT filtering (this pipeline only wants protein conformations
    fit near ligands it actually kept).

    For every cluster_row (one per ligand instance, cluster_reps.csv's own 'placer_file' column
    - the file build_final_model.py actually used to source that instance's protein/ligand
    conformers), pools that placer file's residues into `passed_residues` if despot_passed else
    `rejected_residues`. reset_keys = rejected_residues - passed_residues: residues seen only
    alongside rejected ligand(s), never a surviving one.

    Returns (output_protein_structure, reset_keys) - reset_keys is a sorted list of
    (chain_id, res_num) that were actually reset (i.e. also found in apo_structure); residues in
    reset_keys but absent from apo_structure are left as-is (warned about) rather than dropped.
    """
    cache = {}
    passed_residues, rejected_residues = set(), set()
    for row in cluster_rows:
        placer_file = row.get('placer_file')
        if placer_file is None or (isinstance(placer_file, float) and np.isnan(placer_file)):
            continue
        residues = _placer_file_residues(placer_file, cache)
        if row['despot_passed']:
            passed_residues |= residues
        else:
            rejected_residues |= residues

    reset_keys = sorted(rejected_residues - passed_residues)

    is_lig = structure.resn == 'LIG'
    # A view sharing structure's own underlying atom storage (see replace_residue_in_place) -
    # atoms are reset to apo IN PLACE below, at their original position in the protein's residue
    # order, rather than removed and re-appended at the end.
    output_structure = structure.extract(~is_lig)

    actually_reset = []
    fallback_mask = np.zeros(output_structure.natoms, dtype=bool)
    fallback_pieces = []
    for chain_id, res_num in reset_keys:
        residue_mask = (output_structure.chain == chain_id) & (output_structure.resi == res_num)
        if not np.any(residue_mask):
            continue
        label = f'{chain_id}{res_num}'
        if replace_residue_in_place(output_structure, chain_id, res_num, apo_structure, label):
            actually_reset.append((chain_id, res_num))
            continue
        # Atom-set mismatch - fall back to the old remove-and-append behavior for just this one
        # residue (replace_residue_in_place already printed why).
        apo_residue = apo_structure.extract(f'chain {chain_id} and resid {res_num}')
        if apo_residue.natoms == 0:
            print(f'  WARNING: {chain_id}{res_num} would be reset to apo (only backed by a '
                  f'rejected ligand\'s placer file) but has no apo_structure residue - leaving '
                  f'its current conformation in place.')
            continue
        fallback_mask |= residue_mask
        fallback_pieces.append(apo_residue)
        actually_reset.append((chain_id, res_num))

    if fallback_pieces:
        output_structure = output_structure.extract(~fallback_mask)
        for piece in fallback_pieces:
            output_structure = output_structure.combine(piece)

    if actually_reset:
        labels = ', '.join(f'{c}{r}' for c, r in actually_reset)
        print(f'  Reset {len(actually_reset)} residue(s) to apo (backed only by rejected '
              f'ligand placer file(s), not a surviving one): {labels}')
    else:
        print('  No residues needed resetting to apo.')

    return output_structure, actually_reset


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
              args.final_model_pdb, args.apo_structure):
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
    apo_structure = Structure.fromfile(str(args.apo_structure))
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
                                     [m['normalized_score'] for m in member_infos],
                                     margin_std=args.pareto_margin_std)

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
            rscc = score_rscc(candidate_structure, coor, maps, map_models, rmask,
                               bfactor=args.bfactor)
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

    protein_output, reset_residues = reset_protein_to_apo_where_unbacked(
        structure, apo_structure, cluster_rows)

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

    # --- modified_residues.csv: residues_with_placer_conformers.csv, restricted to residues
    # that still have a non-apo (PLACER-derived) conformation after reset_protein_to_apo_where_
    # unbacked - i.e. minus every residue reset above. Always <= residues_with_placer_conformers
    # .csv, since resetting only ever removes residues, never adds any. ---
    residues_csv = args.residues_with_placer_conformers_csv
    if residues_csv is None:
        residues_csv = args.final_model_pdb.parent.parent / 'residues_with_placer_conformers.csv'
    reset_labels = {f'{chain_id}{res_num}' for chain_id, res_num in reset_residues}
    modified_residues_csv = args.output_pdb.parent / 'modified_residues.csv'
    if residues_csv.is_file():
        with open(residues_csv) as f:
            all_labels = [line.strip() for line in f if line.strip()]
        modified_labels = [label for label in all_labels if label not in reset_labels]
        with open(modified_residues_csv, 'w') as f:
            for label in modified_labels:
                f.write(f'{label}\n')
        print(f'  {len(modified_labels)}/{len(all_labels)} residue(s) from {residues_csv} '
              f'retained a non-apo conformation; wrote {modified_residues_csv}.')
    else:
        print(f'  WARNING: {residues_csv} not found; not writing {modified_residues_csv}.')

    n_kept = sum(1 for r in scores_rows if r['kept'])
    print(f'Kept {n_kept}/{len(scores_rows)} ligand instance(s). Wrote {args.output_pdb}, '
          f'{scores_csv}, {despot_cluster_reps_csv}.')


if __name__ == '__main__':
    main()
