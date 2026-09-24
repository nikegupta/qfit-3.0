import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from qfit import Structure

# matches a residues_with_placer_conformers.csv-style label: one chain letter followed by a
# residue number, e.g. "A162" - same convention as build_final_model.py's
# _write_residue_conformer_list_csv and rscc_common.py's read_residue_conformer_list.
RESIDUE_LABEL_RE = re.compile(r'^([A-Za-z])(-?\d+)$')

# minimum final_rscc - rotamer_rscc for a residue to be reverted to its final_model_refined
# conformation - most of the raw RSCC gap between the two independent refinement runs (stage 6b
# vs stage 7b) is refinement-to-refinement noise rather than a real quality difference (median
# ~0.002 on a real dataset), so a margin well above that noise floor avoids reverting on noise
# alone.
REVERT_MIN_DIFF = 0.1


def build_argparser():
    p = argparse.ArgumentParser(
        description="Sometimes rotamer_optimize.py's own accepted, RSR-refined residues end up "
                    "with a LOWER RSCC than that same residue had in final_model_refined.pdb "
                    "(pre-rotamer-optimization) once real-space refinement has moved things "
                    "around - rotamer_optimize's own acceptance check happens before RSR, "
                    "against the unrefined model, so it can't see this. This also happens to "
                    "residues rotamer_optimize never touched at all, since rsr_rotamer "
                    "refines the full residues_with_placer_conformers.csv selection jointly, "
                    "not just the residues rotamer_optimize accepted - a neighboring residue's "
                    "resampled rotamer can shift an untouched residue's own refined position. "
                    "For every residue in residues_with_placer_conformers.csv, this compares "
                    "that residue's RSCC in final_model_refined_rscc.csv against "
                    "rotamer_refined_rscc.csv and reverts to final_model_refined's conformation "
                    "only if its RSCC is at least REVERT_MIN_DIFF higher (most of the raw gap "
                    "between the two independent refinement runs is noise, not a real quality "
                    "difference) - writing the merged structure to optimized.pdb, the per-residue "
                    "RSCC values actually used (read, never recomputed) to optimized_rscc.csv, "
                    "and the reverted residues alone to reverted_residues.csv."
    )
    p.add_argument(
        'final_model_pdb', type=Path,
        help='Path to final_model_refined.pdb (stage 6, pre-rotamer-optimization).'
    )
    p.add_argument(
        'rotamer_pdb', type=Path,
        help='Path to rotamer_refined.pdb (stage 7b\'s RSR output).'
    )
    p.add_argument(
        'final_rscc_csv', type=Path,
        help='Path to final_model_refined_rscc.csv (stage 6c\'s calc_final_refined_rscc output '
             '- covers every protein residue).'
    )
    p.add_argument(
        'rotamer_rscc_csv', type=Path,
        help='Path to rotamer_refined_rscc.csv (stage 7c\'s calc_rotamer_refined_rscc output - '
             'restricted to residues_with_placer_conformers.csv).'
    )
    p.add_argument(
        'residues_csv', type=Path,
        help='Path to residues_with_placer_conformers.csv (stage 6\'s build_final_model output) '
             '- every residue compared here is restricted to this list, the only residues '
             'rotamer_optimize.py ever touches.'
    )
    p.add_argument(
        'output_pdb', type=Path,
        help='Path to write the merged structure to (optimized.pdb). optimized_rscc.csv is '
             'written alongside it, in the same directory.'
    )
    p.add_argument(
        '--revert_min_diff', type=float, default=REVERT_MIN_DIFF, metavar='<float>',
        help='Minimum final_rscc - rotamer_rscc for a residue to be reverted to its '
             f'final_model_refined conformation (default: {REVERT_MIN_DIFF}).'
    )
    return p


def _residue_rscc_map(rscc_csv):
    """Reads a calc_rscc-style csv (model_idx,residue,rscc) into a {residue_label: rscc} dict.
    Returns {} if the file doesn't exist."""
    if not rscc_csv.is_file():
        return {}
    df = pd.read_csv(rscc_csv)
    return {residue: rscc for residue, rscc in zip(df['residue'], df['rscc']) if pd.notna(rscc)}


def _full_atom_mask(struct, subset_mask):
    """Translates subset_mask - a boolean array aligned to struct's own (possibly
    already-filtered) atom order, i.e. len(subset_mask) == struct.natoms - into a boolean mask
    aligned to struct's full, underlying (pre-selection) atom array.

    Structure.extract() applies a raw (non-string) selection array directly against the object's
    full underlying atom array, not against the object's own current selection - so handing it a
    mask built in struct's own atom order silently selects the wrong atoms wherever that order
    has a gap relative to the full array (e.g. struct is itself already a 'not resname LIG'
    selection). Identical to symmetry_expand.py's own _full_atom_mask (see that module's
    docstring for the full explanation) - duplicated here rather than imported since these are
    independent, self-contained command-line scripts."""
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
    DESPOT scoring) is exactly such a tool: it can see a residue relocated away from its real
    neighbors as a chain break, and cap the "orphaned" residue with a fake N/C-terminus (an
    extra OXT atom carrying real partial charge) even though the residue's actual 3D geometry
    is perfectly bonded to both neighbors. See despot_filter.py's
    reset_protein_to_apo_where_unbacked for the identical fix applied to its own reset residues.

    Relies on Structure.extract() returning a view over the SAME underlying atom storage as
    target_structure (confirmed empirically - mutating an extracted view's .coor/.b/.q mutates
    target_structure itself), so no separate re-assembly/combine() step is needed afterward. Both
    input masks are translated via _full_atom_mask before being handed to extract() - safe
    whether or not target_structure/replacement_structure are themselves already a filtered
    selection (e.g. despot_filter.py's caller passes a 'not resname LIG' selection).

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


def select_optimized_residues(final_structure, rotamer_structure, final_rscc, rotamer_rscc, labels,
                               revert_min_diff=REVERT_MIN_DIFF):
    """For every label ("{chain}{resnum}") in `labels`, compares final_rscc[label] against
    rotamer_rscc[label] and decides which structure that residue's conformation should come from
    in the merged output: 'final_model_refined' only if its RSCC is at least REVERT_MIN_DIFF
    higher than rotamer_refined's; 'rotamer_refined' otherwise (the default - including ties, any
    gap smaller than REVERT_MIN_DIFF, and any label missing from one or both csvs, which is
    warned about and left on rotamer_refined since that is the structure every other
    stage-7-and-later step already treats as current).

    Returns (output_structure, decisions) - decisions is a list of
    {residue, final_rscc, rotamer_rscc, source, rscc} dicts, one per label, in the same order as
    `labels`, where 'rscc' is whichever of the two values was actually kept (matching 'source').
    """
    # Mutated in place below via replace_residue_in_place (falling back to remove-and-append,
    # only for a residue where that isn't possible) - this is the returned output_structure, no
    # separate copy/reassembly needed for the common case.
    output_structure = rotamer_structure

    decisions = []
    fallback_mask = np.zeros(rotamer_structure.natoms, dtype=bool)
    fallback_pieces = []
    for label in labels:
        m = RESIDUE_LABEL_RE.match(label)
        if not m:
            print(f'  WARNING: could not parse residue label {label!r}; skipping.')
            continue
        chain_id, res_num = m.group(1), int(m.group(2))

        f_rscc = final_rscc.get(label)
        r_rscc = rotamer_rscc.get(label)
        if f_rscc is None or r_rscc is None:
            missing = 'final_model_refined_rscc.csv' if f_rscc is None else 'rotamer_refined_rscc.csv'
            print(f'  WARNING: {label} missing from {missing}; keeping rotamer_refined '
                  f'conformation without comparison.')
            source = 'rotamer_refined'
            kept_rscc = r_rscc if r_rscc is not None else f_rscc
        elif f_rscc - r_rscc >= revert_min_diff:
            source = 'final_model_refined'
            kept_rscc = f_rscc
        else:
            source = 'rotamer_refined'
            kept_rscc = r_rscc

        decisions.append({
            'residue': label, 'final_rscc': f_rscc, 'rotamer_rscc': r_rscc,
            'source': source, 'rscc': kept_rscc,
        })

        if source == 'final_model_refined':
            residue_mask = (output_structure.chain == chain_id) & (output_structure.resi == res_num)
            if not np.any(residue_mask):
                print(f'  WARNING: {label} not found in rotamer_refined structure; cannot '
                      f'replace it with final_model_refined\'s conformation.')
                continue
            if replace_residue_in_place(output_structure, chain_id, res_num, final_structure, label):
                continue
            # Atom-set mismatch - fall back to the old remove-and-append behavior for just this
            # one residue (replace_residue_in_place already printed why).
            replacement = final_structure.extract(f'chain {chain_id} and resid {res_num}')
            if replacement.natoms == 0:
                print(f'  WARNING: {label} not found in final_model_refined structure; keeping '
                      f'rotamer_refined conformation instead.')
                continue
            fallback_mask |= residue_mask
            fallback_pieces.append(replacement)

    if fallback_pieces:
        output_structure = output_structure.extract(~fallback_mask)
        for piece in fallback_pieces:
            output_structure = output_structure.combine(piece)

    n_final = sum(1 for d in decisions if d['source'] == 'final_model_refined')
    print(f'  {n_final}/{len(decisions)} residue(s) reverted to final_model_refined (RSCC at '
          f'least {revert_min_diff} higher than rotamer_refined); '
          f'{len(decisions) - n_final} kept from rotamer_refined.')

    return output_structure, decisions


def main():
    args = build_argparser().parse_args()

    for p in (args.final_model_pdb, args.rotamer_pdb, args.final_rscc_csv, args.rotamer_rscc_csv,
              args.residues_csv):
        if not p.is_file():
            sys.exit(f'Error: required file not found: {p}')

    with open(args.residues_csv) as f:
        labels = [line.strip() for line in f if line.strip()]

    final_rscc = _residue_rscc_map(args.final_rscc_csv)
    rotamer_rscc = _residue_rscc_map(args.rotamer_rscc_csv)

    final_structure = Structure.fromfile(str(args.final_model_pdb))
    rotamer_structure = Structure.fromfile(str(args.rotamer_pdb))

    output_structure, decisions = select_optimized_residues(
        final_structure, rotamer_structure, final_rscc, rotamer_rscc, labels,
        args.revert_min_diff)

    args.output_pdb.parent.mkdir(parents=True, exist_ok=True)
    output_structure.tofile(str(args.output_pdb))

    rscc_csv_path = args.output_pdb.parent / 'optimized_rscc.csv'
    rscc_df = pd.DataFrame(
        [{'model_idx': 1, 'residue': d['residue'], 'rscc': d['rscc'], 'source': d['source']}
         for d in decisions],
        columns=['model_idx', 'residue', 'rscc', 'source'],
    )
    rscc_df.to_csv(rscc_csv_path, index=False)

    reverted_csv_path = args.output_pdb.parent / 'reverted_residues.csv'
    reverted_rows = [
        {'residue': d['residue'], 'final_rscc': d['final_rscc'], 'rotamer_rscc': d['rotamer_rscc'],
         'diff': d['final_rscc'] - d['rotamer_rscc']}
        for d in decisions if d['source'] == 'final_model_refined'
    ]
    reverted_df = pd.DataFrame(reverted_rows, columns=['residue', 'final_rscc', 'rotamer_rscc', 'diff'])
    reverted_df.sort_values('diff', ascending=False, inplace=True)
    reverted_df.to_csv(reverted_csv_path, index=False)

    print(f'Wrote {args.output_pdb}, {rscc_csv_path}, {reverted_csv_path}.')


if __name__ == '__main__':
    main()
