import argparse
import string
from pathlib import Path

import numpy as np
from cctbx import crystal
from scipy.spatial import cKDTree

from qfit import Structure


# Pool of new chain ids handed out to symmetry mates, preferring single
# characters before falling back to two-character ids (large structures with
# many mates/chains can exhaust single characters).
_CHAIN_ID_ALPHABET = string.ascii_uppercase + string.ascii_lowercase + string.digits


def build_argparser():
    p = argparse.ArgumentParser(
        description="Add crystallographic symmetry mates of a structure's protein atoms that "
                    "fall near its ligand(s), so that energy calculations on the model see a "
                    "more realistic crystal environment around the ligand. Ligand atoms "
                    "(resname LIG) are never symmetry-expanded, and are written out "
                    "separately, one pdb per ligand instance (with their original chain "
                    "id/residue number, split further by altloc when a residue has any - see "
                    "ligand_output_dir), rather than into the symmetry-expanded protein "
                    "output. The unit cell and space group are always taken from the command "
                    "line, never from the input pdb's own CRYST1 record (which, for this "
                    "pipeline's structures, is never correct). By default the mate-proximity "
                    "target is input_pdb's own ligand atoms - pass --ligand-conformers-pdb to "
                    "target an externally-supplied set of ligand conformers instead (e.g. "
                    "every placer2 conformer, not just one final pose), so the expanded "
                    "protein covers a crystal environment realistic for all of them."
    )
    p.add_argument(
        'input_pdb',
        type=Path,
        help='Path to the input structure.'
    )
    p.add_argument(
        'output_pdb',
        type=Path,
        help='Path to write the input structure\'s protein atoms plus any added symmetry '
             'mates. Ligand atoms are not included here - see ligand_output_dir.'
    )
    p.add_argument(
        'space_group',
        help="Space group to generate symmetry operations from, e.g. 'P212121' or a space "
             "group number."
    )
    p.add_argument('a', type=float, help='Unit cell a (Å)')
    p.add_argument('b', type=float, help='Unit cell b (Å)')
    p.add_argument('c', type=float, help='Unit cell c (Å)')
    p.add_argument('alpha', type=float, help='Unit cell alpha (degrees)')
    p.add_argument('beta', type=float, help='Unit cell beta (degrees)')
    p.add_argument('gamma', type=float, help='Unit cell gamma (degrees)')
    p.add_argument(
        'distance_cutoff',
        type=float,
        help='A symmetry mate is considered only if at least one of its (protein) atoms is '
             'within this distance (Å) of some ligand (resname LIG) atom in the input '
             'structure. Of a considered mate, only the residues with at least one atom '
             'within this distance of a ligand atom are kept - the rest of that mate\'s '
             'residues (too far to matter for a local energy calculation) are discarded, '
             'which keeps the expanded structure small.'
    )
    p.add_argument(
        'ligand_output_dir',
        type=Path,
        help='Directory to write one pdb per ligand (resname LIG) instance found in the '
             'input structure, named lig<chain><resi>.pdb, or lig<chain><resi>-<altloc>.pdb '
             'for each altloc of a residue that has any non-blank altloc (matched with that '
             'residue\'s own blank-altloc atoms, if any) - same altloc-splitting convention '
             'as calc_rscc.py/split_complex_pdbqt.py, so a genuinely disordered ligand '
             'instance gets its own file (and, downstream, its own DESPOT score) instead of '
             'being merged with its other conformer(s). Created if missing. Nothing is '
             'written if the input structure has no LIG atoms.'
    )
    p.add_argument(
        '--strip',
        action='store_true',
        help='Before doing anything else, strip hydrogen atoms off every ligand (resname '
             'LIG) residue and remove explicit water (resname HOH) and DMSO (resname DMS) '
             'residues entirely. Meant for reference-set structures (e.g. a PanDDA model), '
             'which - unlike this pipeline\'s own final_model_refined.pdb - carry explicit '
             'ligand hydrogens and ordered solvent/cryoprotectant molecules that DESPOT '
             'scoring is not set up to handle. A final_model_refined.pdb already has none '
             'of these, so this is a no-op on one; kept as an explicit, toggleable flag '
             'rather than always-on for control.'
    )
    p.add_argument(
        '--ligand-conformers-pdb',
        type=Path,
        nargs='+',
        default=None,
        metavar='PDB',
        help='One or more pdb file(s) of ligand (resname LIG) conformer instances to use as '
             'the mate-proximity target INSTEAD of input_pdb\'s own ligand atoms (e.g. every '
             'placer2 conformer, not just whichever pose ended up in final_model_refined.pdb) '
             '- for scoring many candidate poses against a crystal environment that covers '
             'all of them. input_pdb\'s own ligand atoms are still excluded from the protein '
             'output and still written to ligand_output_dir regardless of this flag; only the '
             'symmetry-mate search target changes. Omit for the default behavior (target is '
             'input_pdb\'s own ligand atoms).'
    )
    return p


def _next_chain_id(used_chain_ids):
    for c in _CHAIN_ID_ALPHABET:
        if c not in used_chain_ids:
            used_chain_ids.add(c)
            return c
    for c1 in _CHAIN_ID_ALPHABET:
        for c2 in _CHAIN_ID_ALPHABET:
            cid = c1 + c2
            if cid not in used_chain_ids:
                used_chain_ids.add(cid)
                return cid
    raise RuntimeError('Ran out of unique chain ids for symmetry mates.')


def _reassign_chain_ids(mate_structure, used_chain_ids):
    """Gives every chain in mate_structure a fresh id not already in
    used_chain_ids (which is updated in place), so a mate's chains never
    collide with the input structure's chains or with an earlier mate's."""
    original_ids = mate_structure.chain.copy()
    mapping = {orig_id: _next_chain_id(used_chain_ids) for orig_id in sorted(set(original_ids))}
    for atom, orig_id in zip(mate_structure.atoms, original_ids):
        atom.chain().id = mapping[orig_id]


def _full_atom_mask(struct, subset_mask):
    """Translates subset_mask - a boolean array aligned to struct's own (possibly
    already-filtered) atom order, i.e. len(subset_mask) == struct.natoms - into a
    boolean mask aligned to struct's full, underlying (pre-selection) atom array.

    Structure.extract() applies a raw (non-string) selection array directly against the
    object's full underlying atom array, not against the object's own current selection -
    so handing it a mask built in struct's own atom order silently selects the wrong
    atoms wherever that order has a gap relative to the full array. struct='not resname
    LIG' (as find_protein_symmetry_mates below builds `protein`) has exactly that kind of
    gap unless every ligand atom happens to sit after every kept atom in the input file -
    true for this pipeline's own final_model_refined.pdb (build_final_model.py always
    appends LIG residues last), but not guaranteed for reference-set/PanDDA-derived
    structures, where a ligand can be interleaved anywhere among the protein chains.
    Translating first makes struct.extract(...) safe regardless of where the excluded
    atoms happened to be."""
    full_mask = np.zeros(struct.total_length, dtype=bool)
    if struct.selection is None:
        full_mask[:] = subset_mask
    else:
        full_indices = np.array(list(struct.selection))
        full_mask[full_indices[subset_mask]] = True
    return full_mask


def find_protein_symmetry_mates(structure, distance_cutoff, ligand_conformers=None):
    """
    Finds every crystallographic symmetry mate of structure's protein atoms
    (resname != LIG) that comes within distance_cutoff of a ligand (resname LIG) atom, and
    returns them as a list of new Structure objects (deep copies, safe to modify/combine
    further).

    ligand_conformers: optional Structure to use as the mate-proximity target INSTEAD of
    structure's own ligand atoms - e.g. every placer2 conformer pooled into one pdb, not just
    whichever single pose ended up in structure itself (see --ligand-conformers-pdb). None
    (default) preserves the original behavior: target is structure.extract('resname LIG').

    Each returned mate is trimmed down to just the residues that have at
    least one atom within distance_cutoff of a ligand atom - not the whole
    symmetry-related copy. distance_cutoff is only used to decide whether a
    symmetry operation produces a mate worth considering at all (see the
    broad-phase search below) and, per residue, whether to keep it; it never
    causes a whole extra mate's worth of far-away atoms to be written out.
    This keeps the expanded structure's atom count close to what's actually
    needed for a local (ligand-centered) energy calculation, rather than
    growing with the size of whichever mates happen to touch the cutoff.
    Each keep_mask is translated via _full_atom_mask before being handed to
    protein.extract() - see that function's docstring for why.

    Candidate symmetry operations are generated with UnitCell's own
    iter_struct_orth_symops (the same broad-phase search qfit.py uses to find
    symmetry-related clash partners), targeted at the ligand rather than the
    protein itself: every space-group symop, combined with neighboring
    unit-cell translations, whose transformed protein centroid could
    plausibly bring the protein within distance_cutoff of the ligand,
    using the protein's and ligand's own radii (centroid to farthest atom)
    as a cushion. Since distance_cutoff is used as that function's cushion,
    this is guaranteed not to miss any true protein-atom-to-ligand-atom
    contact within distance_cutoff, though it can (harmlessly) pass through
    extra candidates that don't survive the true, atom-atom distance check
    below.
    """
    protein = structure.extract('not resname LIG')
    if protein.natoms == 0:
        raise ValueError('No protein (non-LIG) atoms found in structure.')
    ligand = (ligand_conformers if ligand_conformers is not None else structure).extract(
        'resname LIG')
    if ligand.natoms == 0:
        source = 'ligand_conformers' if ligand_conformers is not None else 'structure'
        raise ValueError(f'No ligand (resname LIG) atoms found in {source} - nothing to '
                          f'measure distance_cutoff against.')
    protein_baseline_coor = protein.coor.copy()
    tree = cKDTree(ligand.coor)

    # Per-atom residue id (order-aligned with protein.atoms/coor), used below
    # to expand an "atom within cutoff" mask into a "keep this atom's whole
    # residue" mask. Coordinates are rotated/translated in place each
    # iteration, but atom order - and so this array - never changes.
    residue_id = np.array([
        f'{c}\0{r}\0{ic}' for c, r, ic in zip(protein.chain, protein.resi, protein.icode)
    ])

    mates = []
    for symop in structure.unit_cell.iter_struct_orth_symops(
            protein, target=ligand, cushion=distance_cutoff):
        if symop.is_identity():
            continue
        protein.rotate(symop.R)
        protein.translate(symop.t)
        atom_dists = tree.query(protein.coor, k=1)[0]
        min_dist = atom_dists.min()
        if min_dist <= distance_cutoff:
            residues_in_range = np.unique(residue_id[atom_dists <= distance_cutoff])
            keep_mask = np.isin(residue_id, residues_in_range)
            print(f'  Accepted symmetry mate {len(mates) + 1}: '
                  f'minimum atom-atom distance to ligand = {min_dist:.2f} Å, '
                  f'keeping {keep_mask.sum()}/{protein.natoms} atom(s) '
                  f'({len(residues_in_range)} residue(s)) within {distance_cutoff} Å')
            mates.append(protein.extract(_full_atom_mask(protein, keep_mask)).copy())
        protein.coor = protein_baseline_coor
    return mates


def split_ligand_instances(structure):
    """Splits structure's LIG atoms into one sub-structure per (chain, resi) instance, further
    split by altloc for any residue that has one - same "any non-blank altloc gets its own
    instance" convention as calc_rscc.py/split_complex_pdbqt.py, so a genuinely disordered
    ligand instance (2+ altlocs at the same chain+resi) is never merged with its other
    conformer(s). Each altloc instance is combined with that residue's own blank-altloc atoms
    (shared across all of its conformers), matching split_complex_pdbqt.py's atom selection.

    Returns [(label, sub_structure), ...] sorted by (chain, resi, altloc), label formatted
    '<chain><resi>' or '<chain><resi>-<altloc>' - the same convention
    residue_label_from_key/calc_rscc.py's _residue_label use (see rscc_common.py).

    Every mask below is computed against, and applied to, structure directly (never through an
    intermediate extract() result) - extract() only sets a selection over the ORIGINAL,
    unfiltered atom array, so a mask sized to an already-extracted subset (e.g. one built from
    that subset's own .chain/.resi/.altloc) would be silently misapplied against the wrong
    (much larger) array if handed to a second, chained extract() call."""
    is_lig = structure.resn == 'LIG'
    if not is_lig.any():
        return []

    chain, resi, altloc = structure.chain, structure.resi, structure.altloc
    keys = sorted(set(zip(chain[is_lig], resi[is_lig])))

    instances = []
    for c, r in keys:
        base_mask = is_lig & (chain == c) & (resi == r)
        non_blank_altlocs = sorted(set(altloc[base_mask]) - {''})
        if non_blank_altlocs:
            for al in non_blank_altlocs:
                mask = base_mask & ((altloc == al) | (altloc == ''))
                instances.append((f'{c}{r}-{al}', structure.extract(mask)))
        else:
            instances.append((f'{c}{r}', structure.extract(base_mask)))
    return instances


def collapse_protein_altlocs_to_highest_occupancy(structure):
    """Collapses every non-ligand (resname != LIG) residue's alternate conformations down to a
    single, highest-occupancy conformer. pdb2pqr30 (which protein_to_mol2.sh shells out to for
    DESPOT scoring) has no altloc support and no CLI flag for one - left alone, it silently
    keeps whichever altloc it happens to parse first, regardless of occupancy (confirmed
    empirically: on a real disordered residue it kept the lower-occupancy 'A' conformer over
    the higher-occupancy 'B'). Doing this explicitly and occupancy-aware, before pdb2pqr30 ever
    runs, replaces that silent, occupancy-blind default with a deliberate, logged choice.
    Ligand (resname LIG) atoms are left untouched - see split_ligand_instances for how their
    altlocs are handled instead. A no-op if there's no non-ligand altloc disorder at all."""
    resn = structure.resn
    altloc = structure.altloc
    non_blank = (altloc != '') & (resn != 'LIG')
    if not non_blank.any():
        return structure

    chain, resi, icode, q = structure.chain, structure.resi, structure.icode, structure.q
    chosen = {}
    for c, r, ic, al, occ in zip(chain[non_blank], resi[non_blank], icode[non_blank],
                                  altloc[non_blank], q[non_blank]):
        key = (c, r, ic)
        if key not in chosen or occ > chosen[key][1]:
            chosen[key] = (al, occ)

    keep = ~non_blank
    for i in np.where(non_blank)[0]:
        key = (chain[i], resi[i], icode[i])
        if altloc[i] == chosen[key][0]:
            keep[i] = True

    n_residues = len(chosen)
    print(f'Collapsed {n_residues} disordered non-ligand residue(s) to their highest-occupancy '
          f'altloc before protein_to_mol2.sh/pdb2pqr30 (which cannot represent alternate '
          f'conformations at all).')
    return structure.get_selected_structure(keep)


def main():
    args = build_argparser().parse_args()

    structure = Structure.fromfile(str(args.input_pdb))

    if args.strip:
        n_before = structure.natoms
        keep_mask = ~(
            ((structure.resn == 'LIG') & (structure.e == 'H'))
            | (structure.resn == 'HOH')
            | (structure.resn == 'DMS')
        )
        structure = structure.extract(keep_mask)
        print(f'--strip: removed {n_before - structure.natoms} atom(s) (ligand hydrogens + '
              f'HOH waters + DMS residues); {structure.natoms} atom(s) remain.')
        # extract() only masks atoms - it doesn't rebuild the underlying atom array, so any
        # later selection computed against this (now-smaller) structure's own natoms/properties
        # would be the wrong size to apply against the still-full-sized array beneath it.
        # Rebuild now so every later step (collapse_protein_altlocs_to_highest_occupancy,
        # split_ligand_instances) has a real 1:1 view to select against.
        structure = structure.get_selected_structure(None)

    structure = collapse_protein_altlocs_to_highest_occupancy(structure)

    # Always taken from the command line, never from the input pdb's own CRYST1 record (which,
    # for this pipeline's structures, is never correct/meaningful).
    crystal_symmetry = crystal.symmetry(
        unit_cell=(args.a, args.b, args.c, args.alpha, args.beta, args.gamma),
        space_group_symbol=args.space_group,
    )
    structure.set_crystal_symmetry(crystal_symmetry)
    # Structure.extract()/copy()/etc. all thread crystal_symmetry through via
    # self._kwargs (captured once, at __init__ time) rather than the
    # set_crystal_symmetry() call above, so every extracted/copied piece of
    # `structure` below (protein_output, each mate, each ligand) picks up our
    # crystal_symmetry too, instead of whatever (if anything) the input file's own
    # CRYST1 record parsed to.
    structure._kwargs['crystal_symmetry'] = crystal_symmetry  # pylint: disable=protected-access

    ligand_conformers = None
    ligand_source = args.input_pdb
    if args.ligand_conformers_pdb:
        ligand_conformers = Structure.fromfile(str(args.ligand_conformers_pdb[0]))
        for extra_pdb in args.ligand_conformers_pdb[1:]:
            ligand_conformers = ligand_conformers.combine(Structure.fromfile(str(extra_pdb)))
        ligand_source = ', '.join(str(p) for p in args.ligand_conformers_pdb)

    mates = find_protein_symmetry_mates(structure, args.distance_cutoff, ligand_conformers)
    print(f'Found {len(mates)} protein symmetry mate(s) within {args.distance_cutoff} '
          f'Å of the ligand(s) in {ligand_source}.')

    used_chain_ids = set(structure.chain)
    protein_output = structure.extract('not resname LIG')
    n_original_protein_atoms = protein_output.natoms
    for mate in mates:
        _reassign_chain_ids(mate, used_chain_ids)
        protein_output = protein_output.combine(mate)

    protein_output.tofile(str(args.output_pdb))
    print(f'Wrote {protein_output.natoms} protein atom(s) ({n_original_protein_atoms} original '
          f'+ {protein_output.natoms - n_original_protein_atoms} from symmetry mates) to '
          f'{args.output_pdb}.')

    ligand_instances = split_ligand_instances(structure)
    if ligand_instances:
        args.ligand_output_dir.mkdir(parents=True, exist_ok=True)
        for label, instance in ligand_instances:
            instance_path = args.ligand_output_dir / f'lig{label}.pdb'
            instance.tofile(str(instance_path))
            print(f'Wrote {instance.natoms} ligand atom(s) to {instance_path}.')
        print(f'{len(ligand_instances)} ligand instance(s) written to {args.ligand_output_dir}.')
    else:
        print(f'No ligand (resname LIG) atoms found; nothing written to {args.ligand_output_dir}.')


if __name__ == '__main__':
    main()
