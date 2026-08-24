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
                    "separately, all together in a single pdb (with their original chain "
                    "id/residue number), rather than into the symmetry-expanded protein "
                    "output. The unit cell and space group are always taken from the command "
                    "line, never from the input pdb's own CRYST1 record (which, for this "
                    "pipeline's structures, is never correct)."
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
             'mates. Ligand atoms are not included here - see ligand_output_pdb.'
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
        'ligand_output_pdb',
        type=Path,
        help='Path to write every ligand (resname LIG) atom found in the input structure, '
             'all in a single pdb, keeping each ligand\'s original chain id and residue '
             'number. Not written at all if the input structure has no LIG atoms.'
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


def find_protein_symmetry_mates(structure, distance_cutoff):
    """
    Finds every crystallographic symmetry mate of structure's protein atoms
    (resname != LIG) that comes within distance_cutoff of the input structure's
    own ligand (resname LIG) atoms, and returns them as a list of new
    Structure objects (deep copies, safe to modify/combine further).

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
    ligand = structure.extract('resname LIG')
    if ligand.natoms == 0:
        raise ValueError('No ligand (resname LIG) atoms found in structure - nothing to '
                          'measure distance_cutoff against.')
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

    mates = find_protein_symmetry_mates(structure, args.distance_cutoff)
    print(f'Found {len(mates)} protein symmetry mate(s) within {args.distance_cutoff} '
          f'Å of the ligand(s) in {args.input_pdb}.')

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

    ligand_output = structure.extract('resname LIG')
    if ligand_output.natoms > 0:
        ligand_output.tofile(str(args.ligand_output_pdb))
        print(f'Wrote {ligand_output.natoms} ligand atom(s) to {args.ligand_output_pdb}.')
    else:
        print(f'No ligand (resname LIG) atoms found; {args.ligand_output_pdb} not written.')


if __name__ == '__main__':
    main()
