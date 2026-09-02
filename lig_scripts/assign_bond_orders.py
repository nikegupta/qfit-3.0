#!/usr/bin/env python3
"""
Loads one or more ligand pdbs and the ligand's correct connectivity/bond orders from a SMILES
string, matches the SMILES template onto each instance's structure (by heavy-atom topology, via
rdkit's AssignBondOrdersFromTemplate), and writes out an sdf - one entry per instance - with
each instance's original 3D coordinates but the SMILES's bond orders. Hydrogens are ignored
throughout: the pdb is read with its hydrogens stripped, so both the template (from SMILES) and
each structure (from pdb) are heavy-atom-only, and the output sdf has no hydrogens either.

Each input file may hold either a single already-split ligand instance (e.g.
symmetry_expand.py's ligand_output_dir - one pdb per (chain, resi[, altloc]) instance, already
split apart there so a genuinely disordered instance never shares a file with its other altloc
conformer) or several combined together in one flat file, split apart here by (chain id,
residue number, insertion code) instead (e.g. extract_ligand_conformers.py's pooled ligs.pdb -
one instance per (chain, resnum), no MODEL records; restores a splitting step this script used
to do unconditionally on a single combined file, before every caller was expected to pre-split -
see git history). A file containing exactly one instance names it after the file's own basename
(e.g. 'ligC1.pdb' -> 'ligC1'); a file containing more than one names each
'lig<chain><resnum><icode>' (e.g. 'ligL1', 'ligL2', ... - matching extract_ligand_conformers.py's
documented naming).

AssignBondOrdersFromTemplate only relabels the *order* of bonds that rdkit's own
MolFromPDBBlock already decided exist - it never adds or removes a bond. MolFromPDBBlock's own
connectivity guess comes purely from interatomic distance (there are no CONECT records here),
so a locally distorted instance - e.g. two non-bonded atoms happening to sit close enough
together in this particular pose - can make MolFromPDBBlock perceive a bond that isn't really
there. That either breaks AssignBondOrdersFromTemplate's attempt to reconcile the (now wrong)
topology against the template, or breaks MolFromPDBBlock's own valence check outright (e.g. two
guessed bonds to a fluorine), before the SMILES is ever consulted. A different, chemically
identical instance of the same ligand elsewhere in the same pdb is typically unaffected (its
own geometry doesn't have that particular close contact).

For any instance whose own geometry defeats this normal path, this falls back to reusing the
already-correct topology (bonds + orders) of a sibling instance in the same pdb that DID
succeed, matched atom-by-atom by PDB atom name (every instance of one ligand in this pipeline
shares the same per-atom naming - see build_final_model.py/despot_filter.py's reliance on the
same convention) - only that instance's own 3D coordinates are used, not its geometry-derived
topology. This is a deliberately minimal fix: it recovers an instance only when at least one
sibling in the same pdb succeeded normally, and - like the normal path - does not perceive or
assign stereochemistry (a SMILES string carries no stereo information, and this pipeline's
reference structures don't track which stereoisomer each ligand is, unlike the pipeline's own
poses, which do via their cif restraints file - see cluster_reps.csv's cif_restraints_file
column). A future improvement could assign topology directly from that per-dataset cif's own
_chem_comp_bond table (stereospecific, no template-matching needed at all), rather than a SMILES
template.

Usage:
  assign_bond_orders.py <ligand_pdb>... <smiles> <output_sdf>
"""
import argparse
import sys
from collections import OrderedDict
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import AllChem


def read_ligand_instances(pdb_path):
    """Splits pdb_path's ATOM/HETATM records by their (chain id, residue number, insertion
    code), since one input file may hold either a single already-split ligand instance or
    several combined together in one flat file - see this module's docstring.

    Returns [(name, lines), ...] in first-appearance order. A file containing exactly one
    instance is named after the file's own basename (matching every existing single-instance
    caller's naming, e.g. symmetry_expand's 'ligA103.pdb' -> 'ligA103'); a file containing more
    than one names each 'lig<chain><resnum><icode>' (matching extract_ligand_conformers.py's
    documented naming for its pooled ligs.pdb)."""
    residues = OrderedDict()
    with open(pdb_path) as f:
        for line in f:
            if line.startswith(('ATOM', 'HETATM')):
                chain_id = line[21].strip()
                resnum = line[22:26].strip()
                icode = line[26].strip()
                residues.setdefault((chain_id, resnum, icode), []).append(line)

    if not residues:
        return []
    if len(residues) == 1:
        return [(pdb_path.stem, next(iter(residues.values())))]
    return [
        (f'lig{chain_id}{resnum}{icode}', lines)
        for (chain_id, resnum, icode), lines in residues.items()
    ]


def assign_from_geometry(lines, name, template):
    """Normal path: guesses this instance's own connectivity from its 3D geometry
    (MolFromPDBBlock), then relabels those bonds' orders to match the SMILES template
    (AssignBondOrdersFromTemplate). Returns the resulting mol, or None (after printing a
    warning) if either step fails - see this module's docstring for why a locally distorted
    pose can cause that."""
    mol = Chem.MolFromPDBBlock(''.join(lines), removeHs=True)
    if mol is None:
        print(f'Warning: could not parse structure for {name}; skipping.')
        return None
    try:
        return AllChem.AssignBondOrdersFromTemplate(template, mol)
    except Exception as e:
        print(f'Warning: could not assign bond orders from SMILES onto {name} '
              f'({type(e).__name__}: {e}); skipping.')
        return None


def assign_from_sibling(donor_mol, donor_name, lines, name):
    """Fallback for an instance that assign_from_geometry couldn't handle: reuses donor_mol's
    already-correct topology (bonds + orders, from a sibling instance of the same ligand in the
    same pdb that succeeded normally) as-is, just replacing its conformer with this instance's
    own coordinates - matched atom-by-atom by PDB atom name (donor_mol's atoms keep their
    PDBResidueInfo names from when they were first parsed by MolFromPDBBlock). Returns None
    (after printing a warning) if this instance is missing an atom donor_mol's topology needs
    by name, rather than silently mismatching atoms."""
    coords_by_name = {}
    for line in lines:
        atom_name = line[12:16].strip()
        try:
            x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
        except ValueError:
            continue
        coords_by_name[atom_name] = (x, y, z)

    new_mol = Chem.RWMol(donor_mol)
    conf = Chem.Conformer(new_mol.GetNumAtoms())
    for atom in new_mol.GetAtoms():
        atom_name = atom.GetPDBResidueInfo().GetName().strip()
        if atom_name not in coords_by_name:
            print(f'Warning: could not recover {name} from sibling instance {donor_name} - '
                  f'{name} is missing atom {atom_name!r}; skipping.')
            return None
        conf.SetAtomPosition(atom.GetIdx(), coords_by_name[atom_name])
    new_mol.RemoveAllConformers()
    new_mol.AddConformer(conf)

    try:
        Chem.SanitizeMol(new_mol)
    except Exception as e:
        print(f'Warning: could not recover {name} from sibling instance {donor_name} - '
              f'sanitization failed ({type(e).__name__}: {e}); skipping.')
        return None

    print(f'  Recovered {name}: reused bond topology from sibling instance {donor_name} '
          f'(its own geometry made bond-order assignment fail - see warning above).')
    return new_mol


def build_argparser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('ligand_pdb', nargs='+', type=Path,
                    help='One or more single-instance ligand pdb files (e.g. '
                         'symmetry_expand.py\'s ligand_output_dir/lig*.pdb).')
    p.add_argument('smiles', help='SMILES for the ligand, used as the bond-order template.')
    p.add_argument('output_sdf', help='Path to write the combined multi-instance sdf to.')
    return p


def main():
    args = build_argparser().parse_args()

    template = Chem.MolFromSmiles(args.smiles)
    if template is None:
        sys.exit(f'Could not parse SMILES: {args.smiles!r}')

    instances = []
    for ligand_pdb in args.ligand_pdb:
        file_instances = read_ligand_instances(ligand_pdb)
        if not file_instances:
            print(f'Warning: no ATOM/HETATM records found in {ligand_pdb}; skipping.')
            continue
        instances.extend(file_instances)
    if not instances:
        sys.exit(f'No ATOM/HETATM records found in any of {[str(p) for p in args.ligand_pdb]}')

    writer = Chem.SDWriter(args.output_sdf)
    n_written = 0
    n_recovered = 0
    donor_mol, donor_name = None, None
    failed = []

    for name, lines in instances:
        mol = assign_from_geometry(lines, name, template)
        if mol is None:
            failed.append((name, lines))
            continue
        mol.SetProp('_Name', name)
        writer.write(mol)
        n_written += 1
        if donor_mol is None:
            donor_mol, donor_name = mol, name

    if failed and donor_mol is not None:
        for name, lines in failed:
            mol = assign_from_sibling(donor_mol, donor_name, lines, name)
            if mol is None:
                continue
            mol.SetProp('_Name', name)
            writer.write(mol)
            n_written += 1
            n_recovered += 1

    writer.close()

    if n_written == 0:
        sys.exit(f'No ligand instance could be matched against the SMILES.')
    msg = f'Wrote {n_written} ligand instance(s) to {args.output_sdf}'
    if n_recovered:
        msg += f' ({n_recovered} via sibling-topology fallback)'
    print(msg)


if __name__ == '__main__':
    main()
