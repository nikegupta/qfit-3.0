#!/usr/bin/env python3
"""
Loads a ligand pdb (which may contain more than one copy/instance of the same ligand, e.g.
multiple binding sites - each is matched against the SMILES independently, split by its
original chain id + residue number, not by proximity/connectivity) and its correct
connectivity/bond orders from a SMILES string, matches the SMILES template onto each
instance's structure (by heavy-atom topology, via rdkit's AssignBondOrdersFromTemplate), and
writes out an sdf - one entry per instance, named 'lig<chain><resnum>' (e.g. 'ligC1') - with
each instance's original 3D coordinates but the SMILES's bond orders. Hydrogens are ignored
throughout: the pdb is read with its hydrogens stripped, so both the template (from SMILES)
and each structure (from pdb) are heavy-atom-only, and the output sdf has no hydrogens either.

Usage:
  assign_bond_orders.py <ligand_pdb> <smiles> <output_sdf>
"""
import sys
from collections import OrderedDict

from rdkit import Chem
from rdkit.Chem import AllChem


def read_residue_blocks(pdb_path):
    """
    Splits a pdb's ATOM/HETATM records by their original (chain id, residue number, icode),
    so that separate ligand instances are always kept apart regardless of how close together
    they are in space (unlike relying on rdkit's post-parse connectivity/fragment splitting,
    which is proximity-based and could bridge two nearby instances into one fragment).

    Returns an OrderedDict of {(chain_id, resnum, icode): [pdb lines...]}, in first-appearance
    order.
    """
    residues = OrderedDict()
    with open(pdb_path) as f:
        for line in f:
            if line.startswith(('ATOM', 'HETATM')):
                chain_id = line[21].strip()
                resnum = line[22:26].strip()
                icode = line[26].strip()
                key = (chain_id, resnum, icode)
                residues.setdefault(key, []).append(line)
    return residues


def main():
    if len(sys.argv) != 4:
        sys.exit(f'Usage: {sys.argv[0]} <ligand_pdb> <smiles> <output_sdf>')
    ligand_pdb, smiles, output_sdf = sys.argv[1:4]

    template = Chem.MolFromSmiles(smiles)
    if template is None:
        sys.exit(f'Could not parse SMILES: {smiles!r}')

    residue_blocks = read_residue_blocks(ligand_pdb)
    if not residue_blocks:
        sys.exit(f'No ATOM/HETATM records found in {ligand_pdb}')

    writer = Chem.SDWriter(output_sdf)
    n_written = 0
    for (chain_id, resnum, icode), lines in residue_blocks.items():
        name = f'lig{chain_id}{resnum}{icode}'
        mol = Chem.MolFromPDBBlock(''.join(lines), removeHs=True)
        if mol is None:
            print(f'Warning: could not parse structure for {name}; skipping.')
            continue
        try:
            mol = AllChem.AssignBondOrdersFromTemplate(template, mol)
        except Exception as e:
            print(f'Warning: could not assign bond orders from SMILES {smiles!r} onto '
                  f'{name} ({type(e).__name__}: {e}); skipping.')
            continue
        mol.SetProp('_Name', name)
        writer.write(mol)
        n_written += 1
    writer.close()

    if n_written == 0:
        sys.exit(f'No ligand instance in {ligand_pdb} could be matched against the SMILES.')
    print(f'Wrote {n_written} ligand instance(s) to {output_sdf}')


if __name__ == '__main__':
    main()
