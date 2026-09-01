#!/usr/bin/env python3
"""
apply_smiles_template.py - assigns bond orders (including aromaticity) to a ligand instance
extracted from a crystal structure PDB, using a known-correct SMILES as a template, via RDKit's
AllChem.AssignBondOrdersFromTemplate.

Why this exists: a bare ligand PDB (from split_complex_pdbqt.py) has coordinates but no bond
order/aromaticity information, and obabel's geometry-based perception of that information can
fail - e.g. writing MOL2 aromatic ('ar') bonds for a ring that RDKit (used internally by meeko's
mk_prepare_ligand.py) then can't kekulize, aborting the whole conversion. Overlaying a
known-correct SMILES onto the PDB coordinates sidesteps that perception step entirely.

Hydrogens (absent from the crystal structure) are added with coordinates after template
matching, and the result is written as SDF, which mk_prepare_ligand.py reads directly.

Usage:
  apply_smiles_template.py <ligand_pdb> <smiles> <output_sdf>
"""
import sys

from rdkit import Chem
from rdkit.Chem import AllChem


def main():
    if len(sys.argv) != 4:
        print(__doc__)
        sys.exit(1)
    ligand_pdb, smiles, output_sdf = sys.argv[1], sys.argv[2], sys.argv[3]

    template = Chem.MolFromSmiles(smiles)
    if template is None:
        print(f"Error: could not parse SMILES: {smiles}", file=sys.stderr)
        sys.exit(1)

    pdb_mol = Chem.MolFromPDBFile(ligand_pdb, removeHs=False)
    if pdb_mol is None:
        print(f"Error: could not parse ligand PDB: {ligand_pdb}", file=sys.stderr)
        sys.exit(1)

    try:
        fixed = AllChem.AssignBondOrdersFromTemplate(template, pdb_mol)
    except ValueError as e:
        print(
            f"Error: SMILES template does not match {ligand_pdb} "
            f"(wrong ligand, atom count, or connectivity?): {e}",
            file=sys.stderr,
        )
        sys.exit(1)

    fixed = Chem.AddHs(fixed, addCoords=True)
    Chem.MolToMolFile(fixed, output_sdf)
    print(f"Applied SMILES template ({template.GetNumAtoms()} heavy atoms) -> {output_sdf}")


if __name__ == '__main__':
    main()
