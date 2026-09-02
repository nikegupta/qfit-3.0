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

Connectivity (which atoms are bonded) is perceived by obabel, not RDKit's own
Chem.MolFromPDBFile: RDKit's PDB bonder is a crude fixed-distance cutoff with no valence
awareness, and for any genuinely strained ring (e.g. a 4-membered azetidine, confirmed on a real
ligand) its 1,3-transannular non-bonded distance can be short enough to be misread as an extra
bond, producing a false "Explicit valence ... is greater than permitted" sanitization failure.
obabel's bond perception doesn't make this mistake (verified directly against the same case).
AssignBondOrdersFromTemplate only needs correct connectivity (topology), not correct bond
orders/aromaticity, from this step - it overwrites all of that from the SMILES template.

Hydrogens (absent from the crystal structure) are added with coordinates after template
matching, and the result is written as SDF, which mk_prepare_ligand.py reads directly.

Usage:
  apply_smiles_template.py <ligand_pdb> <smiles> <output_sdf>
"""
import subprocess
import sys
import tempfile
from pathlib import Path

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

    with tempfile.TemporaryDirectory() as tmp_dir:
        perceived_sdf = str(Path(tmp_dir) / "obabel_perceived.sdf")
        result = subprocess.run(
            ["obabel", ligand_pdb, "-O", perceived_sdf, "-h"],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            print(
                f"Error: obabel failed to perceive connectivity for {ligand_pdb}:\n"
                f"{result.stderr}",
                file=sys.stderr,
            )
            sys.exit(1)

        # sanitize=False: obabel's bond ORDERS/aromaticity are not used (overwritten below from
        # the SMILES template) and can fail RDKit's sanitizer (e.g. kekulization) even when its
        # connectivity, which is what we need here, is correct.
        pdb_mol = Chem.MolFromMolFile(perceived_sdf, removeHs=False, sanitize=False)

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
