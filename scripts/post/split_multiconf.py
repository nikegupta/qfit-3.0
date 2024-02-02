#!/usr/bin/env python
"""
This script is used to split a ligand multiconformer pdb file into seperate pdbs for each conformer.
INPUT: PDB structure, chain/residue
OUTPUT: A tab seperated file with information about each residue and the atom type need to calculate cyrstallographic order parameters.

example:
split_multiconf.py ${pdb}.pdb  --residue=${chain_res}

"""

import numpy as np
import pandas as pd
import argparse
import os
from qfit.structure import Structure
from qfit.structure.ligand import _Ligand


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("structure", type=str, help="PDB-file containing structure.")
    p.add_argument("--residue", type=str, help="Chain_ID, Residue_ID for RSCC to be calculated on")
    p.add_argument(
        "-d",
        "--directory",
        default=".",
        metavar="<dir>",
        type=os.path.abspath,
        help="Directory to store results",
    )

    args = p.parse_args()
    return args


args = parse_args()

# Load structure and prepare it
structure = Structure.fromfile(args.structure)
chainid, resi = args.residue.split(",")

# Extract the ligand and altlocs
structure_ligand = structure.extract(f"resi {resi} and chain {chainid}")
altlocs = sorted(set(structure_ligand.altloc) - {''})

# Handle the case where there are no alternate locations
if not altlocs:
    altlocs = ['']  # This will handle the default case

# Extract the common part of the ligand (without alternate locations)
common_structure = structure_ligand.extract("altloc ''")

# Loop over each altloc
for altloc in altlocs:
    # Extract the structure for the current altloc
    if altloc:
        # structure_altloc = structure_ligand.extract(f"altloc {altloc}")
        alt_structure = structure_ligand.extract(f"altloc {altloc}")
        # Combine with common structure
        structure_altloc = common_structure.combine(alt_structure)


    else:
        # structure_altloc = structure_ligand
        structure_altloc = common_structure

    # Prepare the ligand object
    ligand = _Ligand(
        structure_altloc.data,
        structure_altloc._selection,
        link_data=structure_altloc.link_data,
    )
    ligand.altloc = ""
    ligand.q = 1

    # Create a file name for the current altloc
    exte = ".pdb"
    if altloc:
        ligand_name = os.path.join(args.directory, f"ligand_{altloc}{exte}")
    else:
        ligand_name = os.path.join(args.directory, f"ligand{exte}")

    # Save the file
    ligand.tofile(ligand_name)
