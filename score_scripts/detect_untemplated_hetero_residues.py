#!/usr/bin/env python3
"""
Scans a receptor PDB for distinct (chain, resnum, resname) HETATM residue instances whose
resname has no entry in meeko's own built-in residue_templates/ambiguous set (e.g. HEM) - these
are the ones mk_prepare_receptor.py can't build a working ad hoc template for from a bare SDF via
--add_templates (its residue matcher needs the input residue's own bonds, which its PDB/ProDy
readers never perceive for an unrecognized resname - confirmed empirically: intra-residue bonds
are only assigned by copying them from an exact resname match, never independently perceived, so
FindMCS-based matching against any ad hoc template always fails regardless of the template given).
complex_to_pdbqt.sh deletes these residues from the main receptor before running meeko, and preps
each one independently via obabel instead. Waters (HOH/WAT) and any resname meeko does have a
template for (common ions, etc.) are left alone - meeko handles those correctly on its own.

Usage:
  detect_untemplated_hetero_residues.py <receptor_pdb> <output_json>

Writes a JSON list of [chain, resnum, resname] triples (one per untemplated residue instance,
sorted by chain/resnum) to <output_json>.
"""
import json
import os
import sys

import meeko

KNOWN_EXTRA = {'HOH', 'WAT'}


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    receptor_pdb, output_json = sys.argv[1:3]

    templates_path = os.path.join(os.path.dirname(meeko.__file__), 'data', 'residue_chem_templates.json')
    with open(templates_path) as f:
        templates = json.load(f)
    known_resnames = set(templates['residue_templates']) | set(templates['ambiguous']) | KNOWN_EXTRA

    seen = {}
    with open(receptor_pdb) as f:
        for line in f:
            if not line.startswith('HETATM'):
                continue
            resname = line[17:20].strip()
            if resname in known_resnames:
                continue
            chain = line[21:22].strip()
            resnum = int(line[22:26])
            seen[(chain, resnum)] = resname

    result = [[chain, resnum, resname] for (chain, resnum), resname in sorted(seen.items())]
    with open(output_json, 'w') as f:
        json.dump(result, f)

    print(f'{len(result)} untemplated cofactor residue instance(s) found: {result}')


if __name__ == '__main__':
    main()
