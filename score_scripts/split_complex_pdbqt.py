#!/usr/bin/env python3
"""
Splits a protein-ligand complex PDB into a receptor-only PDB (every ATOM/HETATM record whose
resname does NOT match <ligand_resname> - the rest of the protein plus any cofactors, e.g. HEM)
and one ligand-only PDB per ligand instance (every distinct (chain, resi[, altloc]) with at least
one atom matching <ligand_resname>) - a plain byte-column filter/group on the standard PDB fields
(columns 18-20 resname, 17 altloc, 22 chain, 23-26 resi), not a full PDB parse. Mirrors
calc_rscc.py's own _find_residue_keys/_extract_residue altloc-splitting convention (a residue
group with 2+ distinct non-blank altlocs is split into one instance per altloc, combined with
that residue's blank-altloc atoms) so the same residue label (e.g. 'A502' or 'A502-B') can be
used to join this script's per-instance docking scores (see complex_to_pdbqt.sh/
run_gnina_score.sh) against calc_rscc.py's per-residue RSCC scores later (see merge_scores.py).

Runs as plain-text PDB parsing with no qfit/iotbx dependency, so - unlike calc_rscc.py - it can
run in the same conda env as obabel/meeko (see complex_to_pdbqt.sh, which calls this).

Usage:
  split_complex_pdbqt.py <complex_pdb> <ligand_resname> <output_dir>

Writes <output_dir>/receptor.pdb and <output_dir>/ligand_<label>.pdb (one per ligand instance).
"""
import sys
from pathlib import Path


def parse_fields(line):
    return {
        'altloc': line[16:17].strip(),
        'resname': line[17:20].strip(),
        'chain': line[21:22].strip(),
        'resi': int(line[22:26]),
    }


def main():
    if len(sys.argv) != 4:
        print(__doc__)
        sys.exit(1)
    complex_pdb, ligand_resname, output_dir = sys.argv[1], sys.argv[2], Path(sys.argv[3])
    output_dir.mkdir(parents=True, exist_ok=True)

    receptor_lines = []
    ligand_lines_by_key = {}  # (chain, resi) -> [(line, altloc), ...]
    with open(complex_pdb) as f:
        for line in f:
            if not line.startswith(('ATOM', 'HETATM')):
                continue
            fields = parse_fields(line)
            if fields['resname'] == ligand_resname:
                key = (fields['chain'], fields['resi'])
                ligand_lines_by_key.setdefault(key, []).append((line, fields['altloc']))
            else:
                receptor_lines.append(line)

    if not ligand_lines_by_key:
        print(f"Error: no atoms found with resname '{ligand_resname}' in {complex_pdb}", file=sys.stderr)
        sys.exit(1)

    receptor_path = output_dir / 'receptor.pdb'
    with open(receptor_path, 'w') as f:
        f.writelines(receptor_lines)
        f.write('END\n')
    print(f"Receptor (everything else): {len(receptor_lines)} atoms -> {receptor_path}")

    n_instances = 0
    for (chain, resi), lines_altlocs in sorted(ligand_lines_by_key.items()):
        non_blank_altlocs = sorted({a for _, a in lines_altlocs if a != ''})
        instance_altlocs = non_blank_altlocs if non_blank_altlocs else ['']

        for altloc in instance_altlocs:
            if altloc:
                inst_lines = [l for l, a in lines_altlocs if a in (altloc, '')]
                label = f'{chain}{resi}-{altloc}'
            else:
                inst_lines = [l for l, _ in lines_altlocs]
                label = f'{chain}{resi}'
            ligand_path = output_dir / f'ligand_{label}.pdb'
            with open(ligand_path, 'w') as f:
                f.writelines(inst_lines)
                f.write('END\n')
            print(f"Ligand instance {label} ({ligand_resname}): {len(inst_lines)} atoms -> {ligand_path}")
            n_instances += 1

    print(f"{n_instances} ligand instance(s) written.")


if __name__ == '__main__':
    main()
