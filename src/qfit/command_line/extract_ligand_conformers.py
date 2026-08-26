#!/usr/bin/env python3
"""
extract_ligand_conformers.py - pulls every ligand (resname LIG) instance out of every MODEL of
a set of placer2 '<dataset>_backbone_refined_*_refined.pdb' files (PLACER round-2 conformer
ensembles - one file per filter_1-passing backbone/residue window, each file itself a
multi-MODEL pdb with one ligand-conformer sample per MODEL) and combines them into a single,
flat (no MODEL records) multi-instance ligand pdb - one instance per (chain, resnum) pair,
matching the convention this pipeline's own ligand pdbs already use (e.g. symmetry_expand's
ligs.pdb, final_model_refined.pdb's own LIG residues) so it can go straight through the
existing lig_scripts/pdb_to_mol2.sh -> DESPOT scoring path unchanged: assign_bond_orders.py
there splits a ligand pdb into per-instance sdf/mol2 entries purely by (chain, resnum, icode),
not by MODEL, so instances that all share one (chain, resnum) - as every MODEL in a source file
does - would otherwise silently collapse into one broken multi-hundred-atom "residue".

Every extracted instance is written to a fixed chain id and a globally unique, sequentially
assigned residue number (1, 2, 3, ...) - unrelated to the source file's own index or the MODEL
number - and atom serial numbers are renumbered sequentially across the whole output file. A
separate CSV records, for each instance, which source file and MODEL it came from, and the
exact ligand name (e.g. 'ligL1') DESPOT's score_complex.py will report it under - see
assign_bond_orders.py's naming convention (f'lig{chain}{resnum}{icode}'), so a DESPOT score
can be traced back to the placer2 sample it came from.

Used by program.sh's Stage 7a (despot_process_dataset) to score every placer2 conformer against
DESPOT, not just the one pose filter2/build_final_model happened to select - see
despot_filter.py, which reselects the final pose per cluster from this population.

Usage:
  extract_ligand_conformers <placer2_dir> <dataset_name> <output_ligs_pdb> <output_map_csv>

<placer2_dir> is searched (non-recursively) for files matching
'<dataset_name>_backbone_refined_*_refined.pdb'.
"""
import csv
import glob
import os
import re
import sys

CHAIN_ID = 'L'


def iter_lig_instances(pdb_path):
    """Yields (model_number, [pdb lines]) for every MODEL in pdb_path that contains at least
    one resname-LIG ATOM/HETATM record. A file with no MODEL/ENDMDL records at all (single
    conformer) is treated as one implicit model (model_number 1)."""
    model_number = 1
    current_lines = []
    saw_model_record = False
    with open(pdb_path) as f:
        for line in f:
            if line.startswith('MODEL'):
                saw_model_record = True
                model_number = int(line[10:14])
                current_lines = []
            elif line.startswith('ENDMDL'):
                if current_lines:
                    yield model_number, current_lines
                current_lines = []
            elif line.startswith(('ATOM', 'HETATM')):
                resname = line[17:20].strip()
                if resname == 'LIG':
                    current_lines.append(line)
    if not saw_model_record and current_lines:
        yield model_number, current_lines


def main():
    if len(sys.argv) != 5:
        sys.exit(f'Usage: {sys.argv[0]} <placer2_dir> <dataset_name> <output_ligs_pdb> '
                  f'<output_map_csv>')
    placer2_dir, dataset_name, output_ligs_pdb, output_map_csv = sys.argv[1:5]

    pattern = os.path.join(placer2_dir, f'{dataset_name}_backbone_refined_*_refined.pdb')
    source_files = sorted(
        glob.glob(pattern),
        key=lambda p: int(re.search(r'_backbone_refined_(\d+)_refined\.pdb$', p).group(1)),
    )
    if not source_files:
        sys.exit(f'No files matched {pattern}')

    instance_id = 0
    n_atoms_written = 0
    map_rows = []
    with open(output_ligs_pdb, 'w') as out:
        for source_file in source_files:
            for model_number, lines in iter_lig_instances(source_file):
                instance_id += 1
                resnum = instance_id
                if resnum > 9999:
                    sys.exit(f'Too many ligand conformer instances ({resnum}) - resSeq field '
                              f'(4 chars) would overflow.')
                ligand_name = f'lig{CHAIN_ID}{resnum}'
                for line in lines:
                    n_atoms_written += 1
                    if n_atoms_written > 99999:
                        sys.exit(f'Too many atoms ({n_atoms_written}) - atom serial field '
                                  f'(5 chars) would overflow.')
                    new_line = (
                        f'{line[0:6]}{n_atoms_written:5d}{line[11:21]}'
                        f'{CHAIN_ID}{resnum:4d}{line[26:]}'
                    )
                    if not new_line.endswith('\n'):
                        new_line += '\n'
                    out.write(new_line)
                map_rows.append({
                    'instance_id': instance_id,
                    'ligand_name': ligand_name,
                    'chain': CHAIN_ID,
                    'resnum': resnum,
                    'source_file': os.path.basename(source_file),
                    'model_number': model_number,
                    'n_atoms': len(lines),
                })

    if instance_id == 0:
        sys.exit(f'No resname-LIG atoms found in any MODEL of {len(source_files)} file(s) '
                  f'matching {pattern}')

    with open(output_map_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'instance_id', 'ligand_name', 'chain', 'resnum', 'source_file', 'model_number',
            'n_atoms',
        ])
        writer.writeheader()
        writer.writerows(map_rows)

    print(f'Wrote {instance_id} ligand conformer instance(s) ({n_atoms_written} atom(s) total) '
          f'from {len(source_files)} file(s) to {output_ligs_pdb}')
    print(f'Wrote instance -> source mapping to {output_map_csv}')


if __name__ == '__main__':
    main()
