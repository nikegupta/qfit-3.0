from qfit import Structure
import sys
import iotbx.pdb
import iotbx.cif.model
from libtbx import smart_open

def write_mmcif(fname, structure, ligand_name):
    """
    Write a PLACER-compatible mmCIF including _entity, _entity_poly,
    _entity_nonpoly, _entity_poly_seq, and _pdbx_poly_seq_scheme.
    Collapses all waters into a single entity, but keeps ligands separate.
    Does NOT write any _struct_asym loop.
    """

    aa3to1 = {
        "ALA":"A","ARG":"R","ASN":"N","ASP":"D","CYS":"C","GLN":"Q","GLU":"E",
        "GLY":"G","HIS":"H","ILE":"I","LEU":"L","LYS":"K","MET":"M","PHE":"F",
        "PRO":"P","SER":"S","THR":"T","TRP":"W","TYR":"Y","VAL":"V"
    }
    AA3 = set(aa3to1.keys())

    # Build hierarchy just for as_cif_block (but entities come from cif_block)
    atoms = [atom.fetch_labels() for atom in structure.get_selected_atoms()]
    atom_lines = []
    for atom in atoms:
        line = atom.format_atom_record_group()
        if isinstance(line, list):
            atom_lines.extend(line)
        else:
            atom_lines.append(line)

    pdb_in = iotbx.pdb.pdb_input(source_info="qfit_structure", lines=atom_lines)
    hierarchy = pdb_in.construct_hierarchy()
    cif_block = hierarchy.as_cif_block(crystal_symmetry=structure.crystal_symmetry)

    # --- Extract atom_site columns ---
    atom_site = cif_block.get("_atom_site")
    if atom_site is None:
        raise ValueError("No _atom_site loop found in cif_block")

    asym_ids   = cif_block["_atom_site.label_asym_id"]
    resnames   = cif_block["_atom_site.label_comp_id"]
    resseqs    = cif_block["_atom_site.label_seq_id"]
    other_resseqs = cif_block["_atom_site.auth_seq_id"]

    # Group residues by asym_id
    with open(f'{ligand_name}_info.txt','w+') as f:
        chain_residues = {}
        for asym_id, resname, resseq, other_resseq in zip(asym_ids, resnames, resseqs, other_resseqs):
            if resname.strip() == ligand_name:

                print(f"Found ligand {ligand_name} at residue {other_resseq} in chain {asym_id}")
                f.write(f'{asym_id}-{ligand_name}-{other_resseq}\n')

            if resseq in (".", "?", None):  # skip waters/ligands with no seq
                seqnum = None
            else:
                try:
                    seqnum = int(resseq)
                except Exception:
                    seqnum = None
            chain_residues.setdefault(asym_id, []).append((seqnum, resname.strip()))

    # --- Entity bookkeeping ---
    entity_rows = []
    entity_poly_rows = []
    entity_poly_seq_rows = []
    pdbx_poly_seq_rows = []
    entity_nonpoly_rows = []

    entity_counter = 1
    chain_entities = {}
    water_entity_id = None

    for chid, residues in chain_residues.items():
        residues = sorted(set(residues), key=lambda x: (x[0] if x[0] is not None else 1e9))
        if not residues:
            continue
        first_res = residues[0][1]

        # --- polymer chain ---
        if first_res in AA3:
            eid = str(entity_counter)
            entity_counter += 1
            chain_entities[chid] = eid

            entity_rows.append([eid, "polymer", f"polymer chain {chid}", "?", "?"])

            seq = "".join(aa3to1.get(res[1], "X") for res in residues if res[0] is not None)
            entity_poly_rows.append([
                eid, "polypeptide(L)", "no", "no", seq, seq, chid, "?"
            ])

            for (i, (seqnum, resname)) in enumerate(residues, start=1):
                if seqnum is None:
                    continue
                one_letter = aa3to1.get(resname, "X")
                entity_poly_seq_rows.append([eid, str(i), resname, one_letter])
                pdbx_poly_seq_rows.append([eid, str(i), resname, one_letter, chid])

        # --- waters (collapse all) ---
        elif all(r[1] == "HOH" for r in residues):
            if water_entity_id is None:
                water_entity_id = str(entity_counter)
                entity_counter += 1
                entity_rows.append([water_entity_id, "non-polymer", "water", "?", "?"])
                entity_nonpoly_rows.append([water_entity_id, "HOH"])
            chain_entities[chid] = water_entity_id

        # --- ligands ---
        else:
            eid = str(entity_counter)
            entity_counter += 1
            chain_entities[chid] = eid

            entity_rows.append([eid, "non-polymer", f"nonpoly chain {chid}", "?", "?"])
            comp_id = first_res
            entity_nonpoly_rows.append([eid, comp_id])

    # --- Build CIF loops (no struct_asym) ---
    def add_loop(headers, rows):
        if not rows:
            return
        loop = iotbx.cif.model.loop(headers)
        for row in rows:
            loop.add_row(row)
        cif_block.add_loop(loop)

    add_loop((
        "_entity.id", "_entity.type", "_entity.pdbx_description",
        "_entity.formula_weight", "_entity.pdbx_number_of_molecules"
    ), entity_rows)

    add_loop((
        "_entity_poly.entity_id", "_entity_poly.type", "_entity_poly.nstd_linkage",
        "_entity_poly.nstd_monomer", "_entity_poly.pdbx_seq_one_letter_code",
        "_entity_poly.pdbx_seq_one_letter_code_can", "_entity_poly.pdbx_strand_id",
        "_entity_poly.pdbx_target_identifier"
    ), entity_poly_rows)

    add_loop((
        "_entity_nonpoly.entity_id", "_entity_nonpoly.comp_id"
    ), entity_nonpoly_rows)

    add_loop((
        "_entity_poly_seq.entity_id", "_entity_poly_seq.num",
        "_entity_poly_seq.mon_id", "_entity_poly_seq.mon_code"
    ), entity_poly_seq_rows)

    add_loop((
        "_pdbx_poly_seq_scheme.entity_id", "_pdbx_poly_seq_scheme.seq_id",
        "_pdbx_poly_seq_scheme.mon_id", "_pdbx_poly_seq_scheme.mon_code",
        "_pdbx_poly_seq_scheme.pdb_strand_id"
    ), pdbx_poly_seq_rows)

    # Write file
    with smart_open.for_writing(fname, gzip_mode="wt") as f:
        cif_object = iotbx.cif.model.cif()
        cif_object["qfit"] = cif_block
        print(cif_object, file=f)

                            
def main():
    if len(sys.argv) != 3:
        print("Usage: python script.py <cif_file> <ligand_resname>")
        sys.exit(1)

    cif_file = sys.argv[1]
    ligand_name = sys.argv[2]

    structure = Structure.fromfile(cif_file)
    write_mmcif(f'modified_{cif_file}', structure, ligand_name)


if __name__ == "__main__":
    main()