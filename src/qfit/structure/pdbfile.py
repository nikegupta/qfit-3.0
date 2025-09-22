from collections import defaultdict, namedtuple
import itertools
from math import inf
import logging

import numpy as np

import iotbx.pdb
import iotbx.cif.model
from libtbx import smart_open

__all__ = ["read_pdb", "write_pdb", "read_pdb_or_mmcif", "write_mmcif", "ANISOU_FIELDS"]
logger = logging.getLogger(__name__)

ANISOU_FIELDS = ("u00", "u11", "u22", "u01", "u02", "u12")


def _extract_record_type(atoms):
    records = []
    for atom in atoms:
        records.append("ATOM" if not atom.hetero else "HETATM")
    return records


def _extract_link_records(pdb_inp):
    link = defaultdict(list)
    for line in pdb_inp.extract_LINK_records():
        try:
            values = LinkRecord.parse_line(line)
            for field in LinkRecord.fields:
                link[field].append(values[field])
        except Exception as e:
            logger.error(str(e))
            logger.error("read_pdb: could not parse LINK data.")
    return link


def _extract_mmcif_links(mmcif_inp):
    try:
        _ = mmcif_inp.cif_block[LinkRecord.cif_fields[0]]
    except KeyError:
        return {}
    link = {}
    n_rows = len(mmcif_inp.cif_block[LinkRecord.cif_fields[0]])
    for field_name, cif_key, dtype in zip(
        LinkRecord.fields, LinkRecord.cif_fields, LinkRecord.dtypes
    ):

        def _to_value(x):
            if x == "?":
                return ""
            else:
                return dtype(x)
            
        if cif_key not in mmcif_inp.cif_block:
            # Field missing: pad with blanks to preserve row alignment
            link[field_name] = ["" for _ in range(n_rows)]
        else:
            raw_values = mmcif_inp.cif_block[cif_key]
            link[field_name] = [_to_value(x) for x in raw_values]
    return link


def get_pdb_hierarchy(pdb_inp):
    """
    Prepare an iotbx.pdb.hierarchy object from an iotbx.pdb.input object
    """
    pdb_hierarchy = pdb_inp.construct_hierarchy()
    atoms = pdb_hierarchy.atoms()
    atoms.reset_i_seq()
    atoms.reset_serial()
    atoms.reset_tmp()
    atoms.set_chemical_element_simple_if_necessary()
    return pdb_hierarchy


def read_pdb_or_mmcif(fname):
    """
    Parse a PDB or mmCIF file and return the iotbx.pdb object and associated
    content.
    """
    iotbx_in = iotbx.pdb.pdb_input_from_any(
        file_name=fname, source_info=None, raise_sorry_if_format_error=True
    )
    pdb_inp = iotbx_in.file_content()
    link_data = {}
    if iotbx_in.file_format == "pdb":
        link_data = _extract_link_records(pdb_inp)
    else:
        link_data = _extract_mmcif_links(pdb_inp)
    for attr, array in link_data.items():
        link_data[attr] = np.asarray(array)
    input_cls = namedtuple("PDBInput", ["pdb_in", "link_data", "file_format"])
    return input_cls(pdb_inp, link_data, iotbx_in.file_format)


def read_pdb(fname):
    return read_pdb_or_mmcif(fname)


def write_pdb(fname, structure, resolution=None):
    """
    Write a structure to a PDB file using the iotbx.pdb API
    """
    with smart_open.for_writing(fname, gzip_mode="wt") as f:
        if resolution is not None:
            f.write(f"REMARK   2 RESOLUTION.    {resolution:.2f} ANGSTROMS.\n")
        if structure.crystal_symmetry:
            f.write(
                "{}\n".format(
                    iotbx.pdb.format_cryst1_and_scale_records(
                        structure.crystal_symmetry
                    )
                )
            )
        if structure.link_data:
            _write_pdb_link_data(f, structure)
        for atom in structure.get_selected_atoms():
            atom_labels = atom.fetch_labels()
            f.write("{}\n".format(atom_labels.format_atom_record_group()))
        f.write("END")


def _write_pdb_link_data(f, structure):
    for record in zip(*[structure.link_data[x] for x in LinkRecord.fields]):
        record = dict(zip(LinkRecord.fields, record))
        # this will be different if the input was mmCIF
        record["record"] = "LINK"
        if not record["length"]:
            # If the LINK length is 0, then leave it blank.
            # This is a deviation from the PDB standard.
            record["length"] = ""
            fmtstr = LinkRecord.fmtstr.replace("{:>5.2f}", "{:5s}")
            f.write(fmtstr.format(*record.values()))
        else:
            f.write(LinkRecord.fmtstr.format(*record.values()))


def _to_mmcif_link_records(structure):
    if len(structure.link_data) > 0:
        conn_loop = iotbx.cif.model.loop(header=LinkRecord.cif_fields)
        for field_id, cif_key in zip(LinkRecord.fields, LinkRecord.cif_fields):
            for x in structure.link_data[field_id]:
                conn_loop[cif_key].append(str(x))
        return conn_loop
    return None


def load_combined_atoms(*atom_lists):
    """
    Utility to take any number of atom arrays and combine them into a new
    PDB hierarchy.  This is used to combine structures, but also to reorder
    atoms within a new hierarchy (since the hierarchy won't take unsorted
    selections).
    """
    atom_labels = []
    for atoms in atom_lists:
        atom_labels.extend([atom.fetch_labels() for atom in atoms])
    return load_atoms_from_labels(atom_labels)


def load_atoms_from_labels(atom_labels):
    atom_lines = itertools.chain(*[atom.format_atom_record_group().split('\n')
                                   for atom in atom_labels])
    return iotbx.pdb.pdb_input(source_info="qfit_structure",
                               lines=list(atom_lines))


def write_mmcif(fname, structure):
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

    # Remove any old struct_asym content
    # for key in list(cif_block.keys()):
    #     if key.startswith("_struct_asym") or key == "struct_asym":
    #         del cif_block[key]

    if structure.link_data:
        link_loop = _to_mmcif_link_records(structure)
        if link_loop:
            cif_block.add_loop(link_loop)

    # --- Extract atom_site columns ---
    atom_site = cif_block.get("_atom_site")
    if atom_site is None:
        raise ValueError("No _atom_site loop found in cif_block")

    asym_ids   = cif_block["_atom_site.label_asym_id"]
    resnames   = cif_block["_atom_site.label_comp_id"]
    resseqs    = cif_block["_atom_site.label_seq_id"]

    # Group residues by asym_id
    chain_residues = {}
    for asym_id, resname, resseq in zip(asym_ids, resnames, resseqs):
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




class RecordParser:
    """
    Interface class to provide record parsing routines for a PDB file.  This
    is no longer used for parsing ATOM records or crystal symmetry, which are
    handled by CCTBX, but it remains useful as a generic fixed-column-width
    parser for other records that CCTBX leaves unstructured.

    Deriving classes should have class variables for {fields, columns, dtypes,
    fmtstr}.
    """
    fields = tuple()
    columns = tuple()
    dtypes = tuple()
    fmtstr = tuple()
    fmttrs = tuple()

    # prevent assigning additional attributes
    __slots__ = []

    @classmethod
    def parse_line(cls, line):
        """Common interface for parsing a record from a PDB file.

        Args:
            line (str): A record, as read from a PDB file.

        Returns:
            dict[str, Union[str, int, float]]: fields that were parsed
                from the record.
        """
        values = {}
        for field, column, dtype in zip(cls.fields, cls.columns, cls.dtypes):
            try:
                values[field] = dtype(line[slice(*column)].strip())
            except ValueError:
                logger.error(
                    f"RecordParser.parse_line: could not parse "
                    f"{field} ({line[slice(*column)]}) as {dtype}"
                )
                values[field] = dtype()
        return values

    @classmethod
    def format_line(cls, values):
        """Formats record values into a line.

        Args:
            values (Iterable[Union[str, int, float]]): Values to be formatted.
        """
        assert len(values) == len(cls.fields)

        # Helper
        flatten = lambda iterable: sum(iterable, ())

        # Build list of spaces
        column_indices = flatten(cls.columns)
        space_columns = zip(column_indices[1:-1:2], column_indices[2:-1:2])
        space_lengths = map(lambda colpair: colpair[1] - colpair[0], space_columns)
        spaces = map(lambda n: " " * n, space_lengths)

        # Build list of fields
        field_lengths = map(lambda colpair: colpair[1] - colpair[0], cls.columns)
        formatted_values = map(
            lambda args: cls._fixed_length_format(*args),
            zip(values, cls.fmttrs, field_lengths, cls.dtypes),
        )

        # Intersperse formatted values with spaces
        line = itertools.zip_longest(formatted_values, spaces, fillvalue="")
        line = "".join(flatten(line)) + "\n"
        return line

    @staticmethod
    def _fixed_length_format(value, formatter, maxlen, dtype):
        """Formats a value, ensuring the length does not exceed available cols.

        If the value exceeds available length, it will be replaced with a
            "bad value" marker ('X', inf, or 0).

        Args:
            value (Union[str, int, float]): Value to be formatted
            formatter (str): Format-spec string
            maxlen (int): Maximum width of the formatted value
            dtype (type): Type of the field

        Returns:
            str: The formatted value, no wider than maxlen.
        """
        field = formatter.format(value)
        if len(field) > maxlen:
            replacement_field = None
            if dtype is str:
                replacement_field = "X" * maxlen
            elif dtype is float:
                replacement_field = formatter.format(inf)
            elif dtype is int:
                replacement_field = formatter.format(0)
            else:
                raise RuntimeError(f"Can't handle type {dtype} here")
            logger.warning(
                f"{field} exceeds field width {maxlen} chars. "
                f"Using {replacement_field}."
            )
            return replacement_field
        else:
            return field


class LinkRecord(RecordParser):
    # http://www.wwpdb.org/documentation/file-format-content/format33/sect6.html#LINK
    fields = (
        "record",
        "name1",
        "altloc1",
        "resn1",
        "chain1",
        "resi1",
        "icode1",
        "name2",
        "altloc2",
        "resn2",
        "chain2",
        "resi2",
        "icode2",
        "sym1",
        "sym2",
        "length",
    )
    columns = (
        (0, 6),
        (12, 16),
        (16, 17),
        (17, 20),
        (21, 22),
        (22, 26),
        (26, 27),
        (42, 46),
        (46, 47),
        (47, 50),
        (51, 52),
        (52, 56),
        (56, 57),
        (59, 65),
        (66, 72),
        (73, 78),
    )
    dtypes = (
        str,
        str,
        str,
        str,
        str,
        int,
        str,
        str,
        str,
        str,
        str,
        int,
        str,
        str,
        str,
        float,
    )
    fmtstr = (
        "{:<6s}"
        + " " * 6
        + " "
        + "{:<3s}{:1s}{:>3s}"
        + " "
        + "{:1s}{:>4d}{:1s}"
        + " " * 15
        + " "
        + "{:<3s}{:1s}{:>3s}"
        + " "
        + "{:1s}{:>4d}{:1s}"
        + " " * 2
        + "{:>6s} {:>6s} {:>5.2f}"
        + "\n"
    )
    # for mmCIF we need to fetch arrays equivalent to each of the column-based
    # fields in the PDB LINK records
    # https://mmcif.wwpdb.org/dictionaries/mmcif_pdbx_v40.dic/Categories/struct_conn.html
    cif_fields = [
        "_struct_conn.id",  # not really
        "_struct_conn.ptnr1_label_atom_id",
        "_struct_conn.pdbx_ptnr1_label_alt_id",
        "_struct_conn.ptnr1_auth_comp_id",
        "_struct_conn.ptnr1_auth_asym_id",
        "_struct_conn.ptnr1_auth_seq_id",
        "_struct_conn.pdbx_ptnr1_PDB_ins_code",
        "_struct_conn.ptnr2_label_atom_id",
        "_struct_conn.pdbx_ptnr2_label_alt_id",
        "_struct_conn.ptnr2_auth_comp_id",
        "_struct_conn.ptnr2_auth_asym_id",
        "_struct_conn.ptnr2_auth_seq_id",
        "_struct_conn.pdbx_ptnr2_PDB_ins_code",
        "_struct_conn.ptnr1_symmetry",
        "_struct_conn.ptnr2_symmetry",
        "_struct_conn.pdbx_dist_value",
    ]
