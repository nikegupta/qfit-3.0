import os
import re
import csv
import argparse
import tempfile
from pathlib import Path

import numpy as np
import biotite.structure as struc
import biotite.structure.io.pdb as pdb

# Suppress annoying import warnings
devnull = open(os.devnull, 'w')
old_stderr = os.dup(2)
old_stdout = os.dup(1)
os.dup2(devnull.fileno(), 2)
os.dup2(devnull.fileno(), 1)
import coot_headless_api
# Restore stderr/stdout
os.dup2(old_stderr, 2)
os.dup2(old_stdout, 1)
os.close(old_stderr)
os.close(old_stdout)
devnull.close()

EXTRA_FIELDS = ["b_factor", "occupancy", "charge"]


def build_argparser():
    p = argparse.ArgumentParser(
        description=(
            "Real-space refines exactly the residues listed in a CSV (e.g. "
            "residues_with_placer_conformers.csv from build_final_model.py) within a "
            "single 'final model' PDB - one merged model already containing every "
            "protein residue (the PLACER-selected ones plus every apo-fallback one) "
            "plus N numbered LIG residues (resid 1..N). No apo structure or distance "
            "cutoff needs to be supplied or recomputed here: the final model pdb is "
            "already the complete merged structure, and the CSV alone determines the "
            "refinement selection. Every other protein residue already present in the "
            "final model pdb - and every LIG - is used as fixed geometric context but "
            "is not itself refined. Every ligand's own CIF dictionary is imported (LIG "
            "with resid i uses the i-th --cif-list entry) so each ligand is properly "
            "restrained even though none of them are themselves refined."
        )
    )
    p.add_argument("final_model_pdb", type=Path,
                    help="Path to the single-model merged 'final model' PDB: every "
                         "protein residue (PLACER-selected or apo-fallback), plus N "
                         "numbered LIG residues (resid 1..N)")
    p.add_argument("residues_csv", type=Path,
                    help="Path to a headerless CSV listing the residues to refine, one "
                         "per line as '<chain><resnum>' (e.g. 'A101') - the format "
                         "written by build_final_model.py's "
                         "residues_with_placer_conformers.csv")
    p.add_argument("map", type=Path, help="Path to map for refinement (CCP4/MRC format)")
    p.add_argument("output_pdb", type=Path, help="Output path for the refined PDB")

    # Mutually exclusive: either one CIF for every ligand, or a per-ligand list
    # (needed when the different LIG residues in the final PDB are different
    # stereoisomers/ligands, each requiring their own restraints dictionary).
    cif_group = p.add_mutually_exclusive_group(required=True)
    cif_group.add_argument(
        "--cif-restraints", type=Path,
        help="Single CIF restraints file applied to every LIG residue."
    )
    cif_group.add_argument(
        "--cif-list", type=str,
        help="Comma-separated list of CIF paths, one per LIG residue in order (the "
             "LIG with resid i uses the i-th path, 1-indexed). Length must equal the "
             "number of distinct LIG residue numbers in final_model_pdb."
    )
    p.add_argument("--n-cycles", type=int, default=1000, help="Number of refinement cycles (default: 1000)")
    p.add_argument("--map-weight", type=float, default=50.0, help="Weight on map vs geometry (default: 50.0)")
    p.add_argument("--difference-map", action="store_true", default=False, help="Treat map as a difference map")
    p.add_argument("--moved-threshold", type=float, default=0.01,
                    help="Minimum max-atom displacement (Angstroms) for a refined residue "
                         "to be counted as 'moved' (default: 0.01)")
    return p


def suppress_output(func, *args, **kwargs):
    devnull = open(os.devnull, 'w')
    old_stderr, old_stdout = os.dup(2), os.dup(1)
    os.dup2(devnull.fileno(), 2)
    os.dup2(devnull.fileno(), 1)
    try:
        result = func(*args, **kwargs)
    finally:
        os.dup2(old_stderr, 2)
        os.dup2(old_stdout, 1)
        os.close(old_stderr)
        os.close(old_stdout)
        devnull.close()
    return result


def read_single_model_structure(path):
    """Reads a PDB file expected to contain exactly one model's worth of
    atoms, regardless of whether the file actually has an explicit
    MODEL/ENDMDL wrapper. Biotite's get_structure(model=N) requires that many
    literal MODEL records to be present to use N as an index - many
    single-model PDBs (this pipeline's final model and apo files included)
    omit that wrapper entirely, which makes model=1 raise "the file has 0
    models" even though the atom data is right there. Calling get_structure
    without an explicit model number instead lets biotite auto-detect: it
    returns an AtomArray directly for a model-less/single-model file, or an
    AtomArrayStack if the file legitimately contains more than one MODEL
    block - in which case the first model is used.
    """
    pdbfile = pdb.PDBFile.read(str(path))
    structure = pdb.get_structure(pdbfile, extra_fields=EXTRA_FIELDS)
    if isinstance(structure, struc.AtomArrayStack):
        if structure.stack_depth() > 1:
            print(f'WARNING: {path} contains {structure.stack_depth()} models; using the first one')
        structure = structure[0]
    return structure


def get_lig_residues(atom_array):
    """Returns {resid: lig_atom_array} for every distinct LIG residue number
    found in atom_array (e.g. {1: <atoms of LIG 1>, 2: <atoms of LIG 2>, ...})."""
    lig_mask = atom_array.res_name == "LIG"
    lig_atoms = atom_array[lig_mask]
    if len(lig_atoms) == 0:
        raise RuntimeError("No LIG residues found in the final model pdb")

    ligs = {}
    for resid in sorted(set(int(r) for r in lig_atoms.res_id)):
        ligs[resid] = lig_atoms[lig_atoms.res_id == resid]
    return ligs


def get_protein_residues(atom_array):
    """Returns a sorted, de-duplicated list of (chain_id, res_id) for every
    non-LIG (protein) residue present in atom_array - i.e. every protein
    residue in the merged final model pdb, whether it was a PLACER pick or
    an apo fallback."""
    protein_mask = atom_array.res_name != "LIG"
    protein_atoms = atom_array[protein_mask]

    seen = set()
    residues = []
    for atom in protein_atoms:
        key = (atom.chain_id, atom.res_id)
        if key not in seen:
            seen.add(key)
            residues.append(key)
    return residues


def parse_residues_csv(path):
    """Parses a headerless CSV with one residue per line, formatted as
    '<chain><resnum>' (e.g. 'A101') - the format written by
    build_final_model.py's residues_with_placer_conformers.csv. Returns a
    list of (chain_id, res_id) tuples in file order (blank lines skipped).
    """
    pattern = re.compile(r'^([A-Za-z]+)(\d+)$')
    residues = []
    with open(path, newline='') as f:
        for line_num, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue
            match = pattern.match(line)
            if not match:
                raise ValueError(
                    f"{path}:{line_num}: could not parse residue from '{line}' "
                    f"(expected '<chain><resnum>', e.g. 'A101')"
                )
            chain_id, res_id = match.group(1), int(match.group(2))
            residues.append((chain_id, res_id))
    return residues


def residues_to_cids(residues):
    """Build a coot CID selection string ('//chain/resid||...') from (chain_id, res_id) tuples."""
    return "||".join(f"//{chain_id}/{res_id}" for chain_id, res_id in residues)


def build_csv_output_path(output_pdb):
    out_path = Path(output_pdb)
    stem_path = out_path.with_suffix("")
    return stem_path.parent / f"{stem_path.name}_refined_residues.csv"


def build_non_selected_csv_output_path(output_pdb):
    out_path = Path(output_pdb)
    stem_path = out_path.with_suffix("")
    return stem_path.parent / f"{stem_path.name}_non_selected_residues.csv"


def build_atom_coord_map(atom_array):
    """Map (chain_id, res_id, atom_name) -> coord for every atom in the array."""
    coord_map = {}
    for atom in atom_array:
        key = (atom.chain_id, atom.res_id, atom.atom_name)
        coord_map[key] = np.array(atom.coord, dtype=float)
    return coord_map


def compute_residue_displacements(pre_array, post_array, residues):
    """For each (chain_id, res_id) in residues, compute the max per-atom
    displacement between pre_array and post_array.

    Returns a dict: (chain_id, res_id) -> (max_displacement, n_atoms_compared, n_atoms_missing)
    """
    pre_coords = build_atom_coord_map(pre_array)
    post_coords = build_atom_coord_map(post_array)

    # Group atom names by residue once, from the pre-refinement structure
    residue_atom_names = {}
    for atom in pre_array:
        key = (atom.chain_id, atom.res_id)
        residue_atom_names.setdefault(key, []).append(atom.atom_name)

    results = {}
    for chain_id, res_id in residues:
        atom_names = residue_atom_names.get((chain_id, res_id), [])
        max_disp = 0.0
        n_compared = 0
        n_missing = 0
        for atom_name in atom_names:
            key = (chain_id, res_id, atom_name)
            pre_c = pre_coords.get(key)
            post_c = post_coords.get(key)
            if pre_c is None or post_c is None:
                n_missing += 1
                continue
            disp = float(np.linalg.norm(post_c - pre_c))
            if disp > max_disp:
                max_disp = disp
            n_compared += 1
        results[(chain_id, res_id)] = (max_disp, n_compared, n_missing)
    return results


def refine_structure(mc, merged_pdb_path, map_path, cif_per_resid, selection_cid,
                      n_cycles, map_weight, difference_map):
    """Run real-space refinement on the merged final model structure (every
    protein residue plus every ligand), restricted to exactly the residues
    named in selection_cid. Every distinct CIF dictionary in cif_per_resid is
    imported first, so every LIG residue has proper bond/angle restraints
    even though none of them are themselves refined.

    Note: every ligand here shares the residue name "LIG". If two LIG
    residues are chemically different molecules that happen to use the same
    comp_id inside their own CIF files, importing both dictionaries will
    collide (the later import wins for that comp_id) - this is fine when
    every LIG is really the same compound in different poses, but worth
    knowing if the ligands are genuinely distinct chemistries.
    """
    suppress_output(mc.geometry_init_standard)

    seen_cifs = set()
    for cif_path in cif_per_resid.values():
        if cif_path in seen_cifs:
            continue
        if not cif_path.exists():
            raise FileNotFoundError(f"CIF restraints file not found: {cif_path}")
        suppress_output(mc.import_cif_dictionary, str(cif_path), -999999)
        seen_cifs.add(cif_path)

    imol = suppress_output(mc.read_pdb, str(merged_pdb_path))
    if imol < 0:
        raise RuntimeError(f"Failed to read PDB: {merged_pdb_path}")

    imol_map = suppress_output(mc.read_ccp4_map, str(map_path), difference_map)
    if imol_map < 0:
        raise RuntimeError(f"Failed to read map: {map_path}")

    mc.set_imol_refinement_map(imol_map)
    mc.set_map_weight(map_weight)

    if not selection_cid:
        raise RuntimeError("No protein residues selected; nothing to refine")

    # mode="SINGLE" restricts refinement to exactly the residues named in
    # selection_cid (their neighbors, including every LIG, are used as fixed
    # geometric context but are not themselves moved).
    suppress_output(mc.refine_residues_using_atom_cid, imol, selection_cid, "SINGLE", n_cycles)

    return imol


def main():
    p = build_argparser()
    args = p.parse_args()

    final_array = read_single_model_structure(args.final_model_pdb)
    print(f"Loaded final model structure from {args.final_model_pdb} ({len(final_array)} atoms)")

    ligs = get_lig_residues(final_array)
    n_ligs = len(ligs)
    print(f"Found {n_ligs} LIG residue(s): resids {sorted(ligs.keys())}")

    # Resolve per-ligand CIF: either the same file for every LIG, or an
    # explicit ordered list (the i-th --cif-list path applies to the LIG
    # residue with resid i, 1-indexed).
    if args.cif_restraints is not None:
        cif_per_resid = {resid: args.cif_restraints for resid in ligs}
    else:
        cif_paths = [Path(c.strip()) for c in args.cif_list.split(",")]
        if len(cif_paths) != n_ligs:
            raise ValueError(
                f"--cif-list has {len(cif_paths)} entries but "
                f"{args.final_model_pdb} has {n_ligs} distinct LIG residue(s)"
            )
        cif_per_resid = {resid: cif_paths[resid - 1] for resid in ligs}

    missing_cifs = [str(c) for c in set(cif_per_resid.values()) if not c.exists()]
    if missing_cifs:
        raise FileNotFoundError(f"CIF restraints file(s) not found: {missing_cifs}")

    # Every protein residue present in the merged final model pdb - this is
    # the full protein, not just a subset - whether it was a PLACER pick or
    # an apo fallback (that distinction was already made upstream).
    all_protein_residues = get_protein_residues(final_array)
    all_protein_set = set(all_protein_residues)
    print(f"{len(all_protein_residues)} protein residue(s) total in {args.final_model_pdb}")

    # The actual refinement selection comes entirely from the CSV.
    csv_residues = parse_residues_csv(args.residues_csv)
    selected_set = set(csv_residues)

    missing_from_pdb = selected_set - all_protein_set
    if missing_from_pdb:
        raise RuntimeError(
            f"{len(missing_from_pdb)} residue(s) listed in {args.residues_csv} were not "
            f"found in {args.final_model_pdb}: {sorted(missing_from_pdb)[:5]}"
        )

    # Keep refinement order consistent with the pdb's own residue order
    # rather than the CSV's order.
    protein_residues = [r for r in all_protein_residues if r in selected_set]
    selection_cid = residues_to_cids(protein_residues)
    print(f"{len(protein_residues)} of {len(all_protein_residues)} protein residue(s) "
          f"selected for refinement (from {args.residues_csv})")

    # The final model pdb is already the complete merged structure (every
    # protein residue + every ligand) - nothing to splice together here.
    merged_array = final_array.copy()

    csv_path = build_csv_output_path(args.output_pdb)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["chain_id", "residue_number", "max_displacement_A", "moved"])

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        tmp_in = tmpdir / "merged.pdb"
        tmp_out = tmpdir / "refined.pdb"

        tmp_pdb = pdb.PDBFile()
        pdb.set_structure(tmp_pdb, merged_array)
        tmp_pdb.write(str(tmp_in))

        mc = coot_headless_api.molecules_container_t(False)
        imol = refine_structure(
            mc, tmp_in, args.map, cif_per_resid, selection_cid,
            args.n_cycles, args.map_weight, args.difference_map
        )
        suppress_output(mc.write_coordinates, imol, str(tmp_out))
        print(f"Refined {len(protein_residues)} residue(s) across {n_ligs} ligand(s)")

        # Re-load the refined structure and compare pre- vs post-refinement
        # coordinates for exactly the residues that were in the selection,
        # to confirm refinement actually moved them - and moved nothing else.
        refined_array = read_single_model_structure(tmp_out)

        displacements = compute_residue_displacements(merged_array, refined_array, protein_residues)

        moved_count = sum(
            1 for (max_disp, n_compared, n_missing) in displacements.values()
            if max_disp > args.moved_threshold
        )
        print(f"{moved_count}/{len(protein_residues)} selected residue(s) moved "
              f"by more than {args.moved_threshold} A")

        any_missing = [key for key, (_, _, n_missing) in displacements.items() if n_missing > 0]
        if any_missing:
            print(f"WARNING: {len(any_missing)} residue(s) had atoms missing from the "
                  f"pre/post comparison (e.g. {any_missing[:3]})")

        for chain_id, res_id in protein_residues:
            max_disp, n_compared, n_missing = displacements[(chain_id, res_id)]
            moved = max_disp > args.moved_threshold
            csv_writer.writerow([chain_id, res_id, f"{max_disp:.4f}", moved])

        # Every protein residue in the final model pdb that was NOT in the
        # refinement selection. These are fixed context during refinement,
        # so displacement should be ~0, but real space refinement can
        # occasionally nudge neighboring atoms, and this also serves as the
        # scope-leak check below.
        non_selected_residues = [r for r in all_protein_residues if r not in selected_set]

        non_selected_displacements = compute_residue_displacements(
            merged_array, refined_array, non_selected_residues
        )

        non_selected_csv_path = build_non_selected_csv_output_path(args.output_pdb)
        with open(non_selected_csv_path, "w", newline="") as non_selected_csv_file:
            non_selected_csv_writer = csv.writer(non_selected_csv_file)
            non_selected_csv_writer.writerow(["chain_id", "residue_number", "max_displacement_A", "moved"])
            for chain_id, res_id in non_selected_residues:
                max_disp, n_compared, n_missing = non_selected_displacements[(chain_id, res_id)]
                moved = max_disp > args.moved_threshold
                non_selected_csv_writer.writerow([chain_id, res_id, f"{max_disp:.4f}", moved])
        print(f"Wrote {len(non_selected_residues)} non-selected residue(s) and their "
              f"pre-vs-post displacements to {non_selected_csv_path}")

        leaked = [
            (key, max_disp) for key, (max_disp, n_compared, n_missing)
            in non_selected_displacements.items()
            if max_disp > args.moved_threshold
        ]

        if leaked:
            print(f"WARNING: {len(leaked)} residue(s) OUTSIDE the {len(protein_residues)}-residue "
                  f"selection moved by more than {args.moved_threshold} A -- refinement scope "
                  f"leaked beyond selection_cid:")
            for (chain_id, res_id), max_disp in leaked:
                print(f"  {chain_id} {res_id}: {max_disp:.4f} A")
        else:
            print(f"Confirmed: no residues outside the {len(protein_residues)}-residue "
                  f"selection moved.")

        # Sanity check: none of the LIG residues were in selection_cid, so
        # none of them should have moved at all.
        for resid, lig_atoms in ligs.items():
            lig_chain = lig_atoms.chain_id[0]
            lig_key = (lig_chain, resid)
            lig_displacement = compute_residue_displacements(merged_array, refined_array, [lig_key])
            lig_max_disp, lig_n_compared, lig_n_missing = lig_displacement[lig_key]
            if lig_max_disp > args.moved_threshold:
                print(f"NOTE: LIG residue {lig_key} moved by {lig_max_disp:.4f} A "
                      f"during refinement (it was not in the refined selection).")
            else:
                print(f"Confirmed: LIG residue {lig_key} did not move "
                      f"(max displacement {lig_max_disp:.4f} A).")
            if lig_n_missing > 0:
                print(f"WARNING: {lig_n_missing} atom(s) of LIG {resid} missing from pre/post comparison.")

        out_path = args.output_pdb
        out_path.parent.mkdir(parents=True, exist_ok=True)
        final_out_pdb = pdb.PDBFile()
        pdb.set_structure(final_out_pdb, refined_array)
        final_out_pdb.write(str(out_path))
        print(f"Wrote {out_path}")

    csv_file.close()
    print(f"Wrote refined-residue table to {csv_path}")
    print("Done.")


if __name__ == "__main__":
    main()