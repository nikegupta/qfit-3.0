import os
import argparse
import tempfile
from pathlib import Path
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


def build_argparser():
    p = argparse.ArgumentParser(
        description="Refine LIG residues in a multi-model PDB against a map using coot-headless."
    )
    p.add_argument("input_pdb", type=Path, help="Path to the input multi-model PDB file")
    p.add_argument("map", type=Path, help="Path to map for refinement (CCP4/MRC format)")
    p.add_argument("output_pdb", type=Path, help="Path to the output refined multi-model PDB")

    # Mutually exclusive: either one CIF for all models, or a per-model list
    cif_group = p.add_mutually_exclusive_group(required=True)
    cif_group.add_argument(
        "--cif-restraints", type=Path,
        help="Single CIF restraints file applied to all models"
    )
    cif_group.add_argument(
        "--cif-list", type=str,
        help="Comma-separated list of CIF paths, one per model in order"
    )

    p.add_argument("--n-cycles", type=int, default=1000, help="Number of refinement cycles (default: 1000)")
    p.add_argument("--map-weight", type=float, default=50.0, help="Weight on map vs geometry (default: 50.0)")
    p.add_argument("--difference-map", action="store_true", default=False, help="Treat map as a difference map")
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


def get_lig_cids(atom_array):
    """Build coot CID selection strings for all LIG residues in an AtomArray."""
    lig_mask = atom_array.res_name == "LIG"
    lig_atoms = atom_array[lig_mask]
    if len(lig_atoms) == 0:
        return []
    seen = set()
    cids = []
    for atom in lig_atoms:
        key = (atom.chain_id, atom.res_id)
        if key not in seen:
            seen.add(key)
            cids.append(f"//{atom.chain_id}/{atom.res_id}")
    return cids


def transplant_lig_coords(original_array, refined_array):
    """
    Copy LIG atom coordinates from refined_array into original_array.
    Matches atoms by (chain_id, res_id, atom_name).
    Only coordinates are updated; all other annotations come from original.
    """
    refined_lig_mask = refined_array.res_name == "LIG"
    refined_lig = refined_array[refined_lig_mask]

    refined_coords = {}
    for atom in refined_lig:
        key = (atom.chain_id, atom.res_id, atom.atom_name)
        refined_coords[key] = atom.coord.copy()

    if not refined_coords:
        raise RuntimeError("No LIG residues found in refined model")

    updated = original_array.copy()
    for i, atom in enumerate(updated):
        if atom.res_name != "LIG":
            continue
        key = (atom.chain_id, atom.res_id, atom.atom_name)
        if key in refined_coords:
            updated.coord[i] = refined_coords[key]

    return updated


def refine_single_model(mc, pdb_path, cif_restraints, map_path,
                         lig_cids, n_cycles, map_weight, difference_map):
    suppress_output(mc.geometry_init_standard)
    suppress_output(mc.import_cif_dictionary, str(cif_restraints), -999999)

    imol = suppress_output(mc.read_pdb, str(pdb_path))
    if imol < 0:
        raise RuntimeError(f"Failed to read PDB: {pdb_path}")

    imol_map = suppress_output(mc.read_ccp4_map, str(map_path), difference_map)
    if imol_map < 0:
        raise RuntimeError(f"Failed to read map: {map_path}")

    mc.set_imol_refinement_map(imol_map)
    mc.set_map_weight(map_weight)

    if not lig_cids:
        raise RuntimeError("No LIG residues found in model")

    selection = "||".join(lig_cids)
    suppress_output(mc.refine_residues_using_atom_cid, imol, selection, "ALL", n_cycles)

    return imol


def main():
    p = build_argparser()
    args = p.parse_args()

    pdb_file = pdb.PDBFile.read(str(args.input_pdb))
    n_models = pdb.get_model_count(pdb_file)
    print(f"Found {n_models} model(s) in {args.input_pdb}")

    # Resolve per-model CIF list
    if args.cif_restraints is not None:
        # Same CIF for every model
        cif_per_model = [args.cif_restraints] * n_models
    else:
        cif_per_model = [Path(p.strip()) for p in args.cif_list.split(",")]
        if len(cif_per_model) != n_models:
            raise ValueError(
                f"--cif-list has {len(cif_per_model)} entries but PDB has {n_models} models"
            )
        missing = [str(c) for c in cif_per_model if not c.exists()]
        if missing:
            raise FileNotFoundError(f"CIF files not found: {missing}")

    original_arrays = [
        pdb.get_structure(pdb_file, model=i, extra_fields=["b_factor", "occupancy", "charge"])
        for i in range(1, n_models + 1)
    ]

    refined_arrays = []

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        for model_idx, (atom_array, cif_path) in enumerate(
            zip(original_arrays, cif_per_model), start=1
        ):
            print(f"\nRefining model {model_idx} with {cif_path.name}...")

            lig_cids = get_lig_cids(atom_array)
            if not lig_cids:
                print(f"  No LIG residues in model {model_idx}, skipping.")
                refined_arrays.append(atom_array)
                continue
            print(f"  Found {len(lig_cids)} LIG residue(s): {lig_cids}")

            tmp_in  = tmpdir / f"model_{model_idx}_in.pdb"
            tmp_out = tmpdir / f"model_{model_idx}_out.pdb"

            tmp_pdb = pdb.PDBFile()
            pdb.set_structure(tmp_pdb, atom_array)
            tmp_pdb.write(str(tmp_in))

            mc = coot_headless_api.molecules_container_t(False)
            imol = refine_single_model(
                mc, tmp_in, cif_path, args.map,
                lig_cids, args.n_cycles, args.map_weight, args.difference_map
            )
            suppress_output(mc.write_coordinates, imol, str(tmp_out))

            refined_pdb = pdb.PDBFile.read(str(tmp_out))
            refined_array = pdb.get_structure(
                refined_pdb, model=1, extra_fields=["b_factor", "occupancy", "charge"]
            )

            updated_array = transplant_lig_coords(atom_array, refined_array)
            refined_arrays.append(updated_array)
            print(f"  Model {model_idx} done.")

    print(f"\nWriting refined multi-model PDB to {args.output_pdb}")
    with open(args.output_pdb, 'w') as out_f:
        for model_idx, atom_array in enumerate(refined_arrays, start=1):
            tmp_model_pdb = pdb.PDBFile()
            pdb.set_structure(tmp_model_pdb, atom_array)
            lines = tmp_model_pdb.lines
            lines = [l for l in lines if not l.startswith("MODEL")
                     and not l.startswith("ENDMDL")
                     and not l.startswith("END")]
            out_f.write(f"MODEL{model_idx:>9}\n")
            out_f.writelines(l + "\n" if not l.endswith("\n") else l for l in lines)
            out_f.write("ENDMDL\n")
    print("Done.")


if __name__ == "__main__":
    main()