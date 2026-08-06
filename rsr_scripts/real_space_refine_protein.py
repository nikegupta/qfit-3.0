import os
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


class NoNearbyProteinResiduesError(RuntimeError):
    """Raised when no apo protein residues fall within the cutoff distance of
    a model's LIG residue. This is treated as a per-model, skippable
    condition (rather than a fatal error) by the caller in main(), since it
    reflects something about that particular cluster representative (e.g. a
    ligand sitting outside the modeled protein region) rather than a problem
    with the overall run."""
    pass


def build_argparser():
    p = argparse.ArgumentParser(
        description=(
            "For each model in a multi-model PDB (each containing one LIG residue), "
            "transplant that LIG onto a copy of an apo structure (same frame), "
            "real-space refine the apo protein residues near the ligand against a map "
            "(with the ligand present in the same structure, restrained by its own "
            "CIF dictionary so it isn't left as an unrestrained set of free atoms), "
            "and write out one merged/refined PDB per model."
        )
    )
    p.add_argument("input_multimodel_pdb", type=Path, help="Path to the input multi-model PDB file (1 LIG residue per model)")
    p.add_argument("input_apo_pdb", type=Path, help="Path to the input single-model apo PDB file")
    p.add_argument("map", type=Path, help="Path to map for refinement (CCP4/MRC format)")
    p.add_argument("output_pdb", type=Path, help="Output PDB naming pattern; each model's output is named "
                                                  "<stem>_<i><suffix>, e.g. out.pdb -> out_1.pdb, out_2.pdb, ... "
                                                  "(model index i always matches that model's 1-based position "
                                                  "in input_multimodel_pdb, even if earlier/later models were "
                                                  "skipped, so output indices stay in 1:1 correspondence with "
                                                  "the upstream cluster_reps.csv row order)")

    # Mutually exclusive: either one CIF for all models, or a per-model list
    # (needed when different models in the multimodel PDB are different
    # stereoisomers/ligands, each requiring their own restraints dictionary).
    cif_group = p.add_mutually_exclusive_group(required=True)
    cif_group.add_argument(
        "--cif-restraints", type=Path,
        help="Single CIF restraints file applied to every model's LIG residue."
    )
    cif_group.add_argument(
        "--cif-list", type=str,
        help="Comma-separated list of CIF paths, one per model in order (model i uses "
             "the i-th path). Length must equal the number of models in input_multimodel_pdb."
    )
    p.add_argument("--cutoff", type=float, default=10.0,
                    help="Distance cutoff (Angstroms) from the LIG residue used to select protein residues "
                         "for real-space refinement (default: 10.0)")
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


def get_lig_residue(atom_array, model_idx):
    """Extract the single LIG residue from a model's AtomArray."""
    lig_mask = atom_array.res_name == "LIG"
    lig_atoms = atom_array[lig_mask]
    if len(lig_atoms) == 0:
        raise RuntimeError(f"No LIG residue found in model {model_idx}")

    keys = {(a.chain_id, a.res_id) for a in lig_atoms}
    if len(keys) != 1:
        raise RuntimeError(
            f"Expected exactly 1 LIG residue in model {model_idx}, found {len(keys)}: {sorted(keys)}"
        )
    return lig_atoms


def get_protein_residues_within_cutoff(apo_array, lig_coord, cutoff):
    """Return a sorted, de-duplicated list of (chain_id, res_id) tuples for apo
    protein residues with any atom within `cutoff` Angstroms of any LIG atom coordinate."""
    protein_mask = struc.filter_amino_acids(apo_array)
    protein_atoms = apo_array[protein_mask]
    if len(protein_atoms) == 0:
        raise RuntimeError("No protein (amino acid) atoms found in apo structure")

    # Pairwise distances: (n_protein_atoms, n_lig_atoms)
    diff = protein_atoms.coord[:, None, :] - lig_coord[None, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=-1))
    min_dist = dist.min(axis=1)
    close_mask = min_dist <= cutoff
    close_atoms = protein_atoms[close_mask]

    seen = set()
    residues = []
    for atom in close_atoms:
        key = (atom.chain_id, atom.res_id)
        if key not in seen:
            seen.add(key)
            residues.append(key)
    return residues


def residues_to_cids(residues):
    """Build a coot CID selection string ('//chain/resid||...') from (chain_id, res_id) tuples."""
    return "||".join(f"//{chain_id}/{res_id}" for chain_id, res_id in residues)


def build_output_path(output_pattern, model_idx):
    out_path = Path(output_pattern)
    suffix = out_path.suffix if out_path.suffix else ".pdb"
    stem_path = out_path.with_suffix("")
    return stem_path.parent / f"{stem_path.name}_{model_idx}{suffix}"


def build_csv_output_path(output_pattern):
    out_path = Path(output_pattern)
    stem_path = out_path.with_suffix("")
    return stem_path.parent / f"{stem_path.name}_refined_residues.csv"


def build_expected_moved_csv_path(output_pattern, model_idx):
    """Path for the per-model plain list of residues selected for refinement
    (i.e. within --cutoff of that model's LIG, and therefore expected to move),
    e.g. out.pdb -> out_1_expected_moved_residues.csv."""
    out_path = Path(output_pattern)
    stem_path = out_path.with_suffix("")
    return stem_path.parent / f"{stem_path.name}_{model_idx}_expected_moved_residues.csv"


def build_combined_expected_moved_csv_path(output_pattern):
    """Path for the single, non-redundant list of residues selected for
    refinement across every backbone-refined model, named refined_residues.csv
    (in the same directory as the other outputs)."""
    out_path = Path(output_pattern)
    return out_path.with_suffix("").parent / "refined_residues.csv"


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


def refine_model(mc, merged_pdb_path, map_path, cif_restraints, selection_cid,
                  n_cycles, map_weight, difference_map):
    """Run real-space refinement on the merged (apo protein + LIG) structure,
    restricted to exactly the residues named in selection_cid. The ligand's
    CIF dictionary is imported so it has proper bond/angle restraints even
    though it isn't itself being refined."""
    suppress_output(mc.geometry_init_standard)

    if not cif_restraints.exists():
        raise FileNotFoundError(f"CIF restraints file not found: {cif_restraints}")
    suppress_output(mc.import_cif_dictionary, str(cif_restraints), -999999)

    imol = suppress_output(mc.read_pdb, str(merged_pdb_path))
    if imol < 0:
        raise RuntimeError(f"Failed to read PDB: {merged_pdb_path}")

    imol_map = suppress_output(mc.read_ccp4_map, str(map_path), difference_map)
    if imol_map < 0:
        raise RuntimeError(f"Failed to read map: {map_path}")

    mc.set_imol_refinement_map(imol_map)
    mc.set_map_weight(map_weight)

    if not selection_cid:
        raise NoNearbyProteinResiduesError(
            "No protein residues found within cutoff of LIG; nothing to refine"
        )

    # mode="SINGLE" restricts refinement to exactly the residues named in
    # selection_cid (their neighbors, including LIG, are used as fixed
    # geometric context but are not themselves moved). Previously this used
    # mode="ALL", which refines the whole model rather than just the given
    # selection -- that was the root cause of the ligand drifting apart,
    # compounded by the ligand having no CIF dictionary at the time. Both
    # issues are now addressed: scope is restricted to SINGLE, and the CIF
    # dictionary is mandatory so LIG has real bond/angle restraints holding
    # it together even while it sits unrefined in the same structure.
    suppress_output(mc.refine_residues_using_atom_cid, imol, selection_cid, "SINGLE", n_cycles)

    return imol


def main():
    p = build_argparser()
    args = p.parse_args()

    multimodel_pdbfile = pdb.PDBFile.read(str(args.input_multimodel_pdb))
    n_models = pdb.get_model_count(multimodel_pdbfile)
    print(f"Found {n_models} model(s) in {args.input_multimodel_pdb}")

    # Resolve per-model CIF list: either the same file repeated for every
    # model, or an explicit ordered list (one CIF per model, e.g. for
    # datasets with multiple stereoisomers across models).
    if args.cif_restraints is not None:
        cif_per_model = [args.cif_restraints] * n_models
    else:
        cif_per_model = [Path(c.strip()) for c in args.cif_list.split(",")]
        if len(cif_per_model) != n_models:
            raise ValueError(
                f"--cif-list has {len(cif_per_model)} entries but "
                f"{args.input_multimodel_pdb} has {n_models} model(s)"
            )

    missing_cifs = [str(c) for c in cif_per_model if not c.exists()]
    if missing_cifs:
        raise FileNotFoundError(f"CIF restraints file(s) not found: {missing_cifs}")

    apo_pdbfile = pdb.PDBFile.read(str(args.input_apo_pdb))
    apo_array_template = pdb.get_structure(apo_pdbfile, model=1, extra_fields=EXTRA_FIELDS)
    print(f"Loaded apo structure from {args.input_apo_pdb} ({len(apo_array_template)} atoms)")

    # Every residue in the apo structure, refined or not -- the CSV reports
    # a row for each of these per model, regardless of whether that residue
    # fell within --cutoff of the LIG (and was therefore actually refined).
    all_apo_residues = sorted({(a.chain_id, a.res_id) for a in apo_array_template})

    csv_path = build_csv_output_path(args.output_pdb)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(
        ["model_index", "chain_id", "residue_number", "max_displacement_A", "within_cutoff", "moved"]
    )

    # Model indices skipped because no apo protein residue fell within
    # --cutoff of that model's LIG. Tracked so we can print a summary at the
    # end; output files simply aren't written for these indices, which is
    # sufficient on its own to keep output indices in 1:1 correspondence
    # with each model's row position in the upstream cluster_reps.csv (we
    # never renumber remaining models to close the gap).
    skipped_model_indices = []

    # Non-redundant union of residues selected for refinement (i.e. within
    # --cutoff of the LIG) across every processed model, written out at the
    # end as refined_residues.csv.
    all_selected_residues = set()

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        for model_idx in range(1, n_models + 1):
            print(f"\nProcessing model {model_idx}/{n_models}...")

            cif_path = cif_per_model[model_idx - 1]
            print(f"  Using CIF restraints: {cif_path}")

            model_array = pdb.get_structure(multimodel_pdbfile, model=model_idx, extra_fields=EXTRA_FIELDS)
            lig_atoms = get_lig_residue(model_array, model_idx)
            print(f"  Found LIG residue: chain {lig_atoms.chain_id[0]}, resid {lig_atoms.res_id[0]}, "
                  f"{len(lig_atoms)} atoms")

            protein_residues = get_protein_residues_within_cutoff(
                apo_array_template, lig_atoms.coord, args.cutoff
            )

            if not protein_residues:
                print(f"  WARNING: No protein residues found within {args.cutoff} A of LIG "
                      f"in model {model_idx}; skipping this cluster representative. No "
                      f"output PDB or residue-table rows will be written for model "
                      f"{model_idx} (remaining model indices are unaffected).")
                skipped_model_indices.append(model_idx)
                continue

            selected_set = set(protein_residues)
            all_selected_residues.update(protein_residues)
            selection_cid = residues_to_cids(protein_residues)
            print(f"  {len(protein_residues)} protein residue(s) within {args.cutoff} A of LIG "
                  f"(out of {len(all_apo_residues)} total apo residues)")

            # Plain list of the residues selected for refinement (e.g. "A101"),
            # one per line, since these are the ones expected to move.
            expected_moved_path = build_expected_moved_csv_path(args.output_pdb, model_idx)
            expected_moved_path.parent.mkdir(parents=True, exist_ok=True)
            with open(expected_moved_path, "w", newline="") as expected_moved_file:
                expected_moved_writer = csv.writer(expected_moved_file)
                for chain_id, res_id in protein_residues:
                    expected_moved_writer.writerow([f"{chain_id}{res_id}"])
            print(f"  Wrote expected-to-move residue list: {expected_moved_path}")

            merged_array = struc.concatenate([apo_array_template.copy(), lig_atoms])

            tmp_in = tmpdir / f"model_{model_idx}_merged.pdb"
            tmp_out = tmpdir / f"model_{model_idx}_refined.pdb"

            tmp_pdb = pdb.PDBFile()
            pdb.set_structure(tmp_pdb, merged_array)
            tmp_pdb.write(str(tmp_in))

            mc = coot_headless_api.molecules_container_t(False)
            try:
                imol = refine_model(
                    mc, tmp_in, args.map, cif_path, selection_cid,
                    args.n_cycles, args.map_weight, args.difference_map
                )
            except NoNearbyProteinResiduesError as e:
                # Defense in depth: the empty-selection check above should
                # already have caught this, but if this is ever reached via
                # some other path, treat it the same way -- skip this model
                # rather than aborting the whole run.
                print(f"  WARNING: {e}; skipping this cluster representative. No "
                      f"output PDB or residue-table rows will be written for model "
                      f"{model_idx} (remaining model indices are unaffected).")
                skipped_model_indices.append(model_idx)
                continue

            suppress_output(mc.write_coordinates, imol, str(tmp_out))
            print(f"  Refined {len(protein_residues)} residue(s)")

            # Re-load the refined (protein + LIG) structure and compare
            # pre- vs post-refinement coordinates for every residue in the
            # apo structure -- not just the ones that were refined -- so
            # the CSV can report a displacement for every residue and we
            # can confirm refinement moved only the intended selection.
            refined_array = pdb.get_structure(
                pdb.PDBFile.read(str(tmp_out)), model=1, extra_fields=EXTRA_FIELDS
            )

            displacements = compute_residue_displacements(
                apo_array_template, refined_array, all_apo_residues
            )

            moved_count = sum(
                1 for key, (max_disp, n_compared, n_missing) in displacements.items()
                if key in selected_set and max_disp > args.moved_threshold
            )
            print(f"  {moved_count}/{len(protein_residues)} selected residue(s) moved "
                  f"by more than {args.moved_threshold} A")

            any_missing = [key for key, (_, _, n_missing) in displacements.items() if n_missing > 0]
            if any_missing:
                print(f"  WARNING: {len(any_missing)} residue(s) had atoms missing from the "
                      f"pre/post comparison (e.g. {any_missing[:3]})")

            for chain_id, res_id in all_apo_residues:
                max_disp, n_compared, n_missing = displacements[(chain_id, res_id)]
                moved = max_disp > args.moved_threshold
                within_cutoff = (chain_id, res_id) in selected_set
                csv_writer.writerow(
                    [model_idx, chain_id, res_id, f"{max_disp:.4f}", within_cutoff, moved]
                )

            # Sanity check: confirm refinement scope didn't leak beyond
            # selection_cid, using the same displacements already computed
            # above for every apo residue. This is the direct empirical
            # guarantee that mode="SINGLE" refined exactly the intended
            # (within-cutoff) residues and nothing else.
            leaked = [
                (key, max_disp) for key, (max_disp, n_compared, n_missing) in displacements.items()
                if key not in selected_set and max_disp > args.moved_threshold
            ]

            if leaked:
                print(f"  WARNING: {len(leaked)} residue(s) OUTSIDE the {len(protein_residues)}-residue "
                      f"selection moved by more than {args.moved_threshold} A -- refinement scope "
                      f"leaked beyond selection_cid:")
                for (chain_id, res_id), max_disp in leaked:
                    print(f"    {chain_id} {res_id}: {max_disp:.4f} A")
            else:
                print(f"  Confirmed: no residues outside the {len(protein_residues)}-residue "
                      f"selection moved.")

            # Sanity check: the LIG residue was NOT in selection_cid, so it
            # should not have moved at all. Flag it loudly if it did, since
            # that would indicate refinement scope leaked beyond the
            # intended selection (e.g. a mode/CIF-dictionary issue).
            lig_key = (lig_atoms.chain_id[0], lig_atoms.res_id[0])
            lig_displacement = compute_residue_displacements(
                merged_array, refined_array, [lig_key]
            )
            lig_max_disp, lig_n_compared, lig_n_missing = lig_displacement[lig_key]
            if lig_max_disp > args.moved_threshold:
                print(f"  NOTE: LIG residue {lig_key} moved by {lig_max_disp:.4f} A "
                      f"during refinement (it was not in the refined selection).")
            else:
                print(f"  Confirmed: LIG residue {lig_key} did not move "
                      f"(max displacement {lig_max_disp:.4f} A).")
            if lig_n_missing > 0:
                print(f"  WARNING: {lig_n_missing} LIG atom(s) missing from pre/post comparison.")

            out_path = build_output_path(args.output_pdb, model_idx)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            final_pdb = pdb.PDBFile()
            pdb.set_structure(final_pdb, refined_array)
            final_pdb.write(str(out_path))

            print(f"  Wrote {out_path}")

    csv_file.close()
    print(f"\nWrote refined-residue table to {csv_path}")

    combined_path = build_combined_expected_moved_csv_path(args.output_pdb)
    combined_path.parent.mkdir(parents=True, exist_ok=True)
    with open(combined_path, "w", newline="") as combined_file:
        combined_writer = csv.writer(combined_file)
        for chain_id, res_id in sorted(all_selected_residues):
            combined_writer.writerow([f"{chain_id}{res_id}"])
    print(f"Wrote non-redundant list of {len(all_selected_residues)} residue(s) selected across "
          f"all models to {combined_path}")

    if skipped_model_indices:
        print(f"\nSkipped {len(skipped_model_indices)}/{n_models} model(s) with no protein "
              f"residue within {args.cutoff} A of LIG (no output written for these; output "
              f"indices for the remaining models are unchanged, e.g. model 2 skipped out of "
              f"4 still yields output files 1, 3, 4): {skipped_model_indices}")
    else:
        print(f"\nAll {n_models} model(s) processed successfully; none skipped.")

    print("Done.")


if __name__ == "__main__":
    main()