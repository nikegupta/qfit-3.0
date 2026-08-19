import argparse
import glob
from pathlib import Path
import time
import numpy as np
import os
import sys

from qfit import Structure
from qfit import XMap
from qfit.xtal.transformer import get_transformer

import iotbx.pdb

CLASH_VDW_SCALE = 0.75  # fraction of summed VDW radii below which two atoms are considered clashing


class _Tee:
    """
    Minimal write-to-multiple-streams helper. Assigning sys.stdout to a _Tee lets
    every existing print() call in this module keep printing to the console as
    normal while also mirroring the same output to a log file, without having
    to touch each individual print() call.
    """
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()


def build_argparser():
    p = argparse.ArgumentParser()
    p.add_argument(
        'dataset',
        type=Path,
        help='Path to pandas dataset')
    p.add_argument(
        'placer_files',
        type=str,
        help='Glob pattern for all placer files'
    )
    p.add_argument(
        'multimodel_pdb',
        type=Path,
        help='Path to a multimodel pdb containing one or more ligand + binding-site '
             'conformations (e.g. cluster_rep_models.pdb as output by filter_all.py)'
    )
    p.add_argument(
        'apo_structure',
        type=Path,
        help='Path to the apo (ligand-free) PANDDA structure. Used as the fallback '
             'conformation for a residue when no PLACER conformer is '
             'found for it, or when every PLACER conformer of it clashes with a '
             'ligand pose in the multimodel pdb.'
    )
    p.add_argument(
        'output_folder',
        type=str,
        help='name of the output folder.'
    )
    p.add_argument(
        "-r",
        "--resolution",
        default=None,
        metavar="<float>",
        type=float,
        help="Map resolution (Å) (only use when providing CCP4 map files)",
    )
    p.add_argument(
        "--clash_vdw_scale",
        default=CLASH_VDW_SCALE,
        metavar="<float>",
        type=float,
        help="Fraction of the summed VDW radii of two atoms below which they are "
             f"considered clashing (default: {CLASH_VDW_SCALE})",
    )
    return p


class FinalModelBuilder():
    def __init__(self, dataset_dir, placer_files, multimodel_pdb, apo_structure, output_folder,
                 resolution, clash_vdw_scale=CLASH_VDW_SCALE):
        self.dir = dataset_dir
        self.placer_files = placer_files
        self.multimodel_pdb = multimodel_pdb
        self.apo_structure = apo_structure
        self.output_folder = output_folder
        self.resolution = resolution
        self.clash_vdw_scale = clash_vdw_scale

        self._rmask = 0.5 + self.resolution / 3.0 #from qfit

        self._load_event_maps()
        self._load_apo_structure()

        # print(self.__dict__)

    def _load_event_maps(self):
        self.event_maps = {}
        self.event_maps_models = {}
        event_map_files = sorted(self.dir.glob('*-event_*_*-BDC_*_map.native.ccp4'))
        for event_file in event_map_files:
            # Use full filename as key
            event_name = str(event_file).split('/')[-1]  # e.g., "x01325-1-event_1_1-BDC_0.3_map.native.ccp4"
            self.event_maps[event_name] = XMap.fromfile(str(event_file), resolution=self.resolution)

            # make copies for density steps
            event_map_model = self.event_maps[event_name].zeros_like(self.event_maps[event_name])
            event_map_model.set_space_group("P1")
            self.event_maps_models[event_name] = event_map_model

    def _load_apo_structure(self):
        self.apo_model = Structure.fromfile(str(self.apo_structure))

    def _cluster_reps_csv_path(self):
        """cluster_reps.csv sits beside self.multimodel_pdb (cluster_rep_models.pdb) -
        both are written by the same filter/filter2 run, from the same
        cluster_reps dict, so a cluster_rep_models.pdb with N models always
        has a cluster_reps.csv with N data rows, and vice versa."""
        return Path(self.multimodel_pdb).parent / 'cluster_reps.csv'

    def _countClusterReps(self):
        """Returns the number of data rows (i.e. accepted cluster reps) in
        cluster_reps.csv, or 0 if that csv is missing, empty, or header-only -
        which happens when filter/filter2 rejected every candidate for this
        dataset (e.g. every cluster failed the count/rscc/clash cutoffs)."""
        csv_path = self._cluster_reps_csv_path()
        if not csv_path.exists():
            return 0
        with open(csv_path) as f:
            lines = [line for line in f if line.strip()]
        return max(len(lines) - 1, 0)

    def run(self):
        """Rescores the protein binding-site residues around the ligand(s) in a
        multimodel pdb (e.g. filter_all.py's cluster_rep_models.pdb), pooling
        every conformation of each residue across all input placer models, and
        writes out a single merged structure - the best-scoring, non-clashing
        conformation of each residue (falling back to the apo conformation when
        needed), plus every ligand pose from the multimodel pdb - to
        output_folder/final_model.pdb.

        Writes nothing (returns early, no final_model.pdb) if cluster_reps.csv
        (beside multimodel_pdb) has no accepted cluster reps - i.e. filter/
        filter2 rejected every candidate for this dataset. Without that check,
        a dataset in this state would still get a final_model.pdb built from
        placer2's protein-only conformers with zero ligand poses in it, which
        looks superficially complete but can never pass refinement.

        All print() output is mirrored to output_folder/log.txt in addition to
        the console.
        """
        output_folder = str(self.dir) + '/' + self.output_folder
        os.makedirs(output_folder, exist_ok=True)
        log_path = output_folder + '/log.txt'
        log_file = open(log_path, 'w')
        original_stdout = sys.stdout
        sys.stdout = _Tee(original_stdout, log_file)

        try:
            n_cluster_reps = self._countClusterReps()
            print(f'{n_cluster_reps} cluster rep row(s) found in {self._cluster_reps_csv_path()}')
            if n_cluster_reps == 0:
                print('No cluster reps found (filter/filter2 rejected every candidate for this '
                      'dataset) - there is no ligand pose to build a final model around, so '
                      'final_model.pdb is not being written.')
                return

            print(self.multimodel_pdb)
            self.multimodel_models = Structure.fromfile(str(self.multimodel_pdb)).split_models()
            print(f'{len(self.multimodel_models)} model(s) in multimodel pdb')

            #pool every ligand pose's atoms across every model of the multimodel pdb;
            #every one of these poses ends up in the final model together, so a
            #candidate protein residue conformation has to be clash-checked against
            #all of them, not just the pose from its own model
            self._gatherLigandAtoms()

            #find every protein residue in the apo structure. A residue only
            #actually goes through scoring/clash-checking below if PLACER
            #produced at least one conformer of it (see _gatherResidueConformers
            #/ _scoreAndSelectBest); residues with no PLACER conformer are
            #included here too so they end up in the final model, taken
            #directly from the apo structure.
            time0 = time.time()
            self.all_residues = self._determineAllResidues()
            n_residues = sum(len(res_nums) for res_nums in self.all_residues.values())
            print(f'found {n_residues} residue(s) in the apo structure in {time.time() - time0:.2f}s')

            #resolve placer files
            placer_files = sorted(glob.glob(self.placer_files))
            print(f'found {len(placer_files)} placer file(s)')
            if not placer_files:
                print('No placer files found; nothing to rescore.')
                return

            #gather every conformation of each residue across every model of
            #every placer file
            time0 = time.time()
            self.residue_templates, self.residue_conformers = self._gatherResidueConformers(placer_files)
            print(f'gathered residue conformers in {time.time() - time0:.2f}s')

            #flag (not an error - just something to monitor) any residue
            #that wasn't found in ANY placer file at all. A residue
            #missing from *some* placer files is expected and fine; we only
            #need conformations from the ones that do have it. These residues
            #fall back to their apo conformation (see _scoreAndSelectBest).
            missing_residues = [key for key, conformers in self.residue_conformers.items()
                                 if not conformers]
            if missing_residues:
                missing_str = ', '.join(f'{chain_id}{res_num}' for chain_id, res_num in missing_residues)
                print(f'FLAG: {len(missing_residues)} residue(s) had no conformers in '
                      f'any placer file (not a dealbreaker, just flagging for awareness): {missing_str}')

            #score every conformer of every residue (pooled mask per residue, max
            #across event maps), reject any that clash with a ligand pose from the
            #multimodel pdb, and keep the single best-scoring non-clashing conformer
            #per residue - falling back to the apo conformation if none qualify
            time0 = time.time()
            self.best_conformers = self._scoreAndSelectBest(output_folder)
            print(f'scored and selected best conformers in {time.time() - time0:.2f}s')

            #merge the best protein conformations with every ligand pose from the
            #multimodel pdb into a single output structure
            final_model = self._buildFinalModel()
            final_model_path = output_folder + '/final_model.pdb'
            self._write_pdb(final_model, final_model_path)
            print(f'final model written to {final_model_path}')
        finally:
            sys.stdout = original_stdout
            log_file.close()

    def _get_atom_records(self, model):
        """Returns a list of (chain_id, res_num, resname, xyz) for every atom in
        a Structure model, read directly from its iotbx hierarchy."""
        records = []
        for chain in model._pdb_hierarchy.only_model().chains():
            chain_id = chain.id.strip()
            for residue in chain.residue_groups():
                res_num = int(residue.resseq)
                resname = residue.only_atom_group().resname.strip()
                for atom_group in residue.atom_groups():
                    for atom in atom_group.atoms():
                        records.append((chain_id, res_num, resname, np.array(atom.xyz)))
        return records

    def _determineAllResidues(self):
        """Returns {chain_id: sorted [res_nums]} for every protein residue found
        in the apo structure. This is the full set of residues that need to end
        up in the final model - either from a PLACER conformer (if one was
        found for it) or, if not, taken directly from the apo structure (see
        _scoreAndSelectBest).
        """
        residues = {}
        for chain_id, res_num, resname, _ in self._get_atom_records(self.apo_model):
            residues.setdefault(chain_id, set()).add(res_num)

        return {chain_id: sorted(res_nums) for chain_id, res_nums in residues.items()}

    def _gatherLigandAtoms(self):
        """Pools the coordinates and VDW radii of every ligand atom, across every
        model of the multimodel pdb, into flat arrays used for clash-checking
        candidate protein residue conformations (see _clashesWithLigands). Every
        one of these ligand poses is kept in the final model (see
        _buildFinalModel), so a candidate has to be checked against all of them.
        """
        coords = []
        radii = []
        for model in self.multimodel_models:
            ligand = model.extract('resname LIG')
            if ligand.natoms == 0:
                continue
            coords.append(np.asarray(ligand.coor))
            radii.append(np.asarray(ligand.vdw_radius))

        if coords:
            self.ligand_coords = np.concatenate(coords, axis=0)
            self.ligand_vdw_radii = np.concatenate(radii, axis=0)
        else:
            self.ligand_coords = np.zeros((0, 3))
            self.ligand_vdw_radii = np.zeros((0,))

    def _clashesWithLigands(self, template, coor):
        """Returns True if placing `template`'s atoms at coordinates `coor` would
        put any atom within self.clash_vdw_scale * (sum of the two atoms' VDW
        radii) of any ligand atom pooled from the multimodel pdb (see
        _gatherLigandAtoms).
        """
        if self.ligand_coords.shape[0] == 0:
            return False

        residue_vdw = np.asarray(template.vdw_radius)
        diff = coor[:, None, :] - self.ligand_coords[None, :, :]
        dists = np.linalg.norm(diff, axis=-1)
        radii_sum = residue_vdw[:, None] + self.ligand_vdw_radii[None, :]
        clash_threshold = self.clash_vdw_scale * radii_sum

        return bool(np.any(dists < clash_threshold))

    def _get_apo_residue(self, chain_id, res_num):
        """Extracts (chain_id, res_num) from the apo structure. Returns
        (coor, structure) or (None, None) if the apo structure doesn't have
        that residue."""
        residue = self.apo_model.extract(f'chain {chain_id} and resid {res_num}')
        if residue.natoms == 0:
            return None, None
        return residue.coor, residue

    def _gatherResidueConformers(self, placer_files):
        """For every residue in self.all_residues, gathers every
        conformation of that residue found across every model of every input
        placer file. A residue absent from a given placer model is simply
        skipped for that model (no fallback structure is used here).

        Returns:
          residue_templates  : {(chain_id, res_num): Structure} - the apo
                                structure's own copy of that residue. The apo
                                structure is guaranteed to have every residue
                                in self.all_residues (that's where it was
                                enumerated from), and its atom identity/count
                                will always match every PLACER conformer, so
                                it's used as the single canonical template for
                                scoring, clash-checking, and building the
                                final model.
          residue_conformers : {(chain_id, res_num): [(coor, placer_file, model_idx), ...]}
        """
        residue_conformers = {}
        residue_templates = {}

        for chain_id, res_nums in self.all_residues.items():
            for res_num in res_nums:
                residue_conformers[(chain_id, res_num)] = []

                _, apo_template = self._get_apo_residue(chain_id, res_num)
                if apo_template is None:
                    print(f'WARNING: {chain_id}{res_num} was found while enumerating apo '
                          f'residues but could not be re-extracted from the apo structure; '
                          f'this should not happen')
                residue_templates[(chain_id, res_num)] = apo_template

        for placer_file in placer_files:
            print(placer_file)
            models = Structure.fromfile(placer_file).split_models()

            for model_idx, model in enumerate(models):
                for (chain_id, res_num) in residue_conformers:
                    residue = model.extract(f'chain {chain_id} and resid {res_num}')
                    if residue.natoms == 0:
                        continue

                    residue_conformers[(chain_id, res_num)].append(
                        (residue.coor, placer_file, model_idx)
                    )

        return residue_templates, residue_conformers

    def _scoreResidueConformers(self, template, coor_list):
        """Scores every conformer coordinate set of one protein residue against
        every event map, pooling all of that residue's conformers together into
        a single mask per event map (rather than masking each conformer
        separately). Returns one score per conformer: the max RSCC across all
        event maps.
        """
        scaled_bulk_solvent = 0 #from qfit, maybe should be different
        default_bfactor = 20 #can change
        n_conf = len(coor_list)

        per_conformer_scores = [[] for _ in range(n_conf)]

        for event_map_name in list(self.event_maps.keys()):
            #make a transformer for this residue
            transformer = get_transformer("qfit", template, self.event_maps_models[event_map_name])

            #pooled mask covering every conformer of this residue together
            mask = transformer.get_conformers_mask(coor_list, self._rmask)
            target = self.event_maps[event_map_name].array[mask]

            for i, density in enumerate(transformer.get_conformers_densities(coor_list, [default_bfactor] * n_conf)):
                model = density[mask]
                np.maximum(model, scaled_bulk_solvent, out=model)
                correlation_matrix = np.corrcoef(model, target)
                rscc = correlation_matrix[0, 1]
                per_conformer_scores[i].append(rscc)

        return [max(scores) for scores in per_conformer_scores]

    def _scoreAndSelectBest(self, output_folder):
        """For every residue: scores every gathered conformer (see
        _scoreResidueConformers), and - in descending score order - keeps the
        first conformer that doesn't clash with any ligand pose from the
        multimodel pdb (see _clashesWithLigands). If every conformer clashes,
        or no conformer was ever found for the residue, falls back to the apo
        structure's conformation of that residue and prints a message saying so.

        Also writes two CSVs to output_folder:
          - residue_scores.csv: unchanged, one row per residue that had at
            least one PLACER conformer and had a conformer selected from among
            them (i.e. excludes apo fallbacks).
          - residues_with_placer_conformers.csv (new, additional - does not
            replace residue_scores.csv): a plain list of every residue
            ("{chain}{resnum}", one per line, no header) that had at least one
            PLACER conformer, regardless of whether that conformer ended up
            clashing and falling back to apo.

        Returns: {(chain_id, res_num): (best_coor, best_rscc, template)}
        best_rscc is None for residues that fell back to the apo conformation.
        """
        best_conformers = {}
        summary_rows = []
        residues_with_conformers = []

        for (chain_id, res_num), conformers in self.residue_conformers.items():
            template = self.residue_templates[(chain_id, res_num)]

            if template is None:
                # already warned about in _gatherResidueConformers; nothing to
                # score, clash-check, or fall back to for this residue
                print(f'WARNING: {chain_id}{res_num} has no template (apo extraction '
                      f'failed); omitting it from the final model')
                continue

            if not conformers:
                # already flagged in run(); fall back to the apo conformation.
                # template is itself the apo structure's residue, so its own
                # (unmodified) coordinates ARE the apo conformation.
                best_conformers[(chain_id, res_num)] = (template.coor, None, template)
                continue

            residues_with_conformers.append((chain_id, res_num))

            coor_list = [c[0] for c in conformers]
            scores = self._scoreResidueConformers(template, coor_list)

            # try conformers best-scoring first, skipping any that clash with a
            # ligand pose from the multimodel pdb
            order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
            chosen_idx = None
            for idx in order:
                if not self._clashesWithLigands(template, coor_list[idx]):
                    chosen_idx = idx
                    break

            if chosen_idx is not None:
                best_coor, best_placer_file, best_model_idx = conformers[chosen_idx]
                best_rscc = scores[chosen_idx]

                best_conformers[(chain_id, res_num)] = (best_coor, best_rscc, template)
                summary_rows.append((chain_id, res_num, len(conformers), best_rscc,
                                      best_placer_file, best_model_idx))

                print(f'{chain_id}{res_num}: best rscc {best_rscc:.4f} from {best_placer_file} '
                      f'model {best_model_idx} (of {len(conformers)} conformer(s))')
            else:
                print(f'{chain_id}{res_num}: all {len(conformers)} conformer(s) clashed with a '
                      f'ligand pose in the multimodel pdb; a non-clashing conformer could not be '
                      f'found, falling back to apo conformation')
                best_conformers[(chain_id, res_num)] = (template.coor, None, template)

        self._write_residue_scores_csv(summary_rows, output_folder + '/residue_scores.csv')
        self._write_residue_conformer_list_csv(
            residues_with_conformers, output_folder + '/residues_with_placer_conformers.csv'
        )

        return best_conformers

    def _write_residue_scores_csv(self, rows, path):
        with open(path, 'w+') as f:
            f.write('chain,resid,num_conformers,best_rscc,best_placer_file,best_model_idx')
            f.write('\n')
            for chain_id, res_num, num_conformers, best_rscc, best_placer_file, best_model_idx in rows:
                f.write(f'{chain_id},{res_num},{num_conformers},{best_rscc},'
                        f'{best_placer_file},{best_model_idx}')
                f.write('\n')

    def _write_residue_conformer_list_csv(self, residues, path):
        """Writes a plain, headerless list of "{chain}{resnum}" (e.g. "A101"),
        one per line, for every residue that had at least one PLACER conformer.
        This is additional to residue_scores.csv, not a replacement for it.
        """
        with open(path, 'w+') as f:
            for chain_id, res_num in sorted(residues):
                f.write(f'{chain_id}{res_num}')
                f.write('\n')

    def _set_resi(self, structure, resi):
        """Sets the residue number of every atom in `structure` to `resi`.

        Structure.resi is a derived, read-only property (computed from
        atom.parent().parent().resseq_as_int()) with no setter, so the
        residue number has to be changed at the source: the `resseq` field
        on each atom's residue_group in the underlying iotbx hierarchy.
        iotbx.pdb.resseq_encode() handles the standard 4-character
        right-justified formatting (and hybrid-36 encoding, if resi ever
        exceeds 9999).
        """
        resseq = iotbx.pdb.resseq_encode(resi)
        seen = set()
        for atom in structure.atoms:
            residue_group = atom.parent().parent()
            if id(residue_group) in seen:
                continue
            residue_group.resseq = resseq
            seen.add(id(residue_group))

    def _buildFinalModel(self):
        """Merges the best-scoring (or apo-fallback) conformation of every
        residue with every ligand pose found in the multimodel pdb into a
        single Structure, combined in chain/residue-number order so
        final_model.pdb reads out sorted rather than in whatever order
        residues and ligands happened to be processed in."""
        pieces = []

        #best-scoring (or apo-fallback) protein residue conformations
        for (chain_id, res_num), (best_coor, best_rscc, template) in self.best_conformers.items():
            residue_model = template.copy()
            residue_model.coor = best_coor
            residue_model.b = 20

            pieces.append((chain_id, res_num, residue_model))

        #every ligand pose from the multimodel pdb, kept in its original chain
        #but renumbered so its residue number equals the (1-indexed) model
        #number/position it came from in the multimodel pdb - preserving a
        #strict correspondence between each ligand in final_model.pdb and the
        #MODEL record it was pulled from in cluster_rep_models.pdb
        for model_number, model in enumerate(self.multimodel_models, start=1):
            ligand = model.extract('resname LIG')
            if ligand.natoms == 0:
                continue

            ligand = ligand.copy()
            self._set_resi(ligand, model_number)

            ligand_chain_id = self._get_atom_records(ligand)[0][0]
            pieces.append((ligand_chain_id, model_number, ligand))

        #sort by (chain_id, res_num) and combine in that order
        pieces.sort(key=lambda piece: (piece[0], piece[1]))

        final_model = None
        for _, _, structure in pieces:
            final_model = structure if final_model is None else final_model.combine(structure)

        return final_model

    def _write_pdb(self, model, output_path):
        with open(output_path, 'w') as out:
            for atom in model.get_selected_atoms():
                atom_labels = atom.fetch_labels()
                out.write("{}\n".format(atom_labels.format_atom_record_group()))
            out.write("END\n")


def main():
    p = build_argparser()
    args = p.parse_args()
    builder = FinalModelBuilder(args.dataset, args.placer_files, args.multimodel_pdb,
                                 args.apo_structure, args.output_folder, args.resolution,
                                 args.clash_vdw_scale)
    builder.run()


if __name__ == '__main__':
    main()