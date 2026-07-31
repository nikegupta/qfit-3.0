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

DISTANCE_CUTOFF = 10.0  # Å, from any ligand atom in the multimodel pdb


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
        "--distance_cutoff",
        default=DISTANCE_CUTOFF,
        metavar="<float>",
        type=float,
        help="Distance (Å) from any ligand atom in the multimodel pdb within which a "
             f"protein residue is included for re-scoring (default: {DISTANCE_CUTOFF})",
    )
    return p


class FinalModelBuilder():
    def __init__(self, dataset_dir, placer_files, multimodel_pdb, output_folder, resolution,
                 distance_cutoff=DISTANCE_CUTOFF):
        self.dir = dataset_dir
        self.placer_files = placer_files
        self.multimodel_pdb = multimodel_pdb
        self.output_folder = output_folder
        self.resolution = resolution
        self.distance_cutoff = distance_cutoff

        self._rmask = 0.5 + self.resolution / 3.0 #from qfit

        self._load_event_maps()

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

    def run(self):
        """Rescores the protein binding-site residues around the ligand(s) in a
        multimodel pdb (e.g. filter_all.py's cluster_rep_models.pdb), pooling
        every conformation of each residue across all input placer models, and
        writes out a single merged structure - the best-scoring conformation of
        each residue, plus every ligand pose from the multimodel pdb - to
        output_folder/final_model.pdb.

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
            print(self.multimodel_pdb)
            self.multimodel_models = Structure.fromfile(str(self.multimodel_pdb)).split_models()
            print(f'{len(self.multimodel_models)} model(s) in multimodel pdb')

            #find every protein residue within distance_cutoff of any ligand atom,
            #pooled across every model in the multimodel pdb
            time0 = time.time()
            self.binding_site_residues = self._determineBindingSiteByDistance(self.multimodel_models)
            n_residues = sum(len(res_nums) for res_nums in self.binding_site_residues.values())
            print(f'found {n_residues} binding site residue(s) within {self.distance_cutoff} '
                  f'\u00c5 of any ligand pose in {time.time() - time0:.2f}s')

            #resolve placer files
            placer_files = sorted(glob.glob(self.placer_files))
            print(f'found {len(placer_files)} placer file(s)')
            if not placer_files:
                print('No placer files found; nothing to rescore.')
                return

            #gather every conformation of each binding site residue across every
            #model of every placer file
            time0 = time.time()
            self.residue_templates, self.residue_conformers = self._gatherResidueConformers(placer_files)
            print(f'gathered residue conformers in {time.time() - time0:.2f}s')

            #flag (not an error - just something to monitor) any binding site
            #residue that wasn't found in ANY placer file at all. A residue
            #missing from *some* placer files is expected and fine; we only
            #need conformations from the ones that do have it.
            missing_residues = [key for key, conformers in self.residue_conformers.items()
                                 if not conformers]
            if missing_residues:
                missing_str = ', '.join(f'{chain_id}{res_num}' for chain_id, res_num in missing_residues)
                print(f'FLAG: {len(missing_residues)} binding site residue(s) had no conformers in '
                      f'any placer file (not a dealbreaker, just flagging for awareness): {missing_str}')

            #score every conformer of every residue (pooled mask per residue, max
            #across event maps) and keep the single best-scoring conformer per residue
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

    def _determineBindingSiteByDistance(self, models):
        """Finds every protein residue (chain_id, res_num) with at least one atom
        within self.distance_cutoff \u00c5 of any LIG atom, pooled across every
        model in the multimodel pdb - so a residue near the ligand in *any*
        cluster rep is included, even if it isn't near the ligand in every rep.
        """
        residues_in_binding_site = {}

        for model in models:
            records = self._get_atom_records(model)
            ligand_coors = np.array([xyz for (_, _, resname, xyz) in records if resname == 'LIG'])
            if ligand_coors.shape[0] == 0:
                continue

            for chain_id, res_num, resname, xyz in records:
                if resname == 'LIG':
                    continue

                dists = np.linalg.norm(ligand_coors - xyz, axis=1)
                if np.any(dists < self.distance_cutoff):
                    residues_in_binding_site.setdefault(chain_id, set()).add(res_num)

        return {chain_id: sorted(res_nums) for chain_id, res_nums in residues_in_binding_site.items()}

    def _gatherResidueConformers(self, placer_files):
        """For every residue in self.binding_site_residues, gathers every
        conformation of that residue found across every model of every input
        placer file. A residue absent from a given placer model is simply
        skipped for that model (no fallback structure is used here).

        Returns:
          residue_templates  : {(chain_id, res_num): Structure} - the first
                                occurrence of that residue found (in the
                                multimodel pdb, falling back to the first
                                placer conformer), used as the atom-identity
                                template when scoring and when building the
                                final model.
          residue_conformers : {(chain_id, res_num): [(coor, placer_file, model_idx), ...]}
        """
        residue_conformers = {}
        residue_templates = {}

        for chain_id, res_nums in self.binding_site_residues.items():
            for res_num in res_nums:
                residue_conformers[(chain_id, res_num)] = []

                #prefer the multimodel pdb's own copy of this residue as the template
                template = None
                for model in self.multimodel_models:
                    candidate = model.extract(f'chain {chain_id} and resid {res_num}')
                    if candidate.natoms > 0:
                        template = candidate
                        break
                residue_templates[(chain_id, res_num)] = template

        for placer_file in placer_files:
            print(placer_file)
            models = Structure.fromfile(placer_file).split_models()

            for model_idx, model in enumerate(models):
                for (chain_id, res_num) in residue_conformers:
                    residue = model.extract(f'chain {chain_id} and resid {res_num}')
                    if residue.natoms == 0:
                        continue

                    #fall back to the first placer conformer as the template if
                    #this residue was somehow absent from the multimodel pdb itself
                    if residue_templates[(chain_id, res_num)] is None:
                        residue_templates[(chain_id, res_num)] = residue

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
        """For every binding site residue, scores every gathered conformer (see
        _scoreResidueConformers) and keeps the single best-scoring conformer.

        Returns: {(chain_id, res_num): (best_coor, best_rscc, template)}
        """
        best_conformers = {}
        summary_rows = []

        for (chain_id, res_num), conformers in self.residue_conformers.items():
            if not conformers:
                # already flagged in run(); nothing to score for this residue
                continue

            template = self.residue_templates[(chain_id, res_num)]
            coor_list = [c[0] for c in conformers]

            best_per_conformer = self._scoreResidueConformers(template, coor_list)

            best_idx = int(np.argmax(best_per_conformer))
            best_rscc = best_per_conformer[best_idx]
            best_coor, best_placer_file, best_model_idx = conformers[best_idx]

            best_conformers[(chain_id, res_num)] = (best_coor, best_rscc, template)
            summary_rows.append((chain_id, res_num, len(conformers), best_rscc,
                                  best_placer_file, best_model_idx))

            print(f'{chain_id}{res_num}: best rscc {best_rscc:.4f} from {best_placer_file} '
                  f'model {best_model_idx} (of {len(conformers)} conformer(s))')

        self._write_residue_scores_csv(summary_rows, output_folder + '/residue_scores.csv')

        return best_conformers

    def _write_residue_scores_csv(self, rows, path):
        with open(path, 'w+') as f:
            f.write('chain,resid,num_conformers,best_rscc,best_placer_file,best_model_idx')
            f.write('\n')
            for chain_id, res_num, num_conformers, best_rscc, best_placer_file, best_model_idx in rows:
                f.write(f'{chain_id},{res_num},{num_conformers},{best_rscc},'
                        f'{best_placer_file},{best_model_idx}')
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
        """Merges the best-scoring conformation of every binding site residue
        with every ligand pose found in the multimodel pdb into a single
        Structure."""
        final_model = None

        #best-scoring protein residue conformations
        for (chain_id, res_num), (best_coor, best_rscc, template) in self.best_conformers.items():
            residue_model = template.copy()
            residue_model.coor = best_coor
            residue_model.b = 20

            final_model = residue_model if final_model is None else final_model.combine(residue_model)

        #every ligand pose from the multimodel pdb, kept in its original chain
        #but renumbered so its residue number equals the (1-indexed) model
        #number it came from in the multimodel pdb - preserving a strict
        #correspondence between each ligand in final_model.pdb and the
        #MODEL record it was pulled from in cluster_rep_models.pdb
        for model_number, model in enumerate(self.multimodel_models, start=1):
            ligand = model.extract('resname LIG')
            if ligand.natoms == 0:
                continue

            ligand = ligand.copy()
            self._set_resi(ligand, model_number)

            final_model = ligand if final_model is None else final_model.combine(ligand)

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
                                 args.output_folder, args.resolution, args.distance_cutoff)
    builder.run()


if __name__ == '__main__':
    main()