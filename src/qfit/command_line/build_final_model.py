import argparse
import csv
import glob
from collections import namedtuple
from pathlib import Path
import time
import numpy as np
import os
import sys

from qfit import Structure
from qfit import XMap
from qfit.xtal.transformer import get_transformer

import iotbx.pdb

# fraction of summed VDW radii below which two atoms are considered clashing
# (used for sidechain-sidechain clash detection - see _resolveSidechainClashes)
CLASH_VDW_SCALE = 0.75

# looser scale applied instead of CLASH_VDW_SCALE when the clashing pair is
# one N atom and one O atom - a real N-H...O or O-H...N hydrogen bond
# legitimately sits closer than CLASH_VDW_SCALE would otherwise tolerate, so
# without this a lot of real donor/acceptor pairs get misreported as clashes
HBOND_CLASH_VDW_SCALE = 0.6

BACKBONE_ATOM_NAMES = {'N', 'CA', 'C', 'O', 'OXT'}

# safety caps for _resolveSidechainClashes - see its docstring
MAX_CLASH_GROUP_SIZE = 8  # residues; stop absorbing new neighbors past this
MAX_CLASH_GROUP_EXPANSIONS = 10  # rounds of "resolve, then absorb new external clashes"
CLASH_DOMAIN_TOP_K = 25  # candidates considered per residue during joint solving
CLASH_SOLVE_NODE_BUDGET = 200_000  # branch-and-bound search nodes before falling back to ICM

# per-residue candidate pool built by _scoreAndSelectBest and consumed by
# _resolveSidechainClashes. coor: (n_candidates, natoms, 3); mse: (n_candidates,)
# - both indexed identically to placer_file/model_idx (lists, length
# n_candidates). template/sidechain_mask describe the atoms (same for every
# candidate of this residue - see _gatherResidueConformers). fixed residues
# (no PLACER conformer found) get a single candidate: the apo coordinates.
_ResidueCandidates = namedtuple(
    'ResidueCandidates',
    ['coor', 'mse', 'placer_file', 'model_idx', 'template', 'sidechain_mask', 'fixed'],
)


class _NodeBudgetExceeded(Exception):
    """Raised internally by _branchAndBound to abort the search once its node
    budget is exhausted; caught by its caller to trigger the ICM fallback."""


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
             'conformation for a residue when no PLACER conformer is found for it.'
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
        help="Sidechain-sidechain clash detection: fraction of the summed VDW radii "
             "of two sidechain atoms (backbone atoms are never checked) below which "
             f"they are considered clashing (default: {CLASH_VDW_SCALE}). Does not apply "
             "to N/O pairs - see --hbond_clash_vdw_scale.",
    )
    p.add_argument(
        "--hbond_clash_vdw_scale",
        default=HBOND_CLASH_VDW_SCALE,
        metavar="<float>",
        type=float,
        help="Sidechain-sidechain clash detection: same as --clash_vdw_scale, but used "
             "instead of it whenever the pair is one N atom and one O atom, since a real "
             f"hydrogen bond legitimately sits closer than a generic clash (default: "
             f"{HBOND_CLASH_VDW_SCALE})",
    )
    p.add_argument(
        "--max_clash_group_size",
        default=MAX_CLASH_GROUP_SIZE,
        metavar="<int>",
        type=int,
        help="Sidechain-sidechain clash detection: a group of mutually-reselected "
             "clashing residues stops absorbing newly-clashing neighbors once it "
             "would exceed this many residues - the residual clash is logged and "
             f"left unresolved instead (default: {MAX_CLASH_GROUP_SIZE})",
    )
    return p


class FinalModelBuilder():
    def __init__(self, dataset_dir, placer_files, multimodel_pdb, apo_structure, output_folder,
                 resolution, clash_vdw_scale=CLASH_VDW_SCALE,
                 hbond_clash_vdw_scale=HBOND_CLASH_VDW_SCALE,
                 max_clash_group_size=MAX_CLASH_GROUP_SIZE):
        self.dir = dataset_dir
        self.placer_files = placer_files
        self.multimodel_pdb = multimodel_pdb
        self.apo_structure = apo_structure
        self.output_folder = output_folder
        self.resolution = resolution
        self.clash_vdw_scale = clash_vdw_scale
        self.hbond_clash_vdw_scale = hbond_clash_vdw_scale
        self.max_clash_group_size = max_clash_group_size

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

    def _acceptedPlacerFiles(self):
        """Returns the set of placer_file values from cluster_reps.csv's
        placer_file column - the same resolved path strings filter/filter2
        produced from its own copy of the placer_files glob pattern - that
        are the source of an accepted cluster rep, i.e. survived count/rscc/
        per-placer_file-dedup/clash filtering. Empty if cluster_reps.csv is
        missing, empty, or header-only (every candidate was rejected).

        Per-placer_file dedup upstream in filter/filter2 already guarantees
        at most one accepted cluster rep per placer_file, so this set's size
        always equals cluster_reps.csv's row count.
        """
        csv_path = self._cluster_reps_csv_path()
        if not csv_path.exists():
            return set()
        with open(csv_path, newline='') as f:
            return {row['placer_file'] for row in csv.DictReader(f)}

    def _countClusterReps(self):
        """Returns the number of accepted cluster reps in cluster_reps.csv
        (see _acceptedPlacerFiles), or 0 if that csv is missing, empty, or
        header-only - which happens when filter/filter2 rejected every
        candidate for this dataset (e.g. every cluster failed the count/rscc/
        clash cutoffs)."""
        return len(self._acceptedPlacerFiles())

    def run(self):
        """Rescores the protein binding-site residues around the ligand(s) in a
        multimodel pdb (e.g. filter_all.py's cluster_rep_models.pdb), pooling
        every conformation of each residue across all input placer models, and
        writes out a single merged structure - the best-scoring conformation of
        each residue (falling back to the apo conformation when needed), plus
        every ligand pose from the multimodel pdb - to output_folder/final_model.pdb.

        No clash checking is done against the ligand poses at all - a ligand
        pose that badly clashes with the surrounding protein gets reselected
        downstream (DESPOT), and a genuinely correct sidechain rotamer that
        happens to sit close to the true ligand density is expected to be far
        more common than the reverse, so filtering candidate rotamers by
        ligand clash would systematically reject good conformers. Independently
        best-scoring residues can still clash with EACH OTHER though - see
        _resolveSidechainClashes, called from _scoreAndSelectBest.

        Only placer files that are the source of an accepted cluster rep in
        cluster_reps.csv (beside multimodel_pdb - see _acceptedPlacerFiles)
        are used; a placer file whose every candidate was filtered out
        contributed no ligand pose to multimodel_pdb, so its residue
        conformers are excluded rather than pooled in alongside the ones that
        actually informed the final ligand pose(s).

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

            #resolve placer files, then restrict to only those that are the
            #source of an accepted cluster rep in cluster_reps.csv - a placer
            #file whose every candidate was filtered out (count/rscc/
            #per-placer_file-dedup/clash cutoffs) contributed nothing to the
            #ligand pose(s) in multimodel_pdb, so its protein-residue
            #conformers shouldn't be considered here either
            all_placer_files = sorted(glob.glob(self.placer_files))
            accepted_placer_files = self._acceptedPlacerFiles()
            placer_files = [f for f in all_placer_files if f in accepted_placer_files]
            print(f'found {len(all_placer_files)} placer file(s) matching the glob; '
                  f'{len(placer_files)} are the source of an accepted cluster rep '
                  f'(skipping {len(all_placer_files) - len(placer_files)} that are not)')
            if not placer_files:
                print('No placer files are the source of an accepted cluster rep; '
                      'nothing to rescore.')
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

            #score every conformer of every residue (pooled mask per residue, MSE
            #against the first event map only - see _scoreResidueConformers) and
            #independently keep the single best-scoring (lowest MSE) conformer per
            #residue - falling back to the apo conformation if none were found -
            #then resolve any resulting sidechain-sidechain clashes jointly (see
            #_resolveSidechainClashes)
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
        the FIRST event map only (self.event_maps is insertion-ordered by
        _load_event_maps's sorted glob, so this is the lowest-numbered event
        map), pooling all of that residue's conformers together into a single
        mask (rather than masking each conformer separately). Returns one MSE
        score per conformer (map density vs. model density - LOWER is better,
        unlike RSCC).

        Scores against only one map, and by MSE rather than correlation, on
        purpose: this step dominates build_final_model's runtime (one
        correlation per conformer per event map, for every residue), and per
        residue the maps mostly agree on which conformer fits best - trading a
        small amount of accuracy (occasionally picking a conformer the full
        multi-map RSCC comparison would not have) for a large speedup was an
        explicit, deliberate call, not an oversight.
        """
        scaled_bulk_solvent = 0 #from qfit, maybe should be different
        default_bfactor = 20 #can change
        n_conf = len(coor_list)

        event_map_name = next(iter(self.event_maps))

        #make a transformer for this residue
        transformer = get_transformer("qfit", template, self.event_maps_models[event_map_name])

        #pooled mask covering every conformer of this residue together
        mask = transformer.get_conformers_mask(coor_list, self._rmask)
        target = self.event_maps[event_map_name].array[mask]

        per_conformer_scores = []
        for density in transformer.get_conformers_densities(coor_list, [default_bfactor] * n_conf):
            model = density[mask]
            np.maximum(model, scaled_bulk_solvent, out=model)
            mse = np.mean((model - target) ** 2)
            per_conformer_scores.append(mse)

        return per_conformer_scores

    def _scoreAndSelectBest(self, output_folder):
        """For every residue with at least one PLACER conformer: scores every
        gathered conformer (see _scoreResidueConformers - MSE against the
        first event map, lower is better) and independently picks the
        lowest-MSE one, with NO clash checking at this stage (see run()'s
        docstring for why ligand clashes specifically are never checked).
        Residues with no PLACER conformer fall back to their apo conformation.

        Independently-best picks can still clash with EACH OTHER (sidechain
        atoms only - backbone atoms are excluded, since two residues can
        legitimately have been picked from different PLACER models with
        slightly different backbones, which would otherwise look like a
        clash at every peptide bond). _resolveSidechainClashes finds every
        such clashing pair, groups them, and reselects each group jointly to
        the lowest-total-MSE combination with no clash inside the group or
        against any residue outside it.

        Also writes three CSVs to output_folder:
          - residue_scores.csv: one row per residue that had at least one
            PLACER conformer (i.e. excludes apo fallbacks), reflecting the
            FINAL choice after sidechain clash resolution.
          - residues_with_placer_conformers.csv: a plain list of every
            residue ("{chain}{resnum}", one per line, no header) that had at
            least one PLACER conformer.
          - sidechain_clash_groups.csv: one row per resolved sidechain clash
            group (empty if none were found) - see _write_clash_groups_csv.

        Returns: {(chain_id, res_num): (best_coor, best_mse, template)}
        best_mse is None for residues that fell back to the apo conformation.
        """
        self._candidates = {}
        residues_with_conformers = []

        for (chain_id, res_num), conformers in self.residue_conformers.items():
            template = self.residue_templates[(chain_id, res_num)]

            if template is None:
                # already warned about in _gatherResidueConformers; nothing to
                # score or fall back to for this residue
                print(f'WARNING: {chain_id}{res_num} has no template (apo extraction '
                      f'failed); omitting it from the final model')
                continue

            sidechain_mask = ~np.isin(np.asarray(template.name), list(BACKBONE_ATOM_NAMES))

            if not conformers:
                # already flagged in run(); fall back to the apo conformation.
                # A single, immovable candidate (its own apo coordinates) -
                # this residue can still be absorbed into a clash group as a
                # fixed constraint on its neighbors, it just never changes.
                self._candidates[(chain_id, res_num)] = _ResidueCandidates(
                    coor=np.asarray(template.coor)[None, :, :],
                    mse=np.zeros(1),
                    placer_file=[None],
                    model_idx=[None],
                    template=template,
                    sidechain_mask=sidechain_mask,
                    fixed=True,
                )
                continue

            residues_with_conformers.append((chain_id, res_num))

            coor_list = [c[0] for c in conformers]
            scores = self._scoreResidueConformers(template, coor_list)

            self._candidates[(chain_id, res_num)] = _ResidueCandidates(
                coor=np.stack(coor_list, axis=0),
                mse=np.asarray(scores),
                placer_file=[c[1] for c in conformers],
                model_idx=[c[2] for c in conformers],
                template=template,
                sidechain_mask=sidechain_mask,
                fixed=False,
            )

        # independently best (lowest-MSE) pick per residue, ignoring clashes
        chosen_idx = {key: int(np.argmin(cand.mse)) for key, cand in self._candidates.items()}
        initial_idx = dict(chosen_idx)

        group_rows = self._resolveSidechainClashes(chosen_idx)

        best_conformers = {}
        summary_rows = []
        for key, cand in self._candidates.items():
            idx = chosen_idx[key]
            coor = cand.coor[idx]

            if cand.fixed:
                best_conformers[key] = (coor, None, cand.template)
                continue

            mse = float(cand.mse[idx])
            best_conformers[key] = (coor, mse, cand.template)
            summary_rows.append((key[0], key[1], len(cand.mse), mse,
                                  cand.placer_file[idx], cand.model_idx[idx]))

            reassigned = ' (reassigned by sidechain clash resolution)' if idx != initial_idx[key] else ''
            print(f'{key[0]}{key[1]}: best mse {mse:.4f} from {cand.placer_file[idx]} '
                  f'model {cand.model_idx[idx]} (of {len(cand.mse)} conformer(s)){reassigned}')

        self._write_residue_scores_csv(summary_rows, output_folder + '/residue_scores.csv')
        self._write_residue_conformer_list_csv(
            residues_with_conformers, output_folder + '/residues_with_placer_conformers.csv'
        )
        self._write_clash_groups_csv(group_rows, output_folder + '/sidechain_clash_groups.csv')

        return best_conformers

    # ---- sidechain-sidechain clash resolution -----------------------------
    #
    # _scoreAndSelectBest picks every residue's lowest-MSE conformer
    # independently, which can leave pairs of residues whose sidechains
    # clash with each other. The methods below find those pairs, group them
    # by connectivity, and jointly reselect each group to the
    # lowest-total-MSE combination that clashes with nothing - inside the
    # group or out. See _resolveSidechainClashes for the full algorithm and
    # _resolveGroup/_solveGroupAssignment for how one group is actually
    # solved efficiently.

    def _residueReachSpheres(self):
        """Per residue, returns (centroids, reach): reach[key] is the
        distance from centroids[key] to the farthest sidechain atom across
        EVERY gathered candidate conformer of that residue (not just the
        chosen one) - a conservative bounding sphere. Two residues can only
        possibly sidechain-clash if their reach-spheres overlap (plus a
        margin covering the VDW clash threshold), which lets every clash
        search below skip an exact atom-pairwise check for residue pairs
        that are obviously too far apart. Residues with no sidechain atoms
        (e.g. glycine) get reach 0 - they can never clash.
        """
        centroids = {}
        reach = {}
        for key, cand in self._candidates.items():
            if not cand.sidechain_mask.any():
                centroids[key] = cand.coor[0].mean(axis=0)
                reach[key] = 0.0
                continue
            pts = cand.coor[:, cand.sidechain_mask, :].reshape(-1, 3)
            centroid = pts.mean(axis=0)
            centroids[key] = centroid
            reach[key] = float(np.max(np.linalg.norm(pts - centroid, axis=1)))
        return centroids, reach

    def _candidatePairsWithinReach(self, keys_a, centroids, reach, keys_b=None, margin=3.0):
        """Yields (key1, key2) pairs - key1 from keys_a, key2 from keys_b
        (defaults to keys_a itself, in which case each unordered pair is
        yielded once) - whose reach-spheres (see _residueReachSpheres) come
        within `margin` of overlapping. `margin` just needs to conservatively
        cover the largest plausible VDW clash threshold (~2-3 Angstrom for
        two heavy atoms) - it does not need to be exact, since this is only a
        cheap prefilter and every pair it yields still gets an exact
        atom-pairwise check.
        """
        self_pairs = keys_b is None
        keys_b = keys_a if self_pairs else keys_b
        for i, k1 in enumerate(keys_a):
            others = keys_b[i + 1:] if self_pairs else keys_b
            for k2 in others:
                if k1 == k2:
                    continue
                d = np.linalg.norm(centroids[k1] - centroids[k2])
                if d <= reach[k1] + reach[k2] + margin:
                    yield k1, k2

    def _domainCompatibilityMatrix(self, key1, idx1, key2, idx2):
        """Returns an (len(idx1), len(idx2)) boolean matrix: True where
        candidate idx1[i] of residue key1 does NOT sidechain-clash with
        candidate idx2[j] of residue key2 (sidechain atoms only). The
        per-atom-pair threshold is self.clash_vdw_scale * summed VDW radii,
        EXCEPT for an (N, O) atom pair (either order) - a real N-H...O or
        O-H...N hydrogen bond legitimately sits closer than a generic clash
        would tolerate, so those pairs use self.hbond_clash_vdw_scale
        instead. idx1/idx2 are arrays of candidate indices (e.g. a truncated
        top-K domain, or a single index to check one specific pair of
        candidates).
        """
        cand1, cand2 = self._candidates[key1], self._candidates[key2]
        mask1, mask2 = cand1.sidechain_mask, cand2.sidechain_mask
        if not mask1.any() or not mask2.any():
            return np.ones((len(idx1), len(idx2)), dtype=bool)

        coor1 = cand1.coor[idx1][:, mask1, :]  # (n1, a1, 3)
        coor2 = cand2.coor[idx2][:, mask2, :]  # (n2, a2, 3)
        vdw1 = np.asarray(cand1.template.vdw_radius)[mask1]  # (a1,)
        vdw2 = np.asarray(cand2.template.vdw_radius)[mask2]  # (a2,)
        e1 = np.asarray(cand1.template.e)[mask1]  # (a1,) element symbols
        e2 = np.asarray(cand2.template.e)[mask2]  # (a2,)

        vdw_sum = vdw1[:, None] + vdw2[None, :]  # (a1, a2)
        is_n_o_pair = (
            ((e1 == 'N')[:, None] & (e2 == 'O')[None, :])
            | ((e1 == 'O')[:, None] & (e2 == 'N')[None, :])
        )  # (a1, a2)
        scale = np.where(is_n_o_pair, self.hbond_clash_vdw_scale, self.clash_vdw_scale)
        thresh = scale * vdw_sum  # (a1, a2)

        diff = coor1[:, None, :, None, :] - coor2[None, :, None, :, :]  # (n1,n2,a1,a2,3)
        dists = np.linalg.norm(diff, axis=-1)  # (n1,n2,a1,a2)
        clashing = np.any(dists < thresh[None, None, :, :], axis=(2, 3))  # (n1,n2)
        return ~clashing

    def _pairClashes(self, key1, idx1, key2, idx2):
        """Whether residue key1's candidate idx1 sidechain-clashes with
        residue key2's candidate idx2 (both single indices, not arrays)."""
        compat = self._domainCompatibilityMatrix(key1, np.array([idx1]), key2, np.array([idx2]))
        return not compat[0, 0]

    def _findClashingPairs(self, keys, chosen_idx, centroids, reach):
        """Among `keys`' CURRENT choices in chosen_idx, returns every pair
        that sidechain-clashes (after the reach-sphere prefilter)."""
        return [
            (k1, k2) for k1, k2 in self._candidatePairsWithinReach(keys, centroids, reach)
            if self._pairClashes(k1, chosen_idx[k1], k2, chosen_idx[k2])
        ]

    def _externalClashes(self, group, chosen_idx, centroids, reach):
        """Among `group`'s CURRENT choices in chosen_idx, returns every pair
        that sidechain-clashes with a residue outside the group."""
        others = [k for k in self._candidates if k not in group]
        return [
            (k1, k2) for k1, k2 in self._candidatePairsWithinReach(group, centroids, reach, keys_b=others)
            if self._pairClashes(k1, chosen_idx[k1], k2, chosen_idx[k2])
        ]

    def _connectedComponents(self, keys, pairs):
        """Groups `keys` into connected components of the graph formed by
        `pairs` (undirected edges). Keys with no edge at all are omitted -
        only residues that are actually part of some clash end up in a
        returned component."""
        adjacency = {k: set() for k in keys}
        for k1, k2 in pairs:
            adjacency[k1].add(k2)
            adjacency[k2].add(k1)

        seen = set()
        components = []
        for k in keys:
            if k in seen or not adjacency[k]:
                continue
            stack = [k]
            seen.add(k)
            comp = []
            while stack:
                cur = stack.pop()
                comp.append(cur)
                for nb in adjacency[cur]:
                    if nb not in seen:
                        seen.add(nb)
                        stack.append(nb)
            components.append(sorted(comp))
        return components

    def _format_group(self, keys):
        return ', '.join(f'{c}{r}' for c, r in keys)

    def _groupMSE(self, group, chosen_idx):
        return sum(
            float(self._candidates[key].mse[chosen_idx[key]])
            for key in group if not self._candidates[key].fixed
        )

    def _resolveSidechainClashes(self, chosen_idx):
        """Finds every sidechain-sidechain clash among the independently
        chosen (lowest-MSE) conformers in `chosen_idx`, groups clashing
        residues by connectivity, and resolves each group via _resolveGroup -
        mutating chosen_idx in place. Returns a list of per-group summary
        rows for sidechain_clash_groups.csv (see _write_clash_groups_csv).

        `groups` (connected components) are computed once, up front, from
        the INITIAL (pre-resolution) clash graph. But _resolveGroup's own
        expansion can grow one group to fully absorb the residues of a
        different, separately-identified initial component - if that
        happens, that other component is skipped rather than redundantly
        (and wastefully) resolved again from scratch.
        """
        centroids, reach = self._residueReachSpheres()
        keys = list(self._candidates.keys())

        initial_pairs = self._findClashingPairs(keys, chosen_idx, centroids, reach)
        groups = self._connectedComponents(keys, initial_pairs)

        if groups:
            print(f'{len(groups)} sidechain-sidechain clash group(s) found among '
                  f'independently chosen (lowest-MSE) conformers; resolving each jointly.')

        group_rows = []
        settled = set()
        for group in groups:
            if settled.issuperset(group):
                continue
            row = self._resolveGroup(group, chosen_idx, centroids, reach)
            settled.update(row['residues'])
            group_rows.append(row)
        return group_rows

    def _resolveGroup(self, group, chosen_idx, centroids, reach):
        """Jointly reselects one clash group to the lowest-total-MSE
        combination of candidates with no sidechain clash inside the group
        (see _solveGroupAssignment) - mutating chosen_idx in place for every
        member. If the new picks clash with a residue outside the group,
        that residue is absorbed into the group and the whole group is
        resolved again, repeating until stable.

        Two independent ways this can end up NOT fully clash-free, both
        honestly reported in the returned row rather than silently reported
        as success:
          - unresolved: _solveGroupAssignment itself could not find any
            candidate combination (even outside the top-K, see its
            docstring) that eliminates every clash WITHIN the current group -
            e.g. a residue with no PLACER conformer at all blocking every
            candidate of its one real neighbor. Expansion stops immediately;
            there is no neighbor left to absorb that would help.
          - hit_cap: the group WAS fully resolved internally, but doing so
            introduced (or left) a clash against a residue outside the
            group, and absorbing it would exceed MAX_CLASH_GROUP_EXPANSIONS
            (rounds) or self.max_clash_group_size (residues) - so expansion
            stops there instead of growing further or forcing convergence.
        """
        group = list(group)
        original_group = list(group)
        original_mse = self._groupMSE(group, chosen_idx)
        hit_cap = False
        unresolved = False

        for _round in range(MAX_CLASH_GROUP_EXPANSIONS):
            assignment, resolved = self._solveGroupAssignment(group)
            for key, idx in assignment.items():
                chosen_idx[key] = idx

            if not resolved:
                unresolved = True
                still_clashing = [
                    f'{self._format_group([k1])}-{self._format_group([k2])}'
                    for i, k1 in enumerate(group) for k2 in group[i + 1:]
                    if self._pairClashes(k1, chosen_idx[k1], k2, chosen_idx[k2])
                ]
                print(f'WARNING: sidechain clash group [{self._format_group(group)}] could NOT be '
                      f'fully resolved - no combination of candidates (checked up to the full '
                      f'candidate pool of every member) eliminates every clash within the group. '
                      f'Still clashing: {"; ".join(still_clashing)}. Keeping the lowest-total-mse '
                      f'combination found (still clashing) - flagged in '
                      f'sidechain_clash_groups.csv for manual review.')
                break

            external = self._externalClashes(group, chosen_idx, centroids, reach)
            new_members = sorted({k2 for (_, k2) in external if k2 not in group})
            if not new_members:
                break

            if len(group) + len(new_members) > self.max_clash_group_size:
                hit_cap = True
                print(f'WARNING: sidechain clash group [{self._format_group(group)}] would grow '
                      f'past max_clash_group_size={self.max_clash_group_size} residues after '
                      f'absorbing [{self._format_group(new_members)}]; stopping expansion here. '
                      f'Residual clash(es) against [{self._format_group(new_members)}] are left '
                      f'unresolved - see sidechain_clash_groups.csv.')
                break

            group.extend(new_members)
        else:
            hit_cap = True
            print(f'WARNING: sidechain clash group [{self._format_group(group)}] kept absorbing '
                  f'new neighbors past {MAX_CLASH_GROUP_EXPANSIONS} round(s); stopping here. '
                  f'Residual clashes may remain - see sidechain_clash_groups.csv.')

        final_mse = self._groupMSE(group, chosen_idx)
        flagged = hit_cap or unresolved
        verb = 'left with a residual clash' if flagged else 'resolved'
        print(f'sidechain clash group [{self._format_group(group)}] ({len(group)} residue(s), '
              f'{len(original_group)} originally clashing) {verb}: total mse '
              f'{original_mse:.4f} -> {final_mse:.4f}'
              + (' (see warning above)' if flagged else '') + '.')

        return {
            'residues': group,
            'original_residues': original_group,
            'size': len(group),
            'original_size': len(original_group),
            'original_mse': original_mse,
            'final_mse': final_mse,
            'hit_cap': hit_cap,
            'unresolved': unresolved,
        }

    def _assignmentClashFree(self, group, assignment):
        """Whether `assignment` ({key: global_candidate_index}, one per
        member of `group`) has zero pairwise sidechain clash among every
        pair in `group`. `group` is always small (capped by
        max_clash_group_size), so this is a plain O(n^2) check - no
        reach-sphere prefilter needed."""
        return all(
            not self._pairClashes(k1, assignment[k1], k2, assignment[k2])
            for i, k1 in enumerate(group) for k2 in group[i + 1:]
        )

    def _domainsFor(self, group, top_k):
        """{key: candidate indices to consider}, cheapest-first. top_k=None
        means every candidate (no truncation); fixed residues always get
        their single apo candidate regardless of top_k."""
        domains = {}
        for key in group:
            cand = self._candidates[key]
            if cand.fixed:
                domains[key] = np.array([0])
            else:
                order = np.argsort(cand.mse)
                domains[key] = order if top_k is None else order[:top_k]
        return domains

    def _solveGroupAssignmentOverDomains(self, group, domains, domain_label):
        """Solves the joint MSE-minimization problem (see
        _solveGroupAssignment) over exactly the given `domains` - no
        truncation or widening here. Returns ({key: global_candidate_index},
        resolved) where resolved is False if no combination within these
        domains eliminates every pairwise clash (branch-and-bound found
        nothing AND the ICM fallback also didn't land on a compatible
        combination) - the returned assignment is still the best/cheapest
        one found, just not clash-free.
        """
        compat = {}
        for i, key1 in enumerate(group):
            for key2 in group[i + 1:]:
                compat[(key1, key2)] = self._domainCompatibilityMatrix(
                    key1, domains[key1], key2, domains[key2]
                )

        result = self._branchAndBound(group, domains, compat)
        if result is None:
            print(f'  sidechain clash group [{self._format_group(group)}]: exact search over the '
                  f'{domain_label} domain found no fully compatible combination (or exhausted its '
                  f'node budget); falling back to a heuristic (ICM) reassignment.')
            result = self._icmAssignment(group, domains, compat)

        assignment = {key: int(domains[key][local_i]) for key, local_i in result.items()}
        return assignment, self._assignmentClashFree(group, assignment)

    def _solveGroupAssignment(self, group):
        """Returns ({key: chosen_candidate_index}, resolved) for one clash
        group: the combination of candidates (one per residue) that
        minimizes total MSE subject to no pairwise sidechain clash within
        the group - and whether that goal was actually achieved.

        Efficiency: each residue's domain is first truncated to its
        CLASH_DOMAIN_TOP_K cheapest (lowest-MSE) candidates - a conformer far
        down the MSE ranking essentially never wins even when it's
        compatible, so this turns what can be a ~100-300-way domain into a
        ~25-way one with negligible risk of losing the true optimum. Given
        the truncated domains, _branchAndBound does an exact search (DFS,
        most-constrained-residue-first, pruned by an admissible cost bound);
        if that exceeds its node budget, or finds no fully-compatible
        combination at all within the truncated domains, _icmAssignment (a
        fast, always-terminating local-search heuristic) is used instead.

        If even that fails to find a fully compatible combination - which
        does happen: e.g. a residue with no PLACER conformer at all (a fixed,
        single-candidate "domain") can clash with EVERY one of a real
        neighbor's top-K conformers, or two movable residues' only mutually
        compatible pair can simply rank outside the top-K on both sides -
        this retries ONCE with each residue's FULL (untruncated) candidate
        domain before conceding, since that's cheap (see _branchAndBound/
        _icmAssignment's own performance) and can genuinely find a real,
        compatible - if costlier - combination the truncated search missed.
        `resolved` is only False if even the full-domain retry couldn't
        eliminate every internal clash; the caller (_resolveGroup) is
        responsible for surfacing that honestly rather than reporting success.

        Fixed (apo-fallback) residues have a domain of exactly one candidate
        and are handled by the same code with no special-casing.
        """
        top_k_domains = self._domainsFor(group, CLASH_DOMAIN_TOP_K)
        assignment, resolved = self._solveGroupAssignmentOverDomains(
            group, top_k_domains, f'top-{CLASH_DOMAIN_TOP_K}'
        )

        if not resolved:
            full_domains = self._domainsFor(group, top_k=None)
            assignment, resolved = self._solveGroupAssignmentOverDomains(group, full_domains, 'full')

        return assignment, resolved

    def _branchAndBound(self, group, domains, compat, node_budget=CLASH_SOLVE_NODE_BUDGET):
        """Exact DFS branch-and-bound over `domains` (local candidate
        indices per residue), minimizing total MSE subject to `compat`
        (pairwise domain-compatibility matrices - see
        _domainCompatibilityMatrix - keyed by (key1, key2) in `group` order).
        Residues are visited most-constrained-first (smallest domain first);
        within a residue, candidates are tried cheapest-first, and a branch
        is pruned once its partial cost plus the cheapest possible
        completion (each remaining residue's own minimum candidate cost - an
        admissible lower bound, since it ignores compatibility) can no
        longer beat the best solution found so far.

        Returns {key: local_domain_index} for the optimal assignment, or
        None if the node budget was exhausted before one fully-compatible
        assignment was found (including the case where none exists at all
        within these domains).
        """
        order = sorted(group, key=lambda k: len(domains[k]))
        costs = [self._candidates[key].mse[domains[key]] for key in order]
        cheapest_first = [np.argsort(c) for c in costs]

        n = len(order)
        suffix_min = [0.0] * (n + 1)
        for k in range(n - 1, -1, -1):
            suffix_min[k] = suffix_min[k + 1] + float(costs[k].min())

        def get_matrix(k1, k2):
            key1, key2 = order[k1], order[k2]
            if (key1, key2) in compat:
                return compat[(key1, key2)], False
            return compat[(key2, key1)], True

        current = [None] * n
        best = {'assignment': None, 'cost': float('inf')}
        nodes = {'count': 0}

        def compat_ok(k, local_i):
            for prev in range(k):
                m, swapped = get_matrix(prev, k)
                i, j = (current[prev], local_i) if not swapped else (local_i, current[prev])
                if not m[i, j]:
                    return False
            return True

        def dfs(k, cost_so_far):
            if cost_so_far + suffix_min[k] >= best['cost']:
                return
            nodes['count'] += 1
            if nodes['count'] > node_budget:
                raise _NodeBudgetExceeded()
            if k == n:
                best['assignment'] = list(current)
                best['cost'] = cost_so_far
                return
            for local_i in cheapest_first[k]:
                if not compat_ok(k, local_i):
                    continue
                current[k] = local_i
                dfs(k + 1, cost_so_far + float(costs[k][local_i]))
            current[k] = None

        try:
            dfs(0, 0.0)
        except _NodeBudgetExceeded:
            return None

        if best['assignment'] is None:
            return None
        return {key: idx for key, idx in zip(order, best['assignment'])}

    def _icmAssignment(self, group, domains, compat, max_iters=25):
        """Iterated Conditional Modes: a fast, always-terminating heuristic
        for the same joint MSE-minimization problem _branchAndBound solves
        exactly. Starting every residue at its own cheapest candidate,
        repeatedly revisits each residue in `group` in turn and reassigns it
        to its cheapest candidate that's compatible with every OTHER
        residue's CURRENT pick, until a full pass changes nothing (or
        max_iters is hit). May still leave residual clashes if even the
        (already top-K-truncated) domains contain no fully mutually
        compatible combination at all - callers check for that afterward via
        _externalClashes / a subsequent _findClashingPairs pass on the next
        expansion round.

        Returns {key: local_domain_index}.
        """
        def get_matrix(key_a, key_b):
            if (key_a, key_b) in compat:
                return compat[(key_a, key_b)], False
            return compat[(key_b, key_a)], True

        current = {key: 0 for key in group}

        for _ in range(max_iters):
            changed = False
            for key in group:
                costs = self._candidates[key].mse[domains[key]]
                for local_i in np.argsort(costs):
                    ok = True
                    for other in group:
                        if other == key:
                            continue
                        m, swapped = get_matrix(key, other)
                        i, j = (local_i, current[other]) if not swapped else (current[other], local_i)
                        if not m[i, j]:
                            ok = False
                            break
                    if ok:
                        if local_i != current[key]:
                            current[key] = int(local_i)
                            changed = True
                        break
            if not changed:
                break

        return current

    def _write_clash_groups_csv(self, group_rows, path):
        """Writes one row per sidechain-sidechain clash group resolved by
        _resolveSidechainClashes (empty - header only - if none were found):
        which residues ended up jointly reselected (residues) vs. which ones
        were originally found clashing before any group expansion
        (original_residues), and the group's total MSE before/after
        reselection. Two independent residual-clash flags (see
        _resolveGroup's docstring for exactly what each means) -
        unresolved=True or hit_cap=True either one means a clash was left in
        place; check log.txt for the specific residue pair(s) still
        clashing.
        """
        with open(path, 'w+') as f:
            f.write('residues,original_residues,size,original_size,original_mse,final_mse,'
                    'hit_cap,unresolved')
            f.write('\n')
            for row in group_rows:
                residues = ';'.join(f'{c}{r}' for c, r in row['residues'])
                original_residues = ';'.join(f'{c}{r}' for c, r in row['original_residues'])
                f.write(f"{residues},{original_residues},{row['size']},{row['original_size']},"
                        f"{row['original_mse']},{row['final_mse']},{row['hit_cap']},"
                        f"{row['unresolved']}")
                f.write('\n')

    def _write_residue_scores_csv(self, rows, path):
        with open(path, 'w+') as f:
            f.write('chain,resid,num_conformers,best_mse,best_placer_file,best_model_idx')
            f.write('\n')
            for chain_id, res_num, num_conformers, best_mse, best_placer_file, best_model_idx in rows:
                f.write(f'{chain_id},{res_num},{num_conformers},{best_mse},'
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
        for (chain_id, res_num), (best_coor, best_mse, template) in self.best_conformers.items():
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
                                 args.clash_vdw_scale, args.hbond_clash_vdw_scale,
                                 args.max_clash_group_size)
    builder.run()


if __name__ == '__main__':
    main()