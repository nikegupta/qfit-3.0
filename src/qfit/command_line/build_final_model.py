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

from qfit.command_line.sidechain_clash import (
    SidechainClashResolver, CLASH_VDW_SCALE, HBOND_CLASH_VDW_SCALE, MAX_CLASH_GROUP_SIZE,
    MAX_CLASH_GROUP_EXPANSIONS, CLASH_DOMAIN_TOP_K, CLASH_SOLVE_NODE_BUDGET,
)

BACKBONE_ATOM_NAMES = {'N', 'CA', 'C', 'O', 'OXT'}

# per-residue candidate pool built by _scoreAndSelectBest and consumed by
# SidechainClashResolver. coor: (n_candidates, natoms, 3); mse: (n_candidates,)
# - both indexed identically to placer_file/model_idx (lists, length
# n_candidates). template/sidechain_mask describe the atoms (same for every
# candidate of this residue - see _gatherResidueConformers). fixed residues
# (no PLACER conformer found) get a single candidate: the apo coordinates.
_ResidueCandidates = namedtuple(
    'ResidueCandidates',
    ['coor', 'mse', 'placer_file', 'model_idx', 'template', 'sidechain_mask', 'fixed'],
)


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
    p.add_argument(
        "--max_clash_group_expansions",
        default=MAX_CLASH_GROUP_EXPANSIONS,
        metavar="<int>",
        type=int,
        help="Sidechain-sidechain clash detection: rounds of \"resolve, then absorb "
             f"new external clashes\" a group is allowed before giving up (default: "
             f"{MAX_CLASH_GROUP_EXPANSIONS})",
    )
    p.add_argument(
        "--clash_domain_top_k",
        default=CLASH_DOMAIN_TOP_K,
        metavar="<int>",
        type=int,
        help="Sidechain-sidechain clash detection: number of lowest-MSE candidates "
             "considered per residue during joint clash-group solving before falling "
             f"back to the full candidate pool (default: {CLASH_DOMAIN_TOP_K})",
    )
    p.add_argument(
        "--clash_solve_node_budget",
        default=CLASH_SOLVE_NODE_BUDGET,
        metavar="<int>",
        type=int,
        help="Sidechain-sidechain clash detection: branch-and-bound search nodes "
             "allowed per clash group before falling back to a heuristic (ICM) "
             f"reassignment (default: {CLASH_SOLVE_NODE_BUDGET})",
    )
    return p


class FinalModelBuilder():
    def __init__(self, dataset_dir, placer_files, multimodel_pdb, apo_structure, output_folder,
                 resolution, clash_vdw_scale=CLASH_VDW_SCALE,
                 hbond_clash_vdw_scale=HBOND_CLASH_VDW_SCALE,
                 max_clash_group_size=MAX_CLASH_GROUP_SIZE,
                 max_clash_group_expansions=MAX_CLASH_GROUP_EXPANSIONS,
                 clash_domain_top_k=CLASH_DOMAIN_TOP_K,
                 clash_solve_node_budget=CLASH_SOLVE_NODE_BUDGET):
        self.dir = dataset_dir
        self.placer_files = placer_files
        self.multimodel_pdb = multimodel_pdb
        self.apo_structure = apo_structure
        self.output_folder = output_folder
        self.resolution = resolution
        self.clash_vdw_scale = clash_vdw_scale
        self.hbond_clash_vdw_scale = hbond_clash_vdw_scale
        self.max_clash_group_size = max_clash_group_size
        self.max_clash_group_expansions = max_clash_group_expansions
        self.clash_domain_top_k = clash_domain_top_k
        self.clash_solve_node_budget = clash_solve_node_budget

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
        a Structure model, read directly from its iotbx hierarchy.

        resname is read from the residue_group's first atom_group rather than
        via only_atom_group() (which asserts there is exactly one) - a
        residue with 2+ altlocs (e.g. a crystallographically ambiguous ASN/
        GLN/HIS flip) has one atom_group per altloc, all with the same
        resname, so any of them gives the same answer."""
        records = []
        for chain in model._pdb_hierarchy.only_model().chains():
            chain_id = chain.id.strip()
            for residue in chain.residue_groups():
                res_num = int(residue.resseq)
                resname = residue.atom_groups()[0].resname.strip()
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

    def _collapse_apo_template_altloc(self, residue):
        """If `residue` (an apo-extracted template) has 2+ distinct non-blank
        altlocs, returns a copy keeping only the highest-occupancy one (plus
        any blank-altloc atoms, which are shared across altlocs - e.g. a
        residue whose backbone is unsplit but whose sidechain is modeled as
        altloc A/B). No-op (returns `residue` unchanged) if it has at most
        one altloc.

        Why this exists: PLACER always generates a single conformation per
        residue (confirmed empirically - never altloc-split, even when its
        own input template residue is), so a multi-altloc apo template has
        more atoms than every PLACER conformer of that residue ever will.
        Only called on residues that actually have a PLACER conformer to be
        scored against (see _gatherResidueConformers) - a residue with no
        PLACER conformer never reaches the atom-count-sensitive scoring code
        (_scoreResidueConformers), so its apo template is left exactly as-is
        there, altlocs included.
        """
        altlocs = residue.altloc
        non_blank = sorted(set(a for a in altlocs if a))
        if len(non_blank) <= 1:
            return residue

        occupancies = residue.q
        best_altloc = max(non_blank, key=lambda a: np.mean(occupancies[altlocs == a]))
        # A boolean-array extract() on an already-extracted Structure (residue is itself
        # apo_model.extract(...)'s result) corrupts residue-group identity - e.g. .resi
        # silently becomes a bogus small integer instead of the real residue number,
        # which then drops the residue entirely once _buildFinalModel tries to place it
        # by (chain_id, res_num) (confirmed empirically). Two string-based selections
        # (each its own top-level select()) plus combine() - the same pattern
        # calc_rscc.py's _extract_residue already uses for altloc extraction - preserves
        # it correctly.
        alt_structure = residue.extract("altloc", best_altloc, "==")
        blank_structure = residue.extract("altloc", "", "==")
        return alt_structure.combine(blank_structure)

    def _gatherResidueConformers(self, placer_files):
        """For every residue in self.all_residues, gathers every
        conformation of that residue found across every model of every input
        placer file. A residue absent from a given placer model is simply
        skipped for that model (no fallback structure is used here).

        Returns:
          residue_templates  : {(chain_id, res_num): Structure} - the apo
                                structure's own copy of that residue, with
                                its altlocs collapsed to a single one (see
                                _collapse_apo_template_altloc) for any
                                residue that has at least one gathered PLACER
                                conformer - guaranteeing the template's atom
                                count matches every one of that residue's
                                PLACER conformers, since PLACER never
                                generates a multi-altloc residue itself. A
                                residue with zero PLACER conformers keeps its
                                apo template untouched (altlocs included) -
                                it's never atom-count-compared against
                                anything (see _scoreAndSelectBest).
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

        collapsed = []
        for key, conformers in residue_conformers.items():
            template = residue_templates[key]
            if conformers and template is not None and len(set(a for a in template.altloc if a)) > 1:
                residue_templates[key] = self._collapse_apo_template_altloc(template)
                collapsed.append(f'{key[0]}{key[1]}')
        if collapsed:
            print(f'FLAG: {len(collapsed)} scored residue(s) had a multi-altloc apo template '
                  f'collapsed to their higher-occupancy conformer before scoring (PLACER never '
                  f'generates a multi-altloc residue, so the template must match): '
                  f'{", ".join(collapsed)}')

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

        resolver = SidechainClashResolver(
            self._candidates, cost_of=lambda c: c.mse,
            clash_vdw_scale=self.clash_vdw_scale,
            hbond_clash_vdw_scale=self.hbond_clash_vdw_scale,
            max_clash_group_size=self.max_clash_group_size,
            max_clash_group_expansions=self.max_clash_group_expansions,
            clash_domain_top_k=self.clash_domain_top_k,
            clash_solve_node_budget=self.clash_solve_node_budget,
            revert_index=None, group_label='sidechain clash group',
        )
        group_rows = resolver.resolve_sidechain_clashes(chosen_idx)

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
                        f"{row['original_cost']},{row['final_cost']},{row['hit_cap']},"
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
                                 args.max_clash_group_size, args.max_clash_group_expansions,
                                 args.clash_domain_top_k, args.clash_solve_node_budget)
    builder.run()


if __name__ == '__main__':
    main()