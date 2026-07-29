import argparse
import glob
from pathlib import Path
import time
import numpy as np
import heapq
import tempfile
import os
import sys
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
 

from qfit import Structure
from qfit import XMap
from qfit.xtal.transformer import get_transformer

import iotbx.pdb

RSCC_CUTOFF = 0.6


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
        'fit_ligand_files',
        type=str,
        help='Glob pattern for all fit_ligand files'
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
        "-n",
        "--filter_proportion",
        default=0.1,
        metavar="<float>",
        type=float,
        help="Proportion (0-1) of the total number of input conformers "
             "(summed across all placer_files) to filter down to. This is "
             "computed once, from the first iteration's full input pool, and "
             "the resulting integer count is then reused as-is for every "
             "subsequent recursive iteration - it does not shrink as "
             "placer_files get excluded. e.g. 0.1 keeps the best 10%% of the "
             "first iteration's input conformers. (default: 0.1)",
    )
    p.add_argument(
        "--min_cluster_proportion",
        default=0.1,
        metavar="<float>",
        type=float,
        help="Count filtering: minimum number of member conformers a spatial "
             "cluster must have to be kept, expressed as a proportion of the "
             "number of conformers in a single placer_file's coor_set "
             "(assumed to be the same for every placer_file) for that "
             "iteration. e.g. with 100 conformers per placer_file, 0.1 means "
             "a cluster needs more than 10 members to be accepted. Runs "
             "before RSCC filtering; clusters below this are treated as "
             "noise and discarded. (default: 0.1)",
    )
    p.add_argument(
        "--clash_vdw_scale",
        default=0.75,
        metavar="<float>",
        type=float,
        help="Clash detection: two cluster-representative ligand conformers "
             "are considered clashing if any pair of atoms (one from each) "
             "is closer than clash_vdw_scale * (vdw_radius_i + vdw_radius_j), "
             "using each atom's built-in qfit Structure vdw radius. Whenever "
             "two reps clash, the one with the lower RSCC is discarded. "
             "(default: 0.75)",
    )
    p.add_argument(
        "--rep_selection",
        default="score",
        choices=["score", "centroid"],
        help="Method used to pick each spatial cluster's representative "
             "conformer. 'score' (default) picks the conformer with the best "
             "(lowest) map-fit MSE score. 'centroid' picks the conformer "
             "whose ligand geometry is closest (lowest RMSD) to the mean "
             "ligand geometry of the cluster.",
    )
    return p

class Filter():
    def __init__(self, dataset_dir, placer_files, fit_ligand_files, output_folder,
                 resolution, filter_proportion=0.1, min_cluster_proportion=0.1,
                 clash_vdw_scale=0.75, rep_selection='score'):
        self.dir = dataset_dir
        self.placer_files = placer_files
        self.fit_ligand_files = fit_ligand_files
        self.output_folder = output_folder
        self.resolution = resolution

        self._rmask = 0.5 + self.resolution / 3.0 #from qfit

        # Proportion of the *first* iteration's total input conformers to
        # keep for the top-N step. The actual integer count (self.n) is
        # computed once, inside _runIteration, the first time it's None (i.e.
        # on the first iteration), and then reused unchanged for every
        # subsequent iteration - it does not shrink as placer_files get
        # excluded from later iterations.
        self.filter_proportion = filter_proportion
        self.n = None

        # Count filtering: minimum number of conformers required to support a
        # spatial cluster for it to be kept, expressed as a proportion of the
        # number of conformers in a single placer_file's coor_set (assumed to
        # be the same for every placer_file) for that iteration. Runs before
        # RSCC filtering.
        self.min_cluster_proportion = min_cluster_proportion

        # Clash detection: a ligand/protein atom pair clashes if their
        # distance is less than clash_vdw_scale * (sum of their vdw radii).
        self.clash_vdw_scale = clash_vdw_scale

        # How to pick a cluster's representative conformer: 'score' (best
        # map-fit MSE) or 'centroid' (closest to the cluster's mean geometry).
        self.rep_selection = rep_selection

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
        """Recursively filters placer models down to a set of accepted,
        spatially-distinct ligand poses.

        Each iteration:
          1. Scores and clusters the ligand conformers from the remaining
             placer files (see `_runIteration`): top-N selection, spatial
             clustering, count/RSCC/per-placer_file/clash filtering.
          2. Accepts the cluster representatives that pass every filtering
             stage.
          3. Excludes, from the pool of placer files considered in the next
             iteration, every placer file that contributed *any* conformer
             (not just the accepted representative) to a spatial cluster
             that was accepted.

        This repeats until an iteration produces no accepted clusters, or
        there are no placer files left to consider.

        All print() output produced during this method (including from every
        iteration's _runIteration call) is mirrored to
        output_folder/log.txt in addition to the console. Each iteration also
        gets its own iteration_N/log.txt containing just that iteration's
        output, in addition to (not instead of) the combined top-level log.
        """
        self.output_folder_base = str(self.dir) + '/' + self.output_folder
        os.makedirs(self.output_folder_base, exist_ok=True)

        log_path = self.output_folder_base + '/log.txt'
        log_file = open(log_path, 'w')
        original_stdout = sys.stdout
        sys.stdout = _Tee(original_stdout, log_file)

        try:
            remaining_placer_files = list(self.placer_files)
            remaining_fit_ligand_files = list(self.fit_ligand_files)

            # accumulated accepted results across all recursive iterations,
            # keyed by a globally-unique id ("{iteration}_{cluster_id}")
            self.all_accepted_cluster_reps = {}
            self.all_accepted_rsccs = {}
            self.all_accepted_models = {}

            # accumulated *unfiltered* cluster representatives (every spatial
            # cluster produced in every iteration, before any filtering is
            # applied), keyed the same way
            self.all_unfiltered_cluster_reps = {}
            self.all_unfiltered_rsccs = {}

            iteration = 0
            while remaining_placer_files:
                iteration += 1
                print(f'--- Iteration {iteration}: {len(remaining_placer_files)} placer file(s) remaining ---')

                iter_output_folder = os.path.join(self.output_folder_base, f'iteration_{iteration}')
                os.makedirs(iter_output_folder, exist_ok=True)

                # mirror this iteration's output into its own iteration_N/log.txt
                # as well as the combined top-level log already in sys.stdout
                iter_log_path = iter_output_folder + '/log.txt'
                iter_log_file = open(iter_log_path, 'w')
                stdout_before_iteration = sys.stdout
                sys.stdout = _Tee(stdout_before_iteration, iter_log_file)

                try:
                    (accepted_cluster_reps, accepted_rsccs, cluster_placer_file_map, accepted_models,
                     unfiltered_cluster_reps, unfiltered_rsccs) = self._runIteration(
                        remaining_placer_files, remaining_fit_ligand_files, iter_output_folder
                    )
                finally:
                    sys.stdout = stdout_before_iteration
                    iter_log_file.close()

                for cluster_id, entry in unfiltered_cluster_reps.items():
                    global_id = f'{iteration}_{cluster_id}'
                    self.all_unfiltered_cluster_reps[global_id] = entry
                    self.all_unfiltered_rsccs[global_id] = unfiltered_rsccs[cluster_id]

                if not accepted_cluster_reps:
                    print(f'No clusters passed filtering in iteration {iteration}. Stopping recursion.')
                    break

                for cluster_id, entry in accepted_cluster_reps.items():
                    global_id = f'{iteration}_{cluster_id}'
                    self.all_accepted_cluster_reps[global_id] = entry
                    self.all_accepted_rsccs[global_id] = accepted_rsccs[cluster_id]
                    self.all_accepted_models[global_id] = accepted_models[cluster_id]

                # exclude every placer_file that had a conformer land in any
                # accepted spatial cluster, not just the cluster representative
                excluded_placer_files = set()
                for cluster_id in accepted_cluster_reps:
                    excluded_placer_files.update(cluster_placer_file_map.get(cluster_id, set()))

                next_placer_files = []
                next_fit_ligand_files = []
                for pf, flf in zip(remaining_placer_files, remaining_fit_ligand_files):
                    if pf not in excluded_placer_files:
                        next_placer_files.append(pf)
                        next_fit_ligand_files.append(flf)

                if len(next_placer_files) == len(remaining_placer_files):
                    # nothing was excluded even though clusters were accepted;
                    # bail out so we don't loop forever
                    print('Warning: accepted clusters did not exclude any placer files; stopping to avoid an infinite loop.')
                    break

                remaining_placer_files = next_placer_files
                remaining_fit_ligand_files = next_fit_ligand_files

            print(f'Recursive filtering complete after {iteration} iteration(s). '
                  f'Total accepted models: {len(self.all_accepted_cluster_reps)}')

            self._writeCombinedOutputs(self.output_folder_base)
        finally:
            sys.stdout = original_stdout
            log_file.close()

    def _runIteration(self, placer_files, fit_ligand_files, output_folder):
        """Runs one full filtering pass (scoring, top-N selection, spatial
        clustering, and count/RSCC/per-placer_file/clash filtering) over the
        given subset of placer_files/fit_ligand_files, writing per-iteration
        outputs into `output_folder`.

        Returns:
            accepted_cluster_reps: dict cluster_id -> (score, placer_file, index, ligand_coor, cluster_size)
                for clusters that passed every filtering stage.
            accepted_rsccs: dict cluster_id -> rscc, for the accepted clusters above.
            cluster_placer_file_map: dict cluster_id -> set of placer_files that had
                *any* conformer (not just the representative) assigned to that
                spatial cluster. Only populated for accepted cluster_ids.
            accepted_models: dict cluster_id -> structure model (coor/b already set)
                for the accepted cluster representative, so it can be written out
                later without needing to keep this iteration's data around.
            unfiltered_cluster_reps: dict cluster_id -> (score, placer_file, index, ligand_coor, cluster_size)
                for every spatial cluster produced in this iteration, before any
                filtering is applied.
            unfiltered_rsccs: dict cluster_id -> rscc, for every cluster in
                unfiltered_cluster_reps above.
        """
        self.binding_site_residues = {}
        self.base_binding_sites = {}
        self.coor_sets = {}
        self.scores = {}

        #main loop over placer structures
        for placer_file, fit_ligand_file in zip(placer_files, fit_ligand_files):
            print(placer_file)
            models = Structure.fromfile(placer_file).split_models()
            self.base_structure = Structure.fromfile(fit_ligand_file)

            #remove hydrogens from base structure
            self.base_structure = self.base_structure.extract("e", "H", "!=")

            # fixing issues with terminal oxygens
            rename = self.base_structure.extract("name", "OXT", "==")
            rename.name = "O"
            self.base_structure = self.base_structure.extract("name", "OXT", "!=").combine(rename)

            #get set of binding site coors
            self.binding_site_residues.update({placer_file: self._determineBindingSite(models)})
            self.base_binding_sites.update({placer_file: self._getBaseBindingSite(placer_file)})
            self.coor_sets.update({placer_file: self._getBindingSiteConformers(models, placer_file)})

            #convert to density
            self.scores.update({placer_file: self._convertAndScoreLigand(placer_file)})

        #figure out how many models "top N" should actually be, as a
        #proportion of the total input conformers - but only on the *first*
        #iteration. self.n is then reused as-is for every later iteration,
        #so the filter number doesn't shrink as placer_files get excluded.
        total_conformers = sum(len(self.coor_sets[pf]) for pf in self.coor_sets)
        if self.n is None:
            self.n = max(1, int(round(total_conformers * self.filter_proportion)))
            print(f'total input conformers in first iteration: {total_conformers}')
            print(f"filter number fixed at top {self.n} models "
                  f"({self.filter_proportion:.1%} of the first iteration's total); "
                  f"this value will be reused for all subsequent iterations")
        else:
            print(f'total input conformers this iteration: {total_conformers} '
                  f'(reusing fixed filter number from iteration 1: top {self.n} models)')

        #get top N
        top_n = heapq.nsmallest(self.n, ((val, key, idx) for key, lst in self.scores.items() for idx, val in enumerate(lst)))

        #write output files
        os.makedirs(output_folder, exist_ok=True)
        output_path = output_folder + '/filtered_models.pdb'
        output_csv = output_folder + '/scores.csv'
        output_summary = output_folder + '/top_scores.csv'

        #write outputcsv
        with open(output_csv, 'w+') as f:
            f.write('placer_file,index,mse')
            f.write('\n')
            for placer_file in placer_files:
                for i, score in enumerate(self.scores[placer_file]):
                    f.write(f'{placer_file},{i},{score}')
                    f.write('\n')

        #write multimodel output and score csv
        with open(output_summary, 'w+') as f:
            f.write('placer_file,index,mse')
            f.write('\n')
            bs_models = []
            for entry in top_n:
                score = entry[0]
                placer_file = entry[1]
                index = entry[2]

                f.write(f'{placer_file},{index},{score}')
                f.write('\n')

                bs_model = self.base_binding_sites[placer_file].copy()
                bs_model.coor = self.coor_sets[placer_file][index]
                bs_model.b = 20
                bs_models.append(bs_model)

        self._write_multimodel_pdb(bs_models, output_path)

        if not top_n:
            return {}, {}, {}, {}, {}, {}

        #now write out spatially clustered models
        self._spatialClustering(top_n, output_folder)
        self._calcRSCCofClusters()

        print(f'number of reps before filtering: {len(self.cluster_reps)}')

        # snapshot the full, unfiltered set of cluster reps and their rsccs
        # before any filtering below mutates self.cluster_reps, so the full
        # candidate set stays inspectable
        unfiltered_cluster_reps = dict(self.cluster_reps)
        unfiltered_rsccs = dict(self.cluster_rsccs)
        all_cluster_reps_csv = output_folder + '/all_cluster_reps.csv'
        self._write_cluster_reps_csv(unfiltered_cluster_reps, unfiltered_rsccs, all_cluster_reps_csv)
        print(f'all (unfiltered) cluster reps for this iteration written to {all_cluster_reps_csv}')

        #count filtering: drop clusters that are only supported by a small
        #number of conformers, relative to the number of conformers in a
        #single placer_file's coor_set (assumed to be the same for every
        #placer_file in this iteration). e.g. with the default
        #min_cluster_proportion of 0.1 and 100 conformers per placer_file, a
        #cluster needs more than 0.1 * 100 = 10 members to survive; smaller
        #clusters are treated as noise. This runs before the RSCC filter.
        n_per_placer_file = len(next(iter(self.coor_sets.values())))

        filtered_cluster_reps = {}
        for cluster_id in self.cluster_reps:
            num_cluster_members = self.cluster_reps[cluster_id][4]
            if num_cluster_members > n_per_placer_file * self.min_cluster_proportion:
                filtered_cluster_reps.update({cluster_id: self.cluster_reps[cluster_id]})
        self.cluster_reps = filtered_cluster_reps
        passed_count_ids = set(self.cluster_reps.keys())

        print(f'number of reps after count filtering: {len(self.cluster_reps)}')

        #filter cluster_reps by rscc
        rscc_cluster_reps = {}
        for cluster_id in self.cluster_reps:
            if self.cluster_rsccs[cluster_id] > RSCC_CUTOFF:
                rscc_cluster_reps.update({cluster_id: self.cluster_reps[cluster_id]})
        self.cluster_reps = rscc_cluster_reps
        passed_rscc_ids = set(self.cluster_reps.keys())

        print(f'number of reps after rscc filtering: {len(self.cluster_reps)}')

        #filter down to best structure from each placer_file
        filtered_cluster_reps = {}
        visited = []
        for cluster_id in self.cluster_reps:
            placer_file = self.cluster_reps[cluster_id][1]

            if placer_file not in visited:
                visited.append(placer_file)

                best_rscc = 0
                best_cluster_id = None
                for key in self.cluster_reps:
                    if self.cluster_reps[key][1] == placer_file:
                        if self.cluster_rsccs[key] > best_rscc:
                            best_rscc = self.cluster_rsccs[key]
                            best_cluster_id = key

                filtered_cluster_reps.update({best_cluster_id: self.cluster_reps[best_cluster_id]})
        self.cluster_reps = filtered_cluster_reps
        accepted_ids = set(self.cluster_reps.keys())

        print(f'number of reps after file filtering: {len(self.cluster_reps)}')

        #clash filtering: pairwise clash detection between the remaining
        #cluster reps (see _filter_clashes) - whenever two reps clash, the
        #lower-rscc one is greedily dropped.
        self._filter_clashes()

        accepted_rsccs = {cluster_id: self.cluster_rsccs[cluster_id] for cluster_id in self.cluster_reps}

        # map each accepted cluster to the full set of placer_files that had
        # *any* conformer assigned to that spatial cluster (not just the rep) -
        # these are the placer_files that get excluded from the next iteration
        cluster_placer_file_map = {}
        for cluster_id in self.cluster_reps:
            members = self.clusters.get(cluster_id, [])
            cluster_placer_file_map[cluster_id] = {member[1] for member in members}

        #output cluster models - this csv contains only the final,
        #filtered/accepted cluster representatives for this iteration
        cluster_summary = output_folder + '/cluster_reps.csv'
        self._write_cluster_reps_csv(self.cluster_reps, self.cluster_rsccs, cluster_summary)

        #work out, for every raw cluster produced this iteration, exactly why
        #it did or didn't make it into this iteration's final cluster_reps:
        #the first filtering stage it failed, or 'accepted' if it made it all
        #the way through (including surviving clash filtering).
        cluster_status = {}
        for cluster_id in unfiltered_cluster_reps:
            if cluster_id not in passed_count_ids:
                cluster_status[cluster_id] = 'failed_count_cutoff'
            elif cluster_id not in passed_rscc_ids:
                cluster_status[cluster_id] = 'failed_rscc_cutoff'
            elif cluster_id not in accepted_ids:
                cluster_status[cluster_id] = 'lost_per_placer_file_dedup'
            elif cluster_id in self.clash_rejected_ids:
                partner_id = self.clash_partners[cluster_id]
                partner_placer_file = unfiltered_cluster_reps[partner_id][1]
                cluster_status[cluster_id] = (
                    f'failed_clash_filter (vs {partner_placer_file})'
                )
            else:
                cluster_status[cluster_id] = 'accepted'

        #write out full clustering information for every placer_file/index
        #conformer that was scored this iteration, not just the ones that
        #made the top-N cut and were actually clustered, and not just the
        #final accepted representatives - so every conformer can be traced to
        #its cluster, that cluster's representative, and the reason the
        #cluster was accepted or rejected. Conformers that were never passed
        #to _spatialClustering (because they didn't make the top-N score cut)
        #get a 'not_clustered' status.
        cluster_members_csv = output_folder + '/cluster_members.csv'
        self._write_cluster_members_csv(
            self.clusters, self.scores, unfiltered_cluster_reps, unfiltered_rsccs,
            cluster_status, cluster_members_csv
        )
        print(f'cluster membership and rejection reasons for this iteration written to {cluster_members_csv}')

        accepted_models = {}
        cluster_models = []
        for cluster_id in self.cluster_reps:
            placer_file = self.cluster_reps[cluster_id][1]
            index = self.cluster_reps[cluster_id][2]

            cluster_model = self.base_binding_sites[placer_file].copy()
            cluster_model.coor = self.coor_sets[placer_file][index]
            cluster_model.b = 20
            cluster_models.append(cluster_model)
            accepted_models[cluster_id] = cluster_model

        cluster_model_path = output_folder + '/cluster_rep_models.pdb'
        self._write_multimodel_pdb(cluster_models, cluster_model_path)

        return (dict(self.cluster_reps), accepted_rsccs, cluster_placer_file_map, accepted_models,
                unfiltered_cluster_reps, unfiltered_rsccs)

    def _writeCombinedOutputs(self, output_folder_base):
        """Writes the final combined set of accepted cluster representatives,
        collected across every recursive iteration, to cluster_reps.csv and
        cluster_rep_models.pdb at the top level of the output folder. Also
        writes all_cluster_reps.csv at the top level, combining every
        (unfiltered) spatial cluster produced across all iterations, so the
        full candidate set - not just what was ultimately accepted - stays
        inspectable at the top level too.
        """
        cluster_summary = output_folder_base + '/cluster_reps.csv'
        cluster_model_path = output_folder_base + '/cluster_rep_models.pdb'
        all_cluster_summary = output_folder_base + '/all_cluster_reps.csv'

        self._write_cluster_reps_csv(
            self.all_accepted_cluster_reps, self.all_accepted_rsccs, cluster_summary
        )
        self._write_cluster_reps_csv(
            self.all_unfiltered_cluster_reps, self.all_unfiltered_rsccs, all_cluster_summary
        )

        cluster_models = [self.all_accepted_models[global_id]
                           for global_id in self.all_accepted_cluster_reps]
        self._write_multimodel_pdb(cluster_models, cluster_model_path)

    def _write_cluster_reps_csv(self, cluster_reps, cluster_rsccs, path):
        """
        Writes a placer_file,index,mse,cluster,rscc,num_members csv for the
        given cluster_reps/cluster_rsccs dicts. Used for both the
        per-iteration and top-level combined csvs, and for both the
        unfiltered and final filtered sets, so all of these outputs share the
        same columns and can be compared directly - including which
        placer_file and conformer index within it each cluster rep came
        from. The 'cluster' column is whatever key is used in cluster_reps (a
        per-iteration cluster_id, or a globally-unique
        "{iteration}_{cluster_id}" id for the combined csvs).
        """
        with open(path, 'w+') as f:
            f.write('placer_file,index,mse,cluster,rscc,num_members')
            f.write('\n')
            for cluster_id in cluster_reps:
                mse = cluster_reps[cluster_id][0]
                placer_file = cluster_reps[cluster_id][1]
                index = cluster_reps[cluster_id][2]
                rscc = cluster_rsccs[cluster_id]
                num_members = cluster_reps[cluster_id][4]

                f.write(f'{placer_file},{index},{mse},{cluster_id},{rscc},{num_members}')
                f.write('\n')

    def _write_cluster_members_csv(self, clusters, scores, cluster_reps, cluster_rsccs, cluster_status, path):
        """
        Writes a csv covering every placer_file/index conformer that was
        scored in this iteration, with its cluster assignment and enough
        information to trace *why* that cluster's representative was
        accepted or rejected:

          cluster                  : the spatial cluster this conformer belongs
                                      to. Blank if this conformer didn't make
                                      the top-N score cut for this iteration,
                                      since _spatialClustering only ever sees
                                      the top-N subset.
          cluster_rep_placer_file,
          cluster_rep_index        : identifies the model that represents (and
                                      effectively supersedes) this conformer's
                                      cluster - i.e. the cluster's chosen
                                      representative, which is what actually
                                      gets carried forward into RSCC scoring
                                      and filtering
          cluster_rscc             : the RSCC computed for that representative
          cluster_status           : 'accepted', 'failed_count_cutoff',
                                      'failed_rscc_cutoff',
                                      'lost_per_placer_file_dedup', or
                                      'failed_clash_filter (vs <placer_file>)'
                                      - why the representative (and therefore
                                      every member of this cluster) did or
                                      didn't make it into this iteration's
                                      cluster_reps.csv

        Conformers that were never passed to _spatialClustering (because they
        didn't make the top-N score cut) get an empty 'cluster' value and a
        'not_clustered' status, so the full set of input conformers and their
        scores stays inspectable in one place, with cluster membership and
        rejection reasons shown wherever they apply.

        `clusters` is self.clusters (cluster_id -> list of
        (score, placer_file, index, ligand_coor) tuples), scoped to this
        iteration's top-N subset. `scores` is self.scores (placer_file -> list
        of mse scores, one per conformer index) for every placer_file
        considered in this iteration. `cluster_reps`/`cluster_rsccs` should be
        the *unfiltered* snapshots covering every raw cluster_id produced this
        iteration. `cluster_status` maps every raw cluster_id to its final
        disposition for this iteration.
        """
        cluster_of = {}
        for cluster_id, members in clusters.items():
            for score, placer_file, index, ligand_coor in members:
                cluster_of[(placer_file, index)] = cluster_id

        with open(path, 'w+') as f:
            f.write('placer_file,index,mse,cluster,cluster_rep_placer_file,'
                    'cluster_rep_index,cluster_rscc,cluster_status')
            f.write('\n')
            for placer_file, score_list in scores.items():
                for index, mse in enumerate(score_list):
                    cluster_id = cluster_of.get((placer_file, index), '')
                    if cluster_id == '':
                        rep_placer_file = ''
                        rep_index = ''
                        rscc = ''
                        status = 'not_clustered'
                    else:
                        rep_placer_file = cluster_reps[cluster_id][1]
                        rep_index = cluster_reps[cluster_id][2]
                        rscc = cluster_rsccs[cluster_id]
                        status = cluster_status[cluster_id]

                    f.write(f'{placer_file},{index},{mse},{cluster_id},{rep_placer_file},'
                            f'{rep_index},{rscc},{status}')
                    f.write('\n')

    def _calcRSCCofClusters(self):
        self.cluster_rsccs = {}
        for cluster_id in self.cluster_reps:
            placer_file = self.cluster_reps[cluster_id][1]
            ligand_coor = self.cluster_reps[cluster_id][3]

            scaled_bulk_solvent = 0 #from qfit, maybe should be different

            #extract ligand from binding site and coor sets
            ligand = self.base_binding_sites[placer_file].extract('resname LIG')

            #make bfactor array
            default_bfactor = 20 #can change 

            rsccs = []
            for event_map_name in  list(self.event_maps.keys()):
                #make a transformer for this structure
                transformer = get_transformer("qfit", ligand, self.event_maps_models[event_map_name])

                #convert and score this set of rotamers
                mask = transformer.get_conformers_mask([ligand_coor], self._rmask)
                target = self.event_maps[event_map_name].array[mask]

                for density in transformer.get_conformers_densities([ligand_coor],[default_bfactor]):
                    model = density[mask]         
                    np.maximum(model, scaled_bulk_solvent, out=model)  
                    correlation_matrix = np.corrcoef(model, target)
                    rscc = correlation_matrix[0, 1]
                    rsccs.append(rscc)
            
            best_rscc = max(rsccs)
            self.cluster_rsccs.update({cluster_id: best_rscc})

    def _filter_clashes(self):
        """
        Performs pairwise clash detection between every ligand conformer in
        self.cluster_reps (the current filtered set of cluster
        representatives - i.e. those that already passed the count cutoff,
        rscc cutoff, and per-placer_file dedup).

        Whenever two cluster reps clash, the one with the lower RSCC
        (self.cluster_rsccs) is rejected. This is resolved greedily: the
        remaining rep with the highest RSCC is kept, everything that clashes
        with it is rejected, and this repeats among what's left - so a
        rejection by a higher-RSCC neighbor can never be "un-rejected" by a
        separate pairwise comparison against a lower-RSCC one.

        Two ligand atoms are considered clashing if the distance between them
        is less than self.clash_vdw_scale * (vdw_radius_i + vdw_radius_j). Two
        ligand conformers clash if any pair of atoms (one from each) clash.
        All ligand_coor arrays share the same atom ordering, so a single VDW
        radius list (pulled from any base binding site's ligand) applies to
        every conformer.

        Records, for every cluster_id that entered this method, whether it
        was kept or rejected due to clashing:
        self.clash_rejected_ids : set of cluster_ids dropped because a
                                    higher-RSCC rep clashed with them
        self.clash_partners     : cluster_id -> the single kept cluster_id
                                    that caused its rejection (populated only
                                    for rejected cluster_ids)
        """
        self.clash_rejected_ids = set()
        self.clash_partners = {}

        if not self.cluster_reps:
            return

        # VDW radii for the ligand atoms - order matches ligand_coor ordering
        # since all ligand coordinate sets were reordered to match this same
        # reference atom order.
        first_key = list(self.base_binding_sites.keys())[0]
        ligand = self.base_binding_sites[first_key].extract("resname LIG")
        vdw_radii = np.asarray(ligand.vdw_radius)

        cluster_ids = list(self.cluster_reps.keys())
        n = len(cluster_ids)
        ligand_coors = [self.cluster_reps[cid][3] for cid in cluster_ids]

        # sum of radii for every pair of atoms, scaled by the cutoff -> clash
        # distance threshold matrix (n_atoms x n_atoms)
        radii_sum = vdw_radii[:, None] + vdw_radii[None, :]
        clash_threshold = self.clash_vdw_scale * radii_sum

        # build the clash graph: an edge between two cluster reps if ANY pair
        # of atoms (one from each ligand) is closer than its clash threshold
        clash_matrix = np.zeros((n, n), dtype=bool)
        for i in range(n):
            for j in range(i + 1, n):
                diff = ligand_coors[i][:, None, :] - ligand_coors[j][None, :, :]
                dists = np.linalg.norm(diff, axis=-1)
                if np.any(dists < clash_threshold):
                    clash_matrix[i, j] = True
                    clash_matrix[j, i] = True

        n_clashing_pairs = int(clash_matrix.sum() / 2)
        print(f'found {n_clashing_pairs} clashing pairs among {n} cluster reps')

        # greedy resolution: repeatedly keep the highest-rscc remaining rep,
        # reject everything that clashes with it, and continue among the rest
        remaining = set(range(n))
        kept_indices = set()

        while remaining:
            best_i = max(remaining, key=lambda i: self.cluster_rsccs[cluster_ids[i]])
            kept_indices.add(best_i)
            remaining.discard(best_i)

            clashing_with_best = [i for i in remaining if clash_matrix[best_i, i]]
            for i in clashing_with_best:
                rejected_id = cluster_ids[i]
                self.clash_rejected_ids.add(rejected_id)
                self.clash_partners[rejected_id] = cluster_ids[best_i]
                remaining.discard(i)

        kept_ids = {cluster_ids[i] for i in kept_indices}
        self.cluster_reps = {cid: self.cluster_reps[cid] for cid in kept_ids}

        print(f'number of reps after clash filtering: {len(self.cluster_reps)}')

    def _selectClusterRepresentative(self, members):
        """
        Chooses which member of a spatial cluster becomes that cluster's
        representative model. `members` is a list of
        (score, placer_file, index, ligand_coor) tuples belonging to one
        cluster (the same shape as the entries in self.clusters[cluster_id]).

        Two selection strategies are supported, controlled by
        self.rep_selection:

          - 'score' (default, original behavior): pick the member with the
            lowest map-fit MSE score, i.e. the conformer that agrees best
            with the event map.

          - 'centroid': compute the per-atom mean ligand coordinate across
            all members of the cluster (the cluster centroid), then pick the
            member whose ligand coordinates are closest (lowest RMSD) to that
            centroid. This favors the conformer that best represents the
            cluster's consensus geometry, rather than whichever single
            conformer happens to score best against the map.
        """
        if self.rep_selection == 'centroid':
            ligand_coors = np.array([member[3] for member in members])  # (n_members, n_atoms, 3)
            centroid = ligand_coors.mean(axis=0)

            best_rmsd = None
            best_member = None
            for member in members:
                diff = member[3] - centroid
                rmsd = np.sqrt(np.mean(np.sum(diff ** 2, axis=1)))
                if best_rmsd is None or rmsd < best_rmsd:
                    best_rmsd = rmsd
                    best_member = member
            return best_member
        else:
            # original behavior: lowest mse score wins
            return min(members, key=lambda member: member[0])

    def _spatialClustering(self, top_n, output_folder):
        """Spatially clusters the ligands of the top_n models based on RMSD.
        """

        # extract just the ligand coordinates for every entry, tracking provenance
        ligand_coor_sets = []
        entry_labels = []       # human readable "placer_file, index" for dendrogram leaves
        entry_info = []   

        for score, placer_file, index in top_n:
            coor_set = self.coor_sets[placer_file][index]
            ligand_coor = coor_set[-self.ligand_size:, :]

            ligand_coor_sets.append(ligand_coor)
            entry_labels.append(f'{Path(placer_file).stem}_{index}')
            entry_info.append((score,placer_file,index,ligand_coor))

        n_entries = len(ligand_coor_sets)

        # build the pairwise RMSD distance matrix between ligand conformers
        dist_matrix = np.zeros((n_entries, n_entries))
        for i in range(n_entries):
            for j in range(i + 1, n_entries):
                diff = ligand_coor_sets[i] - ligand_coor_sets[j]
                rmsd = np.sqrt(np.mean(np.sum(diff ** 2, axis=1)))
                dist_matrix[i, j] = rmsd
                dist_matrix[j, i] = rmsd

        condensed_dist = squareform(dist_matrix, checks=False)
        linkage_matrix = linkage(condensed_dist, method='average')

        # cut the tree at a 2 A RMSD cutoff: any two leaves are in the same
        # cluster if the RMSD at which their branches merge is <= 2 A.
        rmsd_cutoff = 4.0
        cluster_ids = fcluster(linkage_matrix, t=rmsd_cutoff, criterion='distance')
        self.cluster_assignments = cluster_ids  # 1-indexed cluster id per entry, same order as entry_labels/entry_provenance

        # write out the dendrogram, scoped to this iteration's output folder
        os.makedirs(output_folder, exist_ok=True)
        dendrogram_path = output_folder + '/ligand_dendrogram.png'

        fig, ax = plt.subplots(figsize=(max(8, n_entries * 0.3), 6))
        dendrogram(
            linkage_matrix,
            labels=entry_labels,
            ax=ax,
            leaf_rotation=90,
            color_threshold=rmsd_cutoff,      # color links below this distance by cluster
            above_threshold_color='lightgray' # links merging above cutoff (between clusters)
        )

        ax.set_xlabel('placer_file, index')
        ax.set_ylabel('RMSD (\u00c5)')
        ax.set_title('Hierarchical clustering of top ligand conformers (average linkage, RMSD)')
        fig.tight_layout()
        fig.savefig(dendrogram_path, dpi=200)
        plt.close(fig)

        # full cluster membership (every entry, not just representatives) -
        # kept on self so callers can look up which placer_files contributed
        # to a given cluster, e.g. for exclusion in a subsequent iteration
        clusters = {}
        for i, cluster_id in enumerate(cluster_ids):
            if cluster_id not in clusters:
                clusters.update({cluster_id: []})
            clusters[cluster_id].append(entry_info[i])
        self.clusters = clusters

        # pick each cluster's representative conformer (either the
        # best-scoring member, or the member closest to the cluster centroid
        # - see _selectClusterRepresentative / self.rep_selection)
        self.cluster_reps = {}
        for cluster_id in clusters:
            cluster_size = len(clusters[cluster_id])
            rep = self._selectClusterRepresentative(clusters[cluster_id])
            self.cluster_reps[cluster_id] = rep + (cluster_size,)


    def _determineBindingSite(self, models):
        """Determines where binding site is.
        Binding Site includes all residues in any of the Placer models
        and the ligand.
        """

        time0 = time.time()
        residues_in_binding_site = {}

        for model in models:
            for chain in model._pdb_hierarchy.only_model().chains():
                chain_id = chain.id.strip()

                if chain_id not in residues_in_binding_site:
                    residues_in_binding_site.update({chain_id: []})

                for residue in chain.residue_groups():
                    res_num = int(residue.resseq)
                    if res_num not in residues_in_binding_site[chain_id]:
                        residues_in_binding_site[chain_id].append(res_num)

        # print(f'found binding pocket in {time.time() - time0} seconds')

        return residues_in_binding_site
    
    def _getBaseBindingSite(self, placer_file):
        """Makes a substructure corresponding to all residues in the binding site
        from residues in the base structure.
        """

        time0 = time.time()
        
        heavy_atom_dict = {'GLY': 4, 'ALA': 5, 'VAL': 7, 'LEU': 8, 'ILE': 8,
                           'PRO': 7, 'PHE': 11, 'MET': 8, 'CYS': 6, 'TRP': 14,
                           'SER': 6, 'THR': 7, 'GLN': 9, 'ASN': 8, 'TYR': 12,
                           'HIS': 10, 'LYS': 9, 'ARG': 11, 'ASP': 8, 'GLU': 9}
        
        base_bindingsite = None

        for chain_id in list(self.binding_site_residues[placer_file].keys()):
            for res_num in self.binding_site_residues[placer_file][chain_id]:
                residue = self.base_structure.extract(f"chain {chain_id} and resid {res_num}")
                    
                #getting residue identity is annoyingly long winded
                base_chain = [ch for ch in self.base_structure._pdb_hierarchy.only_model().chains() if ch.id.strip() == chain_id][0]
                res = [res for res in base_chain.residue_groups() if int(res.resseq) == res_num][0]
                res_identity = res.only_atom_group().resname.strip()

                ##This checks if the residue has all the atoms it should in the supplied cif
                if res_identity in heavy_atom_dict:
                    if residue.natoms != heavy_atom_dict[res_identity]:
                        print(f'{res_identity} {chain_id}{res_num} only has {residue.natoms} instead of an expected {heavy_atom_dict[res_identity]} atoms')

                if base_bindingsite is None:
                    base_bindingsite = residue
                else:
                    base_bindingsite = base_bindingsite.combine(residue)

        # print(f'Built base binding site in {time.time() - time0}')

        return base_bindingsite
    
    def _getBindingSiteConformers(self, models, placer_file):
        """This function gets the coors of all conformers of the binding site for each of the placer models
        Since not all placer models contain all residues of the binding site, if a binding site
        residue is not present in the placer model, it takes it from the base model instead."""

        ##The pdb reader will reorder ligand atoms by alphabetical order. We need to undo this and match them to the input structure
        #Practically, since all we care about is extracting the PLACER coordinates in the right order, we don't need to alter the pdb_hierarchy
        #We just need to change the order of the _atoms for each placer model.
        #We also have to do it for each placer model

        # Get reference ligand atom order from target hierarchy traversal
        ref_lig_atom_names = []
        for chain in self.base_structure._pdb_hierarchy.only_model().chains():
            for res in chain.residue_groups():
                if res.only_atom_group().resname.strip() == 'LIG':
                    for atom_group in res.atom_groups():
                        for atom in atom_group.atoms():
                            ref_lig_atom_names.append(atom.name.strip())

        coor_sets = []
        for j,model in enumerate(models):

            # Split placer model atoms into non-ligand and ligand
            non_lig_atoms = []
            lig_atom_dict = {}
            for atom in model._atoms:
                if atom.parent().resname.strip() == 'LIG':
                    lig_atom_dict[atom.name.strip()] = atom
                else:
                    non_lig_atoms.append(atom)

            # Reorder ligand atoms to match reference
            lig_atoms_reordered = [lig_atom_dict[name] for name in ref_lig_atom_names]

            # Remove last n atoms (the ligand) and reappend in correct order
            n_lig = len(ref_lig_atom_names)

            # Split the atom array
            protein_atoms = model._atoms[:-n_lig]  # all but last n
            lig_atom_dict = {atom.name.strip(): atom for atom in model._atoms[-n_lig:]}
            assert len(lig_atom_dict) == n_lig

            # Reorder ligand atoms to match reference
            lig_atoms_reordered = [lig_atom_dict[name] for name in ref_lig_atom_names]

            # Rebuild _atoms array
            reordered_lig = iotbx.pdb.hierarchy.af_shared_atom(lig_atoms_reordered)
            new_atoms = protein_atoms.deep_copy()
            new_atoms.extend(reordered_lig)
            model._atoms = new_atoms

            #get conformers
            coor_set_list = []
            for chain_id in list(self.binding_site_residues[placer_file].keys()):
                for res_num in self.binding_site_residues[placer_file][chain_id]:
                    
                    #get residue from placer model if possible, else get it from the base mdoel
                    residue = model.extract(f'chain {chain_id} and resid {res_num}')
                    if residue.natoms == 0:
                        residue = self.base_structure.extract(f'chain {chain_id} and resid {res_num}')

                    for i in range(residue.coor.shape[0]):
                        coor_set_list.append(residue.coor[i,:])

            coor_set = np.array(coor_set_list)

            #Check for nans in coor set
            nan_flag = np.isnan(coor_set).any()
            if nan_flag:
                raise ValueError('Something went wrong building the placer conformers. At least 1 conformer has nan for a coor value.')

            coor_sets.append(coor_set)

        return coor_sets
    
    def _write_multimodel_pdb(self, models, output_path):
        with open(output_path, 'w') as out:
            for i, model in enumerate(models, start=1):
                out.write(f"MODEL     {i:4d}\n")
                for atom in model.get_selected_atoms():
                    atom_labels = atom.fetch_labels()
                    out.write("{}\n".format(atom_labels.format_atom_record_group()))
                out.write("ENDMDL\n")
            out.write("END\n")   

    def _convertAndScoreLigand(self, placer_file):
        first_event_map_name = list(self.event_maps.keys())[0] #only use the 1st event map right now, could change
        scaled_bulk_solvent = 0 #from qfit, maybe should be different
        coor_set = self.coor_sets[placer_file]

        #extract ligand from binding site and coor sets
        ligand = self.base_binding_sites[placer_file].extract('resname LIG')
        self.ligand_size = ligand.natoms
        ligand_coor_set = [arr[-self.ligand_size:, :] for arr in coor_set]

        #make bfactor array
        default_bfactor = 20 #can change 

        #make a transformer for this structure
        transformer = get_transformer("qfit", ligand, self.event_maps_models[first_event_map_name])
        
        #convert and score this placer_file
        scores = []
        for coor in ligand_coor_set:
            mask = transformer.get_conformers_mask([coor], self._rmask) 

            for density in transformer.get_conformers_densities([coor],[default_bfactor]):
                model = density[mask]
            np.maximum(model,scaled_bulk_solvent,out=model)

            target = self.event_maps[first_event_map_name].array[mask]

            mse = np.mean((model - target) ** 2)
            scores.append(mse)

        return scores
    
    def _convertAndScoreLigandAllEvents(self, placer_file):
        all_scores = {}
        for event_map_name in list(self.event_maps.keys()):
            scaled_bulk_solvent = 0 #from qfit, maybe should be different
            coor_set = self.coor_sets[placer_file]

            #extract ligand from binding site and coor sets
            ligand = self.base_binding_sites[placer_file].extract('resname LIG')
            self.ligand_size = ligand.natoms
            ligand_coor_set = [arr[-self.ligand_size:, :] for arr in coor_set]

            #make bfactor array
            default_bfactor = 20 #can change 

            #make a transformer for this structure
            transformer = get_transformer("qfit", ligand, self.event_maps_models[event_map_name])
            
            #convert and score this placer_file
            scores = []
            for coor in ligand_coor_set:
                mask = transformer.get_conformers_mask([coor], self._rmask) 

                for density in transformer.get_conformers_densities([coor],[default_bfactor]):
                    model = density[mask]
                np.maximum(model,scaled_bulk_solvent,out=model)

                target = self.event_maps[event_map_name].array[mask]

                mse = np.mean((model - target) ** 2)
                scores.append(mse)
            all_scores.update({event_map_name: scores})
        
        return all_scores



def main():
    p = build_argparser()
    args = p.parse_args()
    placer_files = []
    for file in glob.glob(args.placer_files):
        placer_files.append(file)
    placer_files.sort()

    fit_ligand_files = []
    for file in glob.glob(args.fit_ligand_files):
        fit_ligand_files.append(file)
    fit_ligand_files.sort()

    filter = Filter(args.dataset, placer_files, fit_ligand_files, args.output_folder,
                     args.resolution, args.filter_proportion, args.min_cluster_proportion,
                     args.clash_vdw_scale, args.rep_selection)
    filter.run()




if __name__ == '__main__':
    main()