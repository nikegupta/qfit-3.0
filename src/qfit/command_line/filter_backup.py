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
        "--filter_number",
        default=100,
        metavar="<float>",
        type=int,
        help="number of residues to filter down to",
    )
    return p

class Filter():
    def __init__(self, dataset_dir, placer_files, fit_ligand_files, output_folder, resolution, filter_number):
        self.dir = dataset_dir
        self.placer_files = placer_files
        self.fit_ligand_files = fit_ligand_files
        self.output_folder = output_folder
        self.resolution = resolution

        self._rmask = 0.5 + self.resolution / 3.0 #from qfit
        self.n = filter_number #number of models to filter down to
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
        """
        All print() output produced during this method is mirrored to
        output_folder/log.txt in addition to the console.
        """
        # Resolve and create the output folder up front (rather than partway
        # through, as before) so log.txt can capture everything from the very
        # first print() onward, including the per-placer_file progress prints
        # in the main loop below.
        output_folder = str(self.dir) + '/' + self.output_folder
        os.makedirs(output_folder, exist_ok=True)
        log_path = output_folder + '/log.txt'
        log_file = open(log_path, 'w')
        original_stdout = sys.stdout
        sys.stdout = _Tee(original_stdout, log_file)

        try:
            self.binding_site_residues = {}
            self.base_binding_sites = {}
            self.coor_sets = {}
            self.scores = {}
            self.all_scores = {}

            #main loop over placer structures
            for placer_file, fit_ligand_file in zip(self.placer_files, self.fit_ligand_files):
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

                # #look at all event map scores
                # self.all_scores.update({placer_file: self._convertAndScoreLigandAllEvents(placer_file)})

            #get top N
            top_n = heapq.nsmallest(self.n,((val, key, idx) for key, lst in self.scores.items() for idx, val in enumerate(lst)))

            #write output files
            output_path = output_folder + '/filtered_models.pdb'
            output_csv = output_folder + '/scores.csv'
            output_summary = output_folder + '/top_scores.csv'

            # output_event_csv = output_folder + '/event_map_scores.csv'
            # with open(output_event_csv, 'w+') as f:
            #     f.write('eventmap,placer_file,index,mse')
            #     f.write('\n')
            #     for placer_file in list(self.all_scores.keys()):
            #         for event_map in list(self.all_scores[placer_file].keys()):
            #             for i,score in enumerate(self.all_scores[placer_file][event_map]):
            #                 f.write(f'{event_map},{placer_file},{i},{score}')
            #                 f.write('\n')

            #write outputcsv
            with open(output_csv, 'w+') as f:
                f.write('placer_file,index,mse')
                f.write('\n')
                for placer_file in self.placer_files:
                    for i,score in enumerate(self.scores[placer_file]):
                        f.write(f'{placer_file},{i},{score}')
                        f.write('\n')

            #write multimodel output and score csv
            with open(output_summary, 'w+') as f:
                f.write('placer_file,index,mse')
                f.write('\n')
                bs_models = []
                for entry in top_n:
                    # print(entry)
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

            #now write out spatially clustered models
            self._spatialClustering(top_n)
            self._calcRSCCofClusters()

            print(f'number of reps before filtering: {len(self.cluster_reps)}')

            # snapshot the full, unfiltered set of cluster reps and their rsccs
            # before the RSCC cutoff or per-placer_file de-duplication below
            # mutate self.cluster_reps, so the full candidate set stays inspectable
            all_cluster_reps_csv = output_folder + '/all_cluster_reps.csv'
            self._write_cluster_reps_csv(self.cluster_reps, self.cluster_rsccs, all_cluster_reps_csv)
            print(f'all (unfiltered) cluster reps written to {all_cluster_reps_csv}')

            #filter cluster_reps by rscc
            rscc_cluster_reps = {}
            for cluster_id in self.cluster_reps:
                if self.cluster_rsccs[cluster_id] > RSCC_CUTOFF:
                    rscc_cluster_reps.update({cluster_id: self.cluster_reps[cluster_id]})
            self.cluster_reps = rscc_cluster_reps

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
                            if self.cluster_rsccs[cluster_id] > best_rscc:
                                best_rscc = self.cluster_rsccs[cluster_id]
                                best_cluster_id = key

                    filtered_cluster_reps.update({best_cluster_id: self.cluster_reps[best_cluster_id]})
            self.cluster_reps = filtered_cluster_reps

            print(f'number of reps after file filtering: {len(self.cluster_reps)}')

            #output cluster models - this csv contains only the final,
            #filtered/accepted cluster representatives
            cluster_summary = output_folder + '/cluster_reps.csv'
            self._write_cluster_reps_csv(self.cluster_reps, self.cluster_rsccs, cluster_summary)

            #write out full clustering information for every input placer model
            #conformer (every placer_file/index pair that was scored), not just
            #the ones that made the top-N cut and were actually clustered -
            #conformers outside the top N get an empty 'cluster' value since
            #_spatialClustering never considered them
            cluster_members_csv = output_folder + '/cluster_members.csv'
            self._write_cluster_members_csv(self.clusters, self.scores, cluster_members_csv)
            print(f'full cluster membership for every input placer model written to {cluster_members_csv}')

            cluster_models = []
            for cluster_id in self.cluster_reps:
                placer_file = self.cluster_reps[cluster_id][1]
                index = self.cluster_reps[cluster_id][2]

                cluster_model = self.base_binding_sites[placer_file].copy()
                cluster_model.coor = self.coor_sets[placer_file][index]
                cluster_model.b = 20
                cluster_models.append(cluster_model)

            cluster_model_path = output_folder + '/cluster_rep_models.pdb'
            self._write_multimodel_pdb(cluster_models,cluster_model_path)
        finally:
            sys.stdout = original_stdout
            log_file.close()

    def _write_cluster_reps_csv(self, cluster_reps, cluster_rsccs, path):
        """
        Writes a placer_file,index,mse,cluster,rscc,num_members csv for the given
        cluster_reps/cluster_rsccs dicts. Used both for the full unfiltered set of
        cluster representatives and for the final filtered set, so both csvs share
        the same columns and can be compared directly - including which
        placer_file and conformer index within it each cluster rep came from.
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

    def _write_cluster_members_csv(self, clusters, scores, path):
        """
        Writes a cluster,placer_file,index,mse csv covering every input placer
        model conformer that was scored (every placer_file/index pair in
        `scores`), not just the ones that made the top-N cut and were actually
        spatially clustered.

        Conformers that were part of a spatial cluster get that cluster's id
        in the 'cluster' column; conformers that were never passed to
        _spatialClustering (because they didn't make the top-N score cut) get
        an empty 'cluster' value, so the full set of input conformers and
        their scores stays inspectable in one place, with cluster membership
        shown wherever it applies.

        `clusters` is self.clusters as built by _spatialClustering: a dict of
        cluster_id -> list of (score, placer_file, index, ligand_coor) tuples.
        `scores` is self.scores: a dict of placer_file -> list of mse scores,
        one per conformer index, covering every input placer model.
        """
        cluster_of = {}
        for cluster_id, members in clusters.items():
            for score, placer_file, index, ligand_coor in members:
                cluster_of[(placer_file, index)] = cluster_id

        with open(path, 'w+') as f:
            f.write('placer_file,index,mse,cluster')
            f.write('\n')
            for placer_file, score_list in scores.items():
                for index, mse in enumerate(score_list):
                    cluster_id = cluster_of.get((placer_file, index), '')
                    f.write(f'{placer_file},{index},{mse},{cluster_id}')
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
        
    def _spatialClustering(self, top_n):
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

        # write out the dendrogram
        output_folder = str(self.dir) + '/' + self.output_folder
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

        clusters = {}
        for i, cluster_id in enumerate(cluster_ids):
            if cluster_id not in clusters:
                clusters.update({cluster_id: []})
            clusters[cluster_id].append(entry_info[i])
        self.clusters = clusters

        self.cluster_reps = {}
        for cluster_id in clusters:
            cluster_size = len(clusters[cluster_id])

            best_score = 100
            for entry in clusters[cluster_id]:
                if cluster_id not in self.cluster_reps:
                    self.cluster_reps.update({cluster_id: entry + (cluster_size,)})

                current_score = entry[0]
                if current_score < best_score:
                    best_score = current_score
                    self.cluster_reps[cluster_id] = entry + (cluster_size,)


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

    filter = Filter(args.dataset, placer_files, fit_ligand_files, args.output_folder, args.resolution, args.filter_number)
    filter.run()




if __name__ == '__main__':
    main()