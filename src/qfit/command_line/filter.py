import argparse
import glob
from pathlib import Path
import time
import numpy as np
import heapq
import tempfile
import os

from qfit import Structure
from qfit import XMap
from qfit.xtal.transformer import get_transformer

import iotbx.pdb

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

                                  
        print(self.scores)
        #get top N
        top_n = heapq.nsmallest(self.n,((val, key, idx) for key, lst in self.scores.items() for idx, val in enumerate(lst)))
        
        #write output files
        output_folder = str(self.dir) + '/' + self.output_folder
        os.makedirs(output_folder, exist_ok=True)
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
                print(entry)
                score = entry[0]
                placer_file = entry[1]
                index = entry[2]

                f.write(f'{placer_file},{index},{score}')
                f.write('\n')

                bs_model = self.base_binding_sites[placer_file].copy()
                bs_model.coor = self.coor_sets[placer_file][index]
                bs_model.b = 20
                bs_models.append(bs_model)
        
        output_folder = str(self.dir) + '/' + self.output_folder
        os.makedirs(output_folder, exist_ok=True)
        output_path = output_folder + '/filtered_models.pdb'
        self._write_multimodel_pdb(bs_models, output_path)   

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
        ligand_size = ligand.natoms
        ligand_coor_set = [arr[-ligand_size:, :] for arr in coor_set]

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
            ligand_size = ligand.natoms
            ligand_coor_set = [arr[-ligand_size:, :] for arr in coor_set]

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