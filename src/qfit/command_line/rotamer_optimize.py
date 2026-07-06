import argparse
import glob
from pathlib import Path
import time
import numpy as np
import os

from qfit import Structure
from qfit import XMap
from qfit.xtal.transformer import get_transformer
from qfit.samplers import ChiRotator, CBAngleRotator, BisectingAngleRotator

#symetry aware sidechain rmsd calc
def _get_coordinate_rmsd(reference_coordinates, new_coordinate_set, atom_names=None):
    reference_coordinates = np.array(reference_coordinates)
    new_coordinate_set = np.array(new_coordinate_set)
    
    # Build mask to exclude backbone atoms
    backbone_atoms = {"N", "CA", "C", "O"}
    if atom_names is not None:
        sidechain_mask = np.array([name not in backbone_atoms for name in atom_names])
    else:
        sidechain_mask = np.ones(reference_coordinates.shape[0], dtype=bool)
    
    ref_sc = reference_coordinates[sidechain_mask]
    new_sc = new_coordinate_set[:, sidechain_mask, :]
    
    delta = new_sc - ref_sc
    rmsds = np.sqrt(np.square(delta).sum(axis=2).sum(axis=1))
    
    if atom_names is not None:
        atom_names = list(atom_names)
        sc_names = [name for name in atom_names if name not in backbone_atoms]
        flip_pairs = None
        if "CD1" in sc_names and "CD2" in sc_names and "CE1" in sc_names and "CE2" in sc_names:
            flip_pairs = [
                (sc_names.index("CD1"), sc_names.index("CD2")),
                (sc_names.index("CE1"), sc_names.index("CE2")),
            ]
        if flip_pairs is not None:
            flipped = new_sc.copy()
            for i, j in flip_pairs:
                flipped[:, i, :], flipped[:, j, :] = flipped[:, j, :].copy(), flipped[:, i, :].copy()
            delta_flipped = flipped - ref_sc
            rmsds_flipped = np.sqrt(np.square(delta_flipped).sum(axis=2).sum(axis=1))
            rmsds = np.minimum(rmsds, rmsds_flipped)
    
    return min(rmsds)

DEFAULT_RMSD_CUTOFF = 0.2

def build_argparser():
    p = argparse.ArgumentParser()
    p.add_argument(
        'dataset',
        type=Path,
        help='Path to pandas dataset')
    p.add_argument(
        'model_file',
        type=str,
        help='Path to the model file'
    )
    p.add_argument(
        'ligandfit_file',
        type=str,
        help='Path to a ligandift file, doesnt matter which one',
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
    return p

class Uninitialized:
    def __str__(self):
        raise RuntimeError("Variable has not been initialized!")
    def __repr__(self):
        raise RuntimeError("Variable has not been initialized!")

class QFitOptions: #copypasted from qfit.py
    def __init__(self):
        # Sampling options
        self.clash_scaling_factor = 0.75
        self.external_clash = False
        self.dofs_per_iteration = 1
        self.dihedral_stepsize = 6
        self.hydro = False
        self.rmsd_cutoff = DEFAULT_RMSD_CUTOFF

        # QFitRotamericResidueOptions
        # Backbone sampling
        self.sample_backbone = True
        self.neighbor_residues_required = 3
        self.sample_backbone_amplitude = 0.30
        self.sample_backbone_step = 0.1
        self.sample_backbone_sigma = 0.125

        # Sample B-factors
        self.sample_bfactors = True

        # N-CA-CB angle sampling
        self.sample_angle = True
        self.sample_angle_range = 7.5
        self.sample_angle_step = 3.75

        # Rotamer sampling
        self.sample_rotamers = True
        self.rotamer_neighborhood = 24
        self.remove_conformers_below_cutoff = False

class Rotamer_Optimizer():
    def __init__(self, dataset_dir, model_file, ligandfit_file, output_folder, resolution):
        self.dir = dataset_dir
        self.model_file = model_file
        self.ligandfit_file = ligandfit_file
        self.output_path = f"{dataset_dir}/{output_folder}"
        os.makedirs(self.output_path,exist_ok=True)
        self.resolution = resolution
        self.options = QFitOptions()
        self._load_event_maps()
        self._rmask = 0.5 + self.resolution / 3.0 #from qfit

        self.base_structure = Structure.fromfile(self.ligandfit_file)
        self.base_structure = self.base_structure.extract("e", "H", "!=")
        rename = self.base_structure.extract("name", "OXT", "==")
        rename.name = "O"
        self.base_structure = self.base_structure.extract("name", "OXT", "!=").combine(rename)

        self.trim = 5
        self.accept = 1

        self.rscc_cutoff = 0.4

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
        models = Structure.fromfile(self.model_file).split_models()
        self.bs_residues = self._determineBindingSite(models)
        print(self.__dict__)
        # rsccs_all_events = {}
        rsccs = {}
        base_rsccs = {}

        residue_coor_sets = {}
        base_coor_sets = {}
        for chain_id in self.bs_residues:
            for resnum in self.bs_residues[chain_id]:
                current_residue = Uninitialized()
                from_model = np.nan
                retrieved_flag = False

                for i,model in enumerate(models):
                    if not retrieved_flag:
                        try:
                            chain = model[chain_id]
                            current_residue = chain.conformers[0][resnum]
                            print(f'retrieved residue {chain_id},{resnum} from model {i}: {current_residue}')
                            retrieved_flag = True
                            from_model = i
                        except:
                            print(f'couldnt get residue {chain_id},{resnum} from model {i}, trying model {i+1}')

                time0 = time.time()

                #make copy of residue
                if current_residue.type == 'rotamer-residue':
                    (chainid, resi, icode) = current_residue.identifier_tuple
                    resi_selstr = f"chain {chainid} and resi {resi}"
                    if icode:
                        resi_selstr += f" and icode {icode}"
                    structure_new = models[from_model].copy()
                    structure_resi = structure_new.extract(resi_selstr)
                    chain = structure_resi[chainid]
                    conformer = chain.conformers[0]
                    self.current_residue = conformer[current_residue.id]

                    #get rscc/coors for base residue
                    self._coor_set = [self.current_residue.coor]
                    base_rsccs.update({(chainid,resi): self._calc_rscc_all_events()})
                    base_coor_sets.update({(chain_id,resi): self._coor_set})

                    #sample ca-b-y for aromatics
                    self._sample_angle()

                    #sample sidechains chi
                    self._sample_sidechains()
                    
                    #score sidecains to top 1
                    self._convert_and_score_rotamer(self.accept)
                    residue_coor_sets.update({(chain_id,resi): self._coor_set})
                    print(f'residue scored in {time.time() - time0}')

                    #get rscc for top residue
                    rsccs.update({(chainid,resi): self._calc_rscc_all_events()})

                    #legacy code for different evaluations
                    # rsccs = self._calc_rscc_all_events()
                    # rsccs_all_events.update({(chainid, resi): rsccs})
                    # self._write_multimodel_pdb_residue(self.current_residue, self._coor_set, self.output_path + '/test.pdb')

        for model in models:
            self._update_coords(model, residue_coor_sets, base_coor_sets, rsccs)

        output = self.output_path + '/rotamer_optimized.pdb'
        self._write_multimodel_pdb(models, output)

        rscc_output = self.output_path + '/rscc.csv'
        with open(rscc_output,'w+') as f:
            f.write('residue,optimized_rscc,base_rscc,optmized?')
            f.write('\n')
            for residue_tuple in list(rsccs.keys()):
                f.write(f'{residue_tuple[0]}{residue_tuple[1]},')
                f.write(f'{rsccs[residue_tuple]},')
                f.write(f'{base_rsccs[residue_tuple]},')
                if rsccs[residue_tuple] >= self.rscc_cutoff:
                    f.write('yes')
                else:
                    f.write('no')
                f.write('\n')

        # rscc_output = self.output_path + '/rscc.csv'
        # with open(rscc_output,'w+') as f:
        #     f.write('residue,rsccs')
        #     f.write('\n')
        #     for residue_tuple in list(rsccs_all_events.keys()):
        #         f.write(f'{residue_tuple[0]}{residue_tuple[1]},')
        #         for rscc in rsccs_all_events[residue_tuple]:
        #             f.write(f'{rscc}_')
        #         f.write('\n')


    
    def _update_coords(self, model, residue_coor_sets, base_coor_sets, rsccs):
        new_coor = model.coor.copy()
        atom_index = 0
        for chain in model._pdb_hierarchy.only_model().chains():
            chain_id = chain.id.strip()
            for residue_group in chain.residue_groups():
                resi = int(residue_group.resseq)
                n_atoms = sum(
                    len(atom_group.atoms()) for atom_group in residue_group.atom_groups()
                )
                key = (chain_id, resi)
                if key in residue_coor_sets:
                    rscc = rsccs.get(key, 0)
                    if rscc >= self.rscc_cutoff:
                        new_coor[atom_index: atom_index + n_atoms] = residue_coor_sets[key][0]
                    elif key in base_coor_sets:
                        new_coor[atom_index: atom_index + n_atoms] = base_coor_sets[key][0]
                atom_index += n_atoms
        model.coor = new_coor

    def _write_multimodel_pdb(self, models, output_path):
        with open(output_path, 'w') as out:
            for i, model in enumerate(models, start=1):
                out.write(f"MODEL     {i:4d}\n")
                for atom in model.get_selected_atoms():
                    atom_labels = atom.fetch_labels()
                    out.write("{}\n".format(atom_labels.format_atom_record_group()))
                out.write("ENDMDL\n")
            out.write("END\n")  

    def _write_multimodel_pdb_residue(self, residue, coor_set, output_path):
        with open(output_path, 'w') as out:
            for i, coor in enumerate(coor_set, start=1):
                residue.coor = coor
                out.write(f"MODEL     {i:4d}\n")
                for atom in residue.get_selected_atoms():
                    atom_labels = atom.fetch_labels()
                    out.write("{}\n".format(atom_labels.format_atom_record_group()))
                out.write("ENDMDL\n")
            out.write("END\n")

    #this function is an editted version of the code from QfitRotamer
    def _sample_sidechains(self):
        print(f"{self.current_residue.resn[0]}, {self.current_residue.resi[0]}")
        opt = self.options

        if self.current_residue.resn[0] != "PRO":
            sampling_window = np.arange(
                -opt.rotamer_neighborhood,
                opt.rotamer_neighborhood + opt.dihedral_stepsize,
                opt.dihedral_stepsize,
            )
        else:
            sampling_window = [0]

        rotamers = self.current_residue.rotamers
        rotamers.append([self.current_residue.get_chi(i) for i in range(1, self.current_residue.nchi + 1)])

        for chi_index in range(1, self.current_residue.nchi + 1):

            new_coor_set = []
            for coor in self._coor_set:
                self.current_residue.coor = coor
                chis = [self.current_residue.get_chi(i) for i in range(1, chi_index)]
                for rotamer in rotamers:

                # for rotamer in rotamers:
                    if not self.is_same_rotamer(rotamer, chis):
                        continue

                    self.current_residue.set_chi(chi_index, rotamer[chi_index - 1])
                    chi_rotator = ChiRotator(self.current_residue, chi_index)

                    for angle in sampling_window:
                        chi_rotator(angle)
                        if new_coor_set:
                            if _get_coordinate_rmsd(self.current_residue.coor, new_coor_set, self.current_residue.name) >= DEFAULT_RMSD_CUTOFF:
                                new_coor_set.append(self.current_residue.coor.copy())
                        else:
                            new_coor_set.append(self.current_residue.coor.copy())

            print(f'number of conformers to score: {len(new_coor_set)}')
            self._coor_set = new_coor_set
            self._convert_and_score_rotamer(self.trim)

    #this function is largely copy pasted from qfit_rotameric_residue with edits to work with my objects
    def _sample_angle(self):
        # Only operate on aromatics!
        if self.current_residue.resn[0] not in ("TRP", "TYR", "PHE", "HIS"):
            return

        # Define sampling range
        angles = np.arange(
            -self.options.sample_angle_range,
            self.options.sample_angle_range + self.options.sample_angle_step,
            self.options.sample_angle_step,
        )

        # Commence sampling, building on each existing conformer in self._coor_set
        new_coor_set = []
        for coor in self._coor_set:
            self.current_residue.coor = coor
            # Initialize rotator
            perp_rotator = CBAngleRotator(self.current_residue)
            # Rotate about the axis perpendicular to CB-CA and CB-CG vectors
            for perp_angle in angles:
                perp_rotator(perp_angle)
                coor_rotated = self.current_residue.coor
                # Initialize rotator
                bisec_rotator = BisectingAngleRotator(self.current_residue)
                # Rotate about the axis bisecting the CA-CA-CG angle for each angle you sample across the perpendicular axis
                for bisec_angle in angles:
                    self.current_residue.coor = coor_rotated  # Ensure that the second rotation is applied to the updated coordinates from first rotation
                    bisec_rotator(bisec_angle)
                    coor = self.current_residue.coor

                    # Valid, non-clashing conformer found!
                    new_coor_set.append(self.current_residue.coor)

        # Update sampled coords
        self._coor_set = new_coor_set
        self._convert_and_score_rotamer(self.trim)

    def is_same_rotamer(self, rotamer, chis):
        dchi_max = 360 - self.options.rotamer_neighborhood
        for curr_chi, rotamer_chi in zip(chis, rotamer):
            delta_chi = abs(curr_chi - rotamer_chi)
            if dchi_max > delta_chi > self.options.rotamer_neighborhood + 1e-6:
                return False
        return True
    
    def _convert_and_score_rotamer(self, n):
        first_event_map_name = list(self.event_maps.keys())[0] #only use the 1st event map right now, could change
        scaled_bulk_solvent = 0 #from qfit, maybe should be different

        (chainid, resi, icode) = self.current_residue.identifier_tuple
        
        #get residue from base structure
        residue = self.base_structure.extract(f"chain {chainid} and resi {resi}")
        
        #make bfactor array
        default_bfactor = 20
        bfactor_array = []
        for i in range(len(self._coor_set)):
            bfactor_array.append(default_bfactor)

        #initialize transformer
        transformer = get_transformer("qfit", residue, self.event_maps_models[first_event_map_name])
        
        #convert and score this set of rotamers
        scores = []
        rsccs = []
        mask = transformer.get_conformers_mask(self._coor_set, self._rmask)
        target = self.event_maps[first_event_map_name].array[mask]
        for density in transformer.get_conformers_densities(self._coor_set, bfactor_array):
            model = density[mask]
            np.maximum(model, scaled_bulk_solvent, out=model)
            mse = np.mean((model - target) ** 2)
            scores.append(mse)
            
            correlation_matrix = np.corrcoef(model, target)
            rscc = correlation_matrix[0, 1]
            rsccs.append(rscc)

        # Sort by score ascending and filter down
        sorted_indices = np.argsort(scores)
        top_indices = sorted_indices[:n]
        self._coor_set = [self._coor_set[i] for i in top_indices]
        self._rsccs = [rsccs[i] for i in top_indices]

    def _calc_rscc_all_events(self):
        scaled_bulk_solvent = 0 #from qfit, maybe should be different
        rsccs = []
        for event_map_name in list(self.event_maps.keys()): 

            (chainid, resi, icode) = self.current_residue.identifier_tuple
            
            #get residue from base structure
            residue = self.base_structure.extract(f"chain {chainid} and resi {resi}")
            
            #make bfactor array
            default_bfactor = 20
            bfactor_array = []
            for i in range(len(self._coor_set)):
                bfactor_array.append(default_bfactor)

            #initialize transformer
            transformer = get_transformer("qfit", residue, self.event_maps_models[event_map_name])
            
            #convert and score this set of rotamers
            mask = transformer.get_conformers_mask(self._coor_set, self._rmask)
            target = self.event_maps[event_map_name].array[mask]
            for density in transformer.get_conformers_densities(self._coor_set, bfactor_array):
                model = density[mask]         
                np.maximum(model, scaled_bulk_solvent, out=model)  
                correlation_matrix = np.corrcoef(model, target)
                rscc = correlation_matrix[0, 1]
                rsccs.append(rscc)

        top_rscc = max(rsccs)

        return top_rscc
        # return rsccs

    def _determineBindingSite(self, models):
        """Determines where binding site is.
        Binding Site includes all residues in any of the Placer models
        and the ligand.
        """

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


        return residues_in_binding_site
    
def main():
    args = build_argparser().parse_args()
    ro = Rotamer_Optimizer(args.dataset, args.model_file, args.ligandfit_file, args.output_folder, args.resolution)
    ro.run()

if __name__ == '__main__':
    main()