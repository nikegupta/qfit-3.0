import argparse
import numpy as np
import time
from pathlib import Path

from qfit import Structure
from qfit import XMap
from qfit.xtal.transformer import get_transformer

def build_argparser():
    p = argparse.ArgumentParser()
    p.add_argument(
        'dataset',
        type=Path,
        help='Path to pandas dataset')
    p.add_argument(
        'ligand',
        type=Path,
        help='Path to ligand structure file')
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
        "--num_peaks",
        default=5,
        metavar="<int>",
        type=int,
        help="Number of peaks to find (default: 5)",
    )
    p.add_argument(
        "-z",
        "--z_threshold",
        default=5,
        metavar="<float>",
        type=float,
        help="Z-score threshold for peak detection (default: 5)",
    )
    p.add_argument(
        "--sampling",
        required=True,
        help='geometric sampling parameters as an underscore seperated list (ie 3_3_3_3_5_0.5)'
    )
    return p

class LigandPlacer():
    def __init__(self, dataset, ligand_file, resolution, geom_params, num_peaks=5, z_threshold=5):
        # Read in args
        self.dataset = dataset
        self.dataset_name = str(dataset).split('/')[-1]
        self.ligand_file = ligand_file
        self.resolution = resolution
        self.num_peaks = num_peaks
        self.z_threshold = z_threshold
        self.geom_params = geom_params
        self._rmask = 0.5 + self.resolution / 3.0

        #make output folder
        self.output_dir = self.dataset / self.geom_params
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load structures and maps
        self.apo_structure = Structure.fromfile(f'{self.dataset}/{self.dataset_name}-aligned-structure.pdb')
        self.ligand_structure = Structure.fromfile(f'{self.ligand_file}')
        self.zmap = XMap.fromfile(
            f'{self.dataset}/{self.dataset_name}-z_map.native.ccp4',
            resolution=self.resolution
        )
        self._load_event_maps()

        # Initialize a transformer for cartesian/grid conversions
        self.transformer = get_transformer("qfit", self.apo_structure, self.zmap)

    def _load_event_maps(self):
        self.event_maps = {}
        self.event_maps_models = {}
        event_map_files = sorted(self.dataset.glob('*-event_*_*-BDC_*_map.native.ccp4'))
        for event_file in event_map_files:
            # Use full filename as key instead of just event name
            event_name = str(event_file).split('/')[-1]  # e.g., "x01325-1-event_1_1-BDC_0.3_map.native.ccp4"
            self.event_maps[event_name] = XMap.fromfile(str(event_file), resolution=self.resolution)

            # make copies for density steps
            event_map_model = self.event_maps[event_name].zeros_like(self.event_maps[event_name])
            event_map_model.set_space_group("P1")
            self.event_maps_models[event_name] = event_map_model

    def run(self):
        """Fits a ligand to event maps guided by the zmap."""
        self.peaks = self._find_peaks()
        self.centroid_peaks = self._find_event_centroid()
        print(self.centroid_peaks)
        
        best_ligand_score = 10
        merged_structure = self.apo_structure.copy()
        
        # Get ligand center (calculate once)
        ligand_center = self.ligand_structure.coor.mean(axis=0)
        
        for i, (grid_idx, best_peak_coord) in enumerate(self.centroid_peaks):
            print(f"\n=== Peak {i} ===")
            print(f"Placing ligand at: {best_peak_coord}")
            
            # Place ligand center on zmap peak
            self.placed_ligand = self.ligand_structure.copy()
            translation = best_peak_coord - ligand_center
            self.placed_ligand.coor = self.ligand_structure.coor + translation
    
            # Broad geometric sampling of ligand positions
            placed_ligand_coor_set, placed_ligand_b_set = self._geometric_sampling(self.placed_ligand, self.geom_params)
            
            # Detect clashes
            time0 = time.time()
            clash_free_mask = self._detect_clashes(placed_ligand_coor_set, clash_cutoff=0.75)
            
            # Filter to only clash-free conformers
            placed_ligand_coor_set = placed_ligand_coor_set[clash_free_mask]
            placed_ligand_b_set = placed_ligand_b_set[clash_free_mask]
            
            if len(placed_ligand_coor_set) == 0:
                print(f"Peak {i}: All conformers clash, skipping...")
                continue
            print(f"Peak {i}: Using {len(placed_ligand_coor_set)} clash-free conformers")
            print(f"detected clashes in {time.time() - time0}")
            
            # convert ligands to density and score them against event map
            time0 = time.time()
            self._convert(placed_ligand_coor_set, placed_ligand_b_set)
            best_coor_set, ligand_score = self._score_models()
            self.placed_ligand.coor = placed_ligand_coor_set[best_coor_set]
            print(f'converted and scored conformers in {time.time() - time0}')

            # Write out all conformers for this peak
            #commented out because it creates massive files at high sampling rates
            # self._write_conformers(placed_ligand_coor_set, peak_index=i)

            # Check if this ligand is better than all previous ligands
            print(f'{i}, {ligand_score}, {best_coor_set}')
            if ligand_score < best_ligand_score:
                best_ligand_score = ligand_score
                best_ligand = self.placed_ligand
                chosen_peak = grid_idx
            
        # Merge with apo structure
        if best_ligand_score == 10:
            raise ValueError('all ligands had MSE > 10')
        print(f'Merging ligand at grid coor: {chosen_peak}')
        merged_structure = merged_structure.combine(best_ligand)
            
        # Save merged structure
        output_path = f'{self.output_dir}/{self.dataset_name}-ligand_fit.pdb'
        merged_structure.tofile(output_path)
        print(f"Saved to {output_path}")
        print(f"\nSaved {len(self.peaks)} ligands to {output_path}")

    def _grid_to_cartesian(self, grid_idx):
        """
        Converts grid indices to cartesian coordinates
        """
        # Convert tuple to array
        grid_coor = np.array(grid_idx, dtype=float)
        grid_coor += self.transformer.xmap.offset
        grid_coor *= self.transformer.xmap.voxelspacing
        cartesian = np.dot(grid_coor, self.transformer.lattice_to_cartesian.T)
        if not np.allclose(self.transformer.xmap.origin, 0):
            cartesian += self.transformer.xmap.origin
        
        return cartesian

    def _find_symmetry_mate_near_protein(self, peak_coord, protein_center):
        """
        Apply P212121 symmetry operations and unit cell translations to find 
        the symmetry mate closest to the protein.
        
        Args:
            peak_coord: Cartesian coordinates of the peak
            protein_center: Center of the protein structure
            
        Returns:
            best_coord: Cartesian coordinates of the symmetry mate closest to protein
        """
        # Convert peak to fractional coordinates
        peak_frac = self.zmap.unit_cell.orth_to_frac @ peak_coord
        
        # P212121 symmetry operations in fractional coordinates
        symops_frac = [
            lambda xyz: xyz,  # x, y, z
            lambda xyz: np.array([-xyz[0] + 0.5, -xyz[1], xyz[2] + 0.5]),  # -x+1/2, -y, z+1/2
            lambda xyz: np.array([-xyz[0], xyz[1] + 0.5, -xyz[2] + 0.5]),  # -x, y+1/2, -z+1/2
            lambda xyz: np.array([xyz[0] + 0.5, -xyz[1] + 0.5, -xyz[2]])   # x+1/2, -y+1/2, -z
        ]
        
        # Generate all combinations of translations (-1, 0, 1) in x, y, z
        translations = []
        for x in [-1, 0, 1]:
            for y in [-1, 0, 1]:
                for z in [-1, 0, 1]:
                    translations.append(np.array([x, y, z]))
        
        best_distance = float('inf')
        best_coord = None
        best_symop_idx = None
        best_translation = None
        
        # Try all combinations of symmetry operations and translations
        for symop_idx, symop in enumerate(symops_frac):
            # Apply symmetry operation
            sym_frac = symop(peak_frac)
            
            for translation in translations:
                # Apply unit cell translation
                translated_frac = sym_frac + translation
                
                # Convert to Cartesian
                sym_cart = self.zmap.unit_cell.frac_to_orth @ translated_frac
                
                # Calculate distance to protein center
                distance = np.linalg.norm(sym_cart - protein_center)
                
                if distance < best_distance:
                    best_distance = distance
                    best_coord = sym_cart
        
        return best_coord

    def _detect_clashes(self, coor_set, clash_cutoff=0.75):
        """
        Detect clashes between ligand conformers and apo structure backbone atoms.
        
        Args:
            coor_set: 3D numpy array (N_conformers, N_atoms_ligand, 3)
            clash_cutoff: Fraction of sum of VDW radii to use as cutoff (default 0.75)
        
        Returns:
            clash_free_mask: Boolean array of shape (N_conformers,) 
                            True = no clash, False = clash detected
        """
        num_conformers = coor_set.shape[0]
        num_ligand_atoms = coor_set.shape[1]
        
        # Get only backbone atoms from protein (N, CA, C, O)
        backbone_selection = self.apo_structure.select('name', ['N', 'CA', 'C', 'O'])
        backbone_structure = self.apo_structure.extract(backbone_selection)
        
        num_backbone_atoms = backbone_structure.natoms
        
        # Get VDW radii for ligand and backbone
        ligand_vdw = self.ligand_structure.vdw_radius  # Shape: (num_ligand_atoms,)
        backbone_vdw = backbone_structure.vdw_radius    # Shape: (num_backbone_atoms,)
        backbone_coor = backbone_structure.coor         # Shape: (num_backbone_atoms, 3)
        
        # Initialize clash-free mask (assume no clashes initially)
        clash_free_mask = np.ones(num_conformers, dtype=bool)
        
        print(f"Checking clashes for {num_conformers} conformers against {num_backbone_atoms} backbone atoms...")
        
        # Check each conformer
        for conf_idx in range(num_conformers):
            ligand_coor = coor_set[conf_idx]  # Shape: (num_ligand_atoms, 3)
            
            # Compute all pairwise distances between ligand and backbone atoms
            diff = ligand_coor[:, np.newaxis, :] - backbone_coor[np.newaxis, :, :]
            distances = np.linalg.norm(diff, axis=2)  # Shape: (num_ligand_atoms, num_backbone_atoms)
            
            # Compute cutoff distances (sum of VDW radii * cutoff)
            cutoff_distances = (ligand_vdw[:, np.newaxis] + backbone_vdw[np.newaxis, :]) * clash_cutoff
            
            # Check for clashes: any distance less than cutoff
            if np.any(distances < cutoff_distances):
                clash_free_mask[conf_idx] = False
        
        num_clash_free = clash_free_mask.sum()
        num_clashing = num_conformers - num_clash_free
        print(f"Clash detection complete: {num_clash_free} clash-free, {num_clashing} clashing conformers")
        
        return clash_free_mask
    
    def _write_conformers(self, coor_set, peak_index):
        """
        Write all conformers to a multi-model PDB file
        
        Args:
            coor_set: 3D numpy array (N_conformers, N_atoms, 3)
            peak_index: Index of the peak being sampled
        """
        output_path = f'{self.output_dir}/{self.dataset_name}-peak_{peak_index}_conformers.pdb'
        
        num_conformers = coor_set.shape[0]
        
        with open(output_path, 'w') as f:
            for model_num, coor in enumerate(coor_set, start=1):
                # Write MODEL record
                f.write(f"MODEL     {model_num:4d}\n")
                
                # Create a copy of the ligand with new coordinates
                conformer = self.placed_ligand.copy()
                conformer.coor = coor
                
                # Write atoms
                for j, atom in enumerate(conformer.atoms):
                    x, y, z = coor[j]
                    atom_name = atom.name.strip()
                    resname = atom.parent().resname.strip()
                    chain = atom.chain().id.strip()
                    resseq = atom.parent().parent().resseq_as_int()
                    occ = atom.occ
                    bfac = atom.b
                    element = atom.element.strip()
                    
                    f.write(
                        f"HETATM{j+1:5d} {atom_name:^4s} {resname:3s} {chain:1s}{resseq:4d}    "
                        f"{x:8.3f}{y:8.3f}{z:8.3f}{occ:6.2f}{bfac:6.2f}          {element:>2s}\n"
                    )
                
                # Write ENDMDL record
                f.write("ENDMDL\n")

    # def _geometric_sampling(self, placed_ligand, geom_params):
    #     """
    #     Generate conformers by translating and rotating the ligand using spherical coordinates.
        
    #     For each translation direction, apply all possible rotations.
        
    #     Args:
    #         placed_ligand: Structure object with ligand already positioned at peak
    #         geom params splits into:
    #         num_polar_trans: Number of polar angle steps for translation (0 to 180 degrees)
    #         num_azimuthal_trans: Number of azimuthal angle steps for translation (0 to 360 degrees)
    #         num_polar_rot: Number of polar angle steps for rotation (0 to 180 degrees)
    #         num_azimuthal_rot: Number of azimuthal angle steps for rotation (0 to 360 degrees)
    #         translation_range: Max translation in Angstroms
    #         translation_step: Translation increment in Angstroms
        
    #     Returns:
    #         coor_set: 3D numpy array (N_conformers, N_atoms, 3)
    #         b_set: 2D numpy array (N_conformers, N_atoms)
    #     """
    #     import scipy.spatial.transform as sptrans
        
    #     vals = geom_params.split('_')
    #     num_polar_trans, num_azimuthal_trans, num_polar_rot, num_azimuthal_rot = [int(x) for x in vals[:4]]
    #     translation_range, translation_step = float(vals[4]), float(vals[5])

    #     # Get ligand center
    #     ligand_center = placed_ligand.coor.mean(axis=0)
        
    #     # Generate translation distances
    #     translations = np.arange(0, translation_range + translation_step, translation_step)
        
    #     # Calculate number of unique orientations (accounting for pole optimization)
    #     num_trans_orientations = 2 + (num_polar_trans - 2) * num_azimuthal_trans
    #     num_rot_orientations = 2 + (num_polar_rot - 2) * num_azimuthal_rot
        
    #     # Calculate max possible conformers (upper bound)
    #     max_conformers = len(translations) * num_trans_orientations * num_rot_orientations
    #     num_atoms = placed_ligand.coor.shape[0]
        
    #     # Pre-allocate arrays with upper bound
    #     coor_set = np.zeros((max_conformers, num_atoms, 3), dtype=np.float64)
    #     b_set = np.zeros((max_conformers, num_atoms), dtype=np.float64)
        
    #     # Sample polar angles for translation
    #     polar_trans_step = 180.0 / (num_polar_trans - 1) if num_polar_trans > 1 else 180.0
    #     azimuthal_trans_step = 360.0 / num_azimuthal_trans
        
    #     # Sample polar angles for rotation
    #     polar_rot_step = 180.0 / (num_polar_rot - 1) if num_polar_rot > 1 else 180.0
    #     azimuthal_rot_step = 360.0 / num_azimuthal_rot
        
    #     idx = 0
        
    #     # Loop over translation directions
    #     for i_trans in range(num_polar_trans):
    #         theta_trans = i_trans * polar_trans_step
    #         theta_trans_rad = np.deg2rad(theta_trans)
            
    #         # At poles, only sample once for translation
    #         if i_trans == 0 or i_trans == num_polar_trans - 1:
    #             azimuthal_trans_samples = 1
    #         else:
    #             azimuthal_trans_samples = num_azimuthal_trans
            
    #         for j_trans in range(azimuthal_trans_samples):
    #             phi_trans = j_trans * azimuthal_trans_step if azimuthal_trans_samples > 1 else 0
    #             phi_trans_rad = np.deg2rad(phi_trans)
                
    #             # Translation direction vector
    #             translation_direction = np.array([
    #                 np.sin(theta_trans_rad) * np.cos(phi_trans_rad),
    #                 np.sin(theta_trans_rad) * np.sin(phi_trans_rad),
    #                 np.cos(theta_trans_rad)
    #             ])
                
    #             # Apply each translation distance along this direction
    #             for distance in translations:
    #                 # Skip redundant translation directions when distance is 0
    #                 if distance == 0 and not (i_trans == 0 and j_trans == 0):
    #                     continue
                    
    #                 translation_vector = translation_direction * distance
    #                 translated_coor = placed_ligand.coor + translation_vector
                    
    #                 # Now rotate the translated ligand in all orientations
    #                 for i_rot in range(num_polar_rot):
    #                     theta_rot = i_rot * polar_rot_step
    #                     theta_rot_rad = np.deg2rad(theta_rot)
                        
    #                     # At poles, only sample once for rotation
    #                     if i_rot == 0 or i_rot == num_polar_rot - 1:
    #                         azimuthal_rot_samples = 1
    #                     else:
    #                         azimuthal_rot_samples = num_azimuthal_rot
                        
    #                     for j_rot in range(azimuthal_rot_samples):
    #                         phi_rot = j_rot * azimuthal_rot_step if azimuthal_rot_samples > 1 else 0
    #                         phi_rot_rad = np.deg2rad(phi_rot)
                            
    #                         # Rotation direction vector
    #                         rotation_direction = np.array([
    #                             np.sin(theta_rot_rad) * np.cos(phi_rot_rad),
    #                             np.sin(theta_rot_rad) * np.sin(phi_rot_rad),
    #                             np.cos(theta_rot_rad)
    #                         ])
                            
    #                         # Create rotation matrix to align Z-axis with rotation direction
    #                         z_axis = np.array([0, 0, 1])
                            
    #                         if not np.allclose(rotation_direction, z_axis) and not np.allclose(rotation_direction, -z_axis):
    #                             rotation_axis = np.cross(z_axis, rotation_direction)
    #                             rotation_axis_norm = np.linalg.norm(rotation_axis)
                                
    #                             if rotation_axis_norm > 1e-6:
    #                                 rotation_axis = rotation_axis / rotation_axis_norm
    #                                 rotation_angle = np.arccos(np.clip(np.dot(z_axis, rotation_direction), -1.0, 1.0))
    #                                 rotation = sptrans.Rotation.from_rotvec(rotation_axis * rotation_angle)
    #                                 rotation_matrix = rotation.as_matrix()
    #                             else:
    #                                 rotation_matrix = np.eye(3)
    #                         elif np.allclose(rotation_direction, -z_axis):
    #                             rotation_matrix = np.array([
    #                                 [1, 0, 0],
    #                                 [0, -1, 0],
    #                                 [0, 0, -1]
    #                             ])
    #                         else:
    #                             rotation_matrix = np.eye(3)
                            
    #                         # Center the translated coordinates
    #                         translated_center = translated_coor.mean(axis=0)
    #                         centered_coor = translated_coor - translated_center
                            
    #                         # Rotate around the new center
    #                         rotated_coor = centered_coor @ rotation_matrix.T
    #                         rotated_coor += translated_center
                            
    #                         # Store the final coordinates
    #                         coor_set[idx] = rotated_coor
    #                         b_set[idx] = placed_ligand.b
    #                         idx += 1
        
    #     print(f"Generated {idx} conformers "
    #         f"({len(translations)} translations × {num_trans_orientations} translation directions × {num_rot_orientations} rotations)")
        
    #     # Trim to actual size and return
    #     return coor_set[:idx].copy(), b_set[:idx].copy()

    def _geometric_sampling(self, placed_ligand, geom_params): #new version within translation
        """
        Generate conformers by rotating the ligand using spherical coordinates.

        Args:
            placed_ligand: Structure object with ligand already positioned at peak
            geom_params splits into:
            num_polar_rot: Number of polar angle steps for rotation (0 to 180 degrees)
            num_azimuthal_rot: Number of azimuthal angle steps for rotation (0 to 360 degrees)

        Returns:
            coor_set: 3D numpy array (N_conformers, N_atoms, 3)
            b_set: 2D numpy array (N_conformers, N_atoms)
        """
        import scipy.spatial.transform as sptrans

        vals = geom_params.split('_')
        num_polar_rot, num_azimuthal_rot = int(vals[0]), int(vals[1])

        # Calculate number of unique orientations (accounting for pole optimization)
        num_rot_orientations = 2 + (num_polar_rot - 2) * num_azimuthal_rot

        num_atoms = placed_ligand.coor.shape[0]

        # Pre-allocate arrays
        coor_set = np.zeros((num_rot_orientations, num_atoms, 3), dtype=np.float64)
        b_set = np.zeros((num_rot_orientations, num_atoms), dtype=np.float64)

        # Sample polar angles for rotation
        polar_rot_step = 180.0 / (num_polar_rot - 1) if num_polar_rot > 1 else 180.0
        azimuthal_rot_step = 360.0 / num_azimuthal_rot

        idx = 0

        for i_rot in range(num_polar_rot):
            theta_rot = i_rot * polar_rot_step
            theta_rot_rad = np.deg2rad(theta_rot)

            # At poles, only sample once for rotation
            if i_rot == 0 or i_rot == num_polar_rot - 1:
                azimuthal_rot_samples = 1
            else:
                azimuthal_rot_samples = num_azimuthal_rot

            for j_rot in range(azimuthal_rot_samples):
                phi_rot = j_rot * azimuthal_rot_step if azimuthal_rot_samples > 1 else 0
                phi_rot_rad = np.deg2rad(phi_rot)

                # Rotation direction vector
                rotation_direction = np.array([
                    np.sin(theta_rot_rad) * np.cos(phi_rot_rad),
                    np.sin(theta_rot_rad) * np.sin(phi_rot_rad),
                    np.cos(theta_rot_rad)
                ])

                # Create rotation matrix to align Z-axis with rotation direction
                z_axis = np.array([0, 0, 1])

                if not np.allclose(rotation_direction, z_axis) and not np.allclose(rotation_direction, -z_axis):
                    rotation_axis = np.cross(z_axis, rotation_direction)
                    rotation_axis_norm = np.linalg.norm(rotation_axis)

                    if rotation_axis_norm > 1e-6:
                        rotation_axis = rotation_axis / rotation_axis_norm
                        rotation_angle = np.arccos(np.clip(np.dot(z_axis, rotation_direction), -1.0, 1.0))
                        rotation = sptrans.Rotation.from_rotvec(rotation_axis * rotation_angle)
                        rotation_matrix = rotation.as_matrix()
                    else:
                        rotation_matrix = np.eye(3)
                elif np.allclose(rotation_direction, -z_axis):
                    rotation_matrix = np.array([
                        [1, 0, 0],
                        [0, -1, 0],
                        [0, 0, -1]
                    ])
                else:
                    rotation_matrix = np.eye(3)

                # Center, rotate, and restore coordinates
                ligand_center = placed_ligand.coor.mean(axis=0)
                centered_coor = placed_ligand.coor - ligand_center
                rotated_coor = centered_coor @ rotation_matrix.T
                rotated_coor += ligand_center

                coor_set[idx] = rotated_coor
                b_set[idx] = placed_ligand.b
                idx += 1

        print(f"Generated {idx} conformers ({num_rot_orientations} rotations)")

        return coor_set[:idx].copy(), b_set[:idx].copy()

    def _score_models(self):
        """Scores the best ligand conformer using MSE against the target density"""
        best_mse = 10
        best_model = np.nan
        for i, model in enumerate(self._models):
            mse = np.mean((model - self._target) ** 2)
            
            if mse < best_mse:
                best_mse = mse
                best_model = i

        return best_model, best_mse
    
    def _convert(self, placed_ligand_coor_set, placed_ligand_b_set):
        """This function converts the ligand to density for comparison against an event map
        Analogous to _convert in Base_Qfit"""

        # Get first event map name from dictionary
        first_event_map_name = list(self.event_maps.keys())[0]
        
        # Extract 1-BDC value from filename
        import re
        bdc_match = re.search(r'-BDC_([\d.]+)_', first_event_map_name)
        if bdc_match:
            one_minus_bdc = float(bdc_match.group(1))
        else:
            raise ValueError(f"Could not extract 1-BDC value from filename: {first_event_map_name}")
        
        # Calculate scaled bulk solvent value (0.3 is used as a base as it is the default for qfit)
        scaled_bulk_solvent = 0.3 * one_minus_bdc 
        
        # Initialize a transformer for this ligand to do density conversion
        self.ligand_transformer = get_transformer("qfit", self.placed_ligand, 
                                                self.event_maps_models[first_event_map_name])
        
        # Get ligand mask
        mask = self.ligand_transformer.get_conformers_mask(placed_ligand_coor_set, self._rmask) 
        nvalues = mask.sum()

        # Get target
        self._target = self.event_maps[first_event_map_name].array[mask]

        # Get models
        nmodels = len(placed_ligand_coor_set)
        self._models = np.zeros((nmodels, nvalues), float)
        for n, density in enumerate(self.ligand_transformer.get_conformers_densities(
                                    placed_ligand_coor_set, placed_ligand_b_set)):
            model = self._models[n]
            map_values = density[mask]
            model[:] = map_values
            np.maximum(model, scaled_bulk_solvent, out=model)

    def _find_event_centroid(self):
        event_map = self.event_maps[list(self.event_maps.keys())[0]]
        centroid_peaks = []
        protein_center = self.apo_structure.coor.mean(axis=0)
        protein_mask = self.transformer.get_conformers_mask([self.apo_structure.coor],self._rmask)

        for peak in self.peaks:
            peak_coords = peak[0][::-1]  # xyz -> zyx for numpy indexing

            # Flood-fill to find all connected voxels above average density
            # seeded from the peak coordinate
            visited = set()
            queue = [tuple(peak_coords)]

            while queue:
                current = queue.pop()
                if current in visited:
                    continue
                z, y, x = current

                # Bounds check
                if not (0 <= z < event_map.array.shape[0] and
                        0 <= y < event_map.array.shape[1] and
                        0 <= x < event_map.array.shape[2]):
                    continue
                # Threshold check

                if event_map.array[z, y, x] < 0.5 or protein_mask[z, y, x] == True:
                    continue
                visited.add(current)

                # Add 6-connected neighbours
                for dz, dy, dx in [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]:
                    neighbour = (z + dz, y + dy, x + dx)
                    if neighbour not in visited:
                        queue.append(neighbour)

            if not visited:
                continue

            # Centroid as simple mean of zyx coords, returned as xyz
            coords_array = np.array(list(visited))  # shape (N, 3), columns are z, y, x
            centroid_zyx = coords_array.mean(axis=0)
            centroid_xyz = centroid_zyx[::-1]

            centroid_cartesian = self._grid_to_cartesian(centroid_xyz)
            best_centroid_cartesian = self._find_symmetry_mate_near_protein(centroid_cartesian, protein_center)
            centroid_peaks.append((centroid_xyz,best_centroid_cartesian))

        return centroid_peaks


    def _find_peaks(self):
        """
        Finds the n highest peaks in the zmap.
        - Removes symmetry-related duplicates (same z-score)
        - Only includes peaks within 10 Å of Met243 or Leu308 (chain A or B)
        - Continues searching until n_peaks found or no peaks left above threshold
        """
        from scipy.ndimage import maximum_filter
        
        # Find all local maxima above a threshold
        local_max_mask = maximum_filter(self.zmap.array, size=3) == self.zmap.array
        above_threshold_mask = self.zmap.array > self.z_threshold
        peak_mask = local_max_mask & above_threshold_mask
        
        # Get peak indices and values
        peak_indices = np.argwhere(peak_mask)  # Returns (z, y, x) for CCP4 maps
        peak_values = self.zmap.array[peak_mask]
        
        # Sort by peak value (highest first)
        sorted_idx = np.argsort(peak_values)[::-1]
        
        print(f"Found {len(sorted_idx)} total peaks above threshold {self.z_threshold}")
        
        # Get binding site reference atoms (Met243 and Leu308 CA atoms, chains A and B)
        # Build selection string for iotbx
        selection_strings = []
        for chain_id in ['A', 'B']:
            for resseq in [243, 308]:
                selection_strings.append(f"(chain {chain_id} and resseq {resseq} and name CA)")
        
        # Combine with OR
        full_selection_string = " or ".join(selection_strings)
        
        try:
            binding_site_selection = self.apo_structure.select(full_selection_string)
            binding_site_structure = self.apo_structure.extract(binding_site_selection)
            binding_site_coords = binding_site_structure.coor
        except Exception as e:
            binding_site_coords = None
        
        # Get protein center for symmetry mate selection
        protein_center = self.apo_structure.coor.mean(axis=0)
        
        # Track unique z-scores to remove symmetry duplicates
        used_zscores = set()
        unique_peaks = []
        
        # Iterate through ALL peaks until we find n_peaks or run out
        for i in sorted_idx:
            z_score = peak_values[i]
            
            # Round z-score to avoid floating point comparison issues
            z_score_rounded = round(z_score, 3)
            
            # Skip if we've already processed this z-score (symmetry duplicate)
            if z_score_rounded in used_zscores:
                continue
            
            # Convert peak to Cartesian
            z, y, x = peak_indices[i]
            grid_xyz = (x, y, z)
            peak_coord = self._grid_to_cartesian(grid_xyz)
            
            # Find the symmetry mate closest to protein
            best_peak_coord = self._find_symmetry_mate_near_protein(peak_coord, protein_center)
            
            # Check if peak is within 10 Å of binding site
            if binding_site_coords is not None:
                distances = np.linalg.norm(binding_site_coords - best_peak_coord, axis=1)
                min_distance = distances.min()
                
                if min_distance > 10.0:
                    print(f"  Skipping peak (z={z_score:.2f}): {min_distance:.2f} Å from binding site (> 10 Å)")
                    continue
                else:
                    print(f"  Peak accepted: (z={z_score:.2f}): {min_distance:.2f} Å from binding site")
            
            # Add to unique peaks (store grid_xyz, z_score, and best_peak_coord)
            unique_peaks.append((grid_xyz, z_score, best_peak_coord))
            used_zscores.add(z_score_rounded)
            
            # Stop when we have enough unique peaks in the binding site
            if len(unique_peaks) >= self.num_peaks:
                break
        
        if len(unique_peaks) < self.num_peaks:
            print(f"Only found {len(unique_peaks)} peaks (requested {self.num_peaks})")
        
        return unique_peaks


def main():
    p = build_argparser()
    args = p.parse_args()
    placer = LigandPlacer(args.dataset, args.ligand, args.resolution, 
                          args.sampling, args.num_peaks, args.z_threshold)
    placer.run()

if __name__ == '__main__':
    main()
