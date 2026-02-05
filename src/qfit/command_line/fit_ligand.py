import argparse
import numpy as np
import time
from pathlib import Path

from qfit import XMap
from qfit import Structure
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
    return p

class LigandFit():
    def __init__(self, dataset, ligand_file, resolution):
        # Read in args
        self.dataset = dataset
        self.dataset_name = str(dataset).split('/')[-1]
        self.ligand_file = ligand_file
        self.resolution = resolution
        
        # Load structures and maps
        self.apo_structure = Structure.fromfile(f'{self.dataset}/{self.dataset_name}-aligned-structure.pdb')
        self.ligand_structure = Structure.fromfile(f'{self.ligand_file}')
        self.zmap = XMap.fromfile(
            f'{self.dataset}/{self.dataset_name}-z_map.native.ccp4',
            resolution=self.resolution
        )
        self._load_event_maps()

        # FIX THE ORIGIN
        # The Z-map's origin should be shifted by one unit cell in z
        #For some reason the origin is not what the file reader says it is.
        self.zmap.origin = np.array([0, 0, -self.zmap.unit_cell.c])

        #Initialize a transformer for cartesian/grid conversions
        self.transformer = get_transformer("qfit",self.apo_structure,self.zmap)

        #Other params
        self.z_threshold=5
        self.num_peaks=5

    def _load_event_maps(self):
        self.event_maps = {}
        self.event_maps_models = {}
        event_map_files = sorted(self.dataset.glob('*-event_*_*-BDC_*_map.native.ccp4'))
        for event_file in event_map_files:
            # Use full filename as key instead of just event name
            event_name = str(event_file).split('/')[-1]  # e.g., "x01325-1-event_1_1-BDC_0.3_map.native.ccp4"
            self.event_maps[event_name] = XMap.fromfile(str(event_file), resolution=self.resolution)
            #shift origin by z
            self.event_maps[event_name].origin = np.array([0, 0, -self.event_maps[event_name].unit_cell.c])

            #make copies for density steps
            event_map_model = self.event_maps[event_name].zeros_like(self.event_maps[event_name])
            event_map_model.set_space_group("P1")
            self.event_maps_models[event_name] = event_map_model


    def run(self):
        """Fits a ligand to event maps guided by the zmap."""
        self.peaks = self._find_peaks()
        print(self.peaks)
        
        best_ligand_score = 10
        merged_structure = self.apo_structure.copy()
        for i, (grid_idx, z_score) in enumerate(self.peaks):

            # Convert grid to Cartesian
            peak_coord = self._grid_to_cartesian(grid_idx)
            
            # Get ligand center
            ligand_center = self.ligand_structure.coor.mean(axis=0)
            
            # Place ligand center on zmap peak
            self.placed_ligand = self.ligand_structure.copy()
            translation = peak_coord - ligand_center
            self.placed_ligand.coor = self.ligand_structure.coor + translation #ligands will have Bfactor of 20 by default, can change later if necessary
    
            #Broad geometric sampling of ligand positions and
            placed_ligand_coor_set, placed_ligand_b_set = self._geometric_sampling(self.placed_ligand)
            
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
            
            #convert ligands to density and score them against event map
            time0 = time.time()
            self._convert(placed_ligand_coor_set,placed_ligand_b_set)
            best_coor_set, ligand_score = self._score_models()
            self.placed_ligand.coor = placed_ligand_coor_set[best_coor_set]
            print(f'converted and scored conformers in {time.time() - time0}')

            # Write out all conformers for this peak
            self._write_conformers(placed_ligand_coor_set, peak_index=i)

            #Check if this ligand is better than all previous ligands
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
        output_path = f'{self.dataset}/{self.dataset_name}-ligand_fit.pdb'
        merged_structure.tofile(output_path)
        print(f"Saved to {output_path}")

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
        output_path = f'{self.dataset}/{self.dataset_name}-peak_{peak_index}_conformers.pdb'
        
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

    def _geometric_sampling(self, placed_ligand, 
                        num_polar_trans=6, num_azimuthal_trans=12,
                        num_polar_rot=9, num_azimuthal_rot=18,
                        translation_range=5, translation_step=0.5):
        """
        Generate conformers by translating and rotating the ligand using spherical coordinates.
        
        For each translation direction, apply all possible rotations.
        
        Args:
            placed_ligand: Structure object with ligand already positioned at peak
            num_polar_trans: Number of polar angle steps for translation (0 to 180 degrees)
            num_azimuthal_trans: Number of azimuthal angle steps for translation (0 to 360 degrees)
            num_polar_rot: Number of polar angle steps for rotation (0 to 180 degrees)
            num_azimuthal_rot: Number of azimuthal angle steps for rotation (0 to 360 degrees)
            translation_range: Max translation in Angstroms
            translation_step: Translation increment in Angstroms
        
        Returns:
            coor_set: 3D numpy array (N_conformers, N_atoms, 3)
            b_set: 2D numpy array (N_conformers, N_atoms)
        """
        import scipy.spatial.transform as sptrans
        
        # Get ligand center
        ligand_center = placed_ligand.coor.mean(axis=0)
        
        # Generate translation distances
        translations = np.arange(0, translation_range + translation_step, translation_step)
        
        # Calculate number of unique orientations (accounting for pole optimization)
        num_trans_orientations = 2 + (num_polar_trans - 2) * num_azimuthal_trans
        num_rot_orientations = 2 + (num_polar_rot - 2) * num_azimuthal_rot
        
        # Calculate max possible conformers (upper bound)
        max_conformers = len(translations) * num_trans_orientations * num_rot_orientations
        num_atoms = placed_ligand.coor.shape[0]
        
        # Pre-allocate arrays with upper bound
        coor_set = np.zeros((max_conformers, num_atoms, 3), dtype=np.float64)
        b_set = np.zeros((max_conformers, num_atoms), dtype=np.float64)
        
        # Sample polar angles for translation
        polar_trans_step = 180.0 / (num_polar_trans - 1) if num_polar_trans > 1 else 180.0
        azimuthal_trans_step = 360.0 / num_azimuthal_trans
        
        # Sample polar angles for rotation
        polar_rot_step = 180.0 / (num_polar_rot - 1) if num_polar_rot > 1 else 180.0
        azimuthal_rot_step = 360.0 / num_azimuthal_rot
        
        idx = 0
        
        # Loop over translation directions
        for i_trans in range(num_polar_trans):
            theta_trans = i_trans * polar_trans_step
            theta_trans_rad = np.deg2rad(theta_trans)
            
            # At poles, only sample once for translation
            if i_trans == 0 or i_trans == num_polar_trans - 1:
                azimuthal_trans_samples = 1
            else:
                azimuthal_trans_samples = num_azimuthal_trans
            
            for j_trans in range(azimuthal_trans_samples):
                phi_trans = j_trans * azimuthal_trans_step if azimuthal_trans_samples > 1 else 0
                phi_trans_rad = np.deg2rad(phi_trans)
                
                # Translation direction vector
                translation_direction = np.array([
                    np.sin(theta_trans_rad) * np.cos(phi_trans_rad),
                    np.sin(theta_trans_rad) * np.sin(phi_trans_rad),
                    np.cos(theta_trans_rad)
                ])
                
                # Apply each translation distance along this direction
                for distance in translations:
                    # Skip redundant translation directions when distance is 0
                    if distance == 0 and not (i_trans == 0 and j_trans == 0):
                        continue
                    
                    translation_vector = translation_direction * distance
                    translated_coor = placed_ligand.coor + translation_vector
                    
                    # Now rotate the translated ligand in all orientations
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
                            
                            # Center the translated coordinates
                            translated_center = translated_coor.mean(axis=0)
                            centered_coor = translated_coor - translated_center
                            
                            # Rotate around the new center
                            rotated_coor = centered_coor @ rotation_matrix.T
                            rotated_coor += translated_center
                            
                            # Store the final coordinates
                            coor_set[idx] = rotated_coor
                            b_set[idx] = placed_ligand.b
                            idx += 1
        
        print(f"Generated {idx} conformers "
            f"({len(translations)} translations × {num_trans_orientations} translation directions × {num_rot_orientations} rotations)")
        
        # Trim to actual size and return
        return coor_set[:idx].copy(), b_set[:idx].copy()

    def _score_models(self):
        """Scores the best ligand conformer using MSE against the target density"""
        best_mse = 10
        best_model = np.nan
        for i,model in enumerate(self._models):
            mse = np.mean((model - self._target) ** 2)
            
            if mse < best_mse:
                best_mse = mse
                best_model = i

        return best_model,best_mse
    
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
        mask = self.ligand_transformer.get_conformers_mask(placed_ligand_coor_set, rmax=1)  # default rmask is 1
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

    def _grid_to_cartesian(self, grid_idx):
        """
        Converts grid indices to cartesian coordinates
        This is the inverse of _coor_to_grid_coor from the transformer
        """
        # Convert tuple to array
        grid_coor = np.array(grid_idx, dtype=float)
        grid_coor += self.transformer.xmap.offset
        grid_coor *= self.transformer.xmap.voxelspacing
        cartesian = np.dot(grid_coor, self.transformer.lattice_to_cartesian.T)
        if not np.allclose(self.transformer.xmap.origin, 0):
            cartesian += self.transformer.xmap.origin
        
        return cartesian
    
    def _cartesian_to_grid(self, cartesian_idx):
        """Convert from cartesian coords to grid indices. Copied from _coor_to_grid_coor"""
        if np.allclose(self.transformer.xmap.origin, 0):
            cartesian_idx = cartesian_idx
        else:
            cartesian_idx = cartesian_idx - self.transformer.xmap.origin
        grid_coor = np.dot(cartesian_idx, self.transformer.cartesian_to_lattice.T)
        grid_coor /= self.transformer.xmap.voxelspacing
        grid_coor -= self.transformer.xmap.offset

        return grid_coor
        
    def _find_peaks(self):
        """Finds the n highest peaks in the zmap, removing symmetry-related duplicates"""
        from scipy.ndimage import maximum_filter

        # Find all local maxima above a threshold
        local_max_mask = maximum_filter(self.zmap.array, size=3) == self.zmap.array
        above_threshold_mask = self.zmap.array > self.z_threshold
        peak_mask = local_max_mask & above_threshold_mask

        # Filter and score local maxima
        peak_indices = np.argwhere(peak_mask)  # Returns (z, y, x) for CCP4 maps
        peak_values = self.zmap.array[peak_mask]
        sorted_idx = np.argsort(peak_values)[::-1]
        
        # Get protein center for distance calculations
        protein_center = self.apo_structure.coor.mean(axis=0)
        
        # Group peaks by z-score and keep only closest to protein
        unique_peaks = []
        used_zscores = set()
        
        for i in sorted_idx:
            z_score = peak_values[i]
            
            # Round z-score to avoid floating point comparison issues
            z_score_rounded = round(z_score, 3)
            
            # If we've already processed this z-score, skip it
            if z_score_rounded in used_zscores:
                continue
            
            # Find all peaks with this same z-score (symmetry mates)
            same_zscore_indices = sorted_idx[np.isclose(peak_values[sorted_idx], z_score, atol=0.001)]
            
            # Convert all symmetry-related peaks to Cartesian and find closest to protein
            min_distance = float('inf')
            best_peak = None
            
            for idx in same_zscore_indices:
                z, y, x = peak_indices[idx]
                grid_xyz = (x, y, z)
                
                # Convert to Cartesian
                peak_cart = self._grid_to_cartesian(grid_xyz)
                
                # Calculate distance to protein center
                distance = np.linalg.norm(peak_cart - protein_center)
                
                if distance < min_distance:
                    min_distance = distance
                    best_peak = (grid_xyz, z_score)
            
            unique_peaks.append(best_peak)
            used_zscores.add(z_score_rounded)
            
            # Stop when we have enough unique peaks
            if len(unique_peaks) >= self.num_peaks:
                break

        return unique_peaks


def main():
    p = build_argparser()
    args = p.parse_args()
    ligandfitter = LigandFit(args.dataset,args.ligand,args.resolution)
    ligandfitter.run()

if __name__ == '__main__':
    main()