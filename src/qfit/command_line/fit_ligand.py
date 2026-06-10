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
        default=100,
        metavar="<int>",
        type=int,
        help="Number of peaks to find (default: 5)",
    )
    p.add_argument(
        "-z",
        "--z_threshold",
        default=4,
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
        self.ligand_name=str(ligand_file).split('/')[-1].split('.')[0]
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
        
        # Get ligand center (calculate once)
        ligand_center = self.ligand_structure.coor.mean(axis=0)
        
        seen_coords = []
        for i, (grid_idx, best_peak_coord) in enumerate(self.centroid_peaks):
            print(f"\n=== Peak {i} ===")
            merged_structure = self.apo_structure.copy()

            # Skip if too close to a previously processed coordinate
            if any(np.linalg.norm(best_peak_coord - seen) < 0.1 for seen in seen_coords):
                print(f'skipping peak {i}, ligand at {best_peak_coord}')
                continue
            seen_coords.append(best_peak_coord)
            print(f"Placing ligand at: {best_peak_coord}")

            # Place ligand center on zmap peak
            self.placed_ligand = self.ligand_structure.copy()
            translation = best_peak_coord - ligand_center
            self.placed_ligand.coor = self.ligand_structure.coor + translation

            #change ligand to the subsequent chain after protein
            highest_chain = 'A'
            for chain in self.apo_structure._pdb_hierarchy.only_model().chains():
                if chain.id > highest_chain:
                    highest_chain = chain.id

            for chain in self.placed_ligand._pdb_hierarchy.only_model().chains():
                chain.id = chr(ord(highest_chain) + 1)
            merged_structure = merged_structure.combine(self.placed_ligand)
            
            # Save merged structure
            output_path = f'{self.output_dir}/{self.dataset_name}-{self.ligand_name}_{i}.pdb'
            merged_structure.tofile(output_path)
            print(f"Saved to {output_path}")

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

    def _find_event_centroid(self):
        event_map = self.event_maps[list(self.event_maps.keys())[0]]
        centroid_peaks = []
        protein_center = self.apo_structure.coor.mean(axis=0)
        protein_mask = self.transformer.get_conformers_mask([self.apo_structure.coor],self._rmask)
        threshold = event_map.array.mean() + 2 * event_map.array.std() #threhsold is currently at 1 sigma
        print(f'threshold: {threshold}')

        for i, peak in enumerate(self.peaks):
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

                if event_map.array[z, y, x] < threshold or protein_mask[z, y, x] == True:
                    continue
                visited.add(current)

                # Add 6-connected neighbours
                for dz, dy, dx in [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]:
                    neighbour = (z + dz, y + dy, x + dx)
                    if neighbour not in visited:
                        queue.append(neighbour)

            if not visited:
                print(f"peak {i} failed centroid check. ")
                if event_map.array[peak_coords] < threshold:
                    print("Reason: Peak is below threshold")
                if protein_mask[peak_coords] == True:
                    print("Reaons: peak is in protein coords")
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
            print(f"  Peak accepted: (z={z_score:.2f}):")
            
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
