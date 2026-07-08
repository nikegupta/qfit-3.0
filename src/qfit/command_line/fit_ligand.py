import argparse
import numpy as np
import time
from itertools import product
from pathlib import Path
from scipy.spatial import cKDTree

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
        help="Number of peaks to find",
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
        self.rmsd_cutoff = 2

        #make output folder
        self.output_dir = self.dataset / self.geom_params
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load structures and maps.
        #
        # NOTE: "-aligned-structure.pdb", not "-pandda-input.pdb". In
        # principle "-pandda-input.pdb" (PanDDA's verbatim copy of the
        # deposited model, see `CopyInputStructure`) is the file guaranteed
        # to share a coordinate frame with the native maps, since
        # "-aligned-structure.pdb" instead lives in PanDDA's internal
        # reference frame. But for datasets like this collection (very low
        # RMSD to the PanDDA reference -- 0.30 A RMSD between the two files
        # for x00035-1) the two are numerically indistinguishable, and
        # "-pandda-input.pdb" has a practical downside: it already contains
        # any ligands modelled in earlier PanDDA rounds (e.g. chain C
        # residues 1-2 here), which pushes the newly-placed ligand in `run()`
        # onto chain D+ instead of chain C -- breaking downstream tooling
        # that expects new placements at chain C, res 1. "-aligned-structure"
        # has no ligand chain, so the new ligand always lands on chain C.
        self.apo_structure = Structure.fromfile(f'{self.dataset}/{self.dataset_name}-aligned-structure.pdb')
        self.ligand_structure = Structure.fromfile(f'{self.ligand_file}')
        self.zmap = XMap.fromfile(
            f'{self.dataset}/{self.dataset_name}-z_map.native.ccp4',
            resolution=self.resolution
        )
        self._load_event_maps()

        # Initialize a transformer for cartesian/grid conversions
        self.transformer = get_transformer("qfit", self.apo_structure, self.zmap)

        # KDTree over all apo-structure atoms, used to score candidate
        # unit-cell translations of a zmap peak by their actual proximity
        # to the modelled protein (see _translate_to_nearest_protein_copy),
        # rather than by distance to the protein's center of mass alone.
        self._protein_atom_tree = cKDTree(self.apo_structure.coor)
        self._protein_center = self.apo_structure.coor.mean(axis=0)

        # Backbone-only structure and a dedicated transformer, used by
        # _find_event_centroid to build a mask against just the protein
        # backbone instead of every apo-structure atom. Sidechains commonly
        # reposition upon ligand binding, so masking out all protein atoms
        # incorrectly rejects candidate density that only overlaps an apo
        # sidechain position; backbone atoms don't move, so they remain a
        # reliable clash boundary. This needs its own transformer (rather
        # than reusing self.transformer) because get_conformers_mask
        # assigns the passed-in coordinates directly onto its bound
        # structure, which therefore must have the same atom count as
        # those coordinates -- self.transformer is bound to the full
        # apo_structure, not the backbone-only subset.
        self.backbone_structure = self.apo_structure.extract(
            "name CA or name C or name N or name O")
        self.backbone_transformer = get_transformer(
            "qfit", self.backbone_structure, self.zmap)

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
            if any(np.linalg.norm(best_peak_coord - seen) < self.rmsd_cutoff for seen in seen_coords):
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

    def _translate_to_nearest_protein_copy(self, peak_coord):
        """
        PanDDA's native-frame maps (z_map/event_map "*.native.ccp4") are
        written by symmetry-expanding the reference-frame density into the
        dataset's own native unit cell box (see PanDDA's `NativeMapMaker`,
        specifically `symmetrise_map`, which overlays every space-group
        operator's copy of the density into one periodic box). That means
        every crystallographic symmetry mate of a real feature (e.g. a
        ligand-binding pocket) already exists as its own distinct, real
        local maximum somewhere in the map -- the rotational part of the
        symmetry is already "spent" by PanDDA when it wrote the file. What's
        left is exactly what a crystallographic viewer like Coot does when
        it displays one of these periodic maps: it doesn't re-derive
        symmetry mates mathematically, it just wraps the map to whichever
        unit-cell-translated image is nearest whatever is already on
        screen (see PanDDA's `pandda.inspect`, which loads maps/models via
        Coot APIs such as `handle_read_draw_molecule_and_move_molecule_here`
        that do exactly this).

        So, given one specific instance of a peak, we only need to search
        integer unit-cell translations (no rotations) to find the periodic
        image of it that lies closest to our modelled protein. Distinct
        rotational copies of the same feature are handled by trying this on
        *every* same-Z-score instance in `_find_peaks`/`_find_event_centroid`
        and keeping whichever instance's best translation wins -- not by
        applying symmetry operators here.

        The right integer translation isn't found by guessing a fixed
        search window: it's derived directly from the data. `peak_coord`
        and the apo structure are described in the same rigid coordinate
        frame (the same file), so the peak and the protein differ by
        *exactly* a whole number of unit cells in each fractional
        direction, whatever that number happens to be -- rounding the
        difference of their fractional coordinates recovers it in one
        shot, with no assumption about how many cells away it might be.
        A small +/-1 window around that estimate covers the case where the
        peak itself isn't exactly at the protein's own centroid (e.g. it
        sits near a fractional-coordinate boundary and rounds to an
        adjacent cell).

        Args:
            peak_coord: Cartesian coordinates of the peak instance

        Returns:
            (best_coord, best_distance): Cartesian coordinates of the
                unit-cell translate of peak_coord closest to the apo
                structure, and its distance to the nearest protein atom
        """
        unit_cell = self.zmap.unit_cell
        peak_frac = unit_cell.orth_to_frac @ peak_coord

        # Analytically estimate the needed integer translation from the
        # protein's own fractional position, rather than searching a fixed
        # hardcoded window -- this generalizes to any number of unit cells.
        protein_frac_center = unit_cell.orth_to_frac @ self._protein_center
        center_shift = np.round(protein_frac_center - peak_frac)

        best_distance = float('inf')
        best_coord = peak_coord

        for offset in product(range(-1, 2), repeat=3):
            cell_t = center_shift + np.array(offset, dtype=float)
            translated_frac = peak_frac + cell_t
            cart = unit_cell.frac_to_orth @ translated_frac

            # Score by proximity to the nearest actual protein atom, not
            # distance to the protein's center of mass: a wrong (but
            # nearby-in-aggregate) copy can be closer to the centroid of an
            # elongated/asymmetric protein than the copy that truly
            # overlaps it.
            distance, _ = self._protein_atom_tree.query(cart)

            if distance < best_distance:
                best_distance = distance
                best_coord = cart

        return best_coord, best_distance

    def _find_event_centroid(self):
        event_map = self.event_maps[list(self.event_maps.keys())[0]]
        centroid_peaks = []

        # All-atom mask -- walls off flood-fill growth exactly as in the
        # original algorithm, so the density blob's shape/extent doesn't
        # balloon through space currently occupied by any protein atom.
        protein_mask = self.transformer.get_conformers_mask(
            [self.apo_structure.coor], self._rmask)

        # Backbone-only mask (see self.backbone_transformer in __init__) --
        # used only to decide whether a peak clashes with the protein
        # badly enough to discard it outright. Sidechains commonly
        # reposition upon ligand binding, so a peak whose seed happens to
        # coincide with a *sidechain* atom in the apo model isn't a real
        # clash and shouldn't disqualify the peak before flood-fill even
        # gets a chance to explore around it.
        backbone_mask = self.backbone_transformer.get_conformers_mask(
            [self.backbone_structure.coor], self._rmask)

        threshold = event_map.array.mean() + 2 * event_map.array.std() #threhsold is currently at 1 sigma
        print(f'threshold: {threshold}')

        for i, peak in enumerate(self.peaks):
            peak_coords = tuple(peak[0][::-1])  # xyz -> zyx for numpy indexing

            # Clash detection: only reject the peak outright for clashing
            # with the immovable backbone, not for merely sitting near a
            # sidechain (which may not be there once the ligand binds).
            if backbone_mask[peak_coords] == True:
                print(f"peak {i} failed centroid check. Reason: peak clashes with protein backbone")
                continue
            if event_map.array[peak_coords] < threshold:
                print(f"peak {i} failed centroid check. Reason: Peak is below threshold")
                continue

            # Flood-fill outward from the peak to find all connected voxels
            # above threshold. The seed has already passed the (backbone-
            # only) clash check above, so it's seeded unconditionally here;
            # growth beyond the seed still walls off at any protein atom
            # (the original, all-atom protein_mask), keeping the blob's
            # shape/extent the same as the original algorithm.
            visited = {peak_coords}
            queue = [peak_coords]

            while queue:
                current = queue.pop()
                z, y, x = current

                for dz, dy, dx in [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]:
                    neighbour = (z + dz, y + dy, x + dx)
                    if neighbour in visited:
                        continue
                    nz, ny, nx = neighbour

                    # Bounds check
                    if not (0 <= nz < event_map.array.shape[0] and
                            0 <= ny < event_map.array.shape[1] and
                            0 <= nx < event_map.array.shape[2]):
                        continue
                    # Threshold + all-atom protein wall check
                    if event_map.array[neighbour] < threshold or protein_mask[neighbour] == True:
                        continue

                    visited.add(neighbour)
                    queue.append(neighbour)

            # Centroid as simple mean of zyx coords, returned as xyz
            coords_array = np.array(list(visited))  # shape (N, 3), columns are z, y, x
            centroid_zyx = coords_array.mean(axis=0)
            centroid_xyz = centroid_zyx[::-1]

            centroid_cartesian = self._grid_to_cartesian(centroid_xyz)
            best_centroid_cartesian, _ = self._translate_to_nearest_protein_copy(centroid_cartesian)
            centroid_peaks.append((centroid_xyz,best_centroid_cartesian))

        return centroid_peaks


    def _find_peaks(self):
        """
        Finds the n highest peaks in the zmap.

        PanDDA's native maps are pre-symmetry-expanded (see
        `_translate_to_nearest_protein_copy`), so a single real feature
        shows up as several distinct local maxima that all share an
        (effectively) identical Z-score -- one instance per crystallographic
        symmetry mate. Rather than picking one of those instances
        arbitrarily and trying to mathematically regenerate the others via
        symmetry operators, we group all instances of each shared Z-score
        together and run a translation-only search (see
        `_translate_to_nearest_protein_copy`) on *every* instance in the
        group, keeping whichever one lands closest to the modelled protein.
        This mirrors how Coot/pandda.inspect show these maps: no symmetry
        math, just wrapping each already-existing copy to the periodic
        image nearest the model.
        """
        from scipy.ndimage import maximum_filter

        # Find all local maxima above a threshold
        local_max_mask = maximum_filter(self.zmap.array, size=3) == self.zmap.array
        above_threshold_mask = self.zmap.array > self.z_threshold
        peak_mask = local_max_mask & above_threshold_mask

        # Get peak indices and values
        peak_indices = np.argwhere(peak_mask)  # Returns (z, y, x) for CCP4 maps
        peak_values = self.zmap.array[peak_mask]

        # Group all distinct-position peaks that share an (effectively)
        # identical Z-score -- these are the crystallographic symmetry-mate
        # duplicates PanDDA already baked into the map.
        groups = {}
        for grid_zyx, z_score in zip(peak_indices, peak_values):
            z_score_rounded = round(float(z_score), 3)
            groups.setdefault(z_score_rounded, []).append((grid_zyx, z_score))

        print(f"Found {len(peak_indices)} total peaks above threshold {self.z_threshold}, "
              f"grouped into {len(groups)} unique Z-score features")

        unique_peaks = []

        # Walk unique Z-scores from highest to lowest
        for z_score_rounded in sorted(groups.keys(), reverse=True):
            instances = groups[z_score_rounded]

            # Among all same-Z-score instances (symmetry-mate copies), find
            # the one whose translation-only search lands closest to the
            # protein -- that's the copy that actually corresponds to our
            # specific apo structure.
            best_instance = None
            best_coord = None
            best_distance = float('inf')
            for grid_zyx, z_score in instances:
                z, y, x = grid_zyx
                grid_xyz = (x, y, z)
                peak_coord = self._grid_to_cartesian(grid_xyz)
                coord, distance = self._translate_to_nearest_protein_copy(peak_coord)
                if distance < best_distance:
                    best_distance = distance
                    best_coord = coord
                    best_instance = (grid_xyz, z_score)

            grid_xyz, z_score = best_instance
            print(f"  Peak accepted: (z={z_score:.2f}), "
                  f"best of {len(instances)} symmetry-mate instance(s), "
                  f"nearest-protein distance={best_distance:.2f} A")

            unique_peaks.append((grid_xyz, z_score, best_coord))

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
