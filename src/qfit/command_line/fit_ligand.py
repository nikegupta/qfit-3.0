import argparse
import csv
import os
import fcntl
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
        "-z",
        "--z_threshold",
        default=3,
        metavar="<float>",
        type=float,
        help="Minimum peak Z-score of a z-map cluster (PanDDA's min_blob_z_peak) (default: 3)",
    )
    p.add_argument(
        "--blob_contour",
        default=2.5,
        metavar="<float>",
        type=float,
        help="Z-score contour defining z-map clusters: 26-connected regions of searched z-map "
             "voxels at or above this level (default: 2.5, PanDDA's contour_level).",
    )
    p.add_argument(
        "--min_blob_volume",
        default=10.0,
        metavar="<float>",
        type=float,
        help="Minimum z-map cluster volume (Å^3) at --blob_contour, measured over searched "
             "voxels only; smaller clusters are rejected before merging. 0 disables the filter "
             "(default: 10.0, PanDDA's min_blob_volume).",
    )
    p.add_argument(
        "--run_name",
        required=True,
        help='Pipeline run_name, used to name this dataset\'s output subdirectory '
             '(dataset / run_name) - not a numeric sampling parameter despite the historical '
             '"--sampling" flag name this replaces.'
    )
    p.add_argument(
        "--symmetry_tolerance",
        default=None,
        metavar="<float>",
        type=float,
        help="Two z-map clusters are symmetry copies if a space-group operator maps a point of "
             "one to within this distance (Å) of a point of the other (default: PanDDA's "
             "1.05 x sqrt(3) x grid spacing).",
    )
    p.add_argument(
        "--contact_cutoff",
        default=8.0,
        metavar="<float>",
        type=float,
        help="Radius (Å) for protein contacts: symmetry copies are scored by the protein atoms "
             "within this distance of any cluster point, per point, and an event with no "
             "protein atom this close to its event centroid is rejected (default: 8.0).",
    )
    p.add_argument(
        "--merge_distance",
        default=0,
        metavar="<float>",
        type=float,
        help="Z-map clusters whose edges (closest pair of points) come within this distance "
             "(Å) are merged, chaining, into one event (PanDDA's cluster grouping, which uses "
             "5.0); 0 disables (default: 0).",
    )
    p.add_argument(
        "--outer_mask",
        default=6.0,
        metavar="<float>",
        type=float,
        help="Z-map voxels are searched only within this distance (Å) of a protein atom of the "
             "deposited model (default: 6.0, PanDDA's outer_mask).",
    )
    p.add_argument(
        "--symmetry_mask",
        default=3.0,
        metavar="<float>",
        type=float,
        help="Z-map voxels within this distance (Å) of an atom of a crystal-symmetry copy of "
             "the model are not searched (default: 3.0, PanDDA's inner_mask_symmetry).",
    )
    return p

class LigandPlacer():
    def __init__(self, dataset, ligand_file, resolution, run_name, z_threshold=3,
                 blob_contour=2.5, min_blob_volume=10.0,
                 symmetry_tolerance=None, contact_cutoff=8.0,
                 merge_distance=0.0, outer_mask=6.0, symmetry_mask=3.0):
        # Read in args
        self.dataset = dataset
        self.dataset_name = str(dataset).split('/')[-1]
        self.ligand_file = ligand_file
        self.ligand_name=str(ligand_file).split('/')[-1].split('.')[0]
        self.resolution = resolution
        self.z_threshold = z_threshold
        self.blob_contour = blob_contour
        self.min_blob_volume = min_blob_volume
        self.run_name = run_name
        self._rmask = 0.5 + self.resolution / 3.0
        self.symmetry_tolerance = symmetry_tolerance
        self.contact_cutoff = contact_cutoff
        self.merge_distance = merge_distance
        self.outer_mask = outer_mask
        self.symmetry_mask = symmetry_mask

        #make output folder
        self.output_dir = self.dataset / self.run_name
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Per-dataset manifest linking every fit_ligand output pdb directly to
        # the ligand name/file that produced it. 
        self.manifest_path = self.output_dir / 'fit_ligand_manifest.csv'

        # Load structures and maps.
        self.apo_structure = Structure.fromfile(f'{self.dataset}/{self.dataset_name}-aligned-structure.pdb')
        self.ligand_structure = Structure.fromfile(f'{self.ligand_file}')
        self.zmap = XMap.fromfile(
            f'{self.dataset}/{self.dataset_name}-z_map.native.ccp4',
            resolution=self.resolution
        )
        self._load_event_maps()

        # Transformers for cartesian/grid conversions and atom masks. They are
        # bound to a blank copy of the z-map grid, not self.zmap itself:
        # get_conformers_mask writes into, then zeroes, the map its
        # transformer is bound to, which would silently wipe the z-map.
        self._mask_grid = self.zmap.zeros_like(self.zmap)
        self.transformer = get_transformer("qfit", self.apo_structure, self._mask_grid)

        self._protein_center = self.apo_structure.coor.mean(axis=0)

        # Backbone-only structure and a dedicated transformer, used to build
        # the protein mask for the z-map search from just the protein backbone
        # (the event-map flood fill always uses the all-atom mask). Sidechains
        # commonly reposition upon ligand binding, so masking out all protein
        # atoms incorrectly rejects candidate z-map density that only overlaps
        # an apo sidechain position; backbone atoms don't move, so they remain
        # a reliable clash boundary. This needs its own transformer (rather
        # than reusing self.transformer) because get_conformers_mask
        # assigns the passed-in coordinates directly onto its bound
        # structure, which therefore must have the same atom count as
        # those coordinates -- self.transformer is bound to the full
        # apo_structure, not the backbone-only subset.
        self.backbone_structure = self.apo_structure.extract(
            "name CA or name C or name N or name O")
        self.backbone_transformer = get_transformer(
            "qfit", self.backbone_structure, self._mask_grid)

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

    def _record_manifest_entry(self, peak_index, output_path):
        """
        Appends one row to this dataset's fit_ligand_manifest.csv, linking a
        specific output pdb directly to the ligand name and ligand file path
        that produced it (columns: dataset, ligand_name, ligand_file,
        peak_index, output_pdb).

        Because several LigandPlacer runs (one per candidate ligand) can
        write into the same dataset/run_name directory, this appends
        under an exclusive lock (fcntl.flock) rather than each invocation
        managing its own file, and determines whether a header is needed by
        checking the actual file size under the lock (not a separate
        existence check beforehand, which would race against another
        process's first write).
        """
        header = ['dataset', 'ligand_name', 'ligand_file', 'peak_index', 'output_pdb']
        row = [self.dataset_name, self.ligand_name, str(self.ligand_file), peak_index, str(output_path)]

        with open(self.manifest_path, 'a', newline='') as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                write_header = os.fstat(f.fileno()).st_size == 0
                writer = csv.writer(f)
                if write_header:
                    writer.writerow(header)
                writer.writerow(row)
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

    def run(self):
        """Fits a ligand to event maps guided by the zmap."""
        self.centroid_peaks = self._find_symmetry_pooled_centroids()

        # Get ligand center (calculate once)
        ligand_center = self.ligand_structure.coor.mean(axis=0)
        
        seen_coords = []
        for i, (grid_idx, best_peak_coord) in enumerate(self.centroid_peaks):
            print(f"\n=== Peak {i} ===")
            merged_structure = self.apo_structure.copy()

            # Skip if too close to a previously processed coordinate
            # if any(np.linalg.norm(best_peak_coord - seen) < self.rmsd_cutoff for seen in seen_coords):
            #     print(f'skipping peak {i}, ligand at {best_peak_coord}')
            #     continue
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

            self._record_manifest_entry(i, output_path)

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

    # Protein residue names counted by the contact score (HETATM residues such as modelled
    # cofactors are not protein surface a ligand pocket is judged by).
    PROTEIN_RESNAMES = {"ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
                        "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL", "MSE"}

    # Extra box margin (Å) beyond outer_mask, so an event-map blob that extends past the z-map
    # search shell is flood-filled whole rather than cut at the protein-frame box edge.
    FLOOD_FILL_MARGIN = 6.0
    WATER_RESNAMES = {"HOH", "WAT", "DOD"}
    LATTICE_OFFSETS = np.array(list(product(range(-1, 2), repeat=3)), dtype=float)

    @staticmethod
    def _connected_groups(n, edges):
        """Union-find over (i, j) edges; returns groups of indices, each in ascending order,
        ordered by their first index."""
        parent = list(range(n))

        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        for i, j in edges:
            parent[find(j)] = find(i)
        groups = {}
        for i in range(n):
            groups.setdefault(find(i), []).append(i)
        return sorted(groups.values(), key=lambda members: members[0])

    def _symmetry_copy_coordinates(self, reach):
        """Non-water atoms of every crystal-symmetry copy of the model (space-group operators x
        lattice translations, the model itself excluded) lying within `reach` Å of the model's
        bounding box."""
        unit_cell = self.zmap.unit_cell
        coor = self.apo_structure.coor[~np.isin(self.apo_structure.resn, list(self.WATER_RESNAMES))]
        frac = coor @ unit_cell.orth_to_frac.T
        centre = frac.mean(axis=0)
        box_lo, box_hi = coor.min(axis=0) - reach, coor.max(axis=0) + reach
        copies = []
        for op in unit_cell.space_group.iter_symops():
            identity = np.allclose(op.R, np.eye(3)) and np.allclose(op.t, 0)
            image = frac @ op.R.T + op.t
            shift = np.round(centre - image.mean(axis=0))
            for offset in product(range(-2, 3), repeat=3):
                translation = shift + np.array(offset, dtype=float)
                if identity and not translation.any():
                    continue
                cart = (image + translation) @ unit_cell.frac_to_orth.T
                near = np.all((cart >= box_lo) & (cart <= box_hi), axis=1)
                if near.any():
                    copies.append(cart[near])
        return np.concatenate(copies) if copies else np.empty((0, 3))

    def _find_symmetry_pooled_centroids(self):
        """
        PanDDA-style event search around the modelled protein, then an event-map flood fill
        from each event's z-map centroid.

        PanDDA's native maps cover one unit cell and hold every symmetry copy of a feature, and
        the copies' Z-scores are not reliably identical. Like PanDDA, the search is restricted to
        the region around the deposited protein copy, so each feature is (mostly) seen once and
        clusters never have to be matched between copies:
          1. Protein-frame grid: unit-cell grid points, lattice-translated so the box is not
             limited to the cell, within outer_mask of a protein atom, at least symmetry_mask
             from every atom of a crystal-symmetry copy of the model, and outside the backbone
             mask (in place of PanDDA's 1.8 Å all-atom inner mask, so density over apo side
             chains that move on binding is kept). Z and event-map values are read from the
             cell at wrapped indices.
          2. Z-map clusters: 26-connected components of searched points with z >= blob_contour.
             A cluster is kept if its volume is >= min_blob_volume and its peak z >= z_threshold
             (PanDDA's min_blob_volume / min_blob_z_peak).
          3. Merge (merge_distance > 0): clusters whose closest pair of points is
             < merge_distance apart are merged, chaining (PanDDA's cluster grouping).
          4. Symmetry copies (PanDDA's filter_z_clusters_3): two clusters are copies if a
             space-group operator, with lattice translations, maps a point of one to within
             symmetry_tolerance of a point of the other (default 1.05 x sqrt(3) x grid spacing,
             PanDDA's); copies are chained. Each group keeps only the cluster with the most
             protein atoms within contact_cutoff of any of its points, divided by its number of
             points (PanDDA's score; ties go to the higher peak z).
          5. Flood fill: each kept cluster's highest-z point seeds a fill of the event map over
             the protein-frame box: the 6-connected component of points >= the event map's
             mean + 2 sd outside the all-atom protein mask. An event whose peak lies in the
             all-atom mask or below threshold is rejected. The ligand goes at the component's
             mean point. A component reached from two
             clusters is placed once, and an event with no protein atom within contact_cutoff
             of its event centroid is rejected.

        Returns [(seed_grid_xyz, cartesian), ...] for run(), highest cluster peak z first.
        """
        from scipy.ndimage import label

        unit_cell = self.zmap.unit_cell
        frac_to_orth, orth_to_frac = unit_cell.frac_to_orth, unit_cell.orth_to_frac
        cell_shape = np.array(self.zmap.unit_cell_shape)  # xyz
        zarr = self.zmap.array
        event_map = self.event_maps[list(self.event_maps.keys())[0]]
        if zarr.shape[::-1] != tuple(cell_shape) or event_map.array.shape != zarr.shape:
            raise ValueError(f"z-map {zarr.shape} / event map {event_map.array.shape} must both "
                             f"cover exactly one unit cell {tuple(cell_shape[::-1])} (zyx)")
        voxel_volume = abs(np.linalg.det(frac_to_orth)) / np.prod(cell_shape)
        symmetry_tolerance = (self.symmetry_tolerance if self.symmetry_tolerance is not None
                              else 1.05 * np.sqrt(3) * np.max(self.zmap.voxelspacing))

        # 1. protein-frame grid
        protein_sel = np.isin(self.apo_structure.resn, list(self.PROTEIN_RESNAMES))
        protein_coor = self.apo_structure.coor[protein_sel]
        protein_tree = cKDTree(protein_coor)
        margin = self.outer_mask + self.FLOOD_FILL_MARGIN
        # a sphere of radius r spans r x |row i of orth_to_frac| in fractional coordinate i
        margin_idx = margin * np.linalg.norm(orth_to_frac, axis=1) * cell_shape
        protein_idx = (protein_coor @ orth_to_frac.T) * cell_shape
        lo = np.floor(protein_idx.min(axis=0) - margin_idx).astype(int)
        hi = np.ceil(protein_idx.max(axis=0) + margin_idx).astype(int)
        box_shape = tuple((hi - lo + 1)[::-1])  # zyx
        box_xyz = np.stack(np.meshgrid(*[np.arange(l, h + 1) for l, h in zip(lo[::-1], hi[::-1])],
                                       indexing="ij"), axis=-1).reshape(-1, 3)[:, ::-1]
        box_cart = (box_xyz / cell_shape) @ frac_to_orth.T
        wrapped = tuple((box_xyz % cell_shape)[:, ::-1].T)  # zyx indices into the cell arrays

        backbone_box = self.backbone_transformer.get_conformers_mask(
            [self.backbone_structure.coor], self._rmask)[wrapped]
        nearest_protein, _ = protein_tree.query(box_cart, distance_upper_bound=self.outer_mask + 1)
        search = (nearest_protein <= self.outer_mask) & ~backbone_box
        sym_coor = self._symmetry_copy_coordinates(margin)
        if len(sym_coor):
            nearest_sym, _ = cKDTree(sym_coor).query(box_cart[search],
                                                     distance_upper_bound=self.symmetry_mask)
            search[np.flatnonzero(search)[np.isfinite(nearest_sym)]] = False
        print(f"Searching {search.sum()} z-map points within {self.outer_mask} A of the protein, "
              f">= {self.symmetry_mask} A from {len(sym_coor)} symmetry-copy atoms, outside the "
              f"backbone mask (box {box_shape[::-1]} xyz)")

        # 2. z-map clusters, filtered by volume and peak z
        z_box = zarr[wrapped]
        cluster_ids, n_clusters = label((z_box >= self.blob_contour).reshape(box_shape)
                                        & search.reshape(box_shape),
                                        structure=np.ones((3, 3, 3), dtype=bool))
        cluster_ids = cluster_ids.ravel()
        order = np.argsort(cluster_ids, kind="stable")
        bounds = np.cumsum(np.bincount(cluster_ids, minlength=n_clusters + 1))
        clusters = []  # flat box indices of each cluster's points
        n_small = n_low = 0
        for c in range(1, n_clusters + 1):
            points = order[bounds[c - 1]:bounds[c]]
            if len(points) * voxel_volume < self.min_blob_volume:
                n_small += 1
            elif z_box[points].max() < self.z_threshold:
                n_low += 1
            else:
                clusters.append(points)
        print(f"{n_clusters} z-map clusters at z >= {self.blob_contour}: {n_small} below "
              f"{self.min_blob_volume} A^3 (voxel volume {voxel_volume:.4f} A^3), {n_low} with "
              f"peak z < {self.z_threshold}, {len(clusters)} kept")

        # 3. merge clusters whose edges are < merge_distance apart, chaining
        if self.merge_distance > 0 and len(clusters) > 1:
            trees = [cKDTree(box_cart[points]) for points in clusters]
            centres = [box_cart[points].mean(axis=0) for points in clusters]
            radii = [np.linalg.norm(box_cart[points] - centre, axis=1).max()
                     for points, centre in zip(clusters, centres)]
            edges = []
            for i in range(len(clusters)):
                for j in range(i + 1, len(clusters)):
                    # clusters can't be within merge_distance if their bounding spheres aren't
                    if np.linalg.norm(centres[i] - centres[j]) > radii[i] + radii[j] + self.merge_distance:
                        continue
                    gap, _ = trees[i].query(box_cart[clusters[j]], distance_upper_bound=self.merge_distance)
                    if np.any(gap < self.merge_distance):
                        edges.append((i, j))
            groups = self._connected_groups(len(clusters), edges)
            for members in groups:
                if len(members) > 1:
                    print(f"merged z-map clusters {members} (edges < {self.merge_distance} A)")
            clusters = [np.concatenate([clusters[i] for i in members]) for members in groups]
            print(f"{len(clusters)} z-map clusters after merging within {self.merge_distance} A")
        clusters.sort(key=lambda points: -z_box[points].max())
        peak_z = [z_box[points].max() for points in clusters]

        # 4. remove symmetry copies, keeping the cluster(s) with the most protein contacts
        cart = [box_cart[points] for points in clusters]
        frac = [c @ orth_to_frac.T for c in cart]
        trees = [cKDTree(c) for c in cart]
        centres = [c.mean(axis=0) for c in cart]
        radii = [np.linalg.norm(c - centre, axis=1).max() for c, centre in zip(cart, centres)]
        symops = list(unit_cell.space_group.iter_symops())
        edges = []
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                copy = False
                for op in symops:
                    identity = np.allclose(op.R, np.eye(3)) and np.allclose(op.t, 0)
                    image = frac[i] @ op.R.T + op.t
                    shift = np.round(frac[j].mean(axis=0) - image.mean(axis=0))
                    for offset in self.LATTICE_OFFSETS:
                        translation = shift + offset
                        if identity and not translation.any():
                            continue
                        image_cart = (image + translation) @ frac_to_orth.T
                        if (np.linalg.norm(image_cart.mean(axis=0) - centres[j])
                                > radii[i] + radii[j] + symmetry_tolerance):
                            continue
                        gap, _ = trees[j].query(image_cart, distance_upper_bound=symmetry_tolerance)
                        if np.any(np.isfinite(gap)):
                            copy = True
                            break
                    if copy:
                        break
                if copy:
                    edges.append((i, j))
        contacts = []  # protein atoms within contact_cutoff of any cluster point, per point
        for c, tree in zip(cart, trees):
            gap, _ = tree.query(protein_coor, distance_upper_bound=self.contact_cutoff)
            contacts.append(np.isfinite(gap).sum() / len(c))
        sym_groups = self._connected_groups(len(clusters), edges)
        kept = []  # one cluster index per symmetry group
        for members in sym_groups:
            best = max(members, key=lambda i: contacts[i])  # members ascend, so ties -> higher z
            if len(members) > 1:
                print(f"symmetry copies {members} (contacts/point "
                      f"{', '.join(f'{contacts[i]:.2f}' for i in members)}): keeping {best}")
            kept.append(best)
        kept.sort()  # clusters are in descending peak-z order
        print(f"{len(kept)} z-map events after removing symmetry copies "
              f"(tolerance {symmetry_tolerance:.2f} A)")

        # 5. event-map flood fill from each cluster's peak
        event_box = event_map.array[wrapped].reshape(box_shape)
        threshold = event_map.array.mean() + 2 * event_map.array.std()
        print(f'event map threshold: {threshold}')
        protein_box = self.transformer.get_conformers_mask(
            [self.apo_structure.coor], self._rmask)[wrapped]
        components, _ = label((event_box >= threshold) & ~protein_box.reshape(box_shape))
        components = components.ravel()

        centroid_peaks = []
        placed = {}  # event component -> event number it was placed as
        for i in kept:
            points = clusters[i]
            header = (f"event {i} (peak z={peak_z[i]:.2f}, {len(points)} points, "
                      f"{len(points) * voxel_volume:.1f} A^3)")
            seed = points[np.argmax(z_box[points])]
            if protein_box[seed]:
                print(f"{header} rejected: peak seed in all-atom protein mask")
                continue
            if not components[seed]:
                print(f"{header} rejected: peak seed below event map threshold")
                continue
            component = components[seed]
            if component in placed:
                print(f"{header} skipped: same event-map blob as event {placed[component]}")
                continue
            event_cart = box_cart[components == component].mean(axis=0)
            if not protein_tree.query_ball_point(event_cart, self.contact_cutoff):
                print(f"{header} rejected: no protein atom within {self.contact_cutoff} A of "
                      f"the event centroid")
                continue
            placed[component] = i
            print(f"{header} accepted: event centroid at {np.round(event_cart, 2)}, "
                  f"{contacts[i]:.2f} protein contacts/point")
            centroid_peaks.append((box_xyz[seed] % cell_shape, event_cart))

        return centroid_peaks


def main():
    p = build_argparser()
    args = p.parse_args()
    placer = LigandPlacer(args.dataset, args.ligand, args.resolution, args.run_name,
                          z_threshold=args.z_threshold, blob_contour=args.blob_contour,
                          min_blob_volume=args.min_blob_volume,
                          symmetry_tolerance=args.symmetry_tolerance,
                          contact_cutoff=args.contact_cutoff,
                          merge_distance=args.merge_distance, outer_mask=args.outer_mask,
                          symmetry_mask=args.symmetry_mask)
    placer.run()

if __name__ == '__main__':
    main()