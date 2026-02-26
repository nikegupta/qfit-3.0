"""
fit_ligand_zmap.py
==================
Combines zmap-guided peak detection (from LigandPlacer) with Phenix
ligandfit execution, wrapped in a PhenixLigandFitter class that mirrors
the LigandPlacer class architecture.

Workflow
--------
1. Load the apo structure, zmap, and event maps from the dataset directory.
2. Find the top-N zmap peaks above a z-score threshold.
3. For each peak, convert grid index → Cartesian coordinates and apply all
   P2₁2₁2₁ symmetry operations + unit-cell translations (±1) to select the
   symmetry mate closest to the protein centre.
4. Run phenix.ligandfit once per peak against the first event map, passing
   the peak Cartesian coordinates as search_center.
5. Score each ligandfit output against the event map using MSE of the
   model density vs. event-map density (lower = better).
6. Discard candidates with > 3 backbone clashes (VDW-radius-based).
7. Merge the best-scoring, clash-free ligand into the apo structure and
   write the combined PDB with the original header preserved.
8. Copy all ligand_fit_1_*.pdb files to ligandfit_results/ (prefixed with
   the run directory name) then delete all LigandFit_run_* directories.

Usage
-----
    python fit_ligand_zmap.py \\
        /path/to/dataset \\
        /path/to/ligand.pdb \\
        /path/to/ligand.cif \\
        -r 1.8 \\
        [-n 5] \\
        [-z 5] \\
        [--clash_cutoff 0.75]
"""

import argparse
import glob
import os
import re
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
from scipy.ndimage import maximum_filter

from qfit import Structure, XMap
from qfit.xtal.transformer import get_transformer


class PhenixLigandFitter:
    def __init__(self, dataset: Path, ligand: Path, ligand_restraints: Path,
                 resolution: float, num_peaks: int = 5,
                 z_threshold: float = 5.0, clash_cutoff: float = 0.75):
        """
        Parameters
        ----------
        dataset           : Path to the dataset directory.
        ligand            : Path to the ligand PDB file.
        ligand_restraints : Path to the ligand CIF restraints file.
        resolution        : Map resolution in Å.
        num_peaks         : Number of zmap peaks to sample.
        z_threshold       : Minimum z-score for peak detection.
        clash_cutoff      : Fraction of summed VDW radii used as the clash
                            distance threshold.
        """
        self.dataset_path      = Path(dataset)
        self.dataset_name      = self.dataset_path.name
        self.ligand            = Path(ligand)
        self.ligand_restraints = Path(ligand_restraints)
        self.resolution        = resolution
        self.num_peaks         = num_peaks
        self.z_threshold       = z_threshold
        self.clash_cutoff      = clash_cutoff

        self.protein_path = str(
            self.dataset_path / f"{self.dataset_name}-pandda-input.pdb"
        )

        # Load apo structure and zmap
        self.apo_structure = Structure.fromfile(self.protein_path)
        self.zmap = XMap.fromfile(
            str(self.dataset_path / f"{self.dataset_name}-z_map.native.ccp4"),
            resolution=self.resolution,
        )
        self.transformer = get_transformer("qfit", self.apo_structure, self.zmap)

        # Load event maps
        self._load_event_maps()

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self):
        """
        Full fitting pipeline:
          1. Find zmap peaks.
          2. Run ligandfit at each peak.
          3. Filter by backbone clashes.
          4. Score against the event map.
          5. Merge best ligand and write output.
          6. Save ligand PDBs to ligandfit_results/ and clean up run dirs.
        """
        print('running')
        time0 = time.time()
        self.peaks = self._find_peaks()
        if not self.peaks:
            raise ValueError("No zmap peaks found above threshold")
        print(f"Using {len(self.peaks)} peaks")

        # Run ligandfit at every peak; collect {peak_idx: (path, Structure)}
        self.candidate_ligands = self._run_ligandfit()
        if not self.candidate_ligands:
            raise ValueError("All ligandfit runs failed")

        # Clash filter
        self._filter_clashes()
        if not self.candidate_ligands:
            raise ValueError("All candidates removed by clash filter")

        # Score and select best
        best_structure, best_score = self._score_candidates()
        if best_structure is None:
            raise ValueError("Scoring failed for all candidates")

        print(f"\nBest ligand MSE = {best_score:.4f}")

        # Merge and write
        output_path = str(
            self.dataset_path / f"{self.dataset_name}_ligandfit_merged.pdb"
        )
        self._merge_and_write(best_structure, output_path)
        print(f"Saved merged structure to {output_path}")

        # Preserve ligand PDBs then remove the large run directories
        self._save_ligand_results()
        self._cleanup_run_dirs()

        print(f'ran in {time.time()} - {time0}')

    # ------------------------------------------------------------------
    # Private methods
    # ------------------------------------------------------------------

    def _load_event_maps(self):
        """Load all event maps found in the dataset directory."""
        event_map_files = sorted(
            self.dataset_path.glob("*-event_*_*-BDC_*_map.native.ccp4")
        )
        if not event_map_files:
            raise FileNotFoundError(
                f"No event maps found in {self.dataset_path}"
            )
        self.event_maps = {
            str(p): XMap.fromfile(str(p), resolution=self.resolution)
            for p in event_map_files
        }
        # Keep a reference to the first map for ligandfit and scoring
        self.first_event_map_path = str(event_map_files[0])
        self.first_event_map      = self.event_maps[self.first_event_map_path]
        print(f"  Using event map: {event_map_files[0].name}")

    def _find_peaks(self) -> list:
        """
        Find the top-N unique local maxima above z_threshold in the zmap.
        Deduplicates by rounded z-score to remove symmetry-equivalent copies.
        Applies P2₁2₁2₁ symmetry + unit-cell translations to return the
        Cartesian position of each peak closest to the protein centre.

        Returns
        -------
        List of (grid_xyz, z_score, cartesian_coord) tuples.
        """
        local_max  = maximum_filter(self.zmap.array, size=3) == self.zmap.array
        above_thr  = self.zmap.array > self.z_threshold
        peak_mask  = local_max & above_thr

        peak_indices = np.argwhere(peak_mask)   # z, y, x (CCP4 convention)
        peak_values  = self.zmap.array[peak_mask]
        sorted_order = np.argsort(peak_values)[::-1]

        print(f"  Found {len(sorted_order)} peaks above z={self.z_threshold}")

        protein_center = self.apo_structure.coor.mean(axis=0)
        used_zscores   = set()
        peaks          = []

        for i in sorted_order:
            z_score = peak_values[i]
            z_key   = round(float(z_score), 3)
            if z_key in used_zscores:
                continue

            z_idx, y_idx, x_idx = peak_indices[i]
            grid_xyz   = (int(x_idx), int(y_idx), int(z_idx))
            peak_cart  = self._grid_to_cartesian(grid_xyz)
            best_coord = self._symmetry_mate_near_protein(peak_cart, protein_center)

            peaks.append((grid_xyz, float(z_score), best_coord))
            used_zscores.add(z_key)

            if len(peaks) >= self.num_peaks:
                break

        if len(peaks) < self.num_peaks:
            print(f"  Only found {len(peaks)} unique peaks "
                  f"(requested {self.num_peaks})")

        return peaks

    def _run_ligandfit(self) -> dict:
        """
        Run phenix.ligandfit at each peak.

        Returns
        -------
        {peak_idx: (lig_pdb_path, Structure)}  for every successful run.
        """
        candidates = {}
        for peak_idx, (grid_xyz, z_score, peak_coord) in enumerate(self.peaks):
            print(f"\n  Peak {peak_idx}  z={z_score:.2f}  "
                  f"cart={np.round(peak_coord, 2)}")
            lig_path = self._run_ligandfit_for_peak(peak_coord, peak_idx)
            if lig_path is not None:
                candidates[peak_idx] = (lig_path, Structure.fromfile(lig_path))
        return candidates

    def _run_ligandfit_for_peak(self, peak_coord: np.ndarray,
                                 peak_index: int) -> "str | None":
        """
        Run phenix.ligandfit for a single peak and return the path to the
        fitted ligand PDB, or None on failure.
        """
        e_map      = os.path.basename(self.first_event_map_path)
        cx, cy, cz = peak_coord

        cmd = f"""
        source /programs/sbgrid.shrc
        cd {self.dataset_path}
        phenix.ligandfit model={os.path.basename(self.protein_path)} \\
            ligand={self.ligand} \\
            map_in={e_map} \\
            resolution={self.resolution} \\
            number_of_ligands=4 \\
            ligand_cc_min=0.5 \\
            cif_def_file_list={self.ligand_restraints} \\
            cif_already_generated=True \\
            search_center="{cx:.3f},{cy:.3f},{cz:.3f}" \\
            space_group="P 21 21 21" \\
            nproc=5
        """
        log_path = self.dataset_path / f"ligandfit_peak_{peak_index}.log"
        try:
            with open(log_path, "w") as log:
                subprocess.run(cmd, shell=True, executable="/bin/bash", check=True, stdout=log, stderr=log)
                print(f"    Peak {peak_index}: ligandfit succeeded for {e_map}")
        except subprocess.CalledProcessError:
            print(f"    Peak {peak_index}: ligandfit FAILED for {e_map}")
            return None

        # The most-recently modified LigandFit_run_* dir belongs to this call
        run_dirs = sorted(
            glob.glob(str(self.dataset_path / "LigandFit_run_*")),
            key=os.path.getmtime,
        )
        if not run_dirs:
            return None

        ligand_files = glob.glob(os.path.join(run_dirs[-1], "ligand_fit_1_*.pdb"))
        return ligand_files[0] if ligand_files else None

    def _save_ligand_results(self):
        """
        Copy all ligand_fit_1_*.pdb files from every LigandFit_run_* directory
        into <dataset>/ligandfit_results/, prefixing each filename with its
        run directory name so files from different peaks never collide.

        Example output filename:
            ligandfit_results/LigandFit_run_1__ligand_fit_1_1.pdb
        """
        results_dir = self.dataset_path / "ligandfit_results"
        results_dir.mkdir(exist_ok=True)

        run_dirs = sorted(
            glob.glob(str(self.dataset_path / "LigandFit_run_*")),
            key=os.path.getmtime,
        )
        for run_dir in run_dirs:
            run_number = os.path.basename(run_dir)   # e.g. "LigandFit_run_1_"
            for src in glob.glob(os.path.join(run_dir, "ligand_fit_1_*.pdb")):
                dst = results_dir / f"{run_number}_{Path(src).name}"
                shutil.copy2(src, dst)
                print(f"  Saved {dst.name} → ligandfit_results/")

    def _cleanup_run_dirs(self):
        """
        Delete all LigandFit_run_* directories. These are large and are not
        overwritten between runs. All ligand PDBs have already been copied to
        ligandfit_results/ by _save_ligand_results().
        """
        run_dirs = glob.glob(str(self.dataset_path / "LigandFit_run_*"))
        for run_dir in run_dirs:
            shutil.rmtree(run_dir)
            print(f"  Removed {os.path.basename(run_dir)}/")
        n = len(run_dirs)
        print(f"  Cleaned up {n} LigandFit run "
              f"{'directory' if n == 1 else 'directories'}")
        
        for name in ("temp_dir", "PDS"):
            p = self.dataset_path / name
            if p.exists():
                shutil.rmtree(p)

    def _filter_clashes(self):
        """
        Remove candidates with more than 0 backbone-atom clashes from
        self.candidate_ligands in-place.

        A clash between atom pair (i, j) is:
            distance(i, j) < clash_cutoff * (vdw_i + vdw_j)

        VDW radii are read directly from the qfit Structure.vdw_radius
        property — no manual lookup table is needed.
        """
        backbone_sel       = self.apo_structure.select('name', ['N', 'CA', 'C', 'O'])
        backbone_structure = self.apo_structure.extract(backbone_sel)
        backbone_coords    = backbone_structure.coor        # (M, 3)
        backbone_vdw       = backbone_structure.vdw_radius  # (M,)

        for idx, (lig_path, lig_structure) in list(self.candidate_ligands.items()):
            lig_coords = lig_structure.coor        # (N, 3)
            lig_vdw    = lig_structure.vdw_radius  # (N,)

            diff      = lig_coords[:, np.newaxis, :] - backbone_coords[np.newaxis, :, :]
            distances = np.linalg.norm(diff, axis=2)                          # (N, M)
            cutoffs   = self.clash_cutoff * (
                lig_vdw[:, np.newaxis] + backbone_vdw[np.newaxis, :]
            )
            n_clashes = int(np.sum(distances < cutoffs))

            if n_clashes > 0:
                print(f"  Discarding peak {idx} ligand: {n_clashes} backbone clashes")
                del self.candidate_ligands[idx]

    def _score_candidates(self) -> "tuple[Structure | None, float]":
        """
        Score each remaining candidate against the first event map using MSE
        of model density vs. event-map density (lower = better).

        Returns
        -------
        (best_structure, best_score)  — best_structure is None if all fail.
        """
        print(f"\n  Scoring {len(self.candidate_ligands)} clash-free candidates...")

        best_score         = np.inf
        best_lig_structure = None

        for idx, (lig_path, lig_structure) in self.candidate_ligands.items():
            try:
                score = self._score_ligand(lig_structure)
            except Exception as exc:
                print(f"    Peak {idx}: scoring failed ({exc}), skipping")
                continue

            print(f"    Peak {idx}: MSE = {score:.4f}")
            if score < best_score:
                best_score         = score
                best_lig_structure = lig_structure

        return best_lig_structure, best_score

    def _score_ligand(self, lig_structure: "Structure") -> float:
        """
        Convert *lig_structure* to model density and compute its MSE against
        the first event map (lower MSE = better fit).
        """
        event_map_model = self.first_event_map.zeros_like(self.first_event_map)
        event_map_model.set_space_group("P1")

        # Extract BDC value from the event map filename for bulk-solvent scaling
        bdc_match = re.search(
            r"-BDC_([\d.]+)_", os.path.basename(self.first_event_map_path)
        )
        one_minus_bdc = float(bdc_match.group(1)) if bdc_match else 0.3
        scaled_bulk   = 0.3 * one_minus_bdc

        lig_transformer = get_transformer("qfit", lig_structure, event_map_model)

        coor_set = lig_structure.coor[np.newaxis, :, :]  # (1, N, 3)
        b_set    = lig_structure.b[np.newaxis, :]         # (1, N)

        mask   = lig_transformer.get_conformers_mask(coor_set, rmax=1)
        target = self.first_event_map.array[mask]

        nvalues  = mask.sum()
        model    = np.zeros(nvalues, float)
        model[:] = list(
            lig_transformer.get_conformers_densities(coor_set, b_set)
        )[0][mask]
        np.maximum(model, scaled_bulk, out=model)

        return float(np.mean((model - target) ** 2))

    def _merge_and_write(self, best_ligand: "Structure", output_path: str):
        """
        Combine *best_ligand* into the apo structure, write the merged PDB,
        then prepend the original protein header.
        """
        merged = self.apo_structure.combine(best_ligand)
        merged.tofile(output_path)
        self._prepend_header(output_path)

    def _prepend_header(self, output_path: str):
        """Prepend the PDB header from the original protein to *output_path*."""
        header_lines = []
        with open(self.protein_path) as f:
            for line in f:
                if line.startswith(("ATOM", "HETATM")):
                    break
                header_lines.append(line)

        atom_lines = []
        with open(output_path) as f:
            for line in f:
                if line.startswith(("ATOM", "HETATM", "TER", "END")):
                    atom_lines.append(line)

        with open(output_path, "w") as f:
            f.writelines(header_lines)
            f.writelines(atom_lines)

    # ------------------------------------------------------------------
    # Coordinate helpers
    # ------------------------------------------------------------------

    def _grid_to_cartesian(self, grid_xyz: tuple) -> np.ndarray:
        """Convert (x, y, z) grid indices to Cartesian Å coordinates."""
        grid_coor  = np.array(grid_xyz, dtype=float)
        grid_coor += self.transformer.xmap.offset
        grid_coor *= self.transformer.xmap.voxelspacing
        cartesian  = np.dot(grid_coor, self.transformer.lattice_to_cartesian.T)
        if not np.allclose(self.transformer.xmap.origin, 0):
            cartesian += self.transformer.xmap.origin
        return cartesian

    def _symmetry_mate_near_protein(self, peak_coord: np.ndarray,
                                     protein_center: np.ndarray) -> np.ndarray:
        """
        Apply all P2₁2₁2₁ symmetry operations and ±1 unit-cell translations
        and return the Cartesian coordinates of the copy closest to
        *protein_center*.
        """
        peak_frac = self.zmap.unit_cell.orth_to_frac @ peak_coord

        symops = [
            lambda xyz: np.array([ xyz[0],          xyz[1],          xyz[2]        ]),
            lambda xyz: np.array([-xyz[0] + 0.5,    -xyz[1],          xyz[2] + 0.5  ]),
            lambda xyz: np.array([-xyz[0],           xyz[1] + 0.5,   -xyz[2] + 0.5  ]),
            lambda xyz: np.array([ xyz[0] + 0.5,    -xyz[1] + 0.5,  -xyz[2]        ]),
        ]
        translations = [
            np.array([x, y, z])
            for x in (-1, 0, 1)
            for y in (-1, 0, 1)
            for z in (-1, 0, 1)
        ]

        best_dist, best_coord = np.inf, None
        for symop in symops:
            sym_frac = symop(peak_frac)
            for t in translations:
                cart = self.zmap.unit_cell.frac_to_orth @ (sym_frac + t)
                d    = np.linalg.norm(cart - protein_center)
                if d < best_dist:
                    best_dist  = d
                    best_coord = cart

        return best_coord


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_argparser():
    p = argparse.ArgumentParser(
        description="Zmap-guided phenix.ligandfit with density scoring"
    )
    p.add_argument(
        "dataset",
        type=Path,
        help="Path to dataset directory",
    )
    p.add_argument(
        "ligand",
        type=Path,
        help="Path to ligand PDB file",
    )
    p.add_argument(
        "ligand_restraints",
        type=Path,
        help="Path to ligand CIF restraints file",
    )
    p.add_argument(
        "-r", "--resolution",
        required=True,
        metavar="<float>",
        type=float,
        help="Map resolution (Å)",
    )
    p.add_argument(
        "-n", "--num_peaks",
        default=5,
        metavar="<int>",
        type=int,
        help="Number of zmap peaks to sample (default: 5)",
    )
    p.add_argument(
        "-z", "--z_threshold",
        default=5.0,
        metavar="<float>",
        type=float,
        help="Z-score threshold for peak detection (default: 5)",
    )
    p.add_argument(
        "--clash_cutoff",
        default=0.75,
        metavar="<float>",
        type=float,
        help="VDW clash cutoff fraction (default: 0.75)",
    )
    return p


def main():
    args = build_argparser().parse_args()
    fitter = PhenixLigandFitter(
        dataset=args.dataset,
        ligand=args.ligand,
        ligand_restraints=args.ligand_restraints,
        resolution=args.resolution,
        num_peaks=args.num_peaks,
        z_threshold=args.z_threshold,
        clash_cutoff=args.clash_cutoff,
    )
    fitter.run()


if __name__ == "__main__":
    main()