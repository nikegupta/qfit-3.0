import argparse
import re
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

# Matches the BDC value embedded in an event map filename, e.g.
# 'x00407-1-event_1_1-BDC_0.08_map.native.ccp4' -> '0.08'. Same convention as calc_rscc.py.
BDC_PATTERN = re.compile(r'1-BDC_([\d.]+)_')


def parse_bdc(map_filename):
    """Extracts the BDC value (as a string, so '0.080' and '0.08' aren't silently treated as
    equal) from an event map filename. Returns None if the filename doesn't match the expected
    '1-BDC_<value>_' pattern."""
    m = BDC_PATTERN.search(str(map_filename))
    return m.group(1) if m else None


def build_argparser():
    p = argparse.ArgumentParser()
    p.add_argument(
        'dataset',
        type=Path,
        help='Path to pandas dataset')
    p.add_argument(
        'model_file',
        type=str,
        help='Path to the single-model structure to optimize (e.g. final_model.pdb). '
             'residues_with_placer_conformers.csv is expected alongside it in the same folder.'
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

class QFitOptions: #copypasted from qfit.py
    def __init__(self):
        # Sampling options
        self.clash_scaling_factor = 0.75
        self.external_clash = False
        self.dofs_per_iteration = 1
        self.dihedral_stepsize = 12
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
        self.sample_angle_step = 7.5

        # Rotamer sampling
        self.sample_rotamers = True
        self.rotamer_neighborhood = 24
        self.remove_conformers_below_cutoff = False

class Rotamer_Optimizer():
    def __init__(self, dataset_dir, model_file, output_folder, resolution):
        self.dir = dataset_dir
        self.model_file = model_file
        self.output_path = f"{dataset_dir}/{output_folder}"
        os.makedirs(self.output_path,exist_ok=True)
        self.resolution = resolution
        self.options = QFitOptions()
        self._load_event_maps()
        self._rmask = 0.5 + self.resolution / 3.0 #from qfit

        self.base_structure = Structure.fromfile(self.model_file)
        self.base_structure = self.base_structure.extract("e", "H", "!=")

        self.trim = 10

        # Residues scoring below this against the event maps are candidates for optimization;
        # residues already at/above it are left untouched.
        self.rscc_threshold = 0.5
        # An optimized conformer is only accepted if it improves RSCC over the starting
        # conformer by at least this much.
        self.rscc_improvement_threshold = 0.1

    def _load_event_maps(self):
        """Loads every event map for this dataset. Maps sharing the same 1-BDC value are
        identical (same partial-occupancy background subtraction), so only the first one seen
        for a given BDC is loaded; later ones with the same BDC are skipped (see calc_rscc.py,
        same convention)."""
        self.event_maps = {}
        self.event_maps_models = {}
        event_map_files = sorted(self.dir.glob('*-event_*_*-BDC_*_map.native.ccp4'))
        seen_bdcs = set()
        for event_file in event_map_files:
            event_name = event_file.name
            bdc = parse_bdc(event_name)
            if bdc is not None:
                if bdc in seen_bdcs:
                    print(f'Skipping map {event_file}: duplicate BDC={bdc} '
                          f'(another event map with this BDC was already loaded).')
                    continue
                seen_bdcs.add(bdc)

            self.event_maps[event_name] = XMap.fromfile(str(event_file), resolution=self.resolution)

            # make copies for density steps
            event_map_model = self.event_maps[event_name].zeros_like(self.event_maps[event_name])
            event_map_model.set_space_group("P1")
            self.event_maps_models[event_name] = event_map_model

    def _load_binding_site_residues(self):
        """Reads the pipeline-computed list of binding-site residues (those with Placer
        conformers) from residues_with_placer_conformers.csv, expected alongside model_file.
        Each line is a residue label like 'A143' (chain id + residue number)."""
        residues_csv = Path(self.model_file).parent / "residues_with_placer_conformers.csv"
        residues = []
        with open(residues_csv) as f:
            for line in f:
                label = line.strip()
                if not label:
                    continue
                m = re.match(r'^([A-Za-z]+)(\d+)$', label)
                if not m:
                    print(f"Warning: could not parse residue label {label!r} in {residues_csv}; skipping.")
                    continue
                residues.append((m.group(1), int(m.group(2))))
        return residues

    def run(self):
        residues = self._load_binding_site_residues()
        print(f'{len(residues)} binding-site residue(s) to check')

        accepted_coords = {}
        improved_coords = {}
        all_rows = []
        num_improved = 0

        for chain_id, resi in residues:
            resi_selstr = f"chain {chain_id} and resi {resi}"
            structure_new = self.base_structure.copy()
            structure_resi = structure_new.extract(resi_selstr)
            try:
                chain = structure_resi[chain_id]
                current_residue = chain.conformers[0][resi]
            except Exception:
                print(f'Warning: could not retrieve residue {chain_id}{resi} from {self.model_file}; skipping.')
                continue

            if current_residue.type != 'rotamer-residue':
                continue

            time0 = time.time()
            self.current_residue = current_residue

            #get rscc/coors for starting conformer
            self._coor_set = [self.current_residue.coor]
            base_rscc = self._calc_rscc_all_events()
            print(f'{chain_id}{resi}: base_rscc={base_rscc:.3f}')

            if base_rscc >= self.rscc_threshold:
                all_rows.append((chain_id, resi, base_rscc, None, False))
                continue

            #sample ca-b-y for aromatics
            self._sample_angle()

            #sample sidechains chi
            self._sample_sidechains()

            #score sidechains to top 1
            self._convert_and_score_rotamer(1)
            optimized_rscc = self._calc_rscc_all_events()

            print(f'{chain_id}{resi}: base_rscc={base_rscc:.3f} optimized_rscc={optimized_rscc:.3f} '
                  f'({time.time() - time0:.1f}s)')

            accepted = optimized_rscc - base_rscc >= self.rscc_improvement_threshold
            if accepted:
                accepted_coords[(chain_id, resi)] = self._coor_set[0]
                num_improved += 1
            if optimized_rscc > base_rscc:
                improved_coords[(chain_id, resi)] = self._coor_set[0]
            all_rows.append((chain_id, resi, base_rscc, optimized_rscc, accepted))

        # fitted.pdb carries every residue whose resampled conformer improved RSCC at all, even
        # if it didn't clear the acceptance threshold - copy the untouched structure before
        # applying accepted_coords below (accepted_coords is a subset of improved_coords).
        fitted_structure = self.base_structure.copy()
        self._update_coords(fitted_structure, improved_coords)
        fitted_output = self.output_path + '/fitted.pdb'
        self._write_pdb(fitted_structure, fitted_output)

        # rotamer_optimized.pdb only carries residues that cleared the acceptance threshold
        self._update_coords(self.base_structure, accepted_coords)
        output = self.output_path + '/rotamer_optimized.pdb'
        self._write_pdb(self.base_structure, output)

        residue_rscc_output = self.output_path + '/residue_rscc.csv'
        with open(residue_rscc_output, 'w+') as f:
            f.write('residue,initial_rscc,improved_rscc,accepted\n')
            for chain_id, resi, base_rscc, optimized_rscc, accepted in all_rows:
                improved_rscc_str = f'{optimized_rscc}' if optimized_rscc is not None else 'NA'
                f.write(f'{chain_id}{resi},{base_rscc},{improved_rscc_str},{"yes" if accepted else "no"}\n')
        print(f'{num_improved}/{len(all_rows)} residue(s) improved; written to {residue_rscc_output}')

    def _update_coords(self, structure, coords_by_residue):
        new_coor = structure.coor.copy()
        atom_index = 0
        for chain in structure._pdb_hierarchy.only_model().chains():
            chain_id = chain.id.strip()
            for residue_group in chain.residue_groups():
                resi = int(residue_group.resseq)
                n_atoms = sum(
                    len(atom_group.atoms()) for atom_group in residue_group.atom_groups()
                )
                key = (chain_id, resi)
                if key in coords_by_residue:
                    new_coor[atom_index: atom_index + n_atoms] = coords_by_residue[key]
                atom_index += n_atoms
        structure.coor = new_coor

    def _write_pdb(self, structure, output_path):
        with open(output_path, 'w') as out:
            for atom in structure.get_selected_atoms():
                atom_labels = atom.fetch_labels()
                out.write("{}\n".format(atom_labels.format_atom_record_group()))
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

def main():
    args = build_argparser().parse_args()
    ro = Rotamer_Optimizer(args.dataset, args.model_file, args.output_folder, args.resolution)
    ro.run()

if __name__ == '__main__':
    main()
