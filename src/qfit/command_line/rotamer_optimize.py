import argparse
import re
from collections import namedtuple
from pathlib import Path
import time
import numpy as np
import os

from qfit import Structure
from qfit import XMap
from qfit.xtal.transformer import get_transformer
from qfit.samplers import ChiRotator, CBAngleRotator, BisectingAngleRotator

from qfit.command_line.sidechain_clash import (
    SidechainClashResolver, CLASH_VDW_SCALE, HBOND_CLASH_VDW_SCALE, MAX_CLASH_GROUP_SIZE,
    MAX_CLASH_GROUP_EXPANSIONS, CLASH_DOMAIN_TOP_K, CLASH_SOLVE_NODE_BUDGET,
)

# Sidechain-sidechain clash resolution, using the same shared engine (qfit.command_line.
# sidechain_clash) and the same clash variables as build_final_model.py - see
# Rotamer_Optimizer.run()'s Pass 2 for why rotamer_optimize needs this too: each residue's
# optimized rotamer is picked independently against the untouched base structure, so two
# residues that were BOTH independently found to improve can still clash with each other.
BACKBONE_ATOM_NAMES = {'N', 'CA', 'C', 'O', 'OXT'}

# Per-residue candidate pool built by Rotamer_Optimizer.run() and consumed by
# SidechainClashResolver (via cost=-rscc, since the resolver always minimizes). coor:
# (n_candidates, natoms, 3); rscc: (n_candidates,) - both indexed identically, index 0 always
# the true original (untouched) conformation. A residue that didn't pass the sampling threshold
# (base_rscc >= rscc_threshold) is fixed: exactly one candidate. A sampled residue carries its
# full last-chi-angle sampling pool (up to Rotamer_Optimizer.trim conformers, from
# _sample_sidechains' own top-K trim) as candidates 1..N, alongside its original as candidate 0
# - not just the single best one - so clash resolution has real alternatives to pick from
# instead of only "swap or don't". Reverting a group to index 0 (SidechainClashResolver's
# revert_index=0, see Rotamer_Optimizer.run()) is always safe for exactly this reason.
_RotamerCandidates = namedtuple(
    'RotamerCandidates', ['coor', 'rscc', 'template', 'sidechain_mask', 'fixed'],
)

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
    p.add_argument(
        "--rscc_threshold",
        default=0.5,
        metavar="<float>",
        type=float,
        help="Residues scoring below this RSCC against the event maps are candidates for "
             "rotamer resampling; residues already at/above it are left untouched (default: 0.5).",
    )
    p.add_argument(
        "--rscc_improvement_threshold",
        default=0.1,
        metavar="<float>",
        type=float,
        help="A resampled rotamer is only accepted if it improves RSCC over the starting "
             "conformer by at least this much (default: 0.1).",
    )
    p.add_argument(
        "--clash_vdw_scale",
        default=CLASH_VDW_SCALE,
        metavar="<float>",
        type=float,
        help="Sidechain-sidechain clash detection: fraction of the summed VDW radii "
             "of two sidechain atoms (backbone atoms are never checked) below which "
             f"they are considered clashing (default: {CLASH_VDW_SCALE}). Does not apply "
             "to N/O pairs - see --hbond_clash_vdw_scale. Same mechanism/variable as "
             "build_final_model.py's own flag of the same name.",
    )
    p.add_argument(
        "--hbond_clash_vdw_scale",
        default=HBOND_CLASH_VDW_SCALE,
        metavar="<float>",
        type=float,
        help="Sidechain-sidechain clash detection: same as --clash_vdw_scale, but used "
             "instead of it whenever the pair is one N atom and one O atom, since a real "
             f"hydrogen bond legitimately sits closer than a generic clash (default: "
             f"{HBOND_CLASH_VDW_SCALE})",
    )
    p.add_argument(
        "--max_clash_group_size",
        default=MAX_CLASH_GROUP_SIZE,
        metavar="<int>",
        type=int,
        help="Sidechain-sidechain clash detection: a group of mutually-reselected "
             "clashing residues stops absorbing newly-clashing neighbors once it "
             "would exceed this many residues - the residual clash is logged and "
             f"the group reverted instead (default: {MAX_CLASH_GROUP_SIZE})",
    )
    p.add_argument(
        "--max_clash_group_expansions",
        default=MAX_CLASH_GROUP_EXPANSIONS,
        metavar="<int>",
        type=int,
        help="Sidechain-sidechain clash detection: rounds of \"resolve, then absorb "
             f"new external clashes\" a group is allowed before giving up (default: "
             f"{MAX_CLASH_GROUP_EXPANSIONS})",
    )
    p.add_argument(
        "--clash_domain_top_k",
        default=CLASH_DOMAIN_TOP_K,
        metavar="<int>",
        type=int,
        help="Sidechain-sidechain clash detection: number of highest-RSCC candidates "
             "considered per residue during joint clash-group solving before falling "
             f"back to the full candidate pool (default: {CLASH_DOMAIN_TOP_K})",
    )
    p.add_argument(
        "--clash_solve_node_budget",
        default=CLASH_SOLVE_NODE_BUDGET,
        metavar="<int>",
        type=int,
        help="Sidechain-sidechain clash detection: branch-and-bound search nodes "
             "allowed per clash group before falling back to a heuristic (ICM) "
             f"reassignment (default: {CLASH_SOLVE_NODE_BUDGET})",
    )
    return p

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
        self.sample_angle_range = 3.75
        self.sample_angle_step = 7.5

        # Rotamer sampling
        self.sample_rotamers = True
        self.rotamer_neighborhood = 24
        self.remove_conformers_below_cutoff = False

class Rotamer_Optimizer():
    def __init__(self, dataset_dir, model_file, output_folder, resolution,
                 rscc_threshold=0.5, rscc_improvement_threshold=0.1,
                 clash_vdw_scale=CLASH_VDW_SCALE,
                 hbond_clash_vdw_scale=HBOND_CLASH_VDW_SCALE,
                 max_clash_group_size=MAX_CLASH_GROUP_SIZE,
                 max_clash_group_expansions=MAX_CLASH_GROUP_EXPANSIONS,
                 clash_domain_top_k=CLASH_DOMAIN_TOP_K,
                 clash_solve_node_budget=CLASH_SOLVE_NODE_BUDGET):
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

        self.trim = 20

        # Residues scoring below this against the event maps are candidates for optimization;
        # residues already at/above it are left untouched.
        self.rscc_threshold = rscc_threshold
        # An optimized conformer is only accepted if it improves RSCC over the starting
        # conformer by at least this much.
        self.rscc_improvement_threshold = rscc_improvement_threshold

        self.clash_vdw_scale = clash_vdw_scale
        self.hbond_clash_vdw_scale = hbond_clash_vdw_scale
        self.max_clash_group_size = max_clash_group_size
        self.max_clash_group_expansions = max_clash_group_expansions
        self.clash_domain_top_k = clash_domain_top_k
        self.clash_solve_node_budget = clash_solve_node_budget

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

        # Pass 1: score every residue's starting conformer; for those that pass the sampling
        # threshold, resample and keep its FULL last-chi-angle candidate pool (not just the
        # single best one - see _sample_sidechains' own top-K trim), each candidate individually
        # scored (_calc_rscc_per_conformer). No clash consideration yet: this pass's "top pick"
        # (index 0 = original, else the pool's own best-RSCC candidate) and 0.1-threshold
        # accept/reject decision are made exactly as if clashes didn't exist - clash resolution
        # (Pass 2) only ever reopens a residue's pool when its top pick collides with something.
        self._candidates = {}
        top_pick_idx = {}
        step1_accepted = {}
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
            sidechain_mask = ~np.isin(np.asarray(current_residue.name), list(BACKBONE_ATOM_NAMES))

            #get rscc/coors for starting conformer. Snapshot the true original coordinates
            #right here, before any sampling/scoring runs - _sample_angle/_sample_sidechains
            #below repeatedly reassign self.current_residue.coor while exploring candidates, so
            #current_residue.coor read AFTER them is some arbitrary explored candidate, not the
            #starting conformation. current_residue.coor's getter always extracts a fresh numpy
            #array (not a live view), so this copy is safely insulated from those later writes.
            original_coor = self.current_residue.coor
            self._coor_set = [original_coor]
            base_rscc = self._calc_rscc_all_events()
            print(f'{chain_id}{resi}: base_rscc={base_rscc:.3f}')

            key = (chain_id, resi)
            if base_rscc >= self.rscc_threshold:
                # did not pass the threshold for sampling - fixed, single conformation
                self._candidates[key] = _RotamerCandidates(
                    coor=original_coor[None, :, :],
                    rscc=np.array([base_rscc]),
                    template=current_residue,
                    sidechain_mask=sidechain_mask,
                    fixed=True,
                )
                top_pick_idx[key] = 0
                step1_accepted[key] = False
                continue

            #sample ca-b-y for aromatics
            self._sample_angle()

            #sample sidechains chi - self._coor_set is left holding the last chi angle's own
            #candidate pool (up to self.trim conformers), not collapsed to a single best one
            self._sample_sidechains()

            pool_coor = self._coor_set
            pool_rscc = self._calc_rscc_per_conformer(pool_coor)

            coor = np.concatenate([original_coor[None, :, :], np.stack(pool_coor, axis=0)], axis=0)
            rscc = np.concatenate([[base_rscc], pool_rscc])
            top_idx = int(np.argmax(rscc))
            top_rscc = float(rscc[top_idx])
            accepted = top_idx != 0 and (top_rscc - base_rscc >= self.rscc_improvement_threshold)

            print(f'{chain_id}{resi}: base_rscc={base_rscc:.3f} optimized_rscc={top_rscc:.3f} '
                  f'({len(pool_coor)} pool candidate(s), {time.time() - time0:.1f}s)')

            self._candidates[key] = _RotamerCandidates(
                coor=coor, rscc=rscc, template=current_residue,
                sidechain_mask=sidechain_mask, fixed=False,
            )
            top_pick_idx[key] = top_idx
            step1_accepted[key] = accepted

        # Pass 2: resolve sidechain-sidechain clashes. Seeded from the Pass 1 (clash-blind)
        # decision: a residue that passed the threshold on its own starts at its top pick;
        # everything else (never sampled, or sampled but didn't clear the threshold on its own)
        # starts at its original conformation - it never earned a swap in the first place, so it
        # isn't given one just to help a neighbor, though it can still BE a clash partner.
        # _resolveGroup reopens every movable member's full candidate pool (not just its top
        # pick and original) to find the best-scoring clash-free combination.
        chosen_idx = {
            key: (top_pick_idx[key] if step1_accepted.get(key) else 0)
            for key in self._candidates
        }
        resolver = SidechainClashResolver(
            self._candidates, cost_of=lambda c: -c.rscc,
            clash_vdw_scale=self.clash_vdw_scale,
            hbond_clash_vdw_scale=self.hbond_clash_vdw_scale,
            max_clash_group_size=self.max_clash_group_size,
            max_clash_group_expansions=self.max_clash_group_expansions,
            clash_domain_top_k=self.clash_domain_top_k,
            clash_solve_node_budget=self.clash_solve_node_budget,
            revert_index=0, group_label='rotamer clash group',
        )
        resolver.resolve_sidechain_clashes(chosen_idx)

        # Pass 3: redetermine which residues pass, using whatever candidate each one actually
        # ended up on after clash resolution - which may differ from its Pass 1 top pick, since
        # a clash group's best joint combination doesn't have to be each member's own individual
        # best. A residue only counts as accepted if its FINAL candidate both survived clash
        # resolution (isn't index 0) and still clears the 0.1 threshold against its own base_rscc.
        accepted_coords = {}
        improved_coords = {}
        all_rows = []
        num_improved = 0
        for key, cand in self._candidates.items():
            chain_id, resi = key
            if cand.fixed:
                all_rows.append((chain_id, resi, float(cand.rscc[0]), None, False))
                continue

            base_rscc = float(cand.rscc[0])
            top_rscc = float(cand.rscc[top_pick_idx[key]])

            final_idx = chosen_idx[key]
            final_rscc = float(cand.rscc[final_idx])
            accepted = final_idx != 0 and (final_rscc - base_rscc >= self.rscc_improvement_threshold)
            if accepted:
                accepted_coords[key] = cand.coor[final_idx]
                num_improved += 1
            # fitted.pdb is a pure diagnostic (see its comment below) - reports each residue's
            # own Pass 1 top pick, unaffected by clash resolution or the 0.1 threshold.
            if top_rscc > base_rscc:
                improved_coords[key] = cand.coor[top_pick_idx[key]]
            all_rows.append((chain_id, resi, base_rscc, top_rscc, accepted))

        # Defensive reset, belt-and-suspenders on top of the .copy() fixes in
        # _calc_rscc_all_events/_convert_and_score_rotamer: explicitly re-apply every processed
        # residue's true original coordinates to self.base_structure before writing anything.
        # cand.coor[0] is always the original (captured before any sampling ran - see Pass 1),
        # for both fixed and movable residues. This guarantees a rejected residue's own
        # coordinates in the output are exactly its starting ones, regardless of whether some
        # other, still-undiscovered path also mutates self.base_structure during scoring.
        original_coords = {key: cand.coor[0] for key, cand in self._candidates.items()}
        self._update_coords(self.base_structure, original_coords)

        # fitted.pdb carries every residue whose resampled conformer improved RSCC at all, even
        # if it didn't clear the acceptance threshold - copy the (now-clean) untouched structure
        # before applying accepted_coords below (accepted_coords is a subset of improved_coords).
        fitted_structure = self.base_structure.copy()
        self._update_coords(fitted_structure, improved_coords)
        fitted_output = self.output_path + '/fitted.pdb'
        self._write_pdb(fitted_structure, fitted_output)

        # rotamer_optimized.pdb only carries residues that survived clash resolution AND
        # cleared the acceptance threshold
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

        #get residue from base structure. .copy() is required, not cosmetic: extract() shares
        #the base structure's own live atom storage rather than copying it (confirmed via object
        #identity), and get_conformers_mask/get_conformers_densities below write each scored
        #candidate's coordinates onto the passed-in residue as a side effect of building the
        #xray structure they sample density from - without .copy() here, that silently leaves
        #self.base_structure's real, persistent atoms mutated to whatever candidate happened to
        #be scored last, corrupting every residue that gets sampled (accepted or not).
        residue = self.base_structure.extract(f"chain {chainid} and resi {resi}").copy()

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

            #get residue from base structure - .copy() required, see the matching comment in
            #_convert_and_score_rotamer for why (extract() alone shares self.base_structure's
            #live atoms, and the transformer mutates them as a scoring side effect).
            residue = self.base_structure.extract(f"chain {chainid} and resi {resi}").copy()

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

    def _calc_rscc_per_conformer(self, coor_set):
        """Like _calc_rscc_all_events, but returns one RSCC per conformer in coor_set (still the
        max across event maps for each) instead of collapsing every (conformer, event map) pair
        down to a single global max. Used to score every candidate in a residue's last-chi-angle
        sampling pool individually, so clash resolution has real alternatives - not just the
        single best one - to search over. Operates on `coor_set` directly rather than
        self._coor_set so it doesn't disturb whatever the caller is using that for."""
        scaled_bulk_solvent = 0
        n = len(coor_set)
        per_conformer = np.full(n, -np.inf)

        (chainid, resi, icode) = self.current_residue.identifier_tuple
        default_bfactor = 20
        bfactor_array = [default_bfactor] * n

        for event_map_name in list(self.event_maps.keys()):
            # .copy() required - see the matching comment in _convert_and_score_rotamer.
            residue = self.base_structure.extract(f"chain {chainid} and resi {resi}").copy()
            transformer = get_transformer("qfit", residue, self.event_maps_models[event_map_name])

            mask = transformer.get_conformers_mask(coor_set, self._rmask)
            target = self.event_maps[event_map_name].array[mask]
            for i, density in enumerate(transformer.get_conformers_densities(coor_set, bfactor_array)):
                model = density[mask]
                np.maximum(model, scaled_bulk_solvent, out=model)
                correlation_matrix = np.corrcoef(model, target)
                rscc = correlation_matrix[0, 1]
                if rscc > per_conformer[i]:
                    per_conformer[i] = rscc

        return per_conformer

def main():
    args = build_argparser().parse_args()
    ro = Rotamer_Optimizer(args.dataset, args.model_file, args.output_folder, args.resolution,
                            args.rscc_threshold, args.rscc_improvement_threshold,
                            args.clash_vdw_scale, args.hbond_clash_vdw_scale,
                            args.max_clash_group_size, args.max_clash_group_expansions,
                            args.clash_domain_top_k, args.clash_solve_node_budget)
    ro.run()

if __name__ == '__main__':
    main()
