from abc import ABC
import itertools
import logging
import os
from string import ascii_uppercase
from collections import namedtuple

import numpy as np
import tqdm
import timeit
import time
import concurrent.futures


from .backbone import NullSpaceOptimizer
from .relabel import RelabellerOptions, Relabeller
from .samplers import ChiRotator, CBAngleRotator, BisectingAngleRotator
from .solvers import SolverError, get_qp_solver_class, get_miqp_solver_class
from .structure import Structure, Segment, calc_rmsd
from .structure.clash import ClashDetector
from .structure.math import adp_ellipsoid_axes
from .structure.residue import residue_type
from .structure.rotamers import ROTAMERS
from .validator import Validator
from .xtal.transformer import get_transformer

from scipy.spatial.transform import Rotation as R
from scipy.linalg import svd

logger = logging.getLogger(__name__)

MIQPSolutionStats = namedtuple(
    "MIQPSolutionStats", ["threshold", "BIC", "rss", "objective_value", "weights"]
)

DEFAULT_RMSD_CUTOFF = 0.01
MAX_CONFORMERS = 10000
MIN_OCCUPANCY = 0.002

class QFitOptions:
    def __init__(self):
        # General options
        self.directory = "."
        self.verbose = False
        self.debug = False
        self.write_intermediate_conformers = False
        self.label = None
        self.qscore = None
        self.map = None
        self.residue = None
        self.structure = None
        self.em = False
        self.cryo_em_ligand = False
        self.scale_info = None
        self.cryst_info = None
        self.transformer = "cctbx"
        # the FFT routine is different for qfit as well - this flag allows us
        # to experiment with gridding
        self.transformer_map_coeffs = "cctbx"
        self.expand_to_p1 = None

        # Density preparation options
        self.density_cutoff = 0.3
        self.density_cutoff_value = -1
        self.subtract = True
        self.padding = 8.0
        self.waters_clash = True

        # Density creation options
        self.map_type = None
        self.resolution = None
        self.resolution_min = None
        self.scattering = "xray"
        self.omit = False
        self.scale = True
        self.scale_rmask = 1.0
        self.bulk_solvent_level = 0.3

        # Sampling options
        self.clash_scaling_factor = 0.75
        self.external_clash = False
        self.dofs_per_iteration = 1
        self.dihedral_stepsize = 6
        self.hydro = False
        self.rmsd_cutoff = DEFAULT_RMSD_CUTOFF

        # MIQP options
        self.qp_solver = None
        self.miqp_solver = None
        self.cardinality = 5
        self._ligand_cardinality = 3
        self.threshold = 0.20
        self.bic_threshold = True
        self.seg_bic_threshold = True

        ### From QFitRotamericResidueOptions
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

        # General settings
        # Exclude certain atoms always during density and mask creation to
        # influence QP / MIQP. Provide a list of atom names, e.g. ['N', 'CA']
        # TODO not implemented
        self.exclude_atoms = None

        ### From QFitLigandOptions
        self.selection = None
        self.cif_file = None
        # RDKit options
        self.numConf = None
        self.smiles = None
        self.ligand_bic = None
        self.ligand_rmsd = None
        self.rot_range = None
        self.trans_range = None
        self.rotation_step = None
        self.flip_180 = False
        self.ligand_rmsd = None

        ### From QFitSegmentOptions
        self.fragment_length = None
        self.only_segment = False

        ### From QFitProteinOptions
        self.nproc = 1
        self.pdb = None

        ## Placer stuff
        self.placer_ligs = None
        self.target_chain = None
        self.cif_protein_chains = None
        self.pdb_protein_chains = None

    def apply_command_args(self, args):
        for key, value in vars(args).items():
            if hasattr(self, key):
                setattr(self, key, value)
        return self


class _BaseQFit(ABC):
    def __init__(self, conformer, structure, xmap, options, reset_q=True):
        assert options.qp_solver is not None
        assert options.miqp_solver is not None
        self.options = options
        self._set_data(conformer, structure, xmap, reset_q=reset_q)
        if self.options.em == True:
            self.options.scattering = "electron"
            # bulk solvent level is 0 for EM to work with electron SF
            self.options.bulk_solvent_level = 0
            # maximum of 3 conformers can be choosen per residue
            self.options.cardinality = 3

    def _set_data(self, conformer, structure, xmap, reset_q=True):
        """
        Set the basic input data attributes
        conformer: the structure entity being built (e.g. residue)
        structure: the overall input structure
        xmap: XMap object
        """
        self.conformer = conformer
        self.structure = structure
        self.xmap = xmap
        self._initialize_properties(reset_q=reset_q)

    def _initialize_properties(self, reset_q=True):
        """
        Set various internal attributes derived from the input data
        """
        if reset_q:
            self.conformer.q = 1
        self.prng = np.random.default_rng(0)
        self._coor_set = [self.conformer.coor]
        self._occupancies = [self.conformer.q]
        self._bs = [self.conformer.b]
        self._smax = None
        self._simple = True
        self._rmask = 1.5
        self._cd = lambda: NotImplemented
        reso = None
        # if self.xmap.resolution.high is not None:
        #     reso = self.xmap.resolution.high
        # elif self.options.resolution is not None:
        #     reso = self.options.resolution

        # if reso is not None:
        #     self._smax = 1 / (2 * reso)
        #     self._simple = False
        #     self._rmask = 0.5 + reso / 3.0

        # self._smin = None
        # if self.xmap.resolution.low is not None:
        #     self._smin = 1 / (2 * self.xmap.resolution.low)
        # elif self.options.resolution_min is not None:
        #     self._smin = 1 / (2 * self.options.resolution_min)

        # self._xmap_model = self.xmap.zeros_like(self.xmap)
        # self._xmap_model2 = self.xmap.zeros_like(self.xmap)

        # # To speed up the density creation steps, reduce symmetry to P1
        # self._xmap_model.set_space_group("P1")
        # self._xmap_model2.set_space_group("P1")
        # self._voxel_volume = self.xmap.unit_cell.calc_volume()
        # self._voxel_volume /= self.xmap.array.size

    @property
    def directory_name(self):
        dname = self.options.directory
        return dname

    @property
    def file_ext(self):
        # better to get this from the source than rely on it being propagated
        # in the structure object
        path_fields = self.options.structure.split(".")
        if path_fields[-1] == "gz":
            return ".".join(path_fields[-2:])
        return path_fields[-1]

    def get_conformers(self):
        if len(self._occupancies) < len(self._coor_set):
            # Generate an array filled with 1.0 to match the length of the coordinate set
            self._occupancies = [1.0] * len(self._coor_set)
        conformers = []
        for q, coor, b in zip(self._occupancies, self._coor_set, self._bs):
            conformer = self.conformer.copy()
            conformer = conformer.extract(
                f"resi {self.conformer.resi[0]} and " f"chain {self.conformer.chain[0]}"
            )
            conformer.q = q
            conformer.coor = coor
            conformer.b = b
            conformers.append(conformer)
        return conformers

    def _get_transformer(self, *args, **kwds):
        return get_transformer(self.options.transformer, *args, **kwds)

    def _update_transformer(self, conformer):
        self.conformer = conformer
        self._transformer = self._get_transformer(
            conformer,
            self._xmap_model,
            smax=self._smax,
            smin=self._smin,
            simple=self._simple,
            em=self.options.em,
        )
        logger.debug(
            "[_BaseQFit._update_transformer]: Initializing radial density lookup table."
        )
        self._transformer.initialize()

    def _subtract_transformer(self, residue, structure):
        # Select the atoms whose density we are going to subtract:
        subtract_structure = structure.extract_neighbors(residue, self.options.padding)
        if not self.options.waters_clash:
            subtract_structure = subtract_structure.extract("resn", "HOH", "!=")
        logger.debug("Subtracting density for %d atoms", subtract_structure.natoms)

        # Calculate the density that we are going to subtract:
        self._subtransformer = self._get_transformer(
            subtract_structure,
            self._xmap_model2,
            smax=self._smax,
            smin=self._smin,
            simple=self._simple,
            em=self.options.em,
        )
        self._subtransformer.initialize()
        self._subtransformer.reset(full=True)
        self._subtransformer.density()
        if self.options.em == False:
            # Set the lowest values in the map to the bulk solvent level:
            logger.debug("Setting bulk solvent level %f",
                         self.options.bulk_solvent_level)
            np.maximum(
                self._subtransformer.xmap.array,
                self.options.bulk_solvent_level,
                out=self._subtransformer.xmap.array,
            )

        # Subtract the density:
        logger.debug("Histogram of input map: %s",
                     np.histogram(self.xmap.array))
        logger.debug("Histogram of subtracted map: %s",
                     np.histogram(self._subtransformer.xmap.array))
        if self.options.debug:
            self.xmap.tofile("before_subtraction.ccp4")
            logger.debug("Writing subtrated map to subtracted.ccp4")
            self._subtransformer.xmap.tofile("subtracted.ccp4")
        self.xmap.array -= self._subtransformer.xmap.array
        logger.debug("Histogram of output map: %s",
                     np.histogram(self.xmap.array))
        if self.options.debug:
            self.xmap.tofile("after_subtraction.ccp4")

    def _convert(self, save_debug_maps_prefix=None):
        """Convert structures to densities and extract relevant values for (MI)QP."""
        logger.info("Converting conformers to density")
        mask = self._transformer.get_conformers_mask(
            self._coor_set, self._rmask)
        # the mask is a boolean array
        nvalues = mask.sum()
        logger.debug("%d grid points masked out of %s", nvalues, mask.size)
        self._target = self.xmap.array[mask]
        if save_debug_maps_prefix:
            self.xmap.save_mask(mask, f"{save_debug_maps_prefix}_mask.ccp4")
            self.xmap.save_masked_map(mask, f"{save_debug_maps_prefix}_target.ccp4")
        logger.debug("Histogram of current target values: %s",
                     str(np.histogram(self._target)))

        logger.debug(f"Transforming to density for {nvalues} map points")
        nmodels = len(self._coor_set)
        self._models = np.zeros((nmodels, nvalues), float)
        for n, density in enumerate(self._transformer.get_conformers_densities(
                                    self._coor_set, self._bs)):
            if save_debug_maps_prefix:
                self.xmap.save_masked_map(
                    mask,
                    f"{save_debug_maps_prefix}_conformer_{n:05d}.ccp4",
                    map_data=density)
            model = self._models[n]
            map_values = density[mask]
            model[:] = map_values
            np.maximum(model, self.options.bulk_solvent_level, out=model)
        logger.debug("Histogram of final model values: %s",
                     str(np.histogram(self._models[-1])))

    def _solve_qp(self):
        # Create and run solver
        logger.info("Solving QP for %d target values with %d models",
                    len(self._target), len(self._models))
        qp_solver_class = get_qp_solver_class(self.options.qp_solver)
        solver = qp_solver_class(self._target, self._models)
        solver.solve_qp()

        # Update occupancies from solver weights
        self._occupancies = solver.weights  # pylint: disable=no-member

        # Return solver's objective value (|ρ_obs - Σ(ω ρ_calc)|)
        return solver.objective_value

    def _solve_miqp(
        self,
        cardinality,
        threshold,
        loop_range=[1.0, 0.5, 0.33, 0.25, 0.2],
        do_BIC_selection=None,
        segment=None, terminal=True
    ):
        # set loop range differently for EM
        if self.options.em:
            loop_range = [1.0, 0.5, 0.33, 0.25]
        # Set the default (from options) if it hasn't been passed as an argument
        if do_BIC_selection is None:
            do_BIC_selection = self.options.bic_threshold
        if terminal ==  False:
            loop_range = [0.2]
            do_BIC_slection = False

        # Create solver
        logger.info("Solving MIQP for %d target values with %d models",
                    len(self._target), len(self._models))
        miqp_solver_class = get_miqp_solver_class(self.options.miqp_solver)
        assert len(self._models) > 0
        solver = miqp_solver_class(self._target, self._models)

        # Threshold selection by BIC:
        if do_BIC_selection:
            # Iteratively test decreasing values of the threshold parameter tdmin (threshold)
            # to determine if the better fit (RSS) justifies the use of a more complex model (k)
            miqp_solutions = []
            for threshold in loop_range:
                solver.solve_miqp(cardinality=None, threshold=threshold)
                rss = solver.objective_value * self._voxel_volume
                n = len(self._target)
                natoms = self._coor_set[0].shape[0]
                nconfs = np.sum(solver.weights >= MIN_OCCUPANCY)  # pylint: disable=no-member
                model_params_per_atom = 3 + int(self.options.sample_bfactors)
                k = (
                    model_params_per_atom * natoms * nconfs * 0.8
                )  # hyperparameter 0.8 determined to be the best cut off between too many conformations and improving Rfree
                if segment is not None:
                    k = nconfs  # for segment, we only care about the number of conformations come out of MIQP. Considering atoms penalizes this too much
                if self.options.ligand_bic:
                    k = nconfs * natoms
                BIC = n * np.log(rss / n) + k * np.log(n)
                solution = MIQPSolutionStats(
                    threshold=threshold,
                    BIC=BIC,
                    rss=rss,
                    objective_value=solver.objective_value.copy(),
                    weights=solver.weights.copy(),
                )
                miqp_solutions.append(solution)

            # Update occupancies from solver weights
            miqp_solution_lowest_bic = min(miqp_solutions, key=lambda sol: sol.BIC)
            self._occupancies = miqp_solution_lowest_bic.weights  # pylint: disable=no-member
            # Return solver's objective value (|ρ_obs - Σ(ω ρ_calc)|)
            return miqp_solution_lowest_bic.objective_value

        else:
            # Run solver with specified parameters
            solver.solve_miqp(cardinality=cardinality, threshold=threshold)
            # Update occupancies from solver weights
            self._occupancies = solver.weights  # pylint: disable=no-member
            # Return solver's objective value (|ρ_obs - Σ(ω ρ_calc)|)
            return solver.objective_value

    def sample_b(self):
        """Create copies of conformers that vary in B-factor.
        For all conformers selected, create a copy with the B-factor vector by a scaling factor.
        It is intended that this will be run after a QP step (to help save time)
        and before an MIQP step.
        """
        # don't sample b-factors with em
        if not self.options.sample_bfactors or self.options.em:
            return
        logger.info("Sampling B-factors for %s...", self.conformer)
        new_coor = []
        new_bfactor = []
        multiplication_factors = [1.0, 1.3, 1.5, 0.9, 0.5]
        coor_b_pairs = zip(self._coor_set, self._bs)
        for (coor, b), multi in itertools.product(coor_b_pairs, multiplication_factors):
            new_coor.append(coor)
            new_bfactor.append(b * multi)
        self._coor_set = new_coor
        self._bs = new_bfactor

    def _zero_out_most_similar_conformer(self, merge=False):
        """Zero-out the lowest occupancy, most similar conformer.

        Find the most similar pair of conformers, based on backbone RMSD.
        Of these, remove the conformer with the lowest occupancy.
        This is done by setting its occupancy to 0.

        This aims to reduce the 'non-convex objective' errors we encounter during qFit-segment MIQP.
        These errors are likely due to a degenerate conformers, causing a non-invertible matrix.
        """
        n_confs = len(self._coor_set)

        # Make a square matrix for pairwise RMSDs, where
        #   - the lower triangle (and diagonal) are np.inf
        #   - the upper triangle contains the pairwise RMSDs (k=1 to exclude diagonal)
        pairwise_rmsd_matrix = np.zeros((n_confs,) * 2)
        pairwise_rmsd_matrix[np.tril_indices(n_confs)] = np.inf
        for i, j in zip(*np.triu_indices(n_confs, k=1)):
            pairwise_rmsd_matrix[i, j] = calc_rmsd(self._coor_set[i], self._coor_set[j])

        # Which coords have the lowest RMSD?
        #   `idx_low_rmsd` will contain the coordinates of the lowest value in the pairwise matrix
        #   a.k.a. the indices of the closest confs
        idx_low_rmsd = np.array(
            np.unravel_index(
                np.argmin(pairwise_rmsd_matrix), pairwise_rmsd_matrix.shape
            )
        )
        low_rmsd = pairwise_rmsd_matrix[tuple(idx_low_rmsd)]
        logger.debug(
            f"Lowest RMSD between conformers {idx_low_rmsd.tolist()}: {low_rmsd:.06f} Å"
        )

        # Of these, which has the lowest occupancy?
        occs_low_rmsd = self._occupancies[idx_low_rmsd]
        idx_to_zero, idx_to_keep = idx_low_rmsd[occs_low_rmsd.argsort()]

        # Assign conformer we want to remove with an occupancy of 0
        logger.debug(
            f"Zeroing occupancy of conf {idx_to_zero} (of {n_confs}): "
            f"occ={self._occupancies[idx_to_zero]:.06f} vs {self._occupancies[idx_to_keep]:.06f}"
        )
        self._save_intermediate(prefix="qp_remove")
        # Conditionally add the occupancy of the removed conformer to the kept one
        if merge:
            self._occupancies[idx_to_keep] += self._occupancies[idx_to_zero]

        # Set the occupancy of the removed conformer to 0
        self._occupancies[idx_to_zero] = 0

    def _update_conformers(self, cutoff=MIN_OCCUPANCY):
        """Removes conformers with occupancy lower than cutoff.

        Args:
            cutoff (float, optional): Lowest acceptable occupancy for a conformer.
                Cutoff should be in range (0 < cutoff < 1).
        """
        logger.debug("Updating conformers based on occupancy")

        # Check that all arrays match dimensions.
        assert len(self._occupancies) == len(self._coor_set) == len(self._bs)

        # Filter all arrays & lists based on self._occupancies
        # NB: _coor_set and _bs are lists (not arrays). We must compress, not slice.
        filterarray = self._occupancies >= cutoff
        self._occupancies = self._occupancies[filterarray]
        self._coor_set = list(itertools.compress(self._coor_set, filterarray))
        self._bs = list(itertools.compress(self._bs, filterarray))

        logger.debug(f"Remaining valid conformations: {len(self._coor_set)}")

    def _write_intermediate_conformers(self, prefix="conformer", coord_array=None):
        # Use coord_array if provided, otherwise use self._coor_set
        if coord_array is None:
            coord_array = self._coor_set

        for n, coor in enumerate(coord_array):
            self.conformer.coor = coor
            fname = os.path.join(self.directory_name, f"{prefix}_{n}.pdb")
            self.conformer.tofile(fname)

    def _save_intermediate(self, prefix):
        if self.options.write_intermediate_conformers:
            self._write_intermediate_conformers(prefix)

    def write_maps(self):
        """Write out model and difference map."""
        if np.allclose(self.xmap.origin, 0):
            ext = "ccp4"
        else:
            ext = "mrc"

        for q, coor, b in zip(self._occupancies, self._coor_set, self._bs):
            self.conformer.q = q
            self.conformer.coor = coor
            self.conformer.b = b
            self._transformer.density()
        fname = os.path.join(self.directory_name, f"model.{ext}")
        self._transformer.xmap.tofile(fname)
        self._transformer.xmap.array -= self.xmap.array
        fname = os.path.join(self.directory_name, f"diff.{ext}")
        self._transformer.xmap.tofile(fname)
        self._transformer.reset(full=True)

    @property
    def primary_entity(self):
        return self.conformer

    def _get_peptide_bond_exclude_list(self, residue, segment, partner_id):
        """Exclude peptide bonds from clash detector"""
        index = segment.find(partner_id)

        def _get_norm(idx1, idx2):
            xyz1 = residue.get_xyz([idx1])[0]
            xyz2 = segment.get_xyz([idx2])[0]
            return np.linalg.norm(xyz1 - xyz2)

        exclude = []
        if index > 0:
            N_index = residue.select("name", "N")[0]
            N_neighbor = segment.residues[index - 1]
            neighbor_C_index = N_neighbor.select("name", "C")[0]
            if _get_norm(N_index, neighbor_C_index) < 2:
                coor = N_neighbor.get_xyz(neighbor_C_index)
                exclude.append((N_index, coor))
        if index < len(segment.residues) - 1:
            C_index = residue.select("name", "C")[0]
            C_neighbor = segment.residues[index + 1]
            neighbor_N_index = C_neighbor.select("name", "N")[0]
            if _get_norm(C_index, neighbor_N_index) < 2:
                coor = C_neighbor.get_xyz(neighbor_N_index)
                exclude.append((C_index, coor))
        return exclude

    def detect_clashes(self):
        if not hasattr(self, "_cd"):
            raise NotImplementedError("Clash detector needs initialization")
        else:
            self._cd()

    def is_clashing(self):
        return ((self.options.external_clash and
                 (self.detect_clashes() or self.primary_entity.clashes() > 0)) or
                (self.primary_entity.clashes() > 0))

    def is_same_rotamer(self, rotamer, chis):
        # Check if the residue configuration corresponds to the
        # current rotamer
        dchi_max = 360 - self.options.rotamer_neighborhood
        for curr_chi, rotamer_chi in zip(chis, rotamer):
            delta_chi = abs(curr_chi - rotamer_chi)
            if dchi_max > delta_chi > self.options.rotamer_neighborhood:
                return False
        return True

    def is_conformer_below_cutoff(self, coor, active_mask):
        if self.options.remove_conformers_below_cutoff:
            values = self.xmap.interpolate(coor[active_mask])
            mask = self.primary_entity.e[active_mask] != "H"
            if np.min(values[mask]) < self.options.density_cutoff:
                return True
        return False

    def get_sampling_window(self):
        if self.primary_entity.resn[0] != "PRO":
            return np.arange(
                -self.options.rotamer_neighborhood,
                self.options.rotamer_neighborhood + self.options.dihedral_stepsize,
                self.options.dihedral_stepsize,
            )
        else:
            return [0]

    def identify_core_and_sidechain(self, mol):
        """
        Identify branched sections of ligand
        """
        # Get the ring info of the molecule
        ri = mol.GetRingInfo()
        ring_atoms = ri.AtomRings()

        if len(ring_atoms) == 0:  # No rings in the molecule
            # Use the largest connected component as the core
            components = Chem.rdmolops.GetMolFrags(mol, asMols=False)
            core_atoms = max(components, key=len)
        else:
            # Use the largest ring system as the core
            core_atoms = max(ring_atoms, key=len)

        # Identify terminal atoms, atoms bound to no more than one atom & not in the core
        terminal_atoms = [
            atom.GetIdx()
            for atom in mol.GetAtoms()
            if atom.GetDegree() == 1 and atom.GetIdx() not in core_atoms
        ]

        all_side_chain_atoms = []
        # loop through terminal atoms
        for t_atom in terminal_atoms:
            side_chain_atoms = []
            atom = mol.GetAtomWithIdx(t_atom)
            while (
                atom.GetIdx() not in core_atoms
                and atom.GetIdx() not in side_chain_atoms
            ):
                # Ensure the atom is not part of a ring
                if atom.IsInRing():
                    break
                side_chain_atoms.append(atom.GetIdx())
                neighbors = [
                    x.GetIdx()
                    for x in atom.GetNeighbors()
                    if x.GetIdx() not in core_atoms
                    and x.GetIdx() not in side_chain_atoms
                ]
                if not neighbors:  # No more atoms to explore
                    break
                atom = mol.GetAtomWithIdx(
                    neighbors[0]
                )  # Move to the next atom in the chain

            # Check if the side chain is at least 4 atoms long
            if len(side_chain_atoms) >= 4:
                all_side_chain_atoms.extend(side_chain_atoms)
        length_side_chain = len(all_side_chain_atoms)
        return all_side_chain_atoms, length_side_chain

    def apply_translations(self, conformation, translation_range):
        translation_range = int(translation_range)
        translated_conformations = []
        # translate conformers in x, y, z directions based on input range
        for dx in np.linspace(-translation_range, translation_range, num=3):
            for dy in np.linspace(-translation_range, translation_range, num=3):
                for dz in np.linspace(-translation_range, translation_range, num=3):
                    translation_vector = np.array([dx, dy, dz])
                    translated_conformation = conformation + translation_vector
                    translated_conformations.append(translated_conformation)
        return translated_conformations

    def apply_rotations(self, conformation, rotation_range, step):
        rotation_range = int(rotation_range)
        step = int(step)
        rotated_conformations = [conformation]  # Include the original conformation
        center = conformation.mean(axis=0)  # Compute the center of the conformation
        for angle in range(-rotation_range, rotation_range + step, step):
            for axis in ["x", "y", "z"]:
                r = R.from_euler(axis, np.radians(angle), degrees=False)
                rotation_matrix = r.as_matrix()
                # Apply rotation around the center
                rotated_conformation = (
                    np.dot(conformation - center, rotation_matrix.T) + center
                )
                rotated_conformations.append(rotated_conformation)
        return rotated_conformations

class QFitBindingSite(_BaseQFit):
    def __init__(self, conformer, structure, xmap, options):
        super().__init__(conformer, structure, xmap, options)
        self.protein_chains = self._make_protein_chains_key()
        self.residues_in_binding_site_pdb, self.residues_in_binding_site_cif = self._determine_bindingsite()
        self._match_res_identity()
        self.base_binding_site = self._get_base_bindingsite()

    def _make_protein_chains_key(self):
        """This function makes a list of tuples that store which protein chains we will be building the binding site from.
        It relies on the cif_protein_chains and pdb_protein_chains arguments. It is necessary to do this because keeping
        track of how chain information is stored in the pdb structure and the placer model (which follows the cif file)
        is a pain in the ass and idk how to do it automatically.
        
        Getting rid of the need to specify this is an area of potential improvement"""

        protein_chains = []
        for i in range(len(self.options.pdb_protein_chains.split('-'))):
            pdb_chain = self.options.pdb_protein_chains.split('-')[i]
            cif_chain = self.options.cif_protein_chains.split('-')[i]
            protein_chains.append((pdb_chain[0], int(pdb_chain[1:]), cif_chain[0], int(cif_chain[1:])))

        return protein_chains

    def _determine_bindingsite(self):
        """Determines where binding site is.
        Binding Site includes all residues in any of the Placer models
        and the ligand.
        """

        time0 = time.time()
        placer_model = Structure.fromfile(self.options.placer_ligs)
        placer_models = placer_model.split_models()
        residues_in_binding_site_pdb = {}
        residues_in_binding_site_cif = {}

        for model in placer_models:
            for protein_chain in self.protein_chains:
                pdb_chain_id = protein_chain[0]
                pdb_chain_start = protein_chain[1]
                cif_chain_id = protein_chain[2]
                cif_chain_start = protein_chain[3]

                chain = [ch for ch in model._pdb_hierarchy.only_model().chains() if ch.id.strip() == cif_chain_id]

                ###sometimes chains may not be in in all placer models. This filters all cases where than chain does not appear in placer models
                if len(chain) > 0:
                    chain = chain[0]

                    if pdb_chain_id not in residues_in_binding_site_pdb:
                        residues_in_binding_site_pdb.update({pdb_chain_id: []})

                    if cif_chain_id not in residues_in_binding_site_cif:
                        residues_in_binding_site_cif.update({cif_chain_id: []})

                    for residue in chain.residue_groups():
                        res_num = int(residue.resseq)
                        if res_num not in residues_in_binding_site_cif[cif_chain_id]:
                            residues_in_binding_site_cif[cif_chain_id].append(res_num)

                        res_num = res_num + pdb_chain_start - cif_chain_start
                        if res_num not in residues_in_binding_site_pdb[pdb_chain_id]:
                            residues_in_binding_site_pdb[pdb_chain_id].append(res_num)

        print(f'found binding pocket in {time.time() - time0} seconds')

        return residues_in_binding_site_pdb, residues_in_binding_site_cif

    def _match_res_identity(self):
        """This function determines if the residue identity for all residues in the binding
        site match between the placer model and the base structure. It doesnt do anything
        but is a valuable check than translating between the cif and pdb is working well."""

        placer_model = Structure.fromfile(self.options.placer_ligs)
        placer_models = placer_model.split_models()

        base_structure_res_identities = {}
        placer_structure_res_identities = {}

        #build placer model res dict
        for model in placer_models:
            for protein_chain in self.protein_chains:
                cif_chain_id = protein_chain[2]
                placer_chain = [ch for ch in model._pdb_hierarchy.only_model().chains() if ch.id.strip() == cif_chain_id]

                if cif_chain_id not in placer_structure_res_identities:
                    placer_structure_res_identities.update({cif_chain_id: []})

                if len(placer_chain) > 0:
                    placer_chain = placer_chain[0]

                    for res_index in range(len(self.residues_in_binding_site_cif[cif_chain_id])):
                        res = [res for res in placer_chain.residue_groups() if int(res.resseq) == self.residues_in_binding_site_cif[cif_chain_id][res_index]]

                        if len(res) > 0:
                            res = res[0]
                            res_info = int(res.resseq), res.only_atom_group().resname.strip()
                            if res_info not in placer_structure_res_identities[cif_chain_id]:
                                placer_structure_res_identities[cif_chain_id].append(res_info)

        for key in list(placer_structure_res_identities.keys()):
            placer_structure_res_identities[key] = sorted(placer_structure_res_identities[key])

        #build base structure res dict
        for protein_chain in self.protein_chains:
            pdb_chain_id = protein_chain[0]
            base_chain = [ch for ch in self.structure._pdb_hierarchy.only_model().chains() if ch.id.strip() == pdb_chain_id][0]

            if pdb_chain_id not in base_structure_res_identities:
                base_structure_res_identities.update({pdb_chain_id: []})

            for res_index in range(len(self.residues_in_binding_site_pdb[pdb_chain_id])):
                res = [res for res in base_chain.residue_groups() if int(res.resseq) == self.residues_in_binding_site_pdb[pdb_chain_id][res_index]][0]
                res_info = int(res.resseq), res.only_atom_group().resname.strip()
                if res_info not in base_structure_res_identities[pdb_chain_id]:
                    base_structure_res_identities[pdb_chain_id].append(res_info)

        for key in list(base_structure_res_identities.keys()):
            base_structure_res_identities[key] = sorted(base_structure_res_identities[key])

        ###Now assert that the res identities are the same
        for protein_chain in self.protein_chains:
            pdb_chain_id = protein_chain[0]
            cif_chain_id = protein_chain[2]
            try:
                assert len(placer_structure_res_identities[cif_chain_id]) == len(base_structure_res_identities[pdb_chain_id])
            except AssertionError:
                print(f'length of placer residue identity dictionary and base structure residue identity dictionary are different for chain {key}')

            for i in range(len(placer_structure_res_identities[cif_chain_id])):
                try:
                    assert placer_structure_res_identities[cif_chain_id][i][1] == base_structure_res_identities[pdb_chain_id][i][1]
                except AssertionError:
                    print(f'Miss matching residue at placer chain {cif_chain_id} residue {placer_structure_res_identities[cif_chain_id][i][0]} and base structure chain {pdb_chain_id} and residue {base_structure_res_identities[cif_chain_id][i][0]}')
                    print(f'Placer model gives {placer_structure_res_identities[cif_chain_id][i][1]} and base structure gives {base_structure_res_identities[pdb_chain_id][i][1]}')

    def _get_base_bindingsite(self):
        """Makes a substructure corresponding to all residues in the binding site
        from residues in the base structure.
        """

        base_bindingsite = None
        for protein_chain in self.protein_chains:
            pdb_chain_id = protein_chain[0]

            for res_num in self.residues_in_binding_site_pdb[pdb_chain_id]:
                residue = self.structure.extract(f"chain {pdb_chain_id} and resid {res_num}")

                if base_bindingsite is None:
                    base_bindingsite = residue
                else:
                    base_bindingsite = base_bindingsite.combine(residue)

        ####Add in the ligand
        ligand = self.structure.extract(f"chain {self.options.selection.split(',')[0]} and resid {self.options.selection.split(',')[1]}")
        base_bindingsite = base_bindingsite.combine(ligand)

        return base_bindingsite

    def run(self):
        time0=time.time()
        self._getBindingSiteConformers()
        print(f'built binding site conformers in {time.time() - time0}')

    def _getBindingSiteConformers(self):
        """This function gets the coors of all conformers of the binding site for each of the placer models
        Since not all placer models contain all residues of the binding site, if a binding site
        residue is not present in the placer model, it takes it from the base model instead."""
        
        np.set_printoptions(threshold=np.inf)
        placer_model = Structure.fromfile(self.options.placer_ligs)
        placer_models = placer_model.split_models()

        placer_coor_sets = []
        for model in placer_models:
            placer_coor_set = np.full(self.base_binding_site.coor.shape, np.nan)
            index = 0

            for protein_chain in self.protein_chains:
                pdb_chain_id = protein_chain[0]
                cif_chain_id = protein_chain[2]

                placer_chain = [ch for ch in model._pdb_hierarchy.only_model().chains() if ch.id.strip() == cif_chain_id]

                #if this chain exists in this placer model, build with some placer residues
                if len(placer_chain) > 0:
                    placer_chain = placer_chain[0]

                    for res_index in range(len(self.residues_in_binding_site_cif[cif_chain_id])):
                        res = [res for res in placer_chain.residue_groups() if int(res.resseq) == self.residues_in_binding_site_cif[cif_chain_id][res_index]]

                        #if this residue exists in this placer model, use it
                        if len(res) > 0:
                            res = res[0]
                            residue = model.extract(f'chain {cif_chain_id} and resid {res.resseq}')
                            for i in range(residue.coor.shape[0]):
                                placer_coor_set[(index + i),:] = residue.coor[i,:]
                            index += residue.coor.shape[0]
                            index -= 1

                        #else get coords from base structure
                        else:
                            residue = self.structure.extract(f"chain {pdb_chain_id} and resid {self.residues_in_binding_site_pdb[pdb_chain_id][res_index]}")
                            for i in range(residue.coor.shape[0]):
                                placer_coor_set[(index + i),:] = residue.coor[i,:]
                            index += residue.coor.shape[0]

                #else get coords from base structure
                else:
                    for res_index in range(len(self.residues_in_binding_site_pdb[pdb_chain_id])):
                        residue = self.structure.extract(f"chain {pdb_chain_id} and resid {self.residues_in_binding_site_pdb[pdb_chain_id][res_index]}")
                        for i in range(residue.coor.shape[0]):
                            placer_coor_set[(index + i),:] = residue.coor[i,:]
                        index += residue.coor.shape[0]
                        

            ###Add ligand from placer model
            ligand = model.extract(f"chain {self.options.target_chain} and resid {self.options.selection.split(',')[1]}")
            for i in range(ligand.coor.shape[0]):
                placer_coor_set[(index + i),:] = ligand.coor[i,:]

            #Check for nans in coor set
            nan_flag = np.isnan(placer_coor_set).any()
            if nan_flag:
                raise ValueError('Something went wrong building the placer conformers. At least 1 conformer has nan for a coor value.')
            print(nan_flag)
            placer_coor_sets.append(placer_coor_set)





