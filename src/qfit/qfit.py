import itertools
import logging
import os
from sys import argv
import copy
from string import ascii_uppercase
from collections import namedtuple
import subprocess
import numpy as np
import tqdm
import timeit
import time
import concurrent.futures


from .backbone import NullSpaceOptimizer, adp_ellipsoid_axes
from .clash import ClashDetector
from .samplers import ChiRotator, CBAngleRotator, BondRotator, BisectingAngleRotator
from .samplers import CovalentBondRotator, GlobalRotator
from .samplers import RotationSets, Translator
from .solvers import SolverError, get_qp_solver_class, get_miqp_solver_class
from .structure import Structure, _Segment, calc_rmsd
from .structure.residue import residue_type
from .structure.ligand import BondOrder
from .transformer import Transformer
from .validator import Validator
from .volume import XMap
from .scaler import MapScaler
from .relabel import RelabellerOptions, Relabeller
from .structure.rotamers import ROTAMERS

from rdkit import Chem
from rdkit.Chem import AllChem
import random
from scipy.spatial.transform import Rotation as R
import math
from scipy.linalg import svd


logger = logging.getLogger(__name__)

# Create a namedtuple 'class' (struct) which carries info about an MIQP solution
MIQPSolutionStats = namedtuple(
    "MIQPSolutionStats", ["threshold", "BIC", "rss", "objective_value", "weights"]
)


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
        self.rmsd_cutoff = 0.01

        # MIQP options
        self.qp_solver = None
        self.miqp_solver = None
        self.cardinality = 5
        self._ligand_cardinality = 3
        self.threshold = 0.20
        self.bic_threshold = True
        self.seg_bic_threshold = True

        ### From QFitRotamericResidueOptions, QFitCovalentLigandOptions
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

    def apply_command_args(self, args):
        for key, value in vars(args).items():
            if hasattr(self, key):
                setattr(self, key, value)
        return self


class _BaseQFit:
    def __init__(self, conformer, structure, xmap, options):
        self.structure = structure
        self.conformer = conformer
        self.conformer.q = 1
        self.xmap = xmap
        self.options = options
        self.prng = np.random.default_rng(0)
        self._coor_set = [self.conformer.coor]
        self._occupancies = [1.0]
        self._bs = [self.conformer.b]
        self._smax = None
        self._simple = True
        self._rmask = 1.5

        if self.options.em == True:
            self.options.scattering = "electron"  # making sure electron SF are used
            self.options.bulk_solvent_level = (
                0  # bulk solvent level is 0 for EM to work with electron SF
            )
            self.options.cardinality = (
                3  # maximum of 3 conformers can be choosen per residue
            )

        reso = None
        if self.xmap.resolution.high is not None:
            reso = self.xmap.resolution.high
        elif options.resolution is not None:
            reso = options.resolution

        if reso is not None:
            self._smax = 1 / (2 * reso)
            self._simple = False
            self._rmask = 0.5 + reso / 3.0

        self._smin = None
        if self.xmap.resolution.low is not None:
            self._smin = 1 / (2 * self.xmap.resolution.low)
        elif self.options.resolution_min is not None:
            self._smin = 1 / (2 * options.resolution_min)

        self._xmap_model = xmap.zeros_like(self.xmap)
        self._xmap_model2 = xmap.zeros_like(self.xmap)

        # To speed up the density creation steps, reduce symmetry to P1
        self._xmap_model.set_space_group("P1")
        self._xmap_model2.set_space_group("P1")
        self._voxel_volume = self.xmap.unit_cell.calc_volume()
        self._voxel_volume /= self.xmap.array.size

    @property
    def directory_name(self):
        dname = self.options.directory
        return dname

    def get_conformers(self):
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

    def _update_transformer(self, structure):
        self.conformer = structure
        self._transformer = Transformer(
            structure,
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

        # Calculate the density that we are going to subtract:
        self._subtransformer = Transformer(
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
            np.maximum(
                self._subtransformer.xmap.array,
                self.options.bulk_solvent_level,
                out=self._subtransformer.xmap.array,
            )

        # Subtract the density:
        self.xmap.array -= self._subtransformer.xmap.array

    def _convert(self):  # default is to manipulate the maps
        """Convert structures to densities and extract relevant values for (MI)QP."""
        logger.info("Converting conformers to density")
        logger.debug("Masking")
        self._transformer.reset(full=True)
        for n, coor in enumerate(self._coor_set):
            self.conformer.coor = coor
            self._transformer.mask(self._rmask)
        mask = self._transformer.xmap.array > 0
        self._transformer.reset(full=True)

        nvalues = mask.sum()
        self._target = self.xmap.array[mask]

        logger.debug("Density")
        nmodels = len(self._coor_set)
        self._models = np.zeros((nmodels, nvalues), float)
        for n, coor in enumerate(self._coor_set):
            self.conformer.coor = coor
            self.conformer.b = self._bs[n]
            self._transformer.density()
            model = self._models[n]
            model[:] = self._transformer.xmap.array[mask]
            np.maximum(model, self.options.bulk_solvent_level, out=model)
            self._transformer.reset(full=True)
    def _solve_qp(self):
        # Create and run solver
        logger.info("Solving QP")
        qp_solver_class = get_qp_solver_class(self.options.qp_solver)
        solver = qp_solver_class(self._target, self._models)
        solver.solve_qp()

        # Update occupancies from solver weights
        self._occupancies = solver.weights

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
        logger.info("Solving MIQP")
        miqp_solver_class = get_miqp_solver_class(self.options.miqp_solver)
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
                nconfs = np.sum(solver.weights >= 0.002)
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
            self._occupancies = miqp_solution_lowest_bic.weights

            # Return solver's objective value (|ρ_obs - Σ(ω ρ_calc)|)
            return miqp_solution_lowest_bic.objective_value

        else:
            # Run solver with specified parameters
            solver.solve_miqp(cardinality=cardinality, threshold=threshold)

            # Update occupancies from solver weights
            self._occupancies = solver.weights

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
        if (
            self.options.write_intermediate_conformers
        ):  # Output all conformations before we remove them
            self._write_intermediate_conformers(prefix="qp_remove")

        # Conditionally add the occupancy of the removed conformer to the kept one
        if merge:
            self._occupancies[idx_to_keep] += self._occupancies[idx_to_zero]

        # Set the occupancy of the removed conformer to 0
        self._occupancies[idx_to_zero] = 0

    def _update_conformers(self, cutoff=0.002):
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

            data = {}
            for attr in self.conformer.data:
                array1 = getattr(self.conformer, attr)
                data[attr] = array1[self.conformer.active]
            Structure(data).tofile(fname)

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


class QFitRotamericResidue(_BaseQFit):
    def __init__(self, residue, structure, xmap, options):
        super().__init__(residue, structure, xmap, options)
        self.residue = residue
        self.chain = residue.chain[0]
        self.resn = residue.resn[0]
        self.resi, self.icode = residue.id
        self.identifier = f"{self.chain}/{self.resn}{''.join(map(str, residue.id))}"

        # Check if residue has complete heavy atoms. If not, complete it.
        expected_atoms = np.array(self.residue._rotamers["atoms"])
        missing_atoms = np.isin(
            expected_atoms, test_elements=self.residue.name, invert=True
        )
        if np.any(missing_atoms):
            logger.info(
                f"[{self.identifier}] {', '.join(expected_atoms[missing_atoms])} "
                f"are not in structure. Rebuilding residue."
            )
            try:
                self.residue.complete_residue()
            except RuntimeError as e:
                raise RuntimeError(
                    f"[{self.identifier}] Unable to rebuild residue."
                ) from e
            else:
                logger.debug(
                    f"[{self.identifier}] Rebuilt. Now has {', '.join(self.residue.name)} atoms.\n"
                    f"{self.residue.coor}"
                )

            # Rebuild to include the new residue atoms
            index = len(self.structure.record)
            mask = getattr(self.residue, "atomid") >= index
            data = {}
            for attr in self.structure.data:
                data[attr] = np.concatenate(
                    (getattr(structure, attr), getattr(residue, attr)[mask])
                )

            # Create a new Structure, and re-extract the current residue from it.
            #     This ensures the object-tree (i.e. residue.parent, etc.) is correct.
            # Then reinitialise _BaseQFit with these.
            #     This ensures _BaseQFit class attributes (self.residue, but also
            #     self._b, self._coor_set, etc.) come from the rebuilt data-structure.
            #     It is essential to have uniform dimensions on all data before
            #     we begin sampling.
            structure = Structure(data)
            residue = structure[self.chain].conformers[0][residue.id]
            super().__init__(residue, structure, xmap, options)
            self.residue = residue
            if self.options.debug:
                # This should be output with debugging, and shouldn't
                #   require the write_intermediate_conformers option.
                fname = os.path.join(self.directory_name, "rebuilt_residue.pdb")
                self.residue.tofile(fname)

        # If including hydrogens, report if any H are missing
        if options.hydro:
            expected_h_atoms = np.array(self.residue._rotamers["hydrogens"])
            missing_h_atoms = np.isin(
                expected_h_atoms, test_elements=self.residue.name, invert=True
            )
            if np.any(missing_h_atoms):
                logger.warning(
                    f"[{self.identifier}] Missing hydrogens "
                    f"{', '.join(expected_atoms[missing_h_atoms])}."
                )

        # Ensure clash detection matrix is filled.
        self.residue._init_clash_detection(self.options.clash_scaling_factor)

        # Get the segment that the residue belongs to
        self.segment = None
        for segment in self.structure.segments:
            if segment.chain[0] == self.chain and self.residue in segment:
                index = segment.find(self.residue.id)
                if (len(segment[index].name) == len(self.residue.name)) and (
                    segment[index].altloc[-1] == self.residue.altloc[-1]
                ):
                    self.segment = segment
                    logger.info(f"[{self.identifier}] index {index} in {segment}")
                    break
        if self.segment is None:
            rtype = residue_type(self.residue)
            if rtype == "rotamer-residue":
                self.segment = _Segment(
                    self.structure.data,
                    selection=self.residue._selection,
                    parent=self.structure,
                    residues=[self.residue],
                )
                logger.warning(
                    f"[{self.identifier}] Could not determine protein segment. "
                    f"Using independent protein segment."
                )

        # Set up the clash detector, exclude the bonded interaction of the N and
        # C atom of the residue
        if len(self.structure.record) != len(self.residue.record): #if residue == structure, then we do not need clash detector
            self._setup_clash_detector()
        if options.subtract:
            self._subtract_transformer(self.residue, self.structure)
        self._update_transformer(self.residue)

    @property
    def directory_name(self):
        # This is a QFitRotamericResidue, so we're working on a residue.
        # Which residue are we working on?
        resi_identifier = self.residue.shortcode

        dname = os.path.join(super().directory_name, resi_identifier)
        return dname

    def _setup_clash_detector(self):
        residue = self.residue
        segment = self.segment
        index = segment.find(residue.id)
        # Exclude peptide bonds from clash detector
        exclude = []
        if index > 0:
            N_index = residue.select("name", "N")[0]
            N_neighbor = segment.residues[index - 1]
            neighbor_C_index = N_neighbor.select("name", "C")[0]
            if (
                np.linalg.norm(residue._coor[N_index] - segment._coor[neighbor_C_index])
                < 2
            ):
                coor = N_neighbor._coor[neighbor_C_index]
                exclude.append((N_index, coor))
        if index < len(segment.residues) - 1:
            C_index = residue.select("name", "C")[0]
            C_neighbor = segment.residues[index + 1]
            neighbor_N_index = C_neighbor.select("name", "N")[0]
            if (
                np.linalg.norm(residue._coor[C_index] - segment._coor[neighbor_N_index])
                < 2
            ):
                coor = C_neighbor._coor[neighbor_N_index]
                exclude.append((C_index, coor))

        # Obtain atoms with which the residue can clash
        resi, icode = residue.id
        chainid = self.segment.chain[0]
        if icode:
            selection_str = f"not (resi {resi} and icode {icode} and chain {chainid})"
            receptor = self.structure.extract(selection_str)
        else:
            sel_str = f"not (resi {resi} and chain {chainid})"
            receptor = self.structure.extract(sel_str).copy()

        # Find symmetry mates of the receptor
        starting_coor = self.structure.coor.copy()
        iterator = self.xmap.unit_cell.iter_struct_orth_symops
        for symop in iterator(self.structure, target=self.residue, cushion=5):
            if symop.is_identity():
                continue
            logger.debug(
                f"[{self.identifier}] Building symmetry partner for clash_detector: [R|t]\n"
                f"{symop}"
            )
            self.structure.rotate(symop.R)
            self.structure.translate(symop.t)
            receptor = receptor.combine(self.structure)
            self.structure.coor = starting_coor

        self._cd = ClashDetector(
            residue,
            receptor,
            exclude=exclude,
            scaling_factor=self.options.clash_scaling_factor,
        )

    def run(self):
        start_time = timeit.default_timer()
        if self.options.sample_backbone:
            self._sample_backbone()

        if self.options.sample_angle:
            self._sample_angle()

        if self.residue.nchi >= 1 and self.options.sample_rotamers:
            self._sample_sidechain()

        # Check that there are no self-clashes within a conformer
        self.residue.active = True
        self.residue.update_clash_mask()
        new_coor_set = []
        new_bs = []
        for coor, b in zip(self._coor_set, self._bs):
            self.residue.coor = coor
            self.residue.b = b
            if self.options.external_clash:
                if not self._cd() and self.residue.clashes() == 0:
                    new_coor_set.append(coor)
                    new_bs.append(b)
            elif self.residue.clashes() == 0:
                new_coor_set.append(coor)
                new_bs.append(b)
            self._coor_set = new_coor_set
            self._bs = new_bs

        # QP score conformer occupancy
        self._convert()
        self._solve_qp()
        self._update_conformers()
        if self.options.write_intermediate_conformers:
            self._write_intermediate_conformers(prefix="qp_solution_residue")

        # MIQP score conformer occupancy
        self.sample_b()
        self._convert()
        self._solve_miqp(
            threshold=self.options.threshold, cardinality=self.options.cardinality
        )
        self._update_conformers()
        if self.options.write_intermediate_conformers:
            self._write_intermediate_conformers(prefix="miqp_solution_residue")

        # Now that the conformers have been generated, the resulting
        # conformations should be examined via GoodnessOfFit:
        validator = Validator(
            self.xmap, self.xmap.resolution, self.options.directory, em=self.options.em
        )

        if self.xmap.resolution.high < 3.0:
            cutoff = 0.7 + (self.xmap.resolution.high - 0.6) / 3.0
        else:
            cutoff = 0.5 * self.xmap.resolution.high

        self.validation_metrics = validator.GoodnessOfFit(
            self.conformer, self._coor_set, self._occupancies, cutoff
        )
        # End of processing

    def _sample_backbone(self):
        # Check if residue has enough neighboring residues
        index = self.segment.find(self.residue.id)
        # active = self.residue.active
        nn = self.options.neighbor_residues_required
        if index < nn or index + nn > len(self.segment):
            logger.info(
                f"[_sample_backbone] Not enough (<{nn}) neighbor residues: "
                f"lower {index < nn}, upper {index + nn > len(self.segment)}"
            )
            return
        segment = self.segment[(index - nn) : (index + nn + 1)]

        # We will work on CB for all residues, but O for GLY.
        atom_name = "CB"
        if self.residue.resn[0] == "GLY":
            atom_name = "O"

        # Determine directions for backbone sampling
        atom = self.residue.extract("name", atom_name)
        try:
            u_matrix = [
                [atom.u00[0], atom.u01[0], atom.u02[0]],
                [atom.u01[0], atom.u11[0], atom.u12[0]],
                [atom.u02[0], atom.u12[0], atom.u22[0]],
            ]
            directions = adp_ellipsoid_axes(u_matrix)
            logger.debug(f"[_sample_backbone] u_matrix = {u_matrix}")
            logger.debug(f"[_sample_backbone] directions = {directions}")
        except AttributeError:
            logger.info(
                f"[{self.identifier}] Got AttributeError for directions at Cβ. Treating as isotropic B, using Cβ-Cα, C-N,(Cβ-Cα × C-N) vectors."
            )
            # Choose direction vectors as Cβ-Cα, C-N, and then (Cβ-Cα × C-N)
            # Find coordinates of Cα, Cβ, N atoms
            pos_CA = self.residue.extract("name", "CA").coor[0]
            pos_CB = self.residue.extract("name", atom_name).coor[
                0
            ]  # Position of CB for all residues except for GLY, which is position of O
            pos_N = self.residue.extract("name", "N").coor[0]
            # Set x, y, z = Cβ-Cα, Cα-N, (Cβ-Cα × Cα-N)
            vec_x = pos_CB - pos_CA
            vec_y = pos_CA - pos_N
            vec_z = np.cross(vec_x, vec_y)
            # Normalize
            vec_x /= np.linalg.norm(vec_x)
            vec_y /= np.linalg.norm(vec_y)
            vec_z /= np.linalg.norm(vec_z)

            directions = np.vstack([vec_x, vec_y, vec_z])

        # If we are missing a backbone atom in our segment,
        #     use current coords for this residue, and abort.

        # we only want to look for backbone in the segment we are using for inverse kinetmatics, not the entire protein
        for n, residue in enumerate(self.segment.residues[(index - 3) : (index + 3)]):
            for backbone_atom in ["N", "CA", "C", "O"]:
                if backbone_atom not in residue.name:
                    relative_to_residue = n - index
                    logger.warning(
                        f"[{self.identifier}] Missing backbone atom in segment residue {relative_to_residue:+d}."
                    )
                    logger.warning(f"[{self.identifier}] Skipping backbone sampling.")
                    self._coor_set.append(self.segment[index].coor)
                    self._bs.append(self.conformer.b)
                    return

        # Retrieve the amplitudes and stepsizes from options.
        sigma = self.options.sample_backbone_sigma
        bba, bbs = (
            self.options.sample_backbone_amplitude,
            self.options.sample_backbone_step,
        )
        assert bba >= bbs > 0

        # Create an array of amplitudes to scan:
        #   We start from stepsize, making sure to stop only after bba.
        #   Also include negative amplitudes.
        eps = ((bba / bbs) / 2) * np.finfo(
            float
        ).epsneg  # ε to avoid FP errors in arange
        amplitudes = np.arange(start=bbs, stop=bba + bbs - eps, step=bbs)
        amplitudes = np.concatenate([-amplitudes[::-1], amplitudes])

        # Optimize in torsion space to achieve the target atom position
        optimizer = NullSpaceOptimizer(segment)
        start_coor = atom.coor[0]  # We are working on a single atom.
        torsion_solutions = []
        for amplitude, direction in itertools.product(amplitudes, directions):
            endpoint = start_coor + amplitude * direction
            optimize_result = optimizer.optimize(atom_name, endpoint)
            torsion_solutions.append(optimize_result["x"])

        # Capture starting coordinates for the segment, so that we can restart after every rotator
        starting_coor = segment.coor
        for solution in torsion_solutions:
            optimizer.rotator(solution)
            self._coor_set.append(self.segment[index].coor)
            self._bs.append(self.conformer.b)
            segment.coor = starting_coor

        logger.debug(
            f"[_sample_backbone] Backbone sampling generated {len(self._coor_set)} conformers."
        )
        if self.options.write_intermediate_conformers:
            self._write_intermediate_conformers(
                prefix=f"sample_backbone_segment{index:03d}"
            )

    def _sample_angle(self):
        """Sample residue conformations by flexing α-β-γ angle.

        Only operates on residues with large aromatic sidechains
            (Trp, Tyr, Phe, His) where CG is a member of the aromatic ring.
        Here, slight deflections of the ring are likely to lead to better-
            scoring conformers when we scan χ(Cα-Cβ) and χ(Cβ-Cγ).

        This angle does not exist in {Gly, Ala}, and it does not make sense to
            sample this angle in Pro.

        We choose not to do this sampling on smaller amino acids for reason of
            a trade-off in complexity. Sampling the α-β-γ angle increases the
            amount & density of sampled conformations, but is unlikely to yield
            better quality sampling (conformations with good matches to
            electron density). In the case of Arg, the planar aromatic region
            does not start at CG. Later χ angles are more effective at moving
            the guanidinium group.
        """
        # Only operate on aromatics!
        if self.resn not in ("TRP", "TYR", "PHE", "HIS"):
            logger.debug(
                f"[{self.identifier}] Not F/H/W/Y. Cα-Cβ-Cγ angle sampling skipped."
            )
            return

        # Limit active atoms
        active_names = ("N", "CA", "C", "O", "CB", "H", "HA", "CG", "HB2", "HB3")
        selection = self.residue.select("name", active_names)
        self.residue.active = False
        self.residue._active[selection] = True
        self.residue.update_clash_mask()
        active_mask = self.residue.active

        # Define sampling range
        angles = np.arange(
            -self.options.sample_angle_range,
            self.options.sample_angle_range + self.options.sample_angle_step,
            self.options.sample_angle_step,
        )

        # Commence sampling, building on each existing conformer in self._coor_set
        new_coor_set = []
        new_bs = []
        for coor in self._coor_set:
            self.residue.coor = coor
            # Initialize rotator
            perp_rotator = CBAngleRotator(self.residue)
            # Rotate about the axis perpendicular to CB-CA and CB-CG vectors
            for perp_angle in angles:
                perp_rotator(perp_angle)
                coor_rotated = self.residue.coor
                # Initialize rotator
                bisec_rotator = BisectingAngleRotator(self.residue)
                # Rotate about the axis bisecting the CA-CA-CG angle for each angle you sample across the perpendicular axis
                for bisec_angle in angles:
                    self.residue.coor = coor_rotated  # Ensure that the second rotation is applied to the updated coordinates from first rotation
                    bisec_rotator(bisec_angle)
                    coor = self.residue.coor

                    # Move on if these coordinates are unsupported by density
                    if self.options.remove_conformers_below_cutoff:
                        values = self.xmap.interpolate(coor[active_mask])
                        mask = self.residue.e[active_mask] != "H"
                        if np.min(values[mask]) < self.options.density_cutoff:
                            continue

                    # Move on if these coordinates cause a clash
                    if self.options.external_clash:
                        if self._cd() and self.residue.clashes():
                            continue
                    elif self.residue.clashes():
                        continue

                    # Valid, non-clashing conformer found!
                    new_coor_set.append(self.residue.coor)
                    new_bs.append(self.conformer.b)

        # Update sampled coords
        self._coor_set = new_coor_set
        self._bs = new_bs
        logger.debug(f"Bond angle sampling generated {len(self._coor_set)} conformers.")
        if self.options.write_intermediate_conformers:
            self._write_intermediate_conformers(prefix=f"sample_angle")
    
    def _sample_sidechain(self):
        opt = self.options
        start_chi_index = 1
        
        if self.residue.resn[0] != "PRO":
            sampling_window = np.arange(
                -opt.rotamer_neighborhood,
                opt.rotamer_neighborhood + opt.dihedral_stepsize,
                opt.dihedral_stepsize,
            )
        else:
            sampling_window = [0]

        rotamers = self.residue.rotamers
        rotamers.append(
            [self.residue.get_chi(i) for i in range(1, self.residue.nchi + 1)]
        )
        iteration = 0
        while True:
            chis_to_sample = opt.dofs_per_iteration
            if iteration == 0 and (opt.sample_backbone or opt.sample_angle):
                chis_to_sample = max(1, opt.dofs_per_iteration - 1)
            end_chi_index = min(start_chi_index + chis_to_sample, self.residue.nchi + 1)
            iter_coor_set = []
            iter_b_set = (
                []
            )  # track b-factors so that they can be reset along with the coordinates if too many conformers are generated
            for chi_index in range(start_chi_index, end_chi_index):
                # Set active and passive atoms, since we are iteratively
                # building up the sidechain. This updates the internal
                # clash mask.
                self.residue.active = True
                if chi_index < self.residue.nchi:
                    current = self.residue._rotamers["chi-rotate"][chi_index]
                    deactivate = self.residue._rotamers["chi-rotate"][chi_index + 1]
                    selection = self.residue.select("name", deactivate)
                    self.residue._active[selection] = False
                    bs_atoms = list(set(current) - set(deactivate))
                else:
                    bs_atoms = self.residue._rotamers["chi-rotate"][chi_index]

                self.residue.update_clash_mask()
                active = self.residue.active

                logger.info(f"Sampling chi: {chi_index} ({self.residue.nchi})")
                new_coor_set = []
                new_bs = []
                n = 0
                ex = 0
                # For each backbone conformation so far:
                for coor, b in zip(self._coor_set, self._bs):
                    self.residue.coor = coor
                    self.residue.b = b
                    chis = [self.residue.get_chi(i) for i in range(1, chi_index)]
                    # Try each rotamer in the library for this backbone conformation:
                    for rotamer in rotamers:
                        # Check if the current sidechain configuration for this residue
                        # closely matches the rotamer being considered from the library
                        is_this_same_rotamer = True
                        for curr_chi, rotamer_chi in zip(chis, rotamer):
                            diff_chi = abs(curr_chi - rotamer_chi)
                            if (
                                360 - opt.rotamer_neighborhood
                                > diff_chi
                                > opt.rotamer_neighborhood
                            ):
                                is_this_same_rotamer = False
                                break
                        if not is_this_same_rotamer:
                            continue
                        # Set the chi angle to the standard rotamer value.
                        self.residue.set_chi(chi_index, rotamer[chi_index - 1])

                        # Sample around the neighborhood of the rotamer
                        chi_rotator = ChiRotator(self.residue, chi_index)

                        for angle in sampling_window:
                            # Rotate around the chi angle, hitting each of the angle values
                            # in our predetermined, generic chi-angle sampling window
                            n += 1
                            chi_rotator(angle)
                            coor = self.residue.coor

                            # See if this (partial) conformer clashes,
                            # based on a density mask
                            if opt.remove_conformers_below_cutoff:
                                values = self.xmap.interpolate(coor[active])
                                mask = self.residue.e[active] != "H"
                                if np.min(values[mask]) < self.options.density_cutoff:
                                    ex += 1
                                    continue

                            # See if this (partial) conformer clashes (so far),
                            # based on all-atom sterics (if the user wanted that)
                            keep_coor_set = False
                            if self.options.external_clash:
                                if not self._cd() and self.residue.clashes() == 0:
                                    keep_coor_set = True
                            elif self.residue.clashes() == 0:
                                keep_coor_set = True

                            # Based on that, decide whether to keep or reject this (partial) conformer
                            if keep_coor_set:
                                if new_coor_set:
                                    delta = np.array(new_coor_set) - np.array(
                                        self.residue.coor
                                    )
                                    if (
                                        np.sqrt(
                                            min(
                                                np.square((delta))
                                                .sum(axis=2)
                                                .sum(axis=1)
                                            )
                                        )
                                        >= 0.01
                                    ):
                                        new_coor_set.append(self.residue.coor)
                                        new_bs.append(b)
                                    else:
                                        ex += 1
                                else:
                                    new_coor_set.append(self.residue.coor)
                                    new_bs.append(b)
                            else:
                                ex += 1

                iter_coor_set.append(new_coor_set)
                iter_b_set.append(new_bs)
                self._coor_set = new_coor_set
                self._bs = new_bs

                if len(self._coor_set) > 10000:
                    logger.warning(
                        f"[{self.identifier}] Too many conformers generated ({len(self._coor_set)}). Splitting QP scoring."
                    )

                if not self._coor_set:
                    msg = (
                        "No conformers could be generated. Check for initial "
                        "clashes and density support."
                    )
                    raise RuntimeError(msg)

                logger.debug(
                    f"Side chain sampling generated {len(self._coor_set)} conformers"
                )
                if self.options.write_intermediate_conformers:
                    self._write_intermediate_conformers(
                        prefix=f"sample_sidechain_iter{iteration}"
                    )

                if len(self._coor_set) <= 10000:
                    # If <15000 conformers are generated, QP score conformer occupancy normally
                    self._convert()
                    self._solve_qp()
                    self._update_conformers()
                    if self.options.write_intermediate_conformers:
                        self._write_intermediate_conformers(
                            prefix=f"sample_sidechain_iter{iteration}_qp"
                        )
                if len(self._coor_set) > 10000:
                    # If >15000 conformers are generated, split the QP conformer scoring into two
                    temp_coor_set = self._coor_set
                    temp_bs = self._bs

                    # Splitting the arrays into two sections
                    half_index = (
                        len(temp_coor_set) // 2
                    )  # Integer division for splitting
                    section_1_coor = temp_coor_set[:half_index]
                    section_1_bs = temp_bs[:half_index]
                    section_2_coor = temp_coor_set[half_index:]
                    section_2_bs = temp_bs[half_index:]

                    # Process the first section
                    self._coor_set = section_1_coor
                    self._bs = section_1_bs

                    # QP score the first section
                    self._convert()
                    self._solve_qp()
                    self._update_conformers()
                    if self.options.write_intermediate_conformers:
                        self._write_intermediate_conformers(
                            prefix=f"sample_sidechain_iter{iteration}_qp"
                        )

                    # Save results from the first section
                    qp_temp_coor = self._coor_set
                    qp_temp_bs = self._bs

                    # Process the second section
                    self._coor_set = section_2_coor
                    self._bs = section_2_bs

                    # QP score the second section
                    self._convert()
                    self._solve_qp()
                    self._update_conformers()
                    if self.options.write_intermediate_conformers:
                        self._write_intermediate_conformers(
                            prefix=f"sample_sidechain_iter{iteration}_qp"
                        )

                    # Save results from the second section
                    qp_2_temp_coor = self._coor_set
                    qp_2_temp_bs = self._bs

                    # Concatenate the results from both sections
                    self._coor_set = np.concatenate(
                        (qp_temp_coor, qp_2_temp_coor), axis=0
                    )
                    self._bs = np.concatenate((qp_temp_bs, qp_2_temp_bs), axis=0)

                # MIQP score conformer occupancy
                self.sample_b()
                self._convert()
                self._solve_miqp(
                    threshold=self.options.threshold,
                    cardinality=self.options.cardinality,
                )
                self._update_conformers()
                if self.options.write_intermediate_conformers:
                    self._write_intermediate_conformers(
                        prefix=f"sample_sidechain_iter{iteration}_miqp"
                    )

            # Check if we are done
            if chi_index == self.residue.nchi:
                break

            # Use the next chi angle as starting point, except when we are in
            # the first iteration and have selected backbone sampling and we
            # are sampling more than 1 dof per iteration
            increase_chi = not (
                (opt.sample_backbone or opt.sample_angle)
                and iteration == 0
                and opt.dofs_per_iteration > 1
            )
            if increase_chi:
                start_chi_index += 1
            iteration += 1

    def tofile(self):
        # Save the individual conformers
        conformers = self.get_conformers()
        if len(conformers) == 0:
            msg = "No conformers could be generated. " "Using Deposited conformer."
            logger.warning(msg)
            self._coor_set = [self.residue.coor]
            self._bs = [self.residue.b]
            self._occupancies = [1.0]
            conformers = self.get_conformers()
        for n, conformer in enumerate(conformers, start=1):
            fname = os.path.join(self.directory_name, f"conformer_{n}.pdb")
            conformer.tofile(fname)

        # Make a multiconformer residue
        nconformers = len(conformers)
        if nconformers < 1:
            msg = "No conformers could be generated. " "Check for initial clashes."
            raise RuntimeError(msg)
        mc_residue = Structure.fromstructurelike(conformers[0])
        if nconformers == 1:
            mc_residue.altloc = ""
        else:
            mc_residue.altloc = "A"
            for altloc, conformer in zip(ascii_uppercase[1:], conformers[1:]):
                conformer.altloc = altloc
                mc_residue = mc_residue.combine(conformer)
        mc_residue = mc_residue.reorder()

        # Save the multiconformer residue
        logger.info(f"[{self.identifier}] Saving multiconformer_residue.pdb")
        fname = os.path.join(self.directory_name, f"multiconformer_residue.pdb")
        mc_residue.tofile(fname)


class QFitSegment(_BaseQFit):
    """Determines consistent protein segments based on occupancy and
    density fit"""

    def __init__(self, structure, xmap, options):
        self.segment = structure
        self.conformer = structure
        # self.conformer.q = 1
        self.xmap = xmap
        self.options = options
        self.options.bic_threshold = self.options.seg_bic_threshold
        self.fragment_length = options.fragment_length
        self._coor_set = [self.conformer.coor]
        self._occupancies = [self.conformer.q]
        self._bs = [self.conformer.b]
        self.orderings = []
        self.charseq = []
        self._smax = None
        self._simple = True
        self._rmask = 1.5
        reso = None
        if self.xmap.resolution.high is not None:
            reso = self.xmap.resolution.high
        elif options.resolution is not None:
            reso = options.resolution
        if reso is not None:
            self._smax = 1 / (2 * reso)
            self._simple = False
            self._rmask = 0.5 + reso / 3.0

        self._smin = None
        if self.xmap.resolution.low is not None:
            self._smin = 1 / (2 * self.xmap.resolution.low)
        elif self.options.resolution_min is not None:
            self._smin = 1 / (2 * options.resolution_min)

        self._xmap_model = xmap.zeros_like(self.xmap)
        # To speed up the density creation steps, reduce space group symmetry
        # to P1
        self._xmap_model.set_space_group("P1")
        self._voxel_volume = self.xmap.unit_cell.calc_volume()
        self._voxel_volume /= self.xmap.array.size

    def __call__(self):
        logger.info(
            f"Average number of conformers before qfit_segment run: "
            f"{self.segment.average_conformers():.2f}"
        )
        # Extract hetatms
        hetatms = self.segment.extract("record", "HETATM")
        # Create an empty structure:
        multiconformers = Structure.fromstructurelike(
            self.segment.extract("altloc", "Z")
        )
        segment = []

        # Construct progress iterator
        residue_groups = self.segment.extract("record", "ATOM").residue_groups
        residue_groups_pbar = tqdm.tqdm(
            residue_groups,
            total=self.segment.n_residues,
            desc="Building segments",
            unit="res",
            leave=True,
        )

        # Iterate over all residue groups
        for rg in residue_groups_pbar:
            if rg.resn[0] not in ROTAMERS:
                multiconformers = multiconformers.combine(rg)
                continue
            altlocs = np.unique(rg.altloc)
            naltlocs = len(altlocs)
            multiconformer = []
            CA_single = True
            O_single = True
            CA_pos = None
            O_pos = None
            for altloc in altlocs:
                if altloc == "" and naltlocs > 1:
                    continue
                conformer = Structure.fromstructurelike(
                    rg.extract("altloc", (altloc, ""))
                )
                # Reproducing the code in idmulti.cpp:
                if CA_single and O_single:
                    mask = np.isin(conformer.name, ["CA", "O"])
                    if np.sum(mask) > 2:
                        logger.warning(
                            f"Conformer {altloc} of residue "
                            f"{rg.resi[0]} has more than one coordinate "
                            f"for CA/O atoms."
                        )
                        mask = mask[:2]
                    try:
                        CA_single = np.linalg.norm(CA_pos - conformer.coor[mask][0])
                        CA_single = CA_single <= 0.05
                        O_single = np.linalg.norm(O_pos - conformer.coor[mask][1])
                        O_single = O_single <= 0.05
                    except TypeError:
                        CA_pos, O_pos = [coor for coor in conformer.coor[mask]]
                multiconformer.append(conformer)

            # Check to see if the residue has a single conformer:
            if naltlocs == 1:
                # Process the existing segment
                if len(segment):
                    for path in self.find_paths(segment):
                        multiconformers = multiconformers.combine(path)
                segment = []
                # Set the occupancy of all atoms of the residue to 1
                rg.q = np.ones_like(rg.q)
                # Add the single conformer residue to the
                # existing multiconformer:
                multiconformers = multiconformers.combine(rg)

            # Check if we need to collapse the backbone
            elif CA_single and O_single:
                # Process the existing segment
                if len(segment):
                    for path in self.find_paths(segment):
                        multiconformers = multiconformers.combine(path)
                segment = []
                collapsed = multiconformer[:]
                for multi in collapsed:
                    multiconformers = multiconformers.combine(
                        multi.collapse_backbone(multi.resi[0], multi.chain[0])
                    )

            else:
                segment.append(multiconformer)

        # Teardown progress bar
        residue_groups_pbar.close()

        if len(segment):
            logger.debug(f"Running find_paths for segment of length {len(segment)}")
            for path in self.find_paths(segment):
                multiconformers = multiconformers.combine(path)

        logger.info(
            f"Average number of conformers after qfit_segment run: "
            f"{multiconformers.average_conformers():.2f}"
        )
        multiconformers = multiconformers.reorder()
        #         multiconformers = multiconformers.remove_identical_conformers(
        #             self.options.rmsd_cutoff
        #         )
        multiconformers = (
            multiconformers.normalize_occupancy()
        )  # ensure that sum(occupancy) of residues is equal to one.
        logger.info(
            f"Average number of conformers after removal of identical conformers: "
            f"{multiconformers.average_conformers():.2f}"
        )

        # Build an instance of Relabeller
        relab_options = RelabellerOptions()
        relab_options.apply_command_args(
            self.options
        )  # Update RelabellerOptions with QFitSegmentOptions
        relabeller = Relabeller(multiconformers, relab_options)
        multiconformers = relabeller.run()
        multiconformers = multiconformers.combine(hetatms)
        multiconformers = multiconformers.reorder()
        return multiconformers

    def find_paths(self, segment_original):
        segment = segment_original[:]
        fl = self.fragment_length
        possible_conformers = list(map(chr, range(65, 90)))
        possible_conformers = possible_conformers[
            0 : int(round(1.0 / self.options.threshold))
        ]

        while len(segment) > 1:
            n = len(segment)

            fragment_multiconformers = [segment[i : i + fl] for i in range(0, n, fl)]
            segment = []
            for fragment_multiconformer in fragment_multiconformers:
                fragments = []
                # Create all combinations of alternate residue conformers
                for fragment_conformer in itertools.product(*fragment_multiconformer):
                    fragment = fragment_conformer[0].set_backbone_occ()
                    for element in fragment_conformer[1:]:
                        fragment = fragment.combine(element.set_backbone_occ())
                    combine = True
                    for fragment2 in fragments:
                        if calc_rmsd(fragment.coor, fragment2.coor) < 0.05:
                            combine = False
                            break
                    if combine:
                        fragments.append(fragment)

                # We have the fragments, select consistent optimal set
                self._update_transformer(fragments[0])
                self._coor_set = [fragment.coor for fragment in fragments]
                self._bs = [fragment.b for fragment in fragments]

                # QP score segment occupancy
                self._convert()
                self._solve_qp()

                # Run MIQP in a loop, removing the most similar conformer until a solution is found
                while True:
                    # Update conformers
                    fragments = np.array(fragments)
                    mask = self._occupancies >= 0.002

                    # Drop conformers below cutoff
                    _resi_list = list(fragments[0].residues)
                    logger.debug(
                        f"Removing {np.sum(np.invert(mask))} conformers from "
                        f"fragment {_resi_list[0].shortcode}--{_resi_list[-1].shortcode}"
                    )
                    fragments = fragments[mask]
                    self.sample_b()
                    self._occupancies = self._occupancies[mask]
                    self._coor_set = [fragment.coor for fragment in fragments]
                    self._bs = [fragment.b for fragment in fragments]

                    try:
                        # MIQP score segment occupancy
                        self._convert()
                        self._solve_miqp(
                            threshold=self.options.threshold,
                            cardinality=self.options.cardinality,
                            segment=True,
                            do_BIC_selection=False,
                        )
                    except SolverError:
                        # MIQP failed and we need to remove conformers that are close to each other
                        logger.debug("MIQP failed, dropping a fragment-conformer")
                        self._zero_out_most_similar_conformer()  # Remove conformer
                        if self.options.write_intermediate_conformers:
                            self._write_intermediate_conformers(prefix="miqp_kept")
                        continue
                    else:
                        # No Exceptions here! Solvable!
                        break

                # Update conformers for the last time
                mask = self._occupancies >= 0.002

                # Drop conformers below cutoff
                _resi_list = list(fragments[0].residues)
                logger.debug(
                    f"Removing {np.sum(np.invert(mask))} conformers from "
                    f"fragment {_resi_list[0].shortcode}--{_resi_list[-1].shortcode}"
                )
                for fragment, occ in zip(fragments[mask], self._occupancies[mask]):
                    fragment.q = occ
                segment.append(fragments[mask])

        for path, altloc in zip(segment[0], possible_conformers):
            path.altloc = altloc
        return segment[0]

    def print_paths(self, segment):
        # Calculate the path matrix:
        for k, fragment in enumerate(segment):
            path = []
            for i, residue_altloc in enumerate(fragment.altloc):
                if fragment.name[i] == "CA":
                    path.append(residue_altloc)
                    # coor.append(fragment.coor[i])
            logger.info(f"Path {k+1}:\t{path}\t{fragment.q[-1]}")


class QFitLigand(_BaseQFit):
    def __init__(self, ligand, receptor, xmap, options):
        # Initialize using the base qfit class
        super().__init__(ligand, receptor, xmap, options)

        # Populate useful attributes:
        self.ligand = ligand
        self.receptor = receptor
        self.xmap = xmap
        self.options = options
        csf = self.options.clash_scaling_factor
        self._bs = [self.ligand.b]
        self.num_lig_atoms = ligand.natoms

        # External clash detection:
        self._cd = ClashDetector(
            ligand, receptor, scaling_factor=self.options.clash_scaling_factor
        )

        # set up clash detector with receptor symmetry mates include 
        receptor_starting_coor = self.structure.coor.copy()
        iterator = self.xmap.unit_cell.iter_struct_orth_symops
        for symop in iterator(self.structure, target=self.ligand, cushion=5): # loop through sym mate operations
            self.structure.rotate(symop.R) # rotate
            self.structure.translate(symop.t) # translate
            sym_receptor = receptor.combine(self.structure) # add symm mate operation to the receptor
            self.structure.coor = receptor_starting_coor # reset self.structure.coor

        self._sym_mate_cd = ClashDetector(
            ligand,
            sym_receptor,
            scaling_factor=self.options.clash_scaling_factor,
        )

        # Initialize the transformer
        if options.subtract:
            self._subtract_transformer(self.ligand, self.receptor)
        self._update_transformer(self.ligand)
        self._starting_coor_set = [ligand.coor.copy()]
        self._starting_bs = [ligand.b.copy()]

        # Read in ligand pdb file
        self.ligand_pdb_file = "ligand.pdb"

    def run(self):
        ligand = Chem.MolFromPDBFile(self.ligand_pdb_file)
        # total number of conformers to generate
        if self.options.numConf:
            num_gen_conformers = self.options.numConf
        else: 
            if self.num_lig_atoms <= 25: 
                num_gen_conformers = 5000 # default generate 5,000 conformers for small ligands
            else:
                num_gen_conformers = 7000 # defulat geenrate 7,000 conformers for ligands that are not small  
        
        # check if ligand has long branch/side chain
        logger.debug("Testing branching status of ligand")
        branching_test = self.identify_core_and_sidechain(ligand)
        branching_atoms = branching_test[0]
        length_branching = branching_test[1]

        # set cardinality for cryo-EM modeling
        if self.options.cryo_em_ligand:
            self.options._ligand_cardinality = 2

        # run rdkit conformer generator
        logger.info("Starting RDKit conformer generation")

        if not branching_atoms:
            # if there are no branches, qFit will not run branching search or long chain search. Therefore, 3 methods of sampling remain
            num_conf_for_method = round(num_gen_conformers / 3)
        if branching_atoms:
            if length_branching > 30:
                logger.debug("Ligand has long branches, run long chain search")
                num_conf_for_method = round(num_gen_conformers / 5)
            elif length_branching <= 30:
                num_conf_for_method = round(num_gen_conformers / 4)
        self.num_conf_for_method = num_conf_for_method

        if self.options.flip_180:
            self.flip_180()
        else:
            # Run conformer generation functions in parallel
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.options.nproc) as executor:
                futures = [
                    executor.submit(self.random_unconstrained),
                    executor.submit(self.terminal_atom_const),
                    executor.submit(self.spherical_search),
                ]
                if branching_atoms:
                    if length_branching > 30:
                        futures.append(executor.submit(self.branching_search))
                        futures.append(executor.submit(self.long_chain_search))
                    elif length_branching <= 30:
                        futures.append(executor.submit(self.branching_search))

                for future in concurrent.futures.as_completed(futures):
                    try:
                        future.result()
                    except Exception as exc:
                        logger.error(f"Generated an exception: {exc}")

        logger.info(
            f"Number of generated conformers, before scoring: {len(self._coor_set)}"
        )

        if len(self._coor_set) < 1:
            logger.error("qFit-ligand failed to produce a valid conformer.")
            return

        # Make sure b-factor array is the same length as coordinate array
        if len(self._bs) != len(self._coor_set):
            self._bs = np.tile(self._bs[0], (len(self._coor_set), 1))

        # Pre QP RMSD pruning   
        if self.options.ligand_rmsd:
            logger.debug("RMSD pruning step before QP scoring")
            rmsd_threshold = 0.2
            unique_coor_set = []
            unique_bs = []
            
            # Loop through each conformer and check against accepted ones.
            for idx, coor in enumerate(self._coor_set):
                duplicate_found = False
                for unique_coor in unique_coor_set:
                    rmsd = calc_rmsd(coor, unique_coor)
                    if rmsd < rmsd_threshold:
                        duplicate_found = True
                        break
                if not duplicate_found:
                    unique_coor_set.append(coor)
                    unique_bs.append(self._bs[idx])
            
            logger.info(f"Reduced number of conformers from {len(self._coor_set)} to {len(unique_coor_set)} after pruning duplicates.")
            # Update the conformer and b-factor lists.
            self._coor_set = unique_coor_set
            self._bs = unique_bs

        # QP score conformer occupancy
        logger.debug("Converting densities within run.")
        self._convert()
        logger.info("Solving QP within run.")
        self._solve_qp()
        if self.options.write_intermediate_conformers:
                self._write_intermediate_conformers(prefix="pre_qp")
        logger.debug("Updating conformers within run.")
        self._update_conformers()

        # Only conformeres that pass QP scoring will be rotated and translated for additional sampling
        self.rot_trans()

        # MIQP score conformer occupancy
        logger.info("Solving MIQP within run.")
        # Make sure b-factor array is the same length as coordinate array
        if len(self._bs) != len(self._coor_set):
            self._bs = np.tile(self._bs[0], (len(self._coor_set), 1))
        self._convert()
        if self.options.ligand_bic:
            self._solve_miqp(
                threshold=self.options.threshold,
                cardinality=self.options._ligand_cardinality,
                do_BIC_selection=True,
            )
        if not self.options.ligand_bic:
            self._solve_miqp(
                threshold=self.options.threshold,
                cardinality=self.options._ligand_cardinality, do_BIC_selection=False
            )
        self._update_conformers()

        # Ensure there are no duplicate conformers in self._coor_set
        if self.options.ligand_rmsd:
            has_duplicates = False  # Track if duplicates are found

            for i in range(len(self._coor_set)):
                for j in range(i + 1, len(self._coor_set)):
                    rmsd = calc_rmsd(self._coor_set[i], self._coor_set[j])
                    if rmsd < 0.2:
                        has_duplicates = True
                        print(
                            f"RMSD value less than 0.2 between conformers {i} and {j}: {rmsd}"
                        )
                        break  # Exit the inner loop if a duplicate is found
                if has_duplicates:
                    break  # Exit the outer loop if a duplicate is found

            # If duplicates are found, call _zero_out_most_similar_conformer()
            if has_duplicates:
                self._zero_out_most_similar_conformer(merge=True)
                self._update_conformers()
            # Calculate and log the RMSD between all final conformers
            for i in range(len(self._coor_set)):
                for j in range(i + 1, len(self._coor_set)):
                    rmsd = calc_rmsd(self._coor_set[i], self._coor_set[j])
                    logger.info(f"Final RMSD between conformers {i} and {j}: {rmsd}")

        if self.options.write_intermediate_conformers:
            self._write_intermediate_conformers(prefix="miqp_solution")

        # run symmetry mate clash detector on all conformers out of MIQP
        miqp_coor_set = self._coor_set
        final_idx_set = []
        final_coor_set = []
        final_bs = []

        for idx, conf in enumerate(miqp_coor_set):
            b = self._bs
            self.ligand.coor = conf
            self.ligand.b = [b[0]]
            if not self._sym_mate_cd():
                if final_idx_set: 
                    final_idx_set.append(idx)
                    final_coor_set.append(conf)
                    final_bs.append(b[0])
                else: 
                    final_idx_set.append(idx)
                    final_coor_set.append(conf)
                    final_bs.append(b[0])
        self._coor_set = final_coor_set
        self._bs = final_bs


        logger.info(f"Number of final conformers: {len(self._coor_set)}")

    def random_unconstrained(self):
        """
        Run RDKit with the minimum constraints -- only from the bounds matrix
        """
        ligand = Chem.MolFromPDBFile(self.ligand_pdb_file)

        # RDKit is bad at finding the corect bond types from pdb files, but good at doing so from SMILES string. Use SMILES string as templete for corecting bond orders
        ref_mol = Chem.MolFromSmiles(self.options.smiles)

        # Assign bond orders from the template
        ligand = Chem.AllChem.AssignBondOrdersFromTemplate(ref_mol, ligand)
        ligand = Chem.AddHs(ligand)
        num_conformers = (
            self.num_conf_for_method
        )  # Number of conformers you want to generate

        # Create a copy of the 'ligand' object to generate conformers off of. They will later be aligned to 'ligand' object
        mol = Chem.Mol(ligand)

        logger.info(f"Generating {num_conformers} conformers with no constraints")
        Chem.rdDistGeom.EmbedMultipleConfs(
            mol,
            numConfs=num_conformers,
            useBasicKnowledge=True,
            pruneRmsThresh=self.options.rmsd_cutoff,
        )

        # Minimize the energy of each conformer to find most stable structure
        logger.info("Minimizing energy of each conformer with no constraints")
        mp = AllChem.MMFFGetMoleculeProperties(mol)
        for conf_id in mol.GetConformers():
            ff = AllChem.MMFFGetMoleculeForceField(mol, mp, confId=conf_id.GetId())
            ff.Minimize()

        logger.info("Aligning molecules with no constraints")
        # Align the conformers in "mol" to "ligand" to ensure all structures are properly sitting in the binding site
        ligand_crippen_contribs = Chem.rdMolDescriptors._CalcCrippenContribs(ligand)
        mol_crippen_contribs = Chem.rdMolDescriptors._CalcCrippenContribs(mol)

        for conf_id in mol.GetConformers():
            o3a = Chem.rdMolAlign.GetCrippenO3A(
                mol,
                ligand,
                prbCrippenContribs=mol_crippen_contribs,
                refCrippenContribs=ligand_crippen_contribs,
                prbCid=conf_id.GetId(),
            )
            o3a.Align()

        mol = Chem.RemoveHs(mol)
        ligand = Chem.RemoveHs(ligand)

        # Check for internal/external clashes
        if mol.GetNumConformers() == 0:
            logger.error(
                f"Unconstrained search generated no conformers. Moving onto next sampling function."
            )
        if mol.GetNumConformers() != 0:
            # Check for internal/external clashes
            logger.info("Checking for clashes with no constraints")
            # Store the coordinates of each conformer into numpy array
            new_conformer = mol.GetConformers()
            new_coors = []
            for i, conformer in enumerate(new_conformer):
                coords = conformer.GetPositions()
                new_coors.append(coords)

            new_idx_set = []
            new_coor_set = []
            new_bs = []
            # loop through each rdkit generated conformer
            for idx, conf in enumerate(new_coors):
                b = self._bs
                self.ligand.coor = conf
                self.ligand.b = [b[0]]
                if self.options.external_clash:
                    if not self._cd():
                        if (
                            new_idx_set
                        ):  # if there are already conformers in new_idx_set
                            new_idx_set.append(idx)
                            new_coor_set.append(conf)
                            new_bs.append(b[0])
                        else:
                            new_idx_set.append(idx)
                            new_coor_set.append(conf)
                            new_bs.append(b[0])

                elif not self.ligand.clashes():
                    if new_idx_set:  # if there are already conformers in new_idx_set
                        new_idx_set.append(idx)
                        new_coor_set.append(conf)
                        new_bs.append(b[0])
                    else:
                        new_idx_set.append(idx)
                        new_coor_set.append(conf)
                        new_bs.append(b[0])
            
            if self.options.write_intermediate_conformers:
                self._write_intermediate_conformers("unconstrained_sol", new_coor_set)
                
            # Save new conformers to self
            merged_arr = np.concatenate((self._coor_set, new_coor_set), axis=0)
            merged_bs = np.concatenate((self._bs, new_bs), axis=0)

            self._coor_set = merged_arr
            self._bs = merged_bs

            logger.info(
                f"After unconstrained search, there are {len(self._coor_set)} plausible conformers"
            )

            if len(self._coor_set) < 1:
                logger.warning(
                    f"RDKit conformers not sufficiently diverse. Generated: {len(self._coor_set)} conformers"
                )
                return

        return

    def terminal_atom_const(self):
        """
        Identify the terminal atoms of the ligand and learn distances between those atoms and the rest of the molecule. This results in the generated conformers
        having a more reasonable shape, similar to the deposited model.
        """
        ligand = Chem.MolFromPDBFile(self.ligand_pdb_file)

        # RDKit is bad at finding the corect bond types from pdb files, but good at doing so from SMILES string. Use SMILES string as templete for corecting bond orders
        ref_mol = Chem.MolFromSmiles(self.options.smiles)

        # Assign bond orders from the template
        ligand = Chem.AllChem.AssignBondOrdersFromTemplate(ref_mol, ligand)

        # Find the terminal atoms in the ligand, to be used in coordinate map
        terminal_indices = []
        for atom in ligand.GetAtoms():
            if (
                atom.GetDegree() == 1
            ):  # if only one atom is bound to the current atom then it is terminal
                terminal_indices.append(atom.GetIdx())

        ligand = Chem.AddHs(ligand)

        num_conformers = (
            self.num_conf_for_method
        )  # Number of conformers you want to generate
        # Create a copy of the 'ligand' object to generate conformers off of. They will later be aligned to 'ligand' object
        mol = Chem.Mol(ligand)

        logger.info(
            f"Generating {num_conformers} conformers with terminal atom constraints"
        )
        # Generate conformers
        AllChem.EmbedMultipleConfs(
            mol,
            numConfs=num_conformers,
            useBasicKnowledge=True,
            pruneRmsThresh=self.options.rmsd_cutoff,
            coordMap={
                idx: ligand.GetConformer().GetAtomPosition(idx)
                for idx in terminal_indices
            },
        )

        if mol.GetNumConformers() == 0:
            logger.error(
                f"terminal atom constrained search generated no conformers. Moving onto next sampling function."
            )
        if mol.GetNumConformers() != 0:
            # Minimize the energy of each conformer to find most stable structure
            logger.info(
                "Minimizing energy of each conformer with terminal atom constraints"
            )
            mp = AllChem.MMFFGetMoleculeProperties(mol)
            for conf_id in mol.GetConformers():
                ff = AllChem.MMFFGetMoleculeForceField(mol, mp, confId=conf_id.GetId())
                ff.Minimize()

            logger.info("Aligning molecules with terminal atom constraints")
            # Align the conformers in "mol" to "ligand" to ensure all structures are properly sitting in the binding site
            ligand_crippen_contribs = Chem.rdMolDescriptors._CalcCrippenContribs(ligand)
            mol_crippen_contribs = Chem.rdMolDescriptors._CalcCrippenContribs(mol)

            for conf_id in mol.GetConformers():
                o3a = Chem.rdMolAlign.GetCrippenO3A(
                    mol,
                    ligand,
                    prbCrippenContribs=mol_crippen_contribs,
                    refCrippenContribs=ligand_crippen_contribs,
                    prbCid=conf_id.GetId(),
                )
                o3a.Align()

            mol = Chem.RemoveHs(mol)
            ligand = Chem.RemoveHs(ligand)

            # Check for internal/external clashes
            logger.info("Checking for clashes with terminal atom constraints")
            # Store the coordinates of each conformer into numpy array
            new_conformer = mol.GetConformers()
            new_coors = []
            for i, conformer in enumerate(new_conformer):
                coords = conformer.GetPositions()
                new_coors.append(coords)

            new_idx_set = []
            new_coor_set = []
            new_bs = []

            # loop through each rdkit generated conformer
            for idx, conf in enumerate(new_coors):
                b = self._bs
                self.ligand.coor = conf
                self.ligand.b = [b[0]]
                if self.options.external_clash:
                    if not self._cd():
                        if (
                            new_idx_set
                        ):  # if there are already conformers in new_idx_set
                            new_idx_set.append(idx)
                            new_coor_set.append(conf)
                            new_bs.append(b[0])
                        else:
                            new_idx_set.append(idx)
                            new_coor_set.append(conf)
                            new_bs.append(b[0])

                elif not self.ligand.clashes():
                    if new_idx_set:  # if there are already conformers in new_idx_set
                        new_idx_set.append(idx)
                        new_coor_set.append(conf)
                        new_bs.append(b[0])
                    else:
                        new_idx_set.append(idx)
                        new_coor_set.append(conf)
                        new_bs.append(b[0])

            if self.options.write_intermediate_conformers:
                self._write_intermediate_conformers("terminal_atoms", new_coor_set)
                
            # Save new conformers to self
            merged_arr = np.concatenate((self._coor_set, new_coor_set), axis=0)
            merged_bs = np.concatenate((self._bs, new_bs), axis=0)

            self._coor_set = merged_arr
            self._bs = merged_bs

            logger.info(
                f"After terminal atom search, there are {len(self._coor_set)} plausible conformers"
            )

            if len(self._coor_set) < 1:
                logger.warning(
                    f"RDKit conformers not sufficiently diverse. Generated: {len(self._coor_set)} conformers"
                )
                return

        return

    def spherical_search(self):
        """
        Define a sphere around the deposited ligand, where the radius is the maximum distance from the geometric center to any atom in the molecule. Constrain conformer
        generation to be within this sphere. This is a less restrictive constraint, while still biasing RDKit to generate conformers more likely to sit well in binding site
        """
        ligand = Chem.MolFromPDBFile(self.ligand_pdb_file)

        # Create a sphere around the ligand to constrain the conformation generation
        conf = ligand.GetConformer()
        atom_positions = [conf.GetAtomPosition(i) for i in range(ligand.GetNumAtoms())]

        # Calculate the geometric center of the molecule
        geometric_center = np.mean(
            [np.array(atom_position) for atom_position in atom_positions], axis=0
        )
        # Calculate the radius of the sphere (max distance from center to any atom)
        radius = max(
            np.linalg.norm(np.array(atom_position) - geometric_center)
            for atom_position in atom_positions
        )

        # RDKit is bad at finding the corect bond types from pdb files, but good at doing so from SMILES string. Use SMILES string as templete for corecting bond orders
        ref_mol = Chem.MolFromSmiles(self.options.smiles)

        # Assign bond orders from the template
        ligand = Chem.AllChem.AssignBondOrdersFromTemplate(ref_mol, ligand)
        ligand = Chem.AddHs(ligand)
        num_conformers = (
            self.num_conf_for_method
        )  # Number of conformers you want to generate

        # Create a copy of the 'ligand' object to generate conformers off of. They will later be aligned to 'ligand' object
        mol = Chem.Mol(ligand)

        diameter = 2 * radius
        bounds = Chem.rdDistGeom.GetMoleculeBoundsMatrix(ligand)
        # Modify the bounds matrix to set the upper bounds for all atom pairs
        num_atoms = ligand.GetNumAtoms()
        for i in range(num_atoms):
            for j in range(
                i + 1, num_atoms
            ):  # Only need to do this for the upper triangle
                # Set the upper bound for the distance between atoms i and j
                bounds[i, j] = min(bounds[i, j], diameter)

        # Set up the embedding parameters
        ps = Chem.rdDistGeom.EmbedParameters()
        ps = Chem.rdDistGeom.ETKDGv3()
        ps.randomSeed = 0xF00D
        ps.SetBoundsMat(bounds)
        ps.useBasicKnowledge = True
        ps.pruneRmsThresh = self.options.rmsd_cutoff

        logger.info(
            f"Generating {num_conformers} conformers with spherical constraints"
        )
        Chem.rdDistGeom.EmbedMultipleConfs(mol, numConfs=num_conformers, params=ps)

        # Minimize the energy of each conformer to find most stable structure
        logger.info("Minimizing energy of each conformer with spherical constraints")
        mp = AllChem.MMFFGetMoleculeProperties(mol)
        for conf_id in mol.GetConformers():
            ff = AllChem.MMFFGetMoleculeForceField(mol, mp, confId=conf_id.GetId())
            ff.Minimize()

        logger.info("Aligning molecules with spherical constraints")
        # Align the conformers in "mol" to "ligand" to ensure all structures are properly sitting in the binding site
        ligand_crippen_contribs = Chem.rdMolDescriptors._CalcCrippenContribs(ligand)
        mol_crippen_contribs = Chem.rdMolDescriptors._CalcCrippenContribs(mol)

        for conf_id in mol.GetConformers():
            o3a = Chem.rdMolAlign.GetCrippenO3A(
                mol,
                ligand,
                prbCrippenContribs=mol_crippen_contribs,
                refCrippenContribs=ligand_crippen_contribs,
                prbCid=conf_id.GetId(),
            )
            o3a.Align()

        mol = Chem.RemoveHs(mol)
        ligand = Chem.RemoveHs(ligand)

        if mol.GetNumConformers() == 0:
            logger.error(
                f"Spherical search generated no conformers. Moving onto next sampling function."
            )
        if mol.GetNumConformers() != 0:
            # Check for internal/external clashes
            logger.info("Checking for clashes with spherical constraints")
            # Store the coordinates of each conformer into numpy array
            new_conformer = mol.GetConformers()
            new_coors = []
            for i, conformer in enumerate(new_conformer):
                coords = conformer.GetPositions()
                new_coors.append(coords)

            new_idx_set = []
            new_coor_set = []
            new_bs = []
            # loop through each rdkit generated conformer
            for idx, conf in enumerate(new_coors):
                b = self._bs
                self.ligand.coor = conf
                self.ligand.b = [b[0]]
                if self.options.external_clash:
                    if not self._cd():
                        if (
                            new_idx_set
                        ):  # if there are already conformers in new_idx_set
                            new_idx_set.append(idx)
                            new_coor_set.append(conf)
                            new_bs.append(b[0])
                        else:
                            new_idx_set.append(idx)
                            new_coor_set.append(conf)
                            new_bs.append(b[0])
                elif not self.ligand.clashes():
                    if new_idx_set:  # if there are already conformers in new_idx_set
                        new_idx_set.append(idx)
                        new_coor_set.append(conf)
                        new_bs.append(b[0])
                    else:
                        new_idx_set.append(idx)
                        new_coor_set.append(conf)
                        new_bs.append(b[0])

            if self.options.write_intermediate_conformers:
                self._write_intermediate_conformers("blob", new_coor_set)
                
            # Save new conformers to self
            merged_arr = np.concatenate((self._coor_set, new_coor_set), axis=0)
            merged_bs = np.concatenate((self._bs, new_bs), axis=0)

            self._coor_set = merged_arr
            self._bs = merged_bs

            logger.info(
                f"After spherical search, there are {len(self._coor_set)} plausible conformers"
            )

            if len(self._coor_set) < 1:
                logger.warning(
                    f"RDKit conformers not sufficiently diverse. Generated: {len(self._coor_set)} conformers"
                )
                return

        return

    def branching_search(self):
        """
        This function is used to directly address cases where the ligand has branching disorder. Identify the atoms belonging to a branch, and fix all non-branch ligands in place
        (by using coordinate map distance constraints). Allow the branches to randomly sample the conformational space.
        """
        # Make RDKit mol object from the ligand pdb
        # Starting structure
        branching_ligand = Chem.MolFromPDBFile(self.ligand_pdb_file)
        num_branched_confs = self.num_conf_for_method

        # Identify the branching sections of the ligand
        side_chain = self.identify_core_and_sidechain(branching_ligand)
        side_chain_atoms = side_chain[0]

        # Define core_atoms as all atoms not in side_chain_atoms
        core_indices = [
            atom.GetIdx()
            for atom in branching_ligand.GetAtoms()
            if atom.GetIdx() not in side_chain_atoms
        ]
        core_atoms = tuple(core_indices)

        # create reference molecule from smiles string to copy the correct bond order from
        ref_mol = Chem.MolFromSmiles(self.options.smiles)
        branching_ligand = Chem.AllChem.AssignBondOrdersFromTemplate(
            ref_mol, branching_ligand
        )

        # add hydrogens
        branching_ligand = Chem.AddHs(branching_ligand)

        # Create a coordinate map for the core atoms using the original ligand
        coord_map = {
            idx: branching_ligand.GetConformer().GetAtomPosition(idx)
            for idx in core_atoms
        }
        # Create a copy of the molecule for conformer generation
        mol_copy = Chem.Mol(branching_ligand)
        logger.info(
            f"Generating {num_branched_confs} conformers with branched sampling"
        )

        # Generate confromers with coordinate map "fixed"
        AllChem.EmbedMultipleConfs(
            mol_copy,
            numConfs=num_branched_confs,
            coordMap=coord_map,
            useBasicKnowledge=True,
        )

        # Minimize energy of each conformer
        logger.info("Minimizing energy of new conformers with branched sampling")
        mp = AllChem.MMFFGetMoleculeProperties(mol_copy)
        for conf_id in mol_copy.GetConformers():
            ff = AllChem.MMFFGetMoleculeForceField(mol_copy, mp, confId=conf_id.GetId())
            ff.Minimize()

        # Compute Crippen contributions for both molecules to align the generated conformers to each other
        logger.info("Aligning molecules with branched sampling")
        ligand_crippen_contribs = Chem.rdMolDescriptors._CalcCrippenContribs(
            branching_ligand
        )
        mol_crippen_contribs = Chem.rdMolDescriptors._CalcCrippenContribs(mol_copy)
        # Align
        for conf_id in mol_copy.GetConformers():
            o3a = Chem.rdMolAlign.GetCrippenO3A(
                mol_copy,
                branching_ligand,
                prbCrippenContribs=mol_crippen_contribs,
                refCrippenContribs=ligand_crippen_contribs,
                prbCid=conf_id.GetId(),
            )
            o3a.Align()

        mol_copy = Chem.RemoveHs(mol_copy)
        branching_ligand = Chem.RemoveHs(branching_ligand)

        # Check for internal/external clashes
        if mol_copy.GetNumConformers() == 0:
            logger.error(
                f"Branching search generated no conformers. Moving onto next sampling function."
            )
        if mol_copy.GetNumConformers() != 0:
            logger.info("Checking for clashes with branched sampling")
            # Store the coordinates of each conformer into numpy array
            new_conformer = mol_copy.GetConformers()
            new_coors = []
            for i, conformer in enumerate(new_conformer):
                coords = conformer.GetPositions()
                new_coors.append(coords)

            new_idx_set = []
            new_coor_set = []
            new_bs = []

            # loop through each rdkit generated conformer
            for idx, conf in enumerate(new_coors):
                b = self._bs
                self.ligand.coor = conf
                self.ligand.b = [b[0]]
                if self.options.external_clash:
                    if not self._cd():
                        if (
                            new_idx_set
                        ):  # if there are already conformers in new_idx_set
                            new_idx_set.append(idx)
                            new_coor_set.append(conf)
                            new_bs.append(b[0])
                        else:
                            new_idx_set.append(idx)
                            new_coor_set.append(conf)
                            new_bs.append(b[0])
                elif not self.ligand.clashes():
                    if new_idx_set:  # if there are already conformers in new_idx_set
                        new_idx_set.append(idx)
                        new_coor_set.append(conf)
                        new_bs.append(b[0])
                    else:
                        new_idx_set.append(idx)
                        new_coor_set.append(conf)
                        new_bs.append(b[0])

            if self.options.write_intermediate_conformers:
                self._write_intermediate_conformers("branching", new_coor_set)
                
            # Save new conformers to self
            merged_arr = np.concatenate((self._coor_set, new_coor_set), axis=0)
            merged_bs = np.concatenate((self._bs, new_bs), axis=0)

            self._coor_set = merged_arr
            self._bs = merged_bs
            logger.info(
                f"After branched search, there are {len(new_coor_set)} plausible conformers"
            )

            return

    def long_chain_search(self):
        """
        When ligands have long branches with a high number of internal degrees of freedom, a random sampling of the conformational space can lead to
        wildly undersirable configurations (i.e. the generated long branches are not supported by the density). It is useful to implement distance constraints
        for these sections.
        """
        # Starting strucutre from PDB
        ligand = Chem.MolFromPDBFile(self.ligand_pdb_file)
        # Refenerce mol from smiles, used to assign correct bond order
        ref_mol = Chem.MolFromSmiles(self.options.smiles)
        ligand = Chem.AllChem.AssignBondOrdersFromTemplate(ref_mol, ligand)

        # Identify the side chain/branched sections of the ligand
        side_chain = self.identify_core_and_sidechain(ligand)
        side_chain_atoms = side_chain[0]

        ligand = Chem.AddHs(ligand)
        # Set the coordinate map as the brnaches
        coord_map = {
            idx: ligand.GetConformer().GetAtomPosition(idx) for idx in side_chain_atoms
        }

        # Create a copy of the 'ligand' object to generate conformers off of. They will later be aligned to 'ligand' object
        mol = Chem.Mol(ligand)
        logger.info(
            f"Generating {self.num_conf_for_method} conformers with long chain search"
        )
        # Generate conformers
        AllChem.EmbedMultipleConfs(
            mol,
            numConfs=self.num_conf_for_method,
            coordMap=coord_map,
            useBasicKnowledge=True,
        )

        logger.info("Minimizing long chain conformers with long chain search")
        # Minimize the energy of each conformer to find most stable structure
        mp = AllChem.MMFFGetMoleculeProperties(mol)
        for conf_id in mol.GetConformers():
            ff = AllChem.MMFFGetMoleculeForceField(mol, mp, confId=conf_id.GetId())
            ff.Minimize()

        logger.info("Aligning molecules with long chain search")
        ligand_crippen_contribs = Chem.rdMolDescriptors._CalcCrippenContribs(ligand)
        mol_crippen_contribs = Chem.rdMolDescriptors._CalcCrippenContribs(mol)

        for conf_id in mol.GetConformers():
            o3a = Chem.rdMolAlign.GetCrippenO3A(
                mol,
                ligand,
                prbCrippenContribs=mol_crippen_contribs,
                refCrippenContribs=ligand_crippen_contribs,
                prbCid=conf_id.GetId(),
            )
            o3a.Align()

        mol = Chem.RemoveHs(mol)
        ligand = Chem.RemoveHs(ligand)

        if mol.GetNumConformers() == 0:
            logger.error(
                f"Long chain search generated no conformers. Moving onto next sampling function."
            )
        if mol.GetNumConformers() != 0:
            # Check for internal/external clashes
            logger.info("Checking for clashes with long chain search")
            # Store the coordinates of each conformer into numpy array
            new_conformer = mol.GetConformers()
            new_coors = []
            for i, conformer in enumerate(new_conformer):
                coords = conformer.GetPositions()
                new_coors.append(coords)

            new_idx_set = []
            new_coor_set = []
            new_bs = []
            # loop through each rdkit generated conformer
            for idx, conf in enumerate(new_coors):
                b = self._bs
                self.ligand.coor = conf
                self.ligand.b = [b[0]]
                # self._cd()
                if self.options.external_clash:
                    if not self._cd():
                        if (
                            new_idx_set
                        ):  # if there are already conformers in new_idx_set
                            new_idx_set.append(idx)
                            new_coor_set.append(conf)
                            new_bs.append(b[0])
                        else:
                            new_idx_set.append(idx)
                            new_coor_set.append(conf)
                            new_bs.append(b[0])

                elif not self.ligand.clashes():
                    if new_idx_set:  # if there are already conformers in new_idx_set
                        new_idx_set.append(idx)
                        new_coor_set.append(conf)
                        new_bs.append(b[0])
                    else:
                        new_idx_set.append(idx)
                        new_coor_set.append(conf)
                        new_bs.append(b[0])

            if self.options.write_intermediate_conformers:
                self._write_intermediate_conformers("long_chain", new_coor_set)
                
            # Save new conformers to self
            merged_arr = np.concatenate((self._coor_set, new_coor_set), axis=0)
            merged_bs = np.concatenate((self._bs, new_bs), axis=0)

            self._coor_set = merged_arr
            self._bs = merged_bs

            logger.info(
                f"After long chain search, there are {len(new_coor_set)} plausible conformers"
            )

            if len(self._coor_set) < 1:
                logger.warning(
                    f"RDKit conformers not sufficiently diverse. Generated: {len(self._coor_set)} conformers"
                )
                return

        return

    def rot_trans(self):
        """
        Rotate and translate all conformers that pass QP scoring for further sampling of conformational space. Rotate (by default) 15 degrees by 5 degree increments in x, y, z directions
        and translate 0.3 angstroms in x, y, z directions.
        """

        # Initialize empty list to store rotated/translated conformers + b-factors
        extended_coor_set = []
        extended_bs = []
        rotated_coor_set = []
        rotated_bs = []
        new_coor_set = self._coor_set
        new_bs = self._bs

        # rotations
        for conf, b in zip(self._coor_set, self._bs):
            # Apply rotations to each initial conformation
            rotated_conformations = self.apply_rotations(
                conf, self.options.rot_range, self.options.rotation_step
            )
            rotated_coor_set.extend(rotated_conformations)
            rotated_bs.extend(
                [b] * len(rotated_conformations)
            )  # Extend b values for each rotated conformation

        # translations
        for conf, b in zip(self._coor_set, self._bs):
            # Apply translations to each conformation
            translated_conformations = self.apply_translations(
                conf, self.options.trans_range
            )
            extended_coor_set.extend(translated_conformations)
            extended_bs.extend(
                [b] * len(translated_conformations)
            )  # Extend b values for each translated conformation

        self._coor_set = np.concatenate(
            (new_coor_set, rotated_coor_set, extended_coor_set), axis=0
        )
        self._bs = np.concatenate((new_bs, rotated_bs, extended_bs), axis=0)

        logger.info(
            f"After trans/rot search, there are {len(self._coor_set)} plausible conformers"
        )

        # Make sure b-factor array is the same length as coordinate array
        if len(self._bs) != len(self._coor_set):
            self._bs = np.tile(self._bs[0], (len(self._coor_set), 1))
            
        if self.options.write_intermediate_conformers:
                self._write_intermediate_conformers(prefix="trans_rot_sol")
            
        self._convert()
        logger.info("Solving QP after trans and rot search.")
        self._solve_qp()
        logger.debug("Updating conformers after trans and rot search.")
        self._update_conformers()
        # self._write_intermediate_conformers(prefix="trans_rot_sol")

        logger.info(
            f"After rotation and translation QP there are {len(self._coor_set)} conformers"
        )

        if len(self._coor_set) < 1:
            logger.warning(
                f"RDKit conformers not sufficiently diverse. Generated: {len(self._coor_set)} conformers"
            )
            return

    def flip_180(self):
        """
        For each conformer, rotate the molecule 180 degrees around each axis (x, y, z),
        and then apply further rotations within +/- 5 degrees.
        """

        def define_axes_through_PCA(coor_set):
            """
            Determine the principal component axes for the given set of coordinates.
            """
            # Center the coordinates around the mean
            mean = np.mean(coor_set, axis=0)
            centered_data = coor_set - mean

            # Compute the covariance matrix
            covariance_matrix = np.cov(centered_data, rowvar=False)

            # Perform singular value decomposition to find the principal components
            U, s, Vt = svd(covariance_matrix)

            # Principal components are given by the columns of U (or rows of Vt)
            return Vt.T  # Return the principal axes

        def rotation_matrix_180(axis):
            """Generate a rotation matrix for 180 degrees around a given axis."""
            axis = axis / np.linalg.norm(axis)  # Ensure the axis is a unit vector
            return np.eye(3) - 2 * np.outer(axis, axis)

        def apply_rotation(conf, R):
            """Apply rotation matrix R to the conformation."""
            centroid = np.mean(conf, axis=0)
            centered_conf = conf - centroid
            rotated_conf = np.dot(centered_conf, R) + centroid
            return rotated_conf

        coor_set = self._starting_coor_set[0]
        # Get the principal axes
        principal_axes = define_axes_through_PCA(coor_set)

        # Extract each principal axis
        x_axis, y_axis, z_axis = (
            principal_axes[:, 0],
            principal_axes[:, 1],
            principal_axes[:, 2],
        )

        # Rotation matrices for 180 degree rotations around each principal axis
        Rx = rotation_matrix_180(x_axis)
        Ry = rotation_matrix_180(y_axis)
        Rz = rotation_matrix_180(z_axis)

        new_coor = []
        new_bs = []

        # Apply these rotations to input conformation
        logger.info(f"180 degree flipping input ligand")
        for conf, b in zip(self._starting_coor_set, self._starting_bs):
            # Rotate around each principal axis and extend new_coor and new_bs
            for Rot in [Rx, Ry, Rz]:
                # Flip initial conformer
                flipped_conf = apply_rotation(conf, Rot)
                # apply further small rotations around each rotated conformation
                further_rotations_plane = self.apply_rotations(flipped_conf, 10, 2)
                new_coor.extend(further_rotations_plane)
                new_bs.extend([b] * len(further_rotations_plane))

        logger.info(f"Generated {len(new_coor)} flipped conformers")

        if self.options.write_intermediate_conformers:
                self._write_intermediate_conformers("flip_180", new_coor)
            
        self._coor_set = np.array(new_coor)
        self._bs = np.array(new_bs)

        logger.info(f"After Clash Check {len(new_coor)} flipped conformers")

        if len(self._coor_set) < 1:
            logger.warning(
                f"RDKit conformers not sufficiently diverse. Generated: {len(self._coor_set)} conformers"
            )
            return


class QFitCovalentLigand(_BaseQFit):
    def __init__(self, covalent_ligand, receptor, xmap, options):
        self.chain = covalent_ligand.chain[0]
        self.resi = covalent_ligand.resi[0]
        self.covalent_ligand = covalent_ligand
        self._rigid_clusters = covalent_ligand.rigid_clusters()
        if covalent_ligand.covalent_bonds == 0:
            pass
        elif covalent_ligand.covalent_bonds == 1:
            # Extract the information about the residue that the
            # ligand is covalently bonded to:
            (
                partner_chain,
                partner_resi,
                partner_icode,
                self.partner_atom,
            ) = covalent_ligand.covalent_partners[0]
            if partner_icode:
                self.partner_id = (int(partner_resi), partner_icode)
            else:
                self.partner_id = int(partner_resi)

            # Extract the information about the covalent bond itself:
            self.covalent_atom = covalent_ligand.covalent_atoms[0][3]
            self.covalent_bond = [self.partner_atom, self.covalent_atom]

            # Select the partner residue from the receptor:
            sel_str = f"chain {partner_chain} and resi {partner_resi}"
            if partner_icode:
                sel_str = f"{sel_str} and icode {partner_icode}"
            self.covalent_partner = receptor.extract(sel_str)

            # Alter the structure so that covalent ligand and residue are
            # treated as a single residue:
            data = {}
            for attr in receptor.data:
                data[attr] = np.concatenate(
                    (getattr(receptor, attr), getattr(covalent_ligand, attr))
                )
            mask = (data["resi"] == covalent_ligand.resi[0]) & (
                data["chain"] == covalent_ligand.chain[0]
            )
            data["resi"][mask] = self.covalent_partner.resi[0]
            data["chain"][mask] = self.covalent_partner.chain[0]
            data["resn"][mask] = self.covalent_partner.resn[0]
            self.structure = Structure(data)

            # Extract the covalent residue as a Residue object:
            combined_residue = self.structure.extract(sel_str)
            chain = combined_residue[partner_chain]
            conformer = chain.conformers[0]
            self.covalent_residue = conformer[self.partner_id]

            # Initialize using the base qFit class:
            super().__init__(self.covalent_residue, self.structure, xmap, options)

            # Update the transformer:
            self._update_transformer(self.covalent_residue)

            # Identify the bonds in the residue for internal clash detection:
            bonds = covalent_ligand.get_bonds()
            # Add the covalent bond itself to the list:
            bonds.append(self.covalent_bond)
            # Initialize internal clash detection
            self.covalent_residue._init_clash_detection(
                self.options.clash_scaling_factor, bonds
            )

            # Identify the segment to which the covalent residue belongs to:
            self.segment = None
            for segment in self.structure.segments:
                if (
                    segment.chain[0] == partner_chain
                    and self.covalent_residue in segment
                ):
                    self.segment = segment
                    break
            if self.segment is None:
                raise RuntimeError("Could not determine segment.")

            # Set up external clash detection:
            self._setup_clash_detector()
            if options.subtract:
                self._subtract_transformer(self.covalent_residue, self.structure)
            self._update_transformer(self.covalent_residue)

        # More than one covalent bond:
        else:
            pass

    def _setup_clash_detector(self):
        residue = self.covalent_residue
        segment = self.segment
        index = segment.find(self.partner_id)
        # Exclude peptide bonds from clash detector
        exclude = []
        if index > 0:
            N_index = residue.select("name", "N")[0]
            N_neighbor = segment.residues[index - 1]
            neighbor_C_index = N_neighbor.select("name", "C")[0]
            if (
                np.linalg.norm(residue._coor[N_index] - segment._coor[neighbor_C_index])
                < 2
            ):
                coor = N_neighbor._coor[neighbor_C_index]
                exclude.append((N_index, coor))
        if index < len(segment.residues) - 1:
            C_index = residue.select("name", "C")[0]
            C_neighbor = segment.residues[index + 1]
            neighbor_N_index = C_neighbor.select("name", "N")[0]
            if (
                np.linalg.norm(residue._coor[C_index] - segment._coor[neighbor_N_index])
                < 2
            ):
                coor = C_neighbor._coor[neighbor_N_index]
                exclude.append((C_index, coor))
        # Obtain atoms with which the residue can clash
        resi = self.partner_id
        chainid = self.segment.chain[0]
        sel_str = f"not (resi {resi} and chain {chainid})"
        receptor = self.structure.extract(sel_str).copy()
        # Find symmetry mates of the receptor
        starting_coor = self.structure.coor.copy()
        iterator = self.xmap.unit_cell.iter_struct_orth_symops
        for symop in iterator(self.structure, target=self.covalent_residue, cushion=5):
            if symop.is_identity():
                continue
            self.structure.rotate(symop.R)
            self.structure.translate(symop.t)
            receptor = receptor.combine(self.structure)
            self.structure.coor = starting_coor

        self._cd = ClashDetector(
            residue,
            receptor,
            exclude=exclude,
            scaling_factor=self.options.clash_scaling_factor,
        )

    def run(self):
        if self.options.sample_backbone:
            self._sample_backbone()
        if self.options.sample_angle:
            # Is the ligand bound to the backbone or the side chain?
            if self.partner_atom not in ["N", "C", "CA", "O"]:
                self._sample_angle()
        if self.covalent_residue.nchi >= 1 and self.options.sample_rotamers:
            # Is the ligand bound to the backbone or the side chain?
            if self.partner_atom not in ["N", "C", "CA", "O"]:
                self._sample_sidechain()
        if self.options.sample_ligand:
            # Sample around the covalent bond, if rotatable:
            self._sample_covalent_bond()
            # Sample the remaining rotatable bonds of the ligand:
            self._sample_ligand()
        return

    def _sample_backbone(self):
        # Check if residue has enough neighboring residues
        index = self.segment.find(self.partner_id)
        # self.covalent_residue.update_clash_mask()
        nn = self.options.neighbor_residues_required
        if index < nn or index + nn > len(self.segment):
            return

        segment = self.segment[index - nn : index + nn + 1]
        atom_name = "CB"
        if self.covalent_residue.resn[0] == "GLY":
            atom_name = "O"
        atom = self.covalent_residue.extract("name", atom_name)
        try:
            u_matrix = [
                [atom.u00[0], atom.u01[0], atom.u02[0]],
                [atom.u01[0], atom.u11[0], atom.u12[0]],
                [atom.u02[0], atom.u12[0], atom.u22[0]],
            ]
            directions = adp_ellipsoid_axes(u_matrix)
        except AttributeError:
            directions = np.identity(3)

        optimizer = NullSpaceOptimizer(segment)
        start_coor = atom.coor[0]
        torsion_solutions = []
        amplitudes = np.arange(
            0.1,
            self.options.sample_backbone_amplitude + 0.01,
            self.options.sample_backbone_step,
        )

        for amplitude, direction in itertools.product(amplitudes, directions):
            endpoint = start_coor + amplitude * direction
            optimize_result = optimizer.optimize(atom_name, endpoint)
            torsion_solutions.append(optimize_result["x"])

            endpoint = start_coor - amplitude * direction
            optimize_result = optimizer.optimize(atom_name, endpoint)
            torsion_solutions.append(optimize_result["x"])

        starting_coor = segment.coor

        for solution in torsion_solutions:
            optimizer.rotator(solution)
            self._coor_set.append(self.segment[index].coor)
            segment.coor = starting_coor
        # logger.debug(f"Backbone sampling generated {len(self._coor_set)} conformers")

    def _sample_angle(self):
        """Sample residue along the N-CA-CB angle."""
        active_names = ("N", "CA", "C", "O", "CB", "H", "HA")
        selection = self.covalent_residue.select("name", active_names)
        self.covalent_residue._active = False
        self.covalent_residue._active[selection] = True
        self.covalent_residue.update_clash_mask()
        active = self.covalent_residue.active
        angles = np.arange(
            -self.options.sample_angle_range,
            self.options.sample_angle_range + 0.001,
            self.options.sample_angle_step,
        )
        new_coor_set = []
        new_bs = []
        for coor in self._coor_set:
            self.covalent_residue.coor = coor
            rotator = CBAngleRotator(self.covalent_residue)
            for angle in angles:
                rotator(angle)
                coor = self.covalent_residue.coor
                if self.options.remove_conformers_below_cutoff:
                    values = self.xmap.interpolate(coor[active])
                    mask = self.covalent_residue.e[active] != "H"
                    if np.min(values[mask]) < self.options.density_cutoff:
                        continue
                if self.options.external_clash:
                    if self._cd() or self.covalent_residue.clashes():
                        continue
                elif self.covalent_residue.clashes():
                    continue
                new_coor_set.append(self.covalent_residue.coor)
                new_bs.append(self.conformer.b)

        self._coor_set = new_coor_set
        self._bs = new_bs
        # logger.debug(f"Bond angle sampling generated {len(self._coor_set)} conformers.")

    def _sample_sidechain(self):
        opt = self.options
        start_chi_index = 1
        partner_length = self.covalent_partner.name.shape[0]
        if self.covalent_residue.resn[0] != "PRO":
            sampling_window = np.arange(
                -opt.rotamer_neighborhood,
                opt.rotamer_neighborhood + opt.dihedral_stepsize,
                opt.dihedral_stepsize,
            )
        else:
            sampling_window = [0]

        rotamers = self.covalent_residue.rotamers
        rotamers.append(
            [
                self.covalent_residue.get_chi(i)
                for i in range(1, self.covalent_residue.nchi + 1)
            ]
        )
        iteration = 0
        new_bs = []
        for b in self._bs:
            new_bs.append(b)
        self._bs = new_bs

        while True:
            chis_to_sample = opt.dofs_per_iteration
            if iteration == 0 and (opt.sample_backbone or opt.sample_angle):
                chis_to_sample = max(1, opt.dofs_per_iteration - 1)
            end_chi_index = min(
                start_chi_index + chis_to_sample, self.covalent_residue.nchi + 1
            )
            iter_coor_set = []
            for chi_index in range(start_chi_index, end_chi_index):
                # Set active and passive atoms, since we are iteratively
                # building up the sidechain. This updates the internal
                # clash mask.
                self.covalent_residue.active = True
                if chi_index < self.covalent_residue.nchi:
                    deactivate = self.covalent_residue._rotamers["chi-rotate"][
                        chi_index + 1
                    ]
                    selection = self.covalent_residue.select("name", deactivate)
                    self.covalent_residue._active[selection] = False
                    bs_atoms = list(set(current) - set(deactivate))
                else:
                    bs_atoms = self.covalent_residue._rotamers["chi-rotate"][chi_index]
                if self.options.sample_ligand:
                    sel_str = (
                        f"chain {self.covalent_residue.chain[0]} "
                        f"and resi {self.covalent_residue.resi[0]}"
                    )
                    if self.covalent_residue.icode[0]:
                        sel_str = (
                            f"{sel_str} and icode {self.covalent_residue.icode[0]}"
                        )
                    selection = self.covalent_residue.select(sel_str)
                    tmp = copy.deepcopy(self.covalent_residue.active)
                    tmp[partner_length:] = False
                    self.covalent_residue._active[selection] = tmp

                active = self.covalent_residue.active
                self.covalent_residue.update_clash_mask()

                new_coor_set = []
                new_bs = []
                n = 0
                for coor, b in zip(self._coor_set, self._bs):
                    n += 1
                    self.covalent_residue.coor = coor
                    self.covalent_residue.b = b
                    chis = [
                        self.covalent_residue.get_chi(i) for i in range(1, chi_index)
                    ]
                    for rotamer in rotamers:
                        # Check if the residue configuration corresponds to the
                        # current rotamer
                        is_this_rotamer = True
                        for curr_chi, rotamer_chi in zip(chis, rotamer):
                            diff_chi = abs(curr_chi - rotamer_chi)
                            if (
                                360 - opt.rotamer_neighborhood
                                > diff_chi
                                > opt.rotamer_neighborhood
                            ):
                                is_this_rotamer = False
                                break
                        if not is_this_rotamer:
                            continue
                        # Set the chi angle to the standard rotamer value.
                        self.covalent_residue.set_chi(
                            chi_index,
                            rotamer[chi_index - 1],
                            covalent=self.partner_atom,
                            length=partner_length,
                        )

                        # Sample around the neighborhood of the rotamer
                        chi_rotator = ChiRotator(
                            self.covalent_residue,
                            chi_index,
                            covalent=self.partner_atom,
                            length=partner_length,
                        )

                        for angle in sampling_window:
                            n += 1
                            chi_rotator(angle)
                            atoms = self.covalent_residue.name
                            atom_selection = self.covalent_residue.select("name", atoms)
                            coor = self.covalent_residue._coor[atom_selection]
                            if opt.remove_conformers_below_cutoff:
                                values = self.xmap.interpolate(coor[active])
                                mask = self.covalent_residue.e[active] != "H"
                                if np.min(values[mask]) < self.options.density_cutoff:
                                    continue
                            if self.options.external_clash:
                                if (
                                    not self._cd()
                                    and not self.covalent_residue.clashes()
                                ):
                                    if new_coor_set:
                                        delta = np.array(new_coor_set) - np.array(coor)
                                        if (
                                            np.sqrt(
                                                min(
                                                    np.square((delta))
                                                    .sum(axis=2)
                                                    .sum(axis=1)
                                                )
                                            )
                                            >= 0.01
                                        ):
                                            new_coor_set.append(coor)
                                            new_bs.append(b)
                                    else:
                                        new_coor_set.append(coor)
                                        new_bs.append(b)
                            elif self.covalent_residue.clashes() == 0:
                                if new_coor_set:
                                    delta = np.array(new_coor_set) - np.array(coor)
                                    if (
                                        np.sqrt(
                                            min(
                                                np.square((delta))
                                                .sum(axis=2)
                                                .sum(axis=1)
                                            )
                                        )
                                        >= 0.01
                                    ):
                                        new_coor_set.append(coor)
                                        new_bs.append(b)
                                else:
                                    new_coor_set.append(coor)
                                    new_bs.append(b)

                self._coor_set = new_coor_set
                self._bs = new_bs

            if not self._coor_set:
                logger.warning(
                    "No conformers could be generated. Using deposited conformer."
                )
                self._coor_set = [self.residue.coor]
                self._bs = [self.residue.b]
            else:
                logger.info(
                    f"Side chain sampling produced {len(self._coor_set)} conformers"
                )

            if self.options.write_intermediate_conformers:
                self._write_intermediate_conformers(
                    prefix=f"sample_sidechain_iter{iteration}"
                )

            # QP score conformer occupancy
            self._convert()
            self._solve_qp()
            self._update_conformers()
            if self.options.write_intermediate_conformers:
                self._write_intermediate_conformers(
                    prefix=f"sample_sidechain_iter{iteration}_qp"
                )

            # MIQP score conformer occupancy
            self._convert()
            self._solve_miqp(
                threshold=self.options.threshold, cardinality=self.options.cardinality
            )
            self._update_conformers()
            if self.options.write_intermediate_conformers:
                self._write_intermediate_conformers(
                    prefix=f"sample_sidechain_iter{iteration}_miqp"
                )

            # Check if we are done
            if chi_index == self.covalent_residue.nchi:
                break
            # Use the next chi angle as starting point, except when we are in
            # the first iteration and have selected backbone sampling and we
            # are sampling more than 1 dof per iteration
            increase_chi = not (
                (opt.sample_backbone or opt.sample_angle)
                and iteration == 0
                and opt.dofs_per_iteration > 1
            )
            if increase_chi:
                start_chi_index += 1
            iteration += 1

    def _sample_covalent_bond(self):
        opt = self.options
        atoms = self.covalent_bond
        partner_length = self.covalent_partner.name.shape[0]
        self._sampling_range = np.deg2rad(
            np.arange(0, 360, self.options.sample_ligand_stepsize)
        )
        sel_str = f"chain {self.covalent_residue.chain[0]} and resi {self.covalent_residue.resi[0]}"
        if self.covalent_residue.icode[0]:
            sel_str = f"{sel_str} and icode {self.covalent_residue.icode[0]}"
        selection = self.covalent_residue.select(sel_str)
        self.covalent_residue._active[selection] = True
        self.covalent_ligand._active[selection] = True
        bs_atoms = list(set(current))
        new_coor_set = []
        new_bs = []
        n = 0
        for coor, b in zip(self._coor_set, self._bs):
            n += 1
            self.covalent_residue.coor = coor
            self.covalent_residue.b = b
            self.covalent_ligand.coor = coor[partner_length:]
            rotator = CovalentBondRotator(
                self.covalent_residue, self.covalent_ligand, *atoms
            )
            for angle in self._sampling_range:
                n += 1
                new_coor = np.concatenate(
                    (coor[:partner_length], rotator(angle)), axis=0
                )
                if opt.remove_conformers_below_cutoff:
                    values = self.xmap.interpolate(new_coor[active])
                    mask = self.covalent_residue.e[active] != "H"
                    if np.min(values[mask]) < self.options.density_cutoff:
                        continue
                if self.options.external_clash:
                    if not self._cd() and not self.covalent_residue.clashes():
                        if new_coor_set:
                            delta = np.array(new_coor_set) - np.array(new_coor)
                            if (
                                np.sqrt(min(np.square((delta)).sum(axis=2).sum(axis=1)))
                                >= 0.01
                            ):
                                new_coor_set.append(new_coor)
                                new_bs.append(b)
                        else:
                            new_coor_set.append(new_coor)
                            new_bs.append(b)
                elif self.covalent_residue.clashes() == 0:
                    if new_coor_set:
                        delta = np.array(new_coor_set) - np.array(new_coor)
                        if (
                            np.sqrt(min(np.square((delta)).sum(axis=2).sum(axis=1)))
                            >= 0.01
                        ):
                            new_coor_set.append(new_coor)
                            new_bs.append(b)
                    else:
                        new_coor_set.append(new_coor)
                        new_bs.append(b)
        self._coor_set = new_coor_set
        self._bs = new_bs
        self.conformer = self.covalent_residue

        if not self._coor_set:
            msg = (
                "No conformers could be generated. Check for initial "
                "clashes and density support."
            )
            raise RuntimeError(msg)
        else:
            logger.info(
                f"Covalent bond angle sampling generated {len(self._coor_set)} conformers."
            )

        # QP score conformer occupancy
        self._convert()
        self._solve_qp()
        self._update_conformers()
        if self.options.write_intermediate_conformers:
            self._write_intermediate_conformers(prefix="sample_covalent_bond_qp")

        # MIQP score conformer occupancy
        self._convert()
        self._solve_miqp(
            threshold=self.options.threshold, cardinality=self.options.cardinality
        )
        self._update_conformers()
        if self.options.write_intermediate_conformers:
            self._write_intermediate_conformers(prefix="sample_covalent_bond_miqp")

    def _sample_ligand(self):
        opt = self.options

        self._sampling_range = np.deg2rad(
            np.arange(0, 360, self.options.sample_ligand_stepsize)
        )
        nbonds = len(self.covalent_ligand.bond_list)
        if nbonds == 0:
            return

        # This contains an ordered list of bonds, in which the list order
        # was determined by the tree hierarchy starting at the root bond.
        bonds = self.covalent_ligand.bond_list
        partner_length = self.covalent_partner.name.shape[0]
        root = self.covalent_ligand.root

        sel_str = f"chain {self.covalent_residue.chain[0]} and resi {self.covalent_residue.resi[0]}"
        if self.covalent_residue.icode[0]:
            sel_str = f"{sel_str} and icode {self.covalent_residue.icode[0]}"
        selection = self.covalent_residue.select(sel_str)
        iteration = 1
        starting_bond_index = 0
        while True:
            # Identify the bonds that we are going to sample
            end_bond_index = min(
                starting_bond_index + self.options.dofs_per_iteration, nbonds
            )

            # Identify the atoms that are active:
            active = np.ones_like(self.covalent_residue.active, dtype=bool)
            # for bond in bonds[:end_bond_index]:
            #     for cluster in self._rigid_clusters:
            #         if bond[0] in cluster or bond[1] in cluster:
            #             active[partner_length:][cluster] = True
            #             for atom in cluster:
            #                 active[partner_length:][self.covalent_ligand.connectivity[atom]] = True
            self.covalent_residue._active[selection] = active
            self.covalent_ligand._active[selection] = active
            self.covalent_residue.update_clash_mask()

            # Sample each of the bonds in the bond range
            n = 0
            for bond_index in range(starting_bond_index, end_bond_index):
                sampled_bond = bonds[bond_index]
                atoms = [
                    self.covalent_ligand.name[sampled_bond[0]],
                    self.covalent_ligand.name[sampled_bond[1]],
                ]
                new_coor_set = []
                for coor in self._coor_set:
                    self.covalent_residue.coor = coor
                    self.covalent_ligand.coor = coor[partner_length:]
                    rotator = BondRotator(self.covalent_ligand, *atoms)
                    for angle in self._sampling_range:
                        new_coor = np.concatenate(
                            (coor[:partner_length], rotator(angle)), axis=0
                        )
                        if opt.remove_conformers_below_cutoff:
                            values = self.xmap.interpolate(new_coor[active])
                            mask = self.covalent_residue.e[active] != "H"
                            if np.min(values[mask]) < self.options.density_cutoff:
                                continue
                        if self.options.external_clash:
                            if not self._cd() and not self.covalent_residue.clashes():
                                if new_coor_set:
                                    delta = np.array(new_coor_set) - np.array(new_coor)
                                    if (
                                        np.sqrt(
                                            min(
                                                np.square((delta))
                                                .sum(axis=2)
                                                .sum(axis=1)
                                            )
                                        )
                                        >= 0.01
                                    ):
                                        new_coor_set.append(new_coor)
                                else:
                                    new_coor_set.append(new_coor)
                        elif self.covalent_residue.clashes() == 0:
                            if new_coor_set:
                                delta = np.array(new_coor_set) - np.array(new_coor)
                                if (
                                    np.sqrt(
                                        min(np.square((delta)).sum(axis=2).sum(axis=1))
                                    )
                                    >= 0.01
                                ):
                                    new_coor_set.append(new_coor)
                            else:
                                new_coor_set.append(new_coor)
                self._coor_set = new_coor_set

            if not self._coor_set:
                msg = (
                    "No conformers could be generated. Check for initial "
                    "clashes and density support."
                )
                raise RuntimeError(msg)
            else:
                logger.info(
                    f"Ligand sampling {iteration} generated {len(self._coor_set)} conformers"
                )

            # QP score conformer occupancy
            self._convert()
            self._solve_qp()
            self._update_conformers()
            if self.options.write_intermediate_conformers:
                self._write_intermediate_conformers(
                    prefix=f"sample_ligand_iter{iteration}_qp"
                )

            # MIQP score conformer occupancy
            self._convert()
            self._solve_miqp(
                threshold=self.options.threshold, cardinality=self.options.cardinality
            )
            self._update_conformers()
            if self.options.write_intermediate_conformers:
                self._write_intermediate_conformers(
                    prefix=f"sample_ligand_iter{iteration}_miqp"
                )

            # Check if we are done
            if end_bond_index == nbonds:
                break

            # Go to the next bonds to be sampled
            starting_bond_index += self.options.dofs_per_iteration
            iteration += 1

    def get_conformers_covalent(self):
        conformers = []
        for q, coor, b in zip(self._occupancies, self._coor_set, self._bs):
            conformer = self.covalent_residue.copy()
            conformer.coor = coor
            conformer.q = q
            conformer.b = b
            conformers.append(conformer)
        return conformers
