import gc
from .qfit import QFitOptions
from .qfit import QFitRotamericResidue
from .qfit import QFitSegment
import multiprocessing as mp
from tqdm import tqdm
import os.path
import os
import sys
import numpy as np
import time
import argparse
from .custom_argparsers import (
    ToggleActionFlag,
    CustomHelpFormatter,
    ValidateMapFileArgument,
    ValidateStructureFileArgument,
)
import logging
import traceback
import pandas as pd
import itertools as itl
from .logtools import (
    setup_logging,
    log_run_info,
    poolworker_setup_logging,
    QueueListener,
)
from . import MapScaler, Structure, XMap
from .solvers import available_qp_solvers, available_miqp_solvers
from .structure.rotamers import ROTAMERS


logger = logging.getLogger(__name__)
os.environ["OMP_NUM_THREADS"] = "1"


def build_argparser():
    p = argparse.ArgumentParser(
        formatter_class=CustomHelpFormatter, description=__doc__
    )

    p.add_argument(
        "map",
        help="Density map in CCP4 or MRC format, or an MTZ file "
        "containing reflections and phases. For MTZ files "
        "use the --label options to specify columns to read. "
        "For CCP4 files, use the -r to specify resolution.",
        type=str,
        action=ValidateMapFileArgument,
    )
    p.add_argument(
        "structure",
        help="PDB-file containing structure.",
        type=str,
        action=ValidateStructureFileArgument,
    )

    p.add_argument(
        "-em",
        "--cryo_em",
        action="store_true",
        dest="em",
        help="Run qFit with EM options",
    )
    p.add_argument(
        "--residue",
        type=str,
        help="If selected will run qFit Residue"
        "Chain, residue id, and optionally insertion code for "
        "residue in structure, e.g. A,105, or A,105:A.",
    )

    # Map options
    p.add_argument(
        "-l",
        "--label",
        default="2FOFCWT,PH2FOFCWT",
        metavar="<F,PHI>",
        help="MTZ column labels to build density",
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
        "-m",
        "--resolution-min",
        default=None,
        metavar="<float>",
        type=float,
        help="Lower resolution bound (Å) (only use when providing CCP4 map files)",
    )
    p.add_argument(
        "-sb",
        "--no-sampling-b",
        action="store_false",
        dest="sample_bfactors",
        help="Do not sample b-factors within qFit",
    )
    p.add_argument(
        "-o",
        "--omit",
        action="store_true",
        help="Treat map file as an OMIT map in map scaling routines",
    )

    # Map prep options
    p.add_argument(
        "--scale",
        action=ToggleActionFlag,
        dest="scale",
        default=True,
        help="Scale density",
    )
    p.add_argument(
        "-sv",
        "--scale-rmask",
        dest="scale_rmask",
        default=1.0,
        metavar="<float>",
        type=float,
        help="Scaling factor for soft-clash mask radius",
    )
    p.add_argument(
        "-dc",
        "--density-cutoff",
        default=0.3,
        metavar="<float>",
        type=float,
        help="Density values below this value are set to <density-cutoff-value>",
    )
    p.add_argument(
        "-dv",
        "--density-cutoff-value",
        default=-1,
        metavar="<float>",
        type=float,
        help="Density values below <density-cutoff> are set to this value",
    )
    p.add_argument(
        "--subtract",
        action=ToggleActionFlag,
        dest="subtract",
        default=True,
        help="Subtract Fcalc of neighboring residues when running qFit",
    )
    p.add_argument(
        "-pad",
        "--padding",
        default=8.0,
        metavar="<float>",
        type=float,
        help="Padding size for map creation",
    )
    p.add_argument(
        "--waters-clash",
        action=ToggleActionFlag,
        dest="waters_clash",
        default=True,
        help="Consider waters for soft clash detection",
    )

    # Sampling options
    p.add_argument(
        "--backbone",
        action=ToggleActionFlag,
        dest="sample_backbone",
        default=True,
        help="Sample backbone using inverse kinematics",
    )
    p.add_argument(
        "-bbs",
        "--backbone-step",
        default=0.1,
        dest="sample_backbone_step",
        metavar="<float>",
        type=float,
        help="Stepsize for the amplitude of backbone sampling (Å)",
    )
    p.add_argument(
        "-bba",
        "--backbone-amplitude",
        default=0.3,
        dest="sample_backbone_amplitude",
        metavar="<float>",
        type=float,
        help="Maximum backbone amplitude (Å)",
    )
    p.add_argument(
        "-bbv",
        "--backbone-sigma",
        default=0.125,
        dest="sample_backbone_sigma",
        metavar="<float>",
        type=float,
        help="Backbone sampling displacement (Å)",
    )
    p.add_argument(
        "--sample-angle",
        action=ToggleActionFlag,
        dest="sample_angle",
        default=True,
        help="Sample CA-CB-CG angle for aromatic F/H/W/Y residues",
    )
    p.add_argument(
        "-sas",
        "--sample-angle-step",
        default=3.75,
        dest="sample_angle_step",
        metavar="<float>",
        type=float,
        help="CA-CB-CG bond angle sampling step in degrees",
    )
    p.add_argument(
        "-sar",
        "--sample-angle-range",
        default=7.5,
        dest="sample_angle_range",
        metavar="<float>",
        type=float,
        help="CA-CB-CG bond angle sampling range in degrees [-x,x]",
    )
    p.add_argument(
        "--sample-rotamers",
        action=ToggleActionFlag,
        dest="sample_rotamers",
        default=True,
        help="Sample sidechain rotamers",
    )
    p.add_argument(
        "-b",
        "--dofs-per-iteration",
        default=1,
        metavar="<int>",
        type=int,
        help="Number of internal degrees that are sampled/built per iteration",
    )
    p.add_argument(
        "-s",
        "--dihedral-stepsize",
        default=6,
        metavar="<float>",
        type=float,
        help="Stepsize for dihedral angle sampling in degrees",
    )
    p.add_argument(
        "-rn",
        "--rotamer-neighborhood",
        default=24,
        metavar="<float>",
        type=float,
        help="Chi dihedral-angle sampling range around each rotamer in degrees [-x,x]",
    )
    p.add_argument(
        "--remove-conformers-below-cutoff",
        action="store_true",
        dest="remove_conformers_below_cutoff",
        help="Remove conformers during sampling that have atoms "
        "with no density support, i.e. atoms are positioned "
        "at density values below <density-cutoff>",
    )
    p.add_argument(
        "-cf",
        "--clash-scaling-factor",
        default=0.75,
        metavar="<float>",
        type=float,
        help="Set clash scaling factor",
    )
    p.add_argument(
        "-ec",
        "--external-clash",
        action="store_true",
        dest="external_clash",
        help="Enable external clash detection during sampling",
    )
    p.add_argument(
        "-bs",
        "--bulk-solvent-level",
        default=0.3,
        metavar="<float>",
        type=float,
        help="Bulk solvent level in absolute values",
    )
    p.add_argument(
        "-c",
        "--cardinality",
        default=5,
        metavar="<int>",
        type=int,
        help="Cardinality constraint used during MIQP",
    )
    p.add_argument(
        "-t",
        "--threshold",
        default=0.2,
        metavar="<float>",
        type=float,
        help="Threshold constraint used during MIQP",
    )
    p.add_argument(
        "-hy",
        "--hydro",
        action="store_true",
        dest="hydro",
        help="Include hydrogens during calculations",
    )
    p.add_argument(
        "-rmsd",
        "--rmsd-cutoff",
        default=0.01,
        metavar="<float>",
        type=float,
        help="RMSD cutoff for removal of identical conformers",
    )
    p.add_argument(
        "--threshold-selection",
        dest="bic_threshold",
        action=ToggleActionFlag,
        default=True,
        help="Use BIC to select the most parsimonious MIQP threshold",
    )
    p.add_argument(
        "-p",
        "--nproc",
        type=int,
        default=1,
        metavar="<int>",
        help="Number of processors to use",
    )

    # Solver options
    p.add_argument(
        "--qp-solver",
        dest="qp_solver",
        choices=available_qp_solvers.keys(),
        default=next(iter(available_qp_solvers.keys())),
        help="Select the QP solver",
    )
    p.add_argument(
        "--miqp-solver",
        dest="miqp_solver",
        choices=available_miqp_solvers.keys(),
        default=next(iter(available_miqp_solvers.keys())),
        help="Select the MIQP solver",
    )

    # qFit Segment options
    p.add_argument(
        "--only-segment",
        action="store_true",
        dest="only_segment",
        help="Only perform qfit segment",
    )
    p.add_argument(
        "-f",
        "--fragment-length",
        default=4,
        dest="fragment_length",
        metavar="<int>",
        type=int,
        help="Fragment length used during qfit_segment",
    )
    p.add_argument(
        "--segment-threshold-selection",
        action=ToggleActionFlag,
        dest="seg_bic_threshold",
        default=True,
        help="Use BIC to select the most parsimonious MIQP threshold (segment)",
    )

    # EM options
    p.add_argument(
        "-q",
        "--qscore",
        help="Q-score text output file",
    )
    p.add_argument(
        "-q_cutoff",
        "--qscore_cutoff",
        help="Q-score value where we should not model in alternative conformers.",
        default=0.7,
    )

    # Output options
    p.add_argument(
        "-d",
        "--directory",
        default=".",
        metavar="<dir>",
        type=os.path.abspath,
        help="Directory to store results",
    )
    p.add_argument("-v", "--verbose", action="store_true", help="Be verbose")
    p.add_argument(
        "--debug", action="store_true", help="Log as much information as possible"
    )
    p.add_argument(
        "--write_intermediate_conformers",
        action="store_true",
        help="Write intermediate structures to file (useful with debugging)",
    )
    p.add_argument("--pdb", help="Name of the input PDB")

    return p


class QFitProtein:
    def __init__(self, structure, xmap, options):
        self.xmap = xmap
        self.structure = structure
        self.options = options

    def run(self):
        if self.options.pdb is not None:
            self.pdb = self.options.pdb + "_"
        else:
            self.pdb = ""
        # Get information about all backbone atoms. If the input structure has multiple backbones, we will initiate backbone sampling (and all residue sampling) using all backbone coordinates
        if self.options.residue is not None:
            chainid, resi = self.options.residue.split(",")

        if self.options.residue is not None:  # run qFit residue
            multiconformer = self._run_qfit_residue_parallel()
        elif self.options.only_segment:
            multiconformer = self._run_qfit_segment(self.structure)
        else:
            multiconformer = self._run_qfit_residue_parallel()
            multiconformer = self._run_qfit_segment(multiconformer)
        return multiconformer

    def get_map_around_substructure(self, substructure):
        """Make a subsection of the map near the substructure.

        Args:
            substructure (qfit.structure.base_structure._BaseStructure):
                a substructure to carve a map around, commonly a Residue

        Returns:
            qfit.volume.XMap: a new (smaller) map
        """
        return self.xmap.extract(substructure.coor, padding=self.options.padding)

    def _run_qfit_residue_parallel(self):
        """Run qfit independently over all residues."""
        # This function hands out the job in parallel to a Pool of Workers.
        # To create Workers, we will use "forkserver" where possible,
        #     and default to "spawn" elsewhere (e.g. on Windows).
        try:
            ctx = mp.get_context(method="forkserver")
        except ValueError:
            ctx = mp.get_context(method="spawn")

        # Extract non-protein atoms
        self.hetatms = self.structure.extract("record", "HETATM", "==")
        waters = self.structure.extract("resn", "HOH", "==")
        self.hetatms = self.hetatms.combine(waters)
        hetatms = self.hetatms

        # Create a list of residues from single conformations of proteinaceous residues.
        # If we were to loop over all single_conformer_residues, then we end up adding HETATMs in two places
        #    First as we combine multiconformer_residues into multiconformer_model (because they won't be in ROTAMERS)
        #    And then as we re-combine HETATMs back into the multiconformer_model.
        if self.options.residue is not None:
            chainid, resi = self.options.residue.split(",")
            if ":" in resi:
                resi, icode = resi.split(":")
                residue_id = (int(resi), icode)
            elif "_" in resi:
                resi, icode = reis.split("_")
                residue_id = (int(resi), icode)
            else:
                residue_id = int(resi)
                icode = ""
            # Extract the residue:
            residues = list(
                self.structure.extract(f"resi {resi} and chain {chainid}")
                .extract("resn", "HOH", "!=")
                .single_conformer_residues
            )

        else:
            residues = list(
                self.structure.extract("record", "HETATM", "!=")
                .extract("resn", "HOH", "!=")
                .single_conformer_residues
            )

        backbone_coor_dict = {}
        for residue in residues:
            grouped_coords = {}
            grouped_u_matrices = {}
            residue_chain_key = (residue.resi[0], residue.chain[0].replace('"', ""))
            if residue_chain_key not in backbone_coor_dict:
                backbone_coor_dict[residue_chain_key] = {}
            for altloc in np.unique(
                self.structure.extract("chain", residue.chain[0], "==")
                .extract("resi", residue.resi[0], "==")
                .altloc
            ):
                grouped_coords[altloc] = (
                    self.structure.extract("chain", residue.chain[0], "==")
                    .extract("resi", residue.resi[0], "==")
                    .extract("altloc", altloc)
                    .coor
                )
                if residue.resn[0] == "GLY":
                    atom_name = "O"
                else:
                    atom_name = "CB"
                atom = (
                    self.structure.extract("chain", residue.chain[0], "==")
                    .extract("resi", residue.resi[0], "==")
                    .extract("name", atom_name)
                    .extract("altloc", altloc)
                )
                try:
                    u_matrix = [
                        [atom.u00[0], atom.u01[0], atom.u02[0]],
                        [atom.u01[0], atom.u11[0], atom.u12[0]],
                        [atom.u02[0], atom.u12[0], atom.u22[0]],
                    ]
                except:
                    u_matrix = None
                grouped_u_matrices[altloc] = u_matrix
            backbone_coor_dict[residue_chain_key]["coords"] = grouped_coords
            backbone_coor_dict[residue_chain_key]["u_matrices"] = grouped_u_matrices

        # Filter the residues: take only those not containing checkpoints.
        def does_multiconformer_checkpoint_exist(residue):
            fname = os.path.join(
                self.options.directory,
                residue.shortcode,
                "multiconformer_residue.pdb",
            )
            if os.path.exists(fname):
                logger.info(f"Residue {residue.shortcode}: {fname} already exists.")
                return True
            else:
                return False

        residues_to_sample = list(
            itl.filterfalse(does_multiconformer_checkpoint_exist, residues)
        )

        # Print execution stats
        logger.info(f"Residues to sample: {len(residues_to_sample)}")
        logger.info(f"nproc: {self.options.nproc}")

        # Build a Manager, have it construct a Queue. This will conduct
        #   thread-safe and process-safe passing of LogRecords.
        # Then launch a QueueListener Thread to read & handle LogRecords
        #   that are placed on the Queue.
        mgr = mp.Manager()
        logqueue = mgr.Queue()
        listener = QueueListener(logqueue)
        listener.start()

        # Initialise progress bar
        progress = tqdm(
            total=len(residues_to_sample),
            desc="Sampling residues",
            unit="residue",
            unit_scale=True,
            leave=True,
            miniters=1,
        )

        # Define callbacks and error callbacks to be attached to Jobs
        def _cb(result):
            if result:
                logger.info(result)
            progress.update()

        def _error_cb(e):
            tb = "".join(traceback.format_exception(e.__class__, e, e.__traceback__))
            logger.critical(tb)
            progress.update()

        # Here, we calculate alternate conformers for individual residues.
        if self.options.nproc > 1:
            # If multiprocessing, launch a Pool and run Jobs
            with ctx.Pool(processes=self.options.nproc, maxtasksperchild=4) as pool:
                futures = [
                    pool.apply_async(
                        QFitProtein._run_qfit_residue,
                        kwds={
                            "residue": residue,
                            "structure": self.structure,
                            "xmap": self.get_map_around_substructure(residue),
                            "options": self.options,
                            "logqueue": logqueue,
                            "backbone_coor_dict": backbone_coor_dict,
                        },
                        callback=_cb,
                        error_callback=_error_cb,
                    )
                    for residue in residues_to_sample
                ]

                # Make sure all jobs are finished
                # #TODO If a task crashes or is OOM killed, then there is no result.
                #       f.wait waits forever. It would be good to handle this case.
                for f in futures:
                    f.wait()

            # Wait until all workers have completed
            pool.join()

        else:
            # Otherwise, run this in the MainProcess
            for residue in residues_to_sample:
                if residue.resn[0] not in ROTAMERS:
                   self.hetatms = self.hetatms.combine(residue)
                   continue
                else:
                    try:
                        result = QFitProtein._run_qfit_residue(
                            residue=residue,
                            structure=self.structure,
                            xmap=self.get_map_around_substructure(residue),
                            options=self.options,
                            logqueue=logqueue,
                            backbone_coor_dict=backbone_coor_dict,
                        )
                        _cb(result)
                    except Exception as e:
                        _error_cb(e)

        # Close the progressbar
        progress.close()

        # There are no more sub-processes, so we stop the QueueListener
        listener.stop()
        listener.join()

        # Combine all multiconformer residues into one structure, multiconformer_model
        multiconformer_model = None
        if self.options.residue is not None:
            for residue in residues:
                # Load the multiconformer_residue.pdb file
                fname = os.path.join(
                    self.options.directory,
                    residue.shortcode,
                    "multiconformer_residue.pdb",
                )
                if not os.path.exists(fname):
                    logger.warn(
                        f"[{residue.shortcode}] Couldn't find {fname}! "
                        "Will not be present in multiconformer_model.pdb!"
                    )
                    continue
                residue_multiconformer = Structure.fromfile(fname)
                fname = f"{residue.shortcode}_qFit_residue.pdb"
                residue_multiconformer.tofile(fname)

        if self.options.residue is None:
            for residue in residues:
                # Check the residue is a rotameric residue,
                # if not, we won't have a multiconformer_residue.pdb.
                # Make sure to append it to the hetatms object so it stays in the final output.
                if residue.resn[0] not in ROTAMERS:
                    hetatms = hetatms.combine(residue)
                    continue

                # Load the multiconformer_residue.pdb file
                fname = os.path.join(
                    self.options.directory,
                    residue.shortcode,
                    "multiconformer_residue.pdb",
                )
                if not os.path.exists(fname):
                    logger.warn(
                        f"[{residue.shortcode}] Couldn't find {fname}! "
                        "Will not be present in multiconformer_model.pdb!"
                    )
                    continue
                residue_multiconformer = Structure.fromfile(fname)

                # Stitch them together
                if multiconformer_model is None:
                    multiconformer_model = residue_multiconformer
                else:
                    multiconformer_model = multiconformer_model.combine(
                        residue_multiconformer
                    )

            # Write out multiconformer_model.pdb only if in debug mode.
            # This output is not a final qFit output, so it might confuse users.
            if self.options.debug:
                fname = os.path.join(self.options.directory, "multiconformer_model.pdb")
                if self.structure.scale or self.structure.cryst_info:
                    multiconformer_model.tofile(
                        fname, self.structure.scale, self.structure.cryst_info
                    )
                else:
                    multiconformer_model.tofile(fname)

        return multiconformer_model

    def _run_qfit_segment(self, multiconformer):
        self.options.randomize_b = False
        self.options.bic_threshold = self.options.seg_bic_threshold
        if self.options.seg_bic_threshold:
            self.options.fragment_length = 3
        else:
            self.options.threshold = 0.2

        # Extract map of multiconformer in P1
        logger.debug("Extracting map...")
        _then = time.process_time()
        self.xmap = self.get_map_around_substructure(self.structure)
        _now = time.process_time()
        logger.debug(f"Map extraction took {(_now - _then):.03f} s.")

        qfit = QFitSegment(multiconformer, self.xmap, self.options)
        multiconformer = qfit()
        multiconformer = multiconformer.combine(self.hetatms)
        fname = os.path.join(
            self.options.directory, self.pdb + "multiconformer_model2.pdb"
        )
        if self.options.scale or self.options.cryst_info:
            multiconformer.tofile(
                fname, self.options.scale_info, self.options.cryst_info
            )
        else:
            multiconformer.tofile(fname)
        return multiconformer

    @staticmethod
    def _run_qfit_residue(
        residue, structure, xmap, options, logqueue, backbone_coor_dict
    ):
        """Run qfit on a single residue to determine density-supported conformers."""

        # Don't run qfit if we have a ligand or water
        if residue.type != "rotamer-residue":
            raise RuntimeError(
                f"Residue {residue.id}: is not a rotamer-residue. Aborting qfit_residue sampling."
            )

        # Set up logger hierarchy in this subprocess
        poolworker_setup_logging(logqueue)

        # This function is run in a subprocess, so `structure` and `residue` have
        #     been 'copied' (pickled+unpickled) as best as possible.

        # However, `structure`/`residue` objects pickled and passed to subprocesses do
        #     not contain attributes decorated by @_structure_properties.
        #     This decorator attaches 'getter' and 'setter' _local_ functions to the attrs
        #     (defined within, and local to the _structure_properties function).
        #     Local functions are **unpickleable**, and as a result, so are these attrs.
        # This includes:
        #     (record, atomid, name, altloc, resn, chain, resi, icode,
        #      q, b, e, charge, coor, active, u00, u11, u22, u01, u02, u12)
        # Similarly, these objects are also missing attributes wrapped by @property:
        #     (covalent_radius, vdw_radius)
        # Finally, the _selector object is only partially pickleable,
        #     as it contains a few methods that are defined by a local lambda inside
        #     pyparsing._trim_arity().

        # Since all these attributes are attached by __init__ of the
        #     qfit.structure.base_structure._BaseStructure class,
        #     here, we call __init__ again, to make sure these objects are
        #     correctly initialised in a subprocess.
        structure.__init__(
            structure.data,
            selection=structure._selection,
            parent=structure.parent,
        )
        residue.__init__(
            residue.data,
            resi=residue.id[0],
            icode=residue.id[1],
            type=residue.type,
            selection=residue._selection,
            parent=residue.parent,
        )

        # Build the residue results directory
        residue_directory = os.path.join(options.directory, residue.shortcode)
        try:
            os.makedirs(residue_directory)
        except OSError:
            pass

        # Exit early if we have already run qfit for this residue
        fname = os.path.join(residue_directory, "multiconformer_residue.pdb")
        if os.path.exists(fname):
            logger.info(
                f"Residue {residue.shortcode}: {fname} already exists, using this checkpoint."
            )
            return

        # Determine if q-score is too low
        if options.qscore is not None:
            (chainid, resi, icode) = residue._identifier_tuple
            if (
                list(
                    options.qscore[
                        (options.qscore["Res_num"] == resi)
                        & (options.qscore["Chain"] == chainid)
                    ]["Q_sideChain"]
                )[0]
                < options.q_cutoff
            ):
                logger.info(
                    f"Residue {residue.shortcode}: Q-score is too low for this residue. Using deposited structure."
                )
                resi_selstr = f"chain {chainid} and resi {resi}"
                if icode:
                    resi_selstr += f" and icode {icode}"
                structure_new = structure
                structure_resi = structure.extract(resi_selstr)
                chain = structure_resi[chainid]
                conformer = chain.conformers[0]
                residue = conformer[residue.id]
                residue.tofile(fname)
                return

        # Copy the structure
        (chainid, resi, icode) = residue._identifier_tuple
        resi_selstr = f"chain {chainid} and resi {resi}"
        if icode:
            resi_selstr += f" and icode {icode}"
        structure_new = structure
        structure_resi = structure.extract(resi_selstr)
        chain = structure_resi[chainid]
        conformer = chain.conformers[0]
        residue = conformer[residue.id]
        altlocs = sorted(list(set(residue.altloc)))
        if len(altlocs) > 1:
            try:
                altlocs.remove("")
            except ValueError:
                pass
            for altloc in altlocs[1:]:
                sel_str = f"resi {resi} and chain {chainid} and altloc {altloc}"
                sel_str = f"not ({sel_str})"
                structure_new = structure_new.extract(sel_str)

        # add multiple backbone positions

        chain_resi_id = f"{resi}, '{chainid}'"
        chain_resi_id = (resi, chainid)
        options.backbone_coor_dict = backbone_coor_dict[chain_resi_id]
        # Exception handling in case qFit-residue fails:
        qfit = QFitRotamericResidue(residue, structure_new, xmap, options)
        try:
            qfit.run()
        except RuntimeError as e:
            tb = "".join(traceback.format_exception(e.__class__, e, e.__traceback__))
            logger.warning(
                f"[{qfit.identifier}] "
                f"Unable to produce an alternate conformer. "
                f"Using deposited conformer A for this residue."
            )
            logger.info(
                f"[{qfit.identifier}] This is a result of the following exception:\n"
                f"{tb})"
            )
            qfit.conformer = residue.copy()
            qfit._occupancies = [residue.q]
            qfit._coor_set = [residue.coor]
            qfit._bs = [residue.b]

        # Save multiconformer_residue
        qfit.tofile()
        qfit_id = qfit.identifier

        # How many conformers were found?
        n_conformers = len(qfit.get_conformers())

        # Freeing up some memory to avoid memory issues:
        del xmap
        del qfit
        gc.collect()

        # Return a string about the residue that was completed.
        return f"[{qfit_id}]: {n_conformers} conformers"


def prepare_qfit_protein(options):
    """Loads files to build a QFitProtein job."""

    # Load structure and prepare it
    structure = Structure.fromfile(options.structure).reorder()
    options.scale_info = structure.scale
    options.cryst_info = structure.cryst_info
    if not options.hydro:
        structure = structure.extract("e", "H", "!=")

    # fixing issues with terminal oxygens
    rename = structure.extract("name", "OXT", "==")
    rename.name = "O"
    structure = structure.extract("name", "OXT", "!=").combine(rename)

    # Load map and prepare it
    xmap = XMap.fromfile(
        options.map, resolution=options.resolution, label=options.label
    )
    xmap = xmap.canonical_unit_cell()

    # Scale map based on input structure
    if options.scale is True:
        scaler = MapScaler(xmap, em=options.em)
        radius = 1.5
        reso = None
        if xmap.resolution.high is not None:
            reso = xmap.resolution.high
        elif options.resolution is not None:
            reso = options.resolution
        if reso is not None:
            radius = 0.5 + reso / 3.0
        scaler.scale(structure, radius=options.scale_rmask * radius)

    if options.qscore is not None:
        with open(
            options.qscore, "r"
        ) as f:  # not all qscore header are the same 'length'
            for line_n, line_content in enumerate(f):
                if "Q_sideChain" in line_content:
                    break
            start_row = line_n + 1
        options.qscore = pd.read_csv(
            options.qscore,
            sep="\t",
            skiprows=start_row,
            skip_blank_lines=True,
            on_bad_lines="skip",
            header=None,
        )
        options.qscore = options.qscore.iloc[
            :, :6
        ]  # we only care about the first 6 columns
        options.qscore.columns = [
            "Chain",
            "Res",
            "Res_num",
            "Q_backBone",
            "Q_sideChain",
            "Q_residue",
        ]  # rename column names
        options.qscore["Res_num"] = options.qscore["Res_num"].fillna(0).astype(int)

    return QFitProtein(structure, xmap, options)


def main():
    """Default entrypoint for qfit_protein."""

    # Collect and act on arguments
    #   (When args==None, argparse will default to sys.argv[1:])
    p = build_argparser()
    args = p.parse_args(args=None)

    try:
        os.mkdir(args.directory)
    except OSError:
        pass

    # Apply the arguments to options
    options = QFitOptions()
    options.apply_command_args(args)

    # Setup logger
    setup_logging(options=options)
    log_run_info(options, logger)

    # Build a QFitProtein job
    qfit = prepare_qfit_protein(options=options)

    # Run the QFitProtein job
    time0 = time.time()
    multiconformer = qfit.run()
    logger.info(f"Total time: {time.time() - time0}s")
