import logging
import os
import pandas as pd
import time

from qfit.qfit_bindingsite import QFitOptions, QFitBindingSite
from qfit import Structure
from qfit.command_line.common_options import get_base_argparser, load_and_scale_map
from qfit.logtools import (
    setup_logging,
    log_run_info,
    poolworker_setup_logging,
    QueueListener,
)

logger = logging.getLogger(__name__)
os.environ["OMP_NUM_THREADS"] = "1"

def build_argparser():
    p = get_base_argparser(__doc__,
                           default_enable_external_clash=True,
    )
    ##PLACER SPECIFIC ARGS
    p.add_argument(
        "-placer",
        "--placer_ligs",
        type=str,
        help="Path to the PDB file for the imported PLACER sampled ligand ensemble."
    )
    p.add_argument(
        "--ligand",
        "-lig",
        type=str,
        help="Name of the ligand in the placer ensemble",
    )

    return p

def prepare_qfit_bindingsite(options):
    """Loads files to build a QFitBindingSite job."""
    # Load structure and prepare it
    structure = Structure.fromfile(options.structure).reorder()
    if not options.hydro:
        structure = structure.extract("e", "H", "!=")

    # fixing issues with terminal oxygens
    rename = structure.extract("name", "OXT", "==")
    rename.name = "O"
    structure = structure.extract("name", "OXT", "!=").combine(rename)

    xmap = None
    xmap = load_and_scale_map(options, structure)

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

    return QFitBindingSite(structure, structure, xmap, options)


def main():
    p = build_argparser()
    args = p.parse_args()

    # Apply the arguments to options
    options = QFitOptions()
    options.apply_command_args(args)

    # Setup logger
    setup_logging(options=options)
    log_run_info(options, logger)

    # Build a QFitProtein job
    qfit = prepare_qfit_bindingsite(options=options)

    # Run the QFitProtein job
    time0 = time.time()
    multiconformer = qfit.run()
    multiconformer.tofile('multiconformer.pdb')
    logger.info(f"Total time: {time.time() - time0}s")

    #Build whole multiconformer to output to file ala qfit ligand
    time0 = time.time()
    rest_of_model = qfit.getNotBindingSite()
    multiconformer_model = multiconformer.combine(rest_of_model)
    multiconformer.tofile('multiconformer_model.pdb')
    print(f'Built multiconformer file in {time.time() - time0}')
    time0 =time.time()
    multiconformer_model2 = qfit.reorder(multiconformer_model)
    print(f'Reorganized multiconformer in {time.time() - time0}')
    multiconformer_model2.tofile('multiconformer_model2.pdb')
    
