"""
Shared ligand-clash detection, used identically by build_final_model.py (Stage 6) and
rotamer_optimize.py (Stage 7): filters out candidate sidechain conformers that clash with a
surviving ligand pose, BEFORE any density/MSE/RSCC scoring runs on them. Filtering first (rather
than scoring then penalizing/reverting after) is both a correctness fix - a sidechain rotated
directly into a ligand's own unassigned density would otherwise score very well by RSCC/MSE
alone, with nothing else in either script's pipeline ever checking it against the ligand - and a
compute saving, since the expensive map-scoring step never has to run on an excluded candidate.

Per-(residue, ligand-instance) pair, a cheap centroid+reach bounding-sphere prefilter (margin
configurable, default 4.0 A) skips the expensive exact atom-pairwise check whenever the two
bounding spheres can't possibly overlap - confirmed ~3.6x faster than an unconditional
brute-force check against every surviving ligand instance, with identical results, on a real
dataset (x3413/placer2_1 - see mac1/claude_outputs). A residue's bounding sphere is computed
once, pooled across every one of its OWN candidate conformers (not just its current one), so the
prefilter never excludes a pair that the chi-angle/angle sampling could plausibly reach.

The exact atom-pairwise clash test reuses the same convention as sidechain_clash.py's own
domain_compatibility_matrix: a pair clashes when their distance is below clash_vdw_scale times
the summed VDW radii, except an (N, O) pair (either order) which uses the looser
hbond_clash_vdw_scale instead (a real hydrogen bond legitimately sits closer than a generic
clash would tolerate). Only sidechain atoms are ever checked - backbone atoms are exempt, same
as every other clash check in this project.

Two distinct ways to gather "every surviving ligand instance" are provided, because the two
callers' input structures represent ligand instances differently:
  - build_final_model.py's multimodel_pdb (e.g. cluster_rep_models.pdb) has ONE MODEL per
    surviving filter2 cluster_reps.csv row, each with exactly one ligand instance in it (often
    reusing the same chain/resi label across models, e.g. always chain C resi 1 - see
    _buildFinalModel's own renumbering-by-model-number) - so each whole model's own `resname LIG`
    extraction is one instance; see ligand_instances_from_multimodel.
  - rotamer_optimize.py's model_file (final_model.pdb) already has every surviving ligand
    instance MERGED into a single structure, each given its own distinct residue number by
    build_final_model.py's _set_resi - so instances are recovered by grouping that one
    structure's `resname LIG` atoms by (chain_id, resi); see ligand_instances_from_structure.
"""
from collections import namedtuple

import numpy as np

CLASH_VDW_SCALE = 0.75
HBOND_CLASH_VDW_SCALE = 0.6
LIGAND_CLASH_PREFILTER_MARGIN = 4.0

BACKBONE_ATOM_NAMES = {'N', 'CA', 'C', 'O', 'OXT'}

# One per distinct surviving ligand site. coor/vdw/e: (n_atoms,[3]) arrays for that instance's
# own heavy atoms (ligands in this pipeline are already heavy-atom-only - confirmed empirically
# across several datasets, see mac1/claude_outputs). centroid/reach: this instance's own
# bounding sphere (reach = max distance from centroid to any of its own atoms), used by the
# prefilter. chain_id/resi: for logging only.
LigandInstance = namedtuple(
    'LigandInstance', ['coor', 'vdw', 'e', 'centroid', 'reach', 'chain_id', 'resi'],
)


def _bounding_sphere(coor):
    """coor: (..., 3), any shape that reshapes cleanly to (N, 3) - e.g. a single instance's own
    atoms, or a residue's candidates pooled across every sampled conformer. Returns
    (centroid, reach)."""
    flat = np.asarray(coor).reshape(-1, 3)
    centroid = flat.mean(axis=0)
    reach = float(np.max(np.linalg.norm(flat - centroid, axis=1)))
    return centroid, reach


def _make_instance(coor, vdw, e, chain_id, resi):
    centroid, reach = _bounding_sphere(coor)
    return LigandInstance(coor=coor, vdw=vdw, e=e, centroid=centroid, reach=reach,
                           chain_id=chain_id, resi=resi)


def ligand_instances_from_multimodel(models):
    """One LigandInstance per model in `models` (e.g. FinalModelBuilder.multimodel_models) -
    see module docstring for why this is the right granularity for build_final_model.py's own
    multimodel_pdb input. Models with zero LIG atoms are skipped."""
    instances = []
    for model in models:
        lig = model.extract('resname LIG')
        if lig.natoms == 0:
            continue
        chain_id = str(lig.chain[0]) if len(lig.chain) else '?'
        resi = int(lig.resi[0]) if len(lig.resi) else -1
        instances.append(_make_instance(lig.coor, np.asarray(lig.vdw_radius),
                                         np.asarray(lig.e), chain_id, resi))
    return instances


def ligand_instances_from_structure(structure):
    """Groups a single structure's `resname LIG` atoms by (chain_id, resi) - altloc-agnostic,
    same convention used throughout this project's analysis scripts - into one LigandInstance
    per distinct site. Returns [] if the structure has no LIG atoms at all."""
    lig = structure.extract('resname LIG')
    if lig.natoms == 0:
        return []
    chains = np.asarray(lig.chain)
    resis = np.asarray(lig.resi)
    coor = lig.coor
    vdw = np.asarray(lig.vdw_radius)
    e = np.asarray(lig.e)

    instances = []
    for chain_id, resi in sorted(set(zip(chains, resis))):
        mask = (chains == chain_id) & (resis == resi)
        instances.append(_make_instance(coor[mask], vdw[mask], e[mask], str(chain_id), int(resi)))
    return instances


def _clash_mask(coor_arr, vdw, e, lig_coor, lig_vdw, lig_e, clash_vdw_scale, hbond_clash_vdw_scale):
    """coor_arr: (n_candidates, n_atoms, 3) - one residue's candidate sidechain atoms. vdw/e:
    (n_atoms,) matching coor_arr's atom axis. lig_coor/lig_vdw/lig_e: one ligand instance's own
    atoms. Returns (n_candidates,) bool - True if that candidate clashes with ANY given ligand
    atom (same per-atom-pair VDW-sum*scale threshold, N/O hbond exception, as
    sidechain_clash.py's domain_compatibility_matrix)."""
    diff = coor_arr[:, :, None, :] - lig_coor[None, None, :, :]
    dist = np.linalg.norm(diff, axis=-1)  # (n_cand, n_atoms, n_lig_atoms)

    vdw_sum = vdw[:, None] + lig_vdw[None, :]  # (n_atoms, n_lig_atoms)
    is_n = (e[:, None] == 'N')
    is_o = (e[:, None] == 'O')
    lig_is_o = (lig_e[None, :] == 'O')
    lig_is_n = (lig_e[None, :] == 'N')
    hbond_pair = (is_n & lig_is_o) | (is_o & lig_is_n)
    scale = np.where(hbond_pair, hbond_clash_vdw_scale, clash_vdw_scale)
    threshold = vdw_sum * scale  # (n_atoms, n_lig_atoms)

    clash = dist < threshold[None, :, :]
    return clash.any(axis=(1, 2))


def exclude_ligand_clashing_candidates(coor_arr, vdw, e, ligand_instances,
                                        clash_vdw_scale=CLASH_VDW_SCALE,
                                        hbond_clash_vdw_scale=HBOND_CLASH_VDW_SCALE,
                                        margin=LIGAND_CLASH_PREFILTER_MARGIN):
    """coor_arr: (n_candidates, n_atoms, 3) - one residue's candidate SIDECHAIN atoms only
    (caller applies the backbone mask - backbone atoms are never checked, same as every other
    clash check in this project). vdw/e: (n_atoms,) matching coor_arr's atom axis.

    Returns (keep_mask, n_excluded): keep_mask is (n_candidates,) bool, True = does not clash
    with any given ligand instance; n_excluded = int((~keep_mask).sum()).

    Per ligand instance, the cheap centroid+reach prefilter (this residue's own bounding sphere,
    pooled across every candidate in coor_arr - not just one) skips the expensive atom-pairwise
    check entirely when the two bounding spheres can't come within `margin` of overlapping."""
    n_cand = coor_arr.shape[0]
    if n_cand == 0 or not ligand_instances:
        return np.ones(n_cand, dtype=bool), 0

    centroid, reach = _bounding_sphere(coor_arr)
    keep = np.ones(n_cand, dtype=bool)
    for li in ligand_instances:
        d = float(np.linalg.norm(centroid - li.centroid))
        if d > reach + li.reach + margin:
            continue
        clash = _clash_mask(coor_arr, vdw, e, li.coor, li.vdw, li.e,
                             clash_vdw_scale, hbond_clash_vdw_scale)
        keep &= ~clash

    n_excluded = int((~keep).sum())
    return keep, n_excluded


def write_ligand_clash_csv(path, rows):
    """rows: iterable of (chain_id, residue_number, n_candidates_sampled,
    n_excluded_for_ligand_clash). Writes chain_id,residue_number,n_candidates_sampled,
    n_excluded_for_ligand_clash,reset_to_apo_due_to_clash - the last column is "yes" exactly
    when n_excluded_for_ligand_clash == n_candidates_sampled > 0 (every sampled candidate
    clashed, so the residue necessarily fell back to its pre-sampling conformation - apo for
    build_final_model.py, the input model_file conformation for rotamer_optimize.py).

    Returns the number of "yes" rows, for the caller's own console summary line."""
    n_reset = 0
    with open(path, 'w') as f:
        f.write('chain_id,residue_number,n_candidates_sampled,n_excluded_for_ligand_clash,'
                'reset_to_apo_due_to_clash\n')
        for chain_id, resi, n_sampled, n_excluded in rows:
            reset = n_sampled > 0 and n_excluded == n_sampled
            if reset:
                n_reset += 1
            f.write(f'{chain_id},{resi},{n_sampled},{n_excluded},{"yes" if reset else "no"}\n')
    return n_reset
