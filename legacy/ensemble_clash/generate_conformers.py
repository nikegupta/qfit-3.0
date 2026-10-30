"""
generate_conformers - samples a null ensemble of ligand placements in one chain's binding pocket.

Each sample is a random torsional conformer of the dataset's ligand, translated and rotated at
random within the chain's pocket volume's minimum enclosing sphere, and kept only if it lies
inside the pocket volume and doesn't clash with the protein. Samples are drawn until --n-conformers are kept (or --max-candidates-per-conformer x --n-conformers have been tried).
One ensemble covers the pocket of ligand_chain, however many modeled ligand copies (LIG residues)
that chain holds: these are the reference ligands (--reference-resi, default every LIG residue in
the chain), and the first is the template.

  1. Torsion library: --n-torsion-confs conformers embedded from the ligand's SMILES
     (pxr_fragments.csv) with RDKit ETKDGv3 (hydrogens added for embedding, then removed), each
     superposed on the template ligand's heavy atoms (best symmetry-equivalent mapping) and
     written in the template's atom order and names.
  2. Each candidate draws, independently and uniformly:
       - a torsion conformer from the library;
       - a translation of the conformer's centroid from the sampling centre, evenly spread by
         volume within the max radius: direction with cos(polar) and azimuth uniform,
         distance = max_radius * cbrt(u); polar/azimuth rounded to --angle-step degrees and the
         distance to --radius-step Å. The centre and max radius come from the clash ensemble:
         the minimum enclosing sphere of the centroids of every ensemble model's LIG residues in
         this chain (models superposed onto the reference on all CA; centre moved into this
         structure's frame), its radius plus a cushion (default 2 Å). This is a cap, not a
         bound: a pose whose centroid is farther out isn't sampled even if it fits the pocket
         volume (step 3);
       - an orientation: a uniformly random rotation (all three rotational degrees of freedom)
         about the conformer's centroid.
  3. Pocket volume: every heavy atom of a candidate must lie inside ligand_chain's pocket volume
     from the clash ensemble (below): every ensemble model's LIG heavy atoms in that chain, each
     made a sphere of one fixed radius (default 2 Å, the same for every element - a cushion
     around the observed atom positions, not a physical size), with a probe (default 2 Å) rolled
     over the union of those spheres: the volume the probe can't enter, i.e. the spheres with
     the crevices and gaps between them narrower than the probe filled in, and no growth
     outward; on a 0.25 Å grid in the reference structure's frame.
     Candidate atoms are moved into that frame by superposing this structure onto the reference
     on all CA atoms they share.
  4. Clash check: the backbone is rigid, side chains are flexible. The structure's own backbone
     (N, CA, C, O, OXT, all altlocs) is always an obstacle. Binding-site residues (residue
     numbers, chain agnostic, with a heavy atom within 4 Å of a LIG residue in any ensemble
     model) take their side chains from the clash ensemble (--clash-ensemble), built once per run
     by build_clash_ensemble: each binding-site residue's distinct side-chain conformations
     across all ensemble models, each superposed onto the residue's backbone frame on its own N,
     CA and C (so loop or backbone movement between models never counts as a side-chain
     conformation; each altloc is a conformation; conformations missing atoms most others have
     are dropped; de-duplicated within a rotamer tolerance, default 0.3 Å RMSD, raised for any
     residue that would otherwise keep more than 128). Here they are grafted onto this
     structure's backbone (residue frame superposed onto its N, CA, C). Every other residue
     keeps only this structure's own side-chain conformations (altlocs). A candidate is dropped
     if it clashes with the backbone, or if, for some residue, it clashes with every one of that
     residue's side-chain conformations - so any combination of observed rotamers can make room
     for it. Without --clash-ensemble the structure alone is the ensemble (its own ligands give
     the binding site and pocket volume). Obstacles exclude hydrogens, LIG and excluded
     residues; no symmetry mates.
     A ligand-obstacle atom pair clashes when closer than --clash-scale x (sum of vdW radii), or
     --polar-clash-scale x for N-N, N-O and O-O pairs.

Speed: candidates are generated and checked in numpy batches. The pocket volume is a grid
lookup, done first. Residues with a single conformation ("rigid") are ordinary obstacles. Lookup
grids around the pocket mark, per ligand element, the points where an atom certainly clashes with
a rigid residue or with every conformation of a flexible one (cutoffs shrunk by half a voxel
diagonal, so a grid hit is always real). Ligand atoms are checked core-first and a candidate is
dropped at its first certain hit; survivors get exact KD-tree checks, and the last
survivors the exact all-conformations check of the flexible residues (a bitmask of clashed
conformations per residue). Results are exact up to the rotamer de-duplication. It is meant to
run single-threaded (parallelism comes from running pockets side by side): the hot loop avoids
BLAS, and when calling it directly set OPENBLAS_NUM_THREADS=1 (and OMP_/MKL_NUM_THREADS=1), as
generate_null.sh does, so numpy's threaded OpenBLAS doesn't spread other products over every core.

Outputs (in output_dir):
  conformers.pdb            the kept placements, one MODEL each (heavy atoms)
  conformers.csv            per kept placement: torsion conformer, translation polar/azimuth/
                            distance, rotation quaternion (x, y, z, w), centroid
  torsion_library.pdb       the superposed torsion conformers, one MODEL each
  ligand.pdb                the reference ligands (heavy atoms, first altloc), one MODEL each
  obstacles.pdb             the structure's own backbone near the pocket, then every nearby
                            residue's side-chain conformations grafted onto it; flexible side
                            chains as altlocs A, B, ...,
                            occupancy = share of the residue's observed conformations
  ensemble_mask.ccp4        1 inside the vdW volume of any kept placement, 0 elsewhere (see below)
  conformer_summary.json    parameters, sampling centre, clash ensemble (backbone atoms, rigid
                            and flexible side chains), binding site, pocket volume (superposition
                            onto the reference, share of reference ligand atoms inside),
                            candidates tried, acceptance,
                            conformers kept, ensemble volume, each reference ligand's distance
                            from the centre and share of its vdW volume inside the mask, runtime

Ensemble mask: the union of every kept placement's atom vdW spheres, on a grid of the crystal's
own unit cell (cell from the CSV, written as P1) at --mask-spacing, boxed around the ensemble. It
is in the same frame as the model, so any later pose can be tested against it directly:
    from qfit import XMap
    from qfit.command_line.generate_conformers import mask_fraction_inside
    mask = XMap.fromfile('ensemble_mask.ccp4', resolution=1)
    fraction = mask_fraction_inside(mask, xyz, radii)   # share of the pose's vdW volume inside
"""
import argparse
import csv
import json
import pickle
import string
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolAlign
from scipy.ndimage import distance_transform_edt
from scipy.spatial import ConvexHull, QhullError, cKDTree
from scipy.spatial.transform import Rotation

from qfit.structure.vdw_radii import vdwRadiiTable
from qfit.xtal.unitcell import UnitCell
from qfit.xtal.volume import GridParameters, XMap

LIGAND_RESN = 'LIG'
POLAR_ELEMENTS = ('N', 'O')
DEFAULT_VDW = vdwRadiiTable['C']
# Most conformations kept per residue; a flexible residue's conformations are tracked as bits of
# MAX_CONFORMATIONS / 64 uint64 words.
MAX_CONFORMATIONS = 128
MASK_WORDS = MAX_CONFORMATIONS // 64
# Conformations of a residue within this heavy-atom RMSD (Å) count as one.
DEFAULT_ROTAMER_TOLERANCE = 0.3
# Default --max-candidates-per-conformer: candidates tried, at most, per conformer asked for
# (--n-conformers).
MAX_CANDIDATES_PER_CONFORMER = 1_000_000
# Backbone atoms: always the sampled structure's own (rigid); everything else in a residue is its
# side chain, taken from the clash ensemble. FRAME_ATOMS define a residue's backbone frame.
BACKBONE_ATOMS = ('N', 'CA', 'C', 'O', 'OXT')
FRAME_ATOMS = ('N', 'CA', 'C')
# Altloc letters for the conformations of flexible residues in obstacles.pdb.
ALTLOCS = string.ascii_uppercase + string.ascii_lowercase
# Candidates per all-conformations test of the flexible residues (bounds its memory).
FLEXIBLE_CHUNK = 100_000
# Pocket volume: radius (Å) of the sphere every observed ligand heavy atom is made, and of the
# probe rolled over them; and its grid spacing (Å, DEFAULT_VOLUME_SPACING below).
DEFAULT_POCKET_ATOM_RADIUS = 2.0
DEFAULT_POCKET_PROBE_RADIUS = 0.5
# Sampling sphere: cushion (Å) added to the radius of the observed ligand centroids' enclosing
# sphere.
DEFAULT_CENTROID_MARGIN = 2.0
DEFAULT_VOLUME_SPACING = 0.25
# Binding-site residues: a heavy atom within this distance (Å) of a LIG residue in some model.
DEFAULT_BINDING_SITE_CUTOFF = 4.0


def build_argparser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('structure_pdb', help='Structure holding the modeled ligand(s).')
    p.add_argument('fragments_csv',
                   help='CSV with dataset, a, b, c, alpha, beta, gamma, space_group, smiles.')
    p.add_argument('dataset', help='Dataset name (row of fragments_csv).')
    p.add_argument('ligand_chain', help='Chain whose binding pocket is sampled.')
    p.add_argument('output_dir')
    p.add_argument('--reference-resi', type=int, nargs='+',
                   help='Residue numbers of the reference LIG residues in ligand_chain (the first '
                        'is the template). Default: every LIG residue in the chain.')
    p.add_argument('-n', '--n-conformers', type=int, default=1000,
                   help='Number of non-clashing placements to keep. Default: 10000.')
    p.add_argument('--n-torsion-confs', type=int, default=200,
                   help='Size of the ETKDG torsion conformer library. Default: 200.')
    p.add_argument('--radius-step', type=float, default=0.1,
                   help='Translation distances are rounded to this step (Å). Default: 0.1.')
    p.add_argument('--angle-step', type=float, default=1.0,
                   help='Translation polar/azimuth angles are rounded to this step (degrees). '
                        'Default: 1.')
    p.add_argument('--clash-scale', type=float, default=0.75,
                   help='A pair clashes below this fraction of its vdW radius sum. Default: 0.75.')
    p.add_argument('--polar-clash-scale', type=float, default=0.6,
                   help='Same, for N-N, N-O and O-O pairs. Default: 0.6.')
    p.add_argument('--clash-ensemble',
                   help='Clash ensemble file from build_clash_ensemble (built once per run); it '
                        'provides the binding-site residues\' side-chain conformations (grafted '
                        'onto structure_pdb\'s backbone) and the pocket volumes. Default: '
                        'structure_pdb alone (its own side chains and ligands).')
    p.add_argument('--exclude-resn', nargs='+', default=['HOH', 'DMS'],
                   help='Residue names left out of the obstacles (hydrogens and every LIG '
                        'residue are always left out). Default: HOH DMS.')
    p.add_argument('--grid-spacing', type=float, default=0.25,
                   help='Spacing of the lookup grids (Å); only affects speed. Default: 0.25.')
    p.add_argument('--mask-spacing', type=float, default=0.25,
                   help='Grid spacing of ensemble_mask.ccp4 (Å); rounded so it divides each cell '
                        'edge evenly. Default: 0.25.')
    p.add_argument('--batch-size', type=int, default=1_000_000,
                   help='Largest number of candidates generated at once. Default: 1000000.')
    p.add_argument('--max-candidates-per-conformer', type=int,
                   default=MAX_CANDIDATES_PER_CONFORMER,
                   help='Stop after this many candidates per conformer asked for (i.e. after '
                        'this x --n-conformers candidates) even if fewer than --n-conformers '
                        f'were kept. Default: {MAX_CANDIDATES_PER_CONFORMER}.')
    p.add_argument('--seed', type=int, default=None,
                   help='Random seed (0 to 2^31 - 2). Default: a fresh seed from the clock '
                        '(time.time_ns()), recorded in conformer_summary.json so the run can be '
                        'reproduced with --seed.')
    return p


# --- Input -----------------------------------------------------------------------------------

def element_of(line):
    e = line[76:78].strip().upper()
    return e if e else line[12:16].strip()[0].upper()


def coords(lines):
    return np.array([[float(l[30:38]), float(l[38:46]), float(l[46:54])] for l in lines])


def read_crystal_info(csv_path, dataset):
    with open(csv_path, newline='') as f:
        for row in csv.DictReader(f):
            if row['dataset'] == dataset:
                cell = tuple(float(row[k]) for k in ('a', 'b', 'c', 'alpha', 'beta', 'gamma'))
                return cell, row['space_group'], row['smiles']
    raise SystemExit(f'Dataset {dataset} not found in {csv_path}')


def chain_ligands(pdb, chain):
    """Residue numbers of every LIG residue in chain, in file order."""
    resis = []
    for l in open(pdb):
        if (l.startswith(('ATOM', 'HETATM')) and l[17:20].strip() == LIGAND_RESN
                and l[21] == chain and int(l[22:26]) not in resis):
            resis.append(int(l[22:26]))
    return resis


def read_ligand(pdb, chain, resi):
    """Heavy-atom lines of the modeled ligand, first alternate conformer only."""
    lines = [l for l in open(pdb) if l.startswith(('ATOM', 'HETATM'))
             and l[17:20].strip() == LIGAND_RESN and l[21] == chain and int(l[22:26]) == resi
             and l[16] in ' A' and element_of(l) != 'H']
    if not lines:
        raise SystemExit(f'No {LIGAND_RESN} {chain} {resi} heavy atoms in {pdb}')
    return lines


def read_protein(pdb, exclude_resn):
    """{(chain, resi): (resn, {altloc: {atom name: line}})} of the heavy atoms, in file order;
    LIG, hydrogens and exclude_resn residues left out."""
    skip = set(exclude_resn) | {LIGAND_RESN}
    residues = {}
    for l in open(pdb):
        if (not l.startswith(('ATOM', 'HETATM')) or l[17:20].strip() in skip
                or element_of(l) == 'H'):
            continue
        key = (l[21], int(l[22:26]))
        if key not in residues:
            residues[key] = (l[17:20].strip(), {})
        residues[key][1].setdefault(l[16], {})[l[12:16].strip()] = l
    return residues


def conformations(altlocs):
    """A residue's conformations as {atom name: line}: the blank-altloc atoms plus, for each
    altloc letter, that altloc's atoms (one conformation per letter)."""
    blank = altlocs.get(' ', {})
    letters = sorted(a for a in altlocs if a != ' ')
    if not letters:
        return [blank]
    return [{**blank, **altlocs[a]} for a in letters]


def superpose(mobile, target):
    """Rotation r and translation t such that mobile @ r.T + t best fits target (Kabsch)."""
    mc, tc = mobile.mean(axis=0), target.mean(axis=0)
    u, _, vt = np.linalg.svd((mobile - mc).T @ (target - tc))
    d = np.sign(np.linalg.det(u @ vt))
    r = (u @ np.diag([1.0, 1.0, d]) @ vt).T
    return r, tc - mc @ r.T


def ca_coords(pdb):
    """{(chain, resi): CA coordinates} of a PDB file (blank or first altloc)."""
    out = {}
    for l in open(pdb):
        if l.startswith('ATOM') and l[12:16] == ' CA ' and l[16] in ' A':
            out.setdefault((l[21], int(l[22:26])), coords([l])[0])
    return out


def ca_superposition(ref_ca, ca):
    """(r, t, n_ca, rmsd) superposing a structure (ca: ca_coords) onto the reference (ref_ca)
    on all CA atoms they share, matched by chain and residue number; None if fewer than 3."""
    common = sorted(set(ca) & set(ref_ca))
    if len(common) < 3:
        return None
    mobile = np.array([ca[k] for k in common])
    target = np.array([ref_ca[k] for k in common])
    r, t = superpose(mobile, target)
    rmsd = float(np.sqrt(((mobile @ r.T + t - target) ** 2).sum(axis=1).mean()))
    return r, t, len(common), rmsd


def ligand_atoms(pdb):
    """{chain: (xyz, elements)} of the heavy atoms (all altlocs) of every LIG residue."""
    out = defaultdict(list)
    for l in open(pdb):
        if (l.startswith(('ATOM', 'HETATM')) and l[17:20].strip() == LIGAND_RESN
                and element_of(l) != 'H'):
            out[l[21]].append(l)
    return {ch: (coords(lines), [element_of(l) for l in lines]) for ch, lines in out.items()}


def ligand_centroids(pdb):
    """{chain: [centroid of each LIG residue]}: heavy atoms, blank or first altloc (as
    read_ligand)."""
    out = defaultdict(list)
    for l in open(pdb):
        if (l.startswith(('ATOM', 'HETATM')) and l[17:20].strip() == LIGAND_RESN
                and l[16] in ' A' and element_of(l) != 'H'):
            out[(l[21], int(l[22:26]))].append(l)
    centroids = defaultdict(list)
    for (chain, _), lines in out.items():
        centroids[chain].append(coords(lines).mean(axis=0))
    return centroids


def binding_site_resis(pdb, cutoff):
    """Residue numbers of the protein residues (ATOM records) with a heavy atom (any altloc)
    within cutoff of a heavy atom of any LIG residue, in the model's own frame."""
    lig, prot, resis = [], [], []
    for l in open(pdb):
        if not l.startswith(('ATOM', 'HETATM')) or element_of(l) == 'H':
            continue
        if l[17:20].strip() == LIGAND_RESN:
            lig.append(l)
        elif l.startswith('ATOM'):
            prot.append(l)
            resis.append(int(l[22:26]))
    if not lig or not prot:
        return set()
    d = cKDTree(coords(lig)).query(coords(prot), k=1, distance_upper_bound=cutoff)[0]
    return {r for r, x in zip(resis, d) if x <= cutoff}


def pocket_volume(xyz, atom_radius, probe_radius, spacing):
    """The volume a probe of probe_radius rolled over the union of spheres of atom_radius
    around xyz can't enter (a solvent-excluded volume; morphological closing of the spheres
    by the probe): the spheres, plus the crevices and gaps between them narrower than the
    probe; nothing beyond the spheres' outer surface. A point is outside it if it lies within
    probe_radius of a probe centre position, i.e. a point farther than atom_radius +
    probe_radius from every atom. Returns (origin, grid): a boolean grid indexed [x, y, z],
    point (i, j, k) at origin + spacing * (i, j, k); origin is a multiple of spacing."""
    # Padded so the grid's edge is always a probe position (outside every dilated sphere).
    pad = atom_radius + probe_radius + 2 * spacing
    origin = np.floor((xyz.min(axis=0) - pad) / spacing) * spacing
    shape = np.ceil((xyz.max(axis=0) + pad - origin) / spacing).astype(int) + 1
    axes = [origin[i] + spacing * np.arange(shape[i]) for i in range(3)]
    points = np.stack(np.meshgrid(*axes, indexing='ij'), axis=-1).reshape(-1, 3)
    # Where the probe centre can't go: within atom_radius + probe_radius of an atom.
    blocked = within(points, xyz, np.full(len(xyz), atom_radius + probe_radius)).reshape(shape)
    # Distance (Å) of each blocked point to the nearest probe position; inside the volume if
    # more than probe_radius from all of them.
    return origin, distance_transform_edt(blocked, sampling=spacing) > probe_radius


def sphere_through(points):
    """(centre, radius) of the smallest sphere with all of 1 to 4 points on its surface: the
    centre is in their affine span, equidistant from all of them."""
    a = points[0]
    if len(points) == 1:
        return a.copy(), 0.0
    u = points[1:] - a
    lam = np.linalg.lstsq(2 * u @ u.T, (u * u).sum(axis=1), rcond=None)[0]
    x = u.T @ lam
    return a + x, float(np.linalg.norm(x))


def min_enclosing_sphere(points, seed=0):
    """(centre, radius) of the smallest sphere containing every point: Welzl's randomized
    incremental algorithm on the points' convex hull vertices."""
    pts = np.unique(points, axis=0)
    if len(pts) > 4:
        try:
            pts = pts[ConvexHull(pts).vertices]
        except QhullError:  # flat or degenerate: use every point
            pass
    pts = pts[np.random.default_rng(seed).permutation(len(pts))]

    def outside(i, c, r):
        return np.linalg.norm(pts[i] - c) > r + 1e-7

    c, r = sphere_through(pts[:1])
    for i in range(1, len(pts)):
        if not outside(i, c, r):
            continue
        c, r = sphere_through(pts[[i]])
        for j in range(i):
            if not outside(j, c, r):
                continue
            c, r = sphere_through(pts[[i, j]])
            for k in range(j):
                if not outside(k, c, r):
                    continue
                c, r = sphere_through(pts[[i, j, k]])
                for m in range(k):
                    if outside(m, c, r):
                        c, r = sphere_through(pts[[i, j, k, m]])
    return c, r


def write_volume(path, origin, spacing, grid):
    """Writes a boolean grid [x, y, z] (point (i, j, k) at origin + spacing * (i, j, k)) as a
    0/1 CCP4 map, P1 with the grid's box as the cell."""
    shape = np.array(grid.shape)
    uc = UnitCell(*(shape * spacing), 90.0, 90.0, 90.0, space_group='P1')
    array = np.ascontiguousarray(np.transpose(grid, (2, 1, 0)), dtype=np.float64)
    offset = np.rint(origin / spacing).astype(int)
    XMap(array, GridParameters([spacing] * 3, offset=offset),
         unit_cell=uc).write_map_file(str(path))


def write_superposed(reference_pdb, model_pdbs, out_dir):
    """For viewing only: writes every model superposed onto the reference on all CA atoms they
    share (matched by chain and residue number) to out_dir/<name>.pdb (ATOM/HETATM records), plus
    out_dir/superposition.csv (CA count and RMSD per model)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ref = ca_coords(reference_pdb)
    rows = []
    for pdb in model_pdbs:
        fit = ca_superposition(ref, ca_coords(pdb))
        if fit is None:
            print(f'  Warning: {pdb} shares fewer than 3 CA atoms with the reference; not written')
            continue
        r, t, n_ca, rmsd = fit
        with open(out_dir / Path(pdb).name, 'w') as f:
            for l in open(pdb):
                if l.startswith(('ATOM', 'HETATM')):
                    x, y, z = coords([l])[0] @ r.T + t
                    f.write(f'{l[:30]}{x:8.3f}{y:8.3f}{z:8.3f}{l[54:]}')
                elif l.startswith(('TER', 'END')):
                    f.write(l)
        rows.append({'model': Path(pdb).stem, 'n_ca': n_ca, 'ca_rmsd_A': round(rmsd, 3)})
    with open(out_dir / 'superposition.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['model', 'n_ca', 'ca_rmsd_A'])
        writer.writeheader()
        writer.writerows(rows)
    return rows


def conformation_rmsd(a, b):
    """Heavy-atom RMSD of two conformations (names, xyz) over their shared atoms."""
    pos_b = {n: i for i, n in enumerate(b[0])}
    pairs = [(i, pos_b[n]) for i, n in enumerate(a[0]) if n in pos_b]
    if not pairs:
        return np.inf
    ia, ib = np.array(pairs).T
    return float(np.sqrt(((a[1][ia] - b[1][ib]) ** 2).sum(axis=1).mean()))


def cluster_conformations(confs, tolerance):
    """Greedy leader clustering: (representatives, members per representative)."""
    reps, counts = [], []
    for conf in confs:
        for k, rep in enumerate(reps):
            if conformation_rmsd(conf, rep) <= tolerance:
                counts[k] += 1
                break
        else:
            reps.append(conf)
            counts.append(1)
    return reps, counts


def ensemble_residues(reference, model_pdbs, exclude_resn, tolerance, resis):
    """Distinct side-chain conformations of every residue numbered in resis (any chain), from
    every model in model_pdbs. Each conformation (each altloc is one) is superposed onto the residue's backbone
    frame (N, CA, C of the reference structure, parsed: reference; or of the first model that
    has the residue) on its own N, CA and C, and only its side-chain atoms (everything but
    BACKBONE_ATOMS) are kept, so loop or backbone movement between models never counts as a
    conformation. Conformations missing side-chain atoms that at least half of the residue's
    conformations have are dropped (unless none is complete); the rest are de-duplicated within
    tolerance Å RMSD, the tolerance raised in 0.25 Å steps for any residue that would otherwise
    have more than MAX_CONFORMATIONS. Glycines (no side chain) are left out.

    Returns (residues, n_models). Each residue is a dict: key (chain, resi), resn,
    frame (3 x 3: N, CA, C), lines {atom name: PDB line}, confs [(side-chain atom names, xyz)]
    in the frame, most common first, counts (conformations each represents), tolerance (Å)."""
    frames = {}
    for key, (_, altlocs) in reference.items():
        conf = conformations(altlocs)[0]
        if all(n in conf for n in FRAME_ATOMS):
            frames[key] = coords([conf[n] for n in FRAME_ATOMS])
    gathered = {}
    n_models = 0
    for pdb in model_pdbs:
        model = read_protein(pdb, exclude_resn)
        n_models += 1
        for key, (resn, altlocs) in model.items():
            if key[1] not in resis:
                continue
            for conf in conformations(altlocs):
                side = [n for n in conf if n not in BACKBONE_ATOMS]
                if not side or not all(n in conf for n in FRAME_ATOMS):
                    continue
                own_frame = coords([conf[n] for n in FRAME_ATOMS])
                frame = frames.setdefault(key, own_frame)
                r, t = superpose(own_frame, frame)
                g = gathered.setdefault(key, {'resn': resn, 'lines': {}, 'confs': []})
                g['confs'].append((side, coords([conf[n] for n in side]) @ r.T + t))
                for n in conf:
                    g['lines'].setdefault(n, conf[n])

    residues = []
    for key, g in gathered.items():
        confs = g['confs']
        seen = defaultdict(int)
        for names, _ in confs:
            for n in names:
                seen[n] += 1
        core = {n for n, k in seen.items() if 2 * k >= len(confs)}
        confs = [c for c in confs if core <= set(c[0])] or confs
        tol = tolerance
        reps, counts = cluster_conformations(confs, tol)
        while len(reps) > MAX_CONFORMATIONS:
            tol += 0.25
            reps, counts = cluster_conformations(confs, tol)
        order = np.argsort(counts, kind='stable')[::-1]
        residues.append({'key': key, 'resn': g['resn'], 'frame': frames[key],
                         'lines': g['lines'], 'confs': [reps[k] for k in order],
                         'counts': [counts[k] for k in order], 'tolerance': tol})
    return residues, n_models


def own_side_chains(key, resn, confs, tolerance):
    """A residue's own side-chain conformations (confs: conformations()), de-duplicated within
    tolerance, as an ensemble residue (see ensemble_residues; frame None); None if it has no
    side-chain atoms."""
    side = []
    for conf in confs:
        names = [n for n in conf if n not in BACKBONE_ATOMS]
        if names:
            side.append((names, coords([conf[n] for n in names])))
    if not side:
        return None
    reps, counts = cluster_conformations(side, tolerance)
    return {'key': key, 'resn': resn, 'frame': None,
            'lines': {n: l for conf in confs for n, l in conf.items()},
            'confs': reps, 'counts': counts, 'tolerance': tolerance}


# --- Torsion library -------------------------------------------------------------------------

def torsion_library(smiles, ligand_lines, n, seed):
    """(n_confs, n_atoms, 3) heavy-atom coordinates of ETKDG conformers superposed on the modeled
    ligand, in the modeled ligand's atom order."""
    template = Chem.MolFromSmiles(smiles)
    ref = Chem.MolFromPDBBlock(''.join(ligand_lines), removeHs=True)
    ref = AllChem.AssignBondOrdersFromTemplate(template, ref)

    mol = Chem.AddHs(template)
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    cids = list(AllChem.EmbedMultipleConfs(mol, numConfs=n, params=params))
    if not cids:
        raise SystemExit(f'RDKit could not embed any conformers for {smiles}')
    mol = Chem.RemoveHs(mol)
    order = list(mol.GetSubstructMatch(ref))  # conformer atom for each modeled-ligand atom
    library, rmsds = [], []
    for cid in cids:
        rmsds.append(rdMolAlign.GetBestRMS(mol, ref, prbId=cid))  # superposes cid in place
        library.append(mol.GetConformer(cid).GetPositions()[order])
    return np.array(library), np.array(rmsds)


# --- Clash checking --------------------------------------------------------------------------

def grid_points(center, half_width, spacing):
    """(origin, shape, Cartesian points) of a cubic lookup grid around center."""
    n = int(np.ceil(2 * half_width / spacing)) + 1
    origin = center - half_width
    axes = [origin[i] + spacing * np.arange(n) for i in range(3)]
    points = np.stack(np.meshgrid(*axes, indexing='ij'), axis=-1).reshape(-1, 3)
    return origin, np.array([n, n, n]), points


def grid_lookup(grid, origin, spacing, xyz):
    """Grid value at the grid point nearest each position (False outside the grid)."""
    idx = np.rint((xyz - origin) / spacing).astype(np.int64)
    inside = np.all((idx >= 0) & (idx < grid.shape), axis=1)
    out = np.zeros(len(xyz), dtype=bool)
    out[inside] = grid[tuple(idx[inside].T)]
    return out


def pair_cutoffs(element, obs_elements, clash_scale, polar_scale):
    """Clash cutoff between a ligand atom of element and each obstacle atom."""
    obs_radius = np.array([vdwRadiiTable.get(e, DEFAULT_VDW) for e in obs_elements])
    polar = np.isin(obs_elements, POLAR_ELEMENTS) & (element in POLAR_ELEMENTS)
    return (np.where(polar, polar_scale, clash_scale)
            * (vdwRadiiTable.get(element, DEFAULT_VDW) + obs_radius))


class ClashChecker:
    """Clash tests for ligand atoms of given elements against the rigid obstacles."""

    def __init__(self, obs_xyz, obs_elements, lig_elements, clash_scale, polar_scale,
                 center, half_width, spacing):
        self.lig_elements = sorted(set(lig_elements))
        obs_radius = np.array([vdwRadiiTable.get(e, DEFAULT_VDW) for e in obs_elements])
        obs_polar = np.isin(obs_elements, POLAR_ELEMENTS)
        # Obstacles grouped by (vdW radius, polar): every atom in a group has the same cutoff to a
        # given ligand element.
        self.groups = []
        for radius, polar in sorted({(r, p) for r, p in zip(obs_radius, obs_polar)}):
            mask = (obs_radius == radius) & (obs_polar == polar)
            self.groups.append((radius, polar, cKDTree(obs_xyz[mask])))
        # cutoffs[element] = cutoff to each group, in group order.
        self.cutoffs = {}
        for e in self.lig_elements:
            r_e = vdwRadiiTable.get(e, DEFAULT_VDW)
            self.cutoffs[e] = np.array([
                (polar_scale if (polar and e in POLAR_ELEMENTS) else clash_scale) * (r_e + r_o)
                for r_o, polar, _ in self.groups])

        # Lookup grid of certain clashes: a grid point is marked for element e when it is within
        # (cutoff - half voxel diagonal) of an obstacle, so any position rounding to it clashes.
        self.spacing = spacing
        self.origin, self.shape, points = grid_points(center, half_width, spacing)
        self.pad = spacing * np.sqrt(3) / 2
        max_cut = max((c.max() for c in self.cutoffs.values() if len(c)), default=0.0)
        dist = [tree.query(points, k=1, distance_upper_bound=max_cut)[0]
                for _, _, tree in self.groups]
        self.grid = {}
        for e in self.lig_elements:
            hit = np.zeros(len(points), dtype=bool)
            for d, cut in zip(dist, self.cutoffs[e]):
                hit |= d < cut - self.pad
            self.grid[e] = hit.reshape(self.shape)

    def certain_clash(self, xyz, element):
        """True where a ligand atom of this element at xyz certainly clashes (grid lookup)."""
        return grid_lookup(self.grid[element], self.origin, self.spacing, xyz)

    def clash(self, xyz, element):
        """True where a ligand atom of this element at xyz clashes (exact)."""
        out = np.zeros(len(xyz), dtype=bool)
        for (_, _, tree), cut in zip(self.groups, self.cutoffs[element]):
            d = tree.query(xyz, k=1, distance_upper_bound=cut)[0]
            out |= d < cut
        return out


class VolumeChecker:
    """Test for ligand atoms outside a pocket volume (ClashEnsemble.volumes entry, a boolean grid
    in the reference frame); positions are first moved into that frame (x @ r.T + t). Element
    independent."""

    def __init__(self, volume, r, t):
        self.origin, self.spacing, self.grid = volume['origin'], volume['spacing'], volume['grid']
        self.r, self.t = r, t

    def outside(self, xyz, element=None):
        # einsum, not @: a (candidates x 3) @ (3 x 3) product would go to threaded BLAS and
        # spread over every core for no gain.
        ref = np.einsum('ij,kj->ik', xyz, self.r) + self.t
        return ~grid_lookup(self.grid, self.origin, self.spacing, ref)


class FlexibleChecker:
    """All-conformations clash test for the flexible residues (two or more conformations): a
    candidate is dropped if, for some residue, it clashes with every one of its conformations.
    The conformations a candidate clashes with are tracked as bits of MASK_WORDS uint64 words
    per residue."""

    def __init__(self, residues, lig_elements, clash_scale, polar_scale):
        xyz, elements, res, conf = [], [], [], []
        for i, r in enumerate(residues):
            for k, (names, cxyz) in enumerate(r['confs']):
                xyz.append(cxyz)
                elements += [element_of(r['lines'][n]) for n in names]
                res += [i] * len(names)
                conf += [k] * len(names)
        self.xyz = np.vstack(xyz)
        self.res = np.array(res)
        self.conf = np.array(conf)
        self.word = self.conf // 64
        self.bit = np.left_shift(np.uint64(1), (self.conf % 64).astype(np.uint64))
        self.n_confs = [len(r['confs']) for r in residues]
        # full[i, w]: word w of residue i with a bit set for each of its conformations.
        self.full = np.array([[((1 << n) - 1) >> (64 * w) & (2**64 - 1) for w in range(MASK_WORDS)]
                              for n in self.n_confs], dtype=np.uint64)
        self.tree = cKDTree(self.xyz)
        elements = np.array(elements)
        self.cutoffs = {e: pair_cutoffs(e, elements, clash_scale, polar_scale)
                        for e in sorted(set(lig_elements))}

    def add_certain(self, checker):
        """Marks in checker's grids the points where a ligand atom certainly clashes with every
        conformation of some flexible residue."""
        max_cut = max(c.max() for c in self.cutoffs.values())
        for i, n_confs in enumerate(self.n_confs):
            confs = [np.flatnonzero((self.res == i) & (self.conf == k))
                     for k in range(n_confs)]
            # Only points within reach of every conformation can clash with all of them.
            lo = np.max([self.xyz[a].min(axis=0) for a in confs], axis=0) - max_cut
            hi = np.min([self.xyz[a].max(axis=0) for a in confs], axis=0) + max_cut
            ilo = np.clip(np.floor((lo - checker.origin) / checker.spacing).astype(int), 0,
                          checker.shape)
            ihi = np.clip(np.ceil((hi - checker.origin) / checker.spacing).astype(int) + 1, 0,
                          checker.shape)
            if np.any(ihi <= ilo):
                continue
            block = tuple(slice(l, h) for l, h in zip(ilo, ihi))
            axes = [checker.origin[d] + checker.spacing * np.arange(ilo[d], ihi[d])
                    for d in range(3)]
            points = np.stack(np.meshgrid(*axes, indexing='ij'), axis=-1).reshape(-1, 3)
            hit = {e: np.ones(len(points), dtype=bool) for e in self.cutoffs}
            for atoms in confs:
                d = np.linalg.norm(points[:, None, :] - self.xyz[atoms][None, :, :], axis=2)
                for e, cut in self.cutoffs.items():
                    hit[e] &= (d < cut[atoms] - checker.pad).any(axis=1)
            for e in self.cutoffs:
                checker.grid[e][block] |= hit[e].reshape(ihi - ilo)

    def clashes_all(self, xyz, lig_elements):
        """True where a candidate (xyz: candidates x ligand atoms x 3) clashes with every
        conformation of some flexible residue (exact)."""
        masks = np.zeros((len(xyz), len(self.n_confs), MASK_WORDS), dtype=np.uint64)
        for e, cut in self.cutoffs.items():
            cols = [a for a, x in enumerate(lig_elements) if x == e]
            points = xyz[:, cols].reshape(-1, 3)
            pairs = cKDTree(points).sparse_distance_matrix(self.tree, cut.max(),
                                                           output_type='ndarray')
            hit = pairs['v'] < cut[pairs['j']]
            j = pairs['j'][hit]
            np.bitwise_or.at(masks, (pairs['i'][hit] // len(cols), self.res[j], self.word[j]),
                             self.bit[j])
        return (masks == self.full).all(axis=2).any(axis=1)


class ClashEnsemble:
    """Built once per run (build_clash_ensemble) from the ensemble models:
      binding_site: {residue number: models in which it is a binding-site residue} (chain
        agnostic; a heavy atom within binding_site_cutoff of a LIG residue);
      residues: every binding-site residue's distinct side-chain conformations, each held
        relative to the residue's backbone frame (N, CA, C); each generate_conformers run grafts
        them onto its own structure's backbone (placed());
      volumes: {chain: pocket volume}, the probe-closed atom-sphere volume (pocket_volume) of
        every model's LIG heavy atoms in that chain, each model superposed onto the reference on
        all CA atoms:
        dict of origin, spacing, grid (bool [x, y, z]) in the reference frame, n_ligand_atoms,
        and the sampling sphere: center (reference frame), observed_radius and max_radius, the
        minimum enclosing sphere of the models' ligand centroids in that chain, its radius plus
        centroid_margin; n_centroids;
      reference_ca: the reference's CA coordinates (to superpose a sampled structure onto it)."""

    def __init__(self, reference_pdb, model_pdbs, rotamer_tolerance, exclude_resn,
                 spacing=DEFAULT_VOLUME_SPACING,
                 binding_site_cutoff=DEFAULT_BINDING_SITE_CUTOFF,
                 pocket_atom_radius=DEFAULT_POCKET_ATOM_RADIUS,
                 pocket_probe_radius=DEFAULT_POCKET_PROBE_RADIUS,
                 centroid_margin=DEFAULT_CENTROID_MARGIN):
        start = time.time()
        self.reference_pdb = str(Path(reference_pdb).resolve())
        self.reference_ca = ca_coords(reference_pdb)
        self.binding_site = defaultdict(int)
        ligands = defaultdict(lambda: ([], []))
        centroids = defaultdict(list)
        n_aligned = 0
        for pdb in model_pdbs:
            for resi in binding_site_resis(pdb, binding_site_cutoff):
                self.binding_site[resi] += 1
            fit = ca_superposition(self.reference_ca, ca_coords(pdb))
            if fit is None:
                print(f'  Warning: {pdb} shares fewer than 3 CA atoms with the reference; its '
                      f'ligands are left out of the pocket volumes')
                continue
            r, t, _, _ = fit
            n_aligned += 1
            for chain, (xyz, elements) in ligand_atoms(pdb).items():
                ligands[chain][0].append(xyz @ r.T + t)
                ligands[chain][1].extend(elements)
            for chain, cents in ligand_centroids(pdb).items():
                centroids[chain] += [c @ r.T + t for c in cents]
        self.binding_site = dict(sorted(self.binding_site.items()))
        self.volumes = {}
        for chain, (xyz, _) in sorted(ligands.items()):
            xyz = np.vstack(xyz)
            # Atoms repeated across models (to 0.01 Å) add nothing.
            _, keep = np.unique(np.round(xyz, 2), axis=0, return_index=True)
            origin, grid = pocket_volume(xyz[keep], pocket_atom_radius, pocket_probe_radius,
                                         spacing)
            # Sampling sphere: the observed ligand centroids' minimum enclosing sphere, plus a
            # cushion.
            center, radius = min_enclosing_sphere(np.array(centroids[chain]))
            self.volumes[chain] = {'origin': origin, 'spacing': spacing, 'grid': grid,
                                   'n_ligand_atoms': len(xyz), 'center': center,
                                   'observed_radius': radius,
                                   'max_radius': radius + centroid_margin,
                                   'n_centroids': len(centroids[chain])}
        self.residues, n_models = ensemble_residues(read_protein(reference_pdb, exclude_resn),
                                                    model_pdbs, exclude_resn, rotamer_tolerance,
                                                    set(self.binding_site))
        self.info = {
            'reference': self.reference_pdb,
            'n_models': n_models,
            'binding_site_cutoff_A': binding_site_cutoff,
            'n_binding_site_residues': len(self.binding_site),
            'n_residues': len(self.residues),
            'n_conformations': sum(len(r['confs']) for r in self.residues),
            'rotamer_tolerance_A': rotamer_tolerance,
            'max_conformations': MAX_CONFORMATIONS,
            'exclude_resn': sorted(exclude_resn),
            'pocket_atom_radius_A': pocket_atom_radius,
            'pocket_probe_radius_A': pocket_probe_radius,
            'centroid_margin_A': centroid_margin,
            'volume_spacing_A': spacing,
            'n_models_superposed': n_aligned,
            'pocket_volumes_A3': {ch: round(float(v['grid'].sum()) * spacing ** 3, 1)
                                  for ch, v in self.volumes.items()},
            'pocket_spheres': {ch: {'center': np.round(v['center'], 3).tolist(),
                                    'max_radius_A': round(float(v['max_radius']), 3),
                                    'observed_radius_A': round(float(v['observed_radius']), 3),
                                    'n_ligands': v['n_centroids']}
                               for ch, v in self.volumes.items()},
            'build_time_s': round(time.time() - start, 1),
        }

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(path):
        with open(path, 'rb') as f:
            return pickle.load(f)

    def placed(self, target, center, cutoff):
        """Obstacles of a sampled structure (target, parsed), keeping atoms/residues within
        cutoff of center: (rigid lines, side-chain residues).
          rigid lines: the structure's own backbone (BACKBONE_ATOMS, all altlocs).
          side-chain residues: for binding-site residues, the ensemble's side-chain
            conformations, grafted onto the structure's backbone (residue frame superposed onto
            its N, CA, C); for every other residue (or one that can't be grafted), the
            structure's own side-chain conformations (altlocs)."""
        ensemble = {res['key']: res for res in self.residues}
        tolerance = self.info['rotamer_tolerance_A']
        rigid, residues = {}, []
        for key, (resn, altlocs) in target.items():
            own = conformations(altlocs)
            for conf in own:
                for n, line in conf.items():
                    if n in BACKBONE_ATOMS:
                        rigid[line] = None  # dict: blank-altloc atoms appear in every altloc
            res = ensemble.get(key)
            if res is not None and all(n in own[0] for n in FRAME_ATOMS):
                r, t = superpose(res['frame'], coords([own[0][n] for n in FRAME_ATOMS]))
                res = {**res, 'confs': [(names, xyz @ r.T + t) for names, xyz in res['confs']]}
            else:
                res = own_side_chains(key, resn, own, tolerance)
            if res is not None and min(np.linalg.norm(xyz - center, axis=1).min()
                                       for _, xyz in res['confs']) <= cutoff:
                residues.append(res)
        rigid = [l for l in rigid if np.linalg.norm(coords([l])[0] - center) <= cutoff]
        return rigid, residues


# --- Sampling --------------------------------------------------------------------------------

def draw_candidates(rng, n, n_lib, max_radius, radius_step, angle_step):
    """Random torsion index, translation (polar, azimuth, distance, vector) and rotation."""
    conf = rng.integers(n_lib, size=n)
    polar = np.degrees(np.arccos(rng.uniform(-1, 1, size=n)))
    azimuth = rng.uniform(0, 360, size=n)
    distance = max_radius * np.cbrt(rng.uniform(0, 1, size=n))
    polar = np.round(polar / angle_step) * angle_step
    azimuth = np.round(azimuth / angle_step) * angle_step % 360
    distance = np.round(distance / radius_step) * radius_step
    t, a = np.radians(polar), np.radians(azimuth)
    shift = distance[:, None] * np.stack(
        [np.sin(t) * np.cos(a), np.sin(t) * np.sin(a), np.cos(t)], axis=1)
    rotation = Rotation.random(n, rng=rng)
    return conf, polar, azimuth, distance, shift, rotation


def place_atom(library, lib_centroids, center, conf, rot_m, shift, atom):
    """Coordinates of one ligand atom for each candidate."""
    local = library[conf, atom] - lib_centroids[conf]
    return np.einsum('bij,bj->bi', rot_m, local) + center + shift


def place_all(library, lib_centroids, center, conf, rot_m, shift):
    """Coordinates of every ligand atom for each candidate (candidates x atoms x 3)."""
    local = library[conf] - lib_centroids[conf][:, None]
    return np.einsum('bij,baj->bai', rot_m, local) + center + shift[:, None]


def sample(library, lig_elements, checks, flexible, center, args, rng):
    """checks: per-atom tests (xyz, element) -> dropped, applied in order, atom by atom on the
    survivors; flexible: FlexibleChecker (or None) applied to the final survivors."""
    lib_centroids = library.mean(axis=1)
    # Core-first: atoms closest to the centroid fail most often, so check them first.
    order = np.argsort(np.linalg.norm(library[0] - lib_centroids[0], axis=1))
    kept = []
    n_kept, n_tried = 0, 0
    while n_kept < args.n_conformers and n_tried < args.max_candidates:
        acceptance = n_kept / n_tried if n_kept else 1e-4
        want = int(1.2 * (args.n_conformers - n_kept) / acceptance)
        batch = int(np.clip(want, 10_000, args.batch_size))
        batch = min(batch, args.max_candidates - n_tried)
        conf, polar, azimuth, distance, shift, rotation = draw_candidates(
            rng, batch, len(library), args.max_radius, args.radius_step, args.angle_step)
        rot_m = rotation.as_matrix()
        n_tried += batch

        # Grid passes then exact passes, atom by atom on the shrinking set of survivors, then
        # the all-conformations test of the flexible residues on whole candidates.
        alive = np.arange(batch)
        for check in checks:
            for atom in order:
                if not len(alive):
                    break
                xyz = place_atom(library, lib_centroids, center, conf[alive], rot_m[alive],
                                 shift[alive], atom)
                alive = alive[~check(xyz, lig_elements[atom])]
        if flexible is not None and len(alive):
            keep = np.zeros(len(alive), dtype=bool)
            for s in range(0, len(alive), FLEXIBLE_CHUNK):
                part = alive[s:s + FLEXIBLE_CHUNK]
                xyz = place_all(library, lib_centroids, center, conf[part], rot_m[part],
                                shift[part])
                keep[s:s + len(part)] = ~flexible.clashes_all(xyz, lig_elements)
            alive = alive[keep]

        # Keep only as many as still needed. Candidates after the last one kept were never
        # needed, so they don't count as tried (keeps the acceptance rate unbiased).
        needed = args.n_conformers - n_kept
        if len(alive) > needed:
            alive = alive[:needed]
            n_tried -= batch - (int(alive[-1]) + 1)
        if len(alive):
            kept.append((conf[alive], polar[alive], azimuth[alive], distance[alive],
                         shift[alive], rotation[alive]))
            n_kept += len(alive)
        print(f'  {n_tried:,} candidates tried, {n_kept:,} kept', flush=True)

    if not kept:
        return None, n_tried
    conf, polar, azimuth, distance, shift = (np.concatenate([k[i] for k in kept])
                                             for i in range(5))
    rotation = Rotation.concatenate([k[5] for k in kept])
    xyz = place_all(library, lib_centroids, center, conf, rotation.as_matrix(), shift)
    return dict(conf=conf, polar=polar, azimuth=azimuth, distance=distance,
                quat=rotation.as_quat(), xyz=xyz), n_tried


# --- Ensemble mask ---------------------------------------------------------------------------

def within(points, xyz, radii):
    """True where a point is within radii[i] of some atom i."""
    hit = np.zeros(len(points), dtype=bool)
    for r in np.unique(radii):
        d = cKDTree(xyz[radii == r]).query(points, k=1, distance_upper_bound=r)[0]
        hit |= np.isfinite(d)
    return hit


def grid_box(uc, shape, xyz, radius):
    """Grid indices (x, y, z order) of the box of the unit-cell grid holding every point within
    radius of xyz, and those points' Cartesian coordinates. The index padding is doubled so it
    also covers non-orthogonal cells."""
    spacing = np.array([uc.a, uc.b, uc.c]) / shape
    pad = int(np.ceil(2 * radius / spacing.min())) + 1
    g = (xyz @ uc.orth_to_frac.T) * shape
    lo = np.floor(g.min(axis=0)).astype(int) - pad
    hi = np.ceil(g.max(axis=0)).astype(int) + pad
    ijk = np.stack(np.meshgrid(*[np.arange(l, h + 1) for l, h in zip(lo, hi)], indexing='ij'),
                   axis=-1)
    points = (ijk.reshape(-1, 3) / shape) @ uc.frac_to_orth.T
    return lo, ijk.shape[:3], points


def ensemble_mask(xyz, radii, cell, spacing):
    """XMap (0/1) of the union of vdW spheres of every placement, xyz (n, atoms, 3), on the unit
    cell's grid at about spacing Å."""
    uc = UnitCell(*cell, space_group='P1')
    abc = np.array(cell[:3])
    shape = np.maximum(np.round(abc / spacing).astype(int), 1)
    atoms = xyz.reshape(-1, 3)
    atom_r = np.tile(radii, len(xyz))
    # Many placements share atom positions to within 0.01 Å; drop the repeats.
    _, keep = np.unique(np.column_stack([np.round(atoms, 2), atom_r]), axis=0,
                        return_index=True)
    atoms, atom_r = atoms[keep], atom_r[keep]
    lo, dims, points = grid_box(uc, shape, atoms, atom_r.max())
    occupied = within(points, atoms, atom_r).reshape(dims)  # [x, y, z]
    array = np.ascontiguousarray(np.transpose(occupied, (2, 1, 0)), dtype=np.float64)
    return XMap(array, GridParameters(abc / shape, offset=lo), unit_cell=uc)


def mask_fraction_inside(mask, xyz, radii):
    """Fraction of a pose's vdW volume (union of spheres of radii around xyz, sampled on the
    mask's grid) that lies inside the mask. Points outside the mask's box count as outside."""
    xyz, radii = np.asarray(xyz, float), np.asarray(radii, float)
    shape = mask.unit_cell_shape
    lo, dims, points = grid_box(mask.unit_cell, shape, xyz, radii.max())
    in_pose = within(points, xyz, radii)
    ijk = np.stack(np.meshgrid(*[np.arange(l, l + n) for l, n in zip(lo, dims)],
                               indexing='ij'), axis=-1).reshape(-1, 3)[in_pose]
    local = ijk - np.asarray(mask.offset)
    nz, ny, nx = mask.array.shape
    in_box = np.all((local >= 0) & (local < [nx, ny, nz]), axis=1)
    inside = np.zeros(len(local), dtype=bool)
    inside[in_box] = mask.array[local[in_box, 2], local[in_box, 1], local[in_box, 0]] > 0.5
    return float(inside.mean()) if len(inside) else 0.0


# --- Output ----------------------------------------------------------------------------------

def write_models(path, lines, models):
    with open(path, 'w') as f:
        for i, xyz in enumerate(models, start=1):
            f.write(f'MODEL     {i:4d}\n')
            for line, (x, y, z) in zip(lines, xyz):
                f.write(f'{line[:16]} {line[17:30]}{x:8.3f}{y:8.3f}{z:8.3f}{line[54:]}')
            f.write('ENDMDL\n')
        f.write('END\n')


def obstacle_lines(residues):
    """PDB lines of every residue's conformations: rigid residues as one conformation, flexible
    ones as altlocs (most common first, up to len(ALTLOCS)), occupancy = share of the residue's
    observed conformations each represents."""
    out = []
    for r in residues:
        flexible = len(r['confs']) > 1
        total = sum(r['counts'])
        for k, ((names, xyz), count) in enumerate(zip(r['confs'], r['counts'])):
            if k == len(ALTLOCS):
                break
            alt = ALTLOCS[k] if flexible else ' '
            for name, (x, y, z) in zip(names, xyz):
                line = r['lines'][name]
                out.append(f'{line[:16]}{alt}{line[17:30]}{x:8.3f}{y:8.3f}{z:8.3f}'
                           f'{count / total:6.2f}{line[60:]}')
    return out


def main():
    args = build_argparser().parse_args()
    args.max_candidates = args.max_candidates_per_conformer * args.n_conformers
    start = time.time()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.seed is None:
        args.seed = time.time_ns() % (2**31 - 1)  # fits RDKit's 32-bit int seed
    rng = np.random.default_rng(args.seed)
    print(f'Random seed: {args.seed}')

    cell, _, smiles = read_crystal_info(args.fragments_csv, args.dataset)
    resis = args.reference_resi or chain_ligands(args.structure_pdb, args.ligand_chain)
    if not resis:
        raise SystemExit(f'No {LIGAND_RESN} residues in chain {args.ligand_chain} of '
                         f'{args.structure_pdb}')
    args.reference_resi = resis
    references = [read_ligand(args.structure_pdb, args.ligand_chain, r) for r in resis]
    lig_lines = references[0]  # template: atom names/order of every output model
    lig_elements = [element_of(l) for l in lig_lines]

    # Clash ensemble, the chain's pocket volume and this structure's superposition onto the
    # reference (rot, shift: structure -> reference frame), which places the volume's sampling
    # sphere.
    if args.clash_ensemble:
        ensemble = ClashEnsemble.load(args.clash_ensemble)
    else:
        ensemble = ClashEnsemble(args.structure_pdb, [args.structure_pdb],
                                 DEFAULT_ROTAMER_TOLERANCE, args.exclude_resn)
    volume = ensemble.volumes.get(args.ligand_chain)
    if volume is None:
        raise SystemExit(f'The clash ensemble has no pocket volume for chain {args.ligand_chain} '
                         f'(no LIG residues in that chain in any ensemble model)')
    fit = ca_superposition(ensemble.reference_ca, ca_coords(args.structure_pdb))
    if fit is None:
        raise SystemExit(f'{args.structure_pdb} shares fewer than 3 CA atoms with the reference '
                         f'{ensemble.reference_pdb}')
    rot, shift, n_ca, ca_rmsd = fit
    center = (volume['center'] - shift) @ rot
    args.max_radius = float(volume['max_radius'])
    print(f'{args.dataset} chain {args.ligand_chain}: sampling centre {np.round(center, 3)} '
          f'(reference frame {np.round(volume["center"], 3)}), max radius '
          f'{args.max_radius:.2f} Å, SMILES {smiles}')
    for r, lines in zip(resis, references):
        c = coords(lines).mean(axis=0)
        print(f'  reference {LIGAND_RESN} {args.ligand_chain} {r}: {len(lines)} heavy atoms, '
              f'centroid {np.round(c, 3)}, {np.linalg.norm(c - center):.2f} Å from the centre')

    library, lib_rmsd = torsion_library(smiles, lig_lines, args.n_torsion_confs, args.seed)
    print(f'Torsion library: {len(library)} conformers, RMSD to modeled ligand '
          f'{lib_rmsd.min():.2f}-{lib_rmsd.max():.2f} Å (median {np.median(lib_rmsd):.2f})')

    # Region any placed ligand atom can reach, and obstacles that can touch it.
    extent = np.linalg.norm(library - library.mean(axis=1, keepdims=True), axis=2).max()
    reach = args.max_radius + extent
    max_cut = max(args.clash_scale, args.polar_clash_scale) * 2 * max(vdwRadiiTable.values())
    region = reach + max_cut + 1.0
    half_width = reach + args.grid_spacing

    # Obstacles: this structure's own backbone (rigid) plus the clash ensemble's side-chain
    # conformations grafted onto it.
    t1 = time.time()
    target = read_protein(args.structure_pdb, args.exclude_resn)
    rigid_lines, residues = ensemble.placed(target, center, region)
    rigid = [res for res in residues if len(res['confs']) == 1]
    flexible_residues = [res for res in residues if len(res['confs']) > 1]
    raised = {f'{res["resn"]}{res["key"][0]}{res["key"][1]}': res['tolerance']
              for res in residues if res['tolerance'] > ensemble.info['rotamer_tolerance_A']}
    print(f'Clash ensemble ({args.clash_ensemble or "this structure only"}): '
          f'{ensemble.info["n_models"]} models, {len(ensemble.binding_site)} binding-site residue '
          f'numbers; {len(rigid_lines)} backbone atoms of this structure within {region:.1f} Å; '
          f'side chains of {len(residues)} '
          f'residues: {len(rigid)} single-conformation, {len(flexible_residues)} flexible '
          f'({sum(len(res["confs"]) for res in flexible_residues)} conformations)'
          + (f'; tolerance raised for {raised}' if raised else ''))

    obs_xyz = np.vstack([coords(rigid_lines)] + [res['confs'][0][1] for res in rigid]) \
        if rigid_lines or rigid else np.zeros((0, 3))
    obs_elements = np.array([element_of(l) for l in rigid_lines]
                            + [element_of(res['lines'][n])
                               for res in rigid for n in res['confs'][0][0]])
    checker = ClashChecker(obs_xyz, obs_elements, lig_elements, args.clash_scale,
                           args.polar_clash_scale, center, half_width, args.grid_spacing)
    flexible = None
    if flexible_residues:
        flexible = FlexibleChecker(flexible_residues, lig_elements, args.clash_scale,
                                   args.polar_clash_scale)
        flexible.add_certain(checker)
    setup_s = time.time() - t1
    print(f'Clash grid: {checker.shape[0]}^3 points at {args.grid_spacing} Å '
          f'(ensemble placed and grids built in {setup_s:.1f} s)')

    # Pocket volume: candidate atoms are moved into the reference frame (rot, shift above).
    in_volume = VolumeChecker(volume, rot, shift)
    ref_atoms = np.vstack([coords(lines) for lines in references])
    ref_inside = 100 * float(1 - in_volume.outside(ref_atoms).mean())
    volume_summary = {'chain': args.ligand_chain,
                      'pocket_atom_radius_A': ensemble.info.get('pocket_atom_radius_A'),
                      'pocket_probe_radius_A': ensemble.info.get('pocket_probe_radius_A'),
                      # Clash ensembles built before the probe-closed volumes.
                      **({'pocket_margin_A': ensemble.info['pocket_margin_A']}
                         if 'pocket_margin_A' in ensemble.info else {}),
                      'spacing_A': volume['spacing'],
                      'volume_A3': round(float(volume['grid'].sum()) * volume['spacing'] ** 3, 1),
                      'n_ligand_atoms': volume['n_ligand_atoms'],
                      'superposition_n_ca': n_ca, 'superposition_ca_rmsd_A': round(ca_rmsd, 3),
                      'reference_atoms_inside_percent': round(ref_inside, 1)}
    print(f'Pocket volume (chain {args.ligand_chain}): {volume_summary["volume_A3"]:.0f} Å^3; '
          f'this structure superposed onto the reference on {n_ca} CA (RMSD {ca_rmsd:.2f} Å); '
          f'{ref_inside:.1f}% of the reference ligand atoms inside')

    checks = [in_volume.outside, checker.certain_clash, checker.clash]
    result, n_tried = sample(library, lig_elements, checks, flexible, center, args, rng)
    n_kept = 0 if result is None else len(result['conf'])

    with open(out_dir / 'ligand.pdb', 'w') as f:
        for i, lines in enumerate(references, start=1):
            f.write(f'MODEL     {i:4d}\n')
            f.writelines(f'{l[:16]} {l[17:]}' for l in lines)
            f.write('ENDMDL\n')
        f.write('END\n')
    write_models(out_dir / 'torsion_library.pdb', lig_lines, library)
    (out_dir / 'obstacles.pdb').write_text(
        ''.join(rigid_lines) + ''.join(obstacle_lines(residues)) + 'END\n')
    with open(out_dir / 'conformers.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['model', 'torsion_conf', 'trans_polar_deg', 'trans_azimuth_deg',
                         'trans_distance_A', 'quat_x', 'quat_y', 'quat_z', 'quat_w',
                         'centroid_x', 'centroid_y', 'centroid_z'])
        if result is not None:
            write_models(out_dir / 'conformers.pdb', lig_lines, result['xyz'])
            centroids = result['xyz'].mean(axis=1)
            for i in range(n_kept):
                writer.writerow([i + 1, int(result['conf'][i]), result['polar'][i],
                                 result['azimuth'][i], round(float(result['distance'][i]), 3),
                                 *np.round(result['quat'][i], 6), *np.round(centroids[i], 3)])

    reference_summary = []
    for r, lines in zip(resis, references):
        c = coords(lines).mean(axis=0)
        reference_summary.append({'resi': r, 'n_heavy_atoms': len(lines),
                                  'centroid': np.round(c, 3).tolist(),
                                  'centroid_to_center_A': round(float(np.linalg.norm(c - center)),
                                                                3),
                                  'percent_inside': None})
    mask_summary = None
    if result is not None:
        lig_radii = np.array([vdwRadiiTable.get(e, DEFAULT_VDW) for e in lig_elements])
        mask = ensemble_mask(result['xyz'], lig_radii, cell, args.mask_spacing)
        mask.write_map_file(str(out_dir / 'ensemble_mask.ccp4'))
        voxel_volume = abs(np.linalg.det(mask.unit_cell.frac_to_orth)) / np.prod(
            mask.unit_cell_shape)
        n_voxels = int((mask.array > 0.5).sum())
        mask_summary = {
            'file': 'ensemble_mask.ccp4',
            'spacing_A': np.round(mask.voxelspacing, 4).tolist(),
            'grid_shape_xyz': list(mask.array.shape[::-1]),
            'n_voxels': n_voxels,
            'volume_A3': round(n_voxels * voxel_volume, 1),
        }
        for ref, lines in zip(reference_summary, references):
            radii = np.array([vdwRadiiTable.get(element_of(l), DEFAULT_VDW) for l in lines])
            ref['percent_inside'] = round(
                100 * mask_fraction_inside(mask, coords(lines), radii), 1)
        print(f'Ensemble mask: {mask_summary["volume_A3"]:.0f} Å^3; reference ligands inside: '
              + ', '.join(f'{args.ligand_chain}{r["resi"]} {r["percent_inside"]}%'
                          for r in reference_summary))

    summary = {
        'dataset': args.dataset,
        'chain': args.ligand_chain,
        'sampling_center': np.round(center, 3).tolist(),
        'sampling_center_reference_frame': np.round(volume['center'], 3).tolist(),
        'max_radius_A': round(args.max_radius, 3),
        # Template ligand (first reference): atom names/order of every output model.
        'ligand': {'chain': args.ligand_chain, 'resi': resis[0], 'smiles': smiles,
                   'n_heavy_atoms': len(lig_lines)},
        'reference_ligands': reference_summary,
        'params': {k: v for k, v in vars(args).items()
                   if k not in ('structure_pdb', 'fragments_csv', 'output_dir')},
        'torsion_library': {'n_conformers': len(library),
                            'rmsd_to_modeled_A': {'min': float(lib_rmsd.min()),
                                                  'median': float(np.median(lib_rmsd)),
                                                  'max': float(lib_rmsd.max())}},
        'clash_ensemble': {'file': args.clash_ensemble, **ensemble.info,
                           'n_backbone_atoms_near': len(rigid_lines),
                           'n_side_chains_near': len(residues), 'n_rigid_side_chains': len(rigid),
                           'n_flexible_residues': len(flexible_residues),
                           'n_conformations_near': sum(len(res['confs']) for res in residues),
                           'tolerance_raised': raised, 'setup_s': round(setup_s, 1)},
        'binding_site': {'cutoff_A': ensemble.info.get('binding_site_cutoff_A'),
                         'resis': list(ensemble.binding_site)},
        'pocket_volume': volume_summary,
        'candidates_tried': n_tried,
        'conformers_kept': n_kept,
        'acceptance': n_kept / n_tried if n_tried else 0.0,
        'complete': n_kept >= args.n_conformers,
        'ensemble_mask': mask_summary,
        'runtime_s': round(time.time() - start, 1),
        'files': {'conformers': 'conformers.pdb' if n_kept else None,
                  'conformers_csv': 'conformers.csv', 'torsion_library': 'torsion_library.pdb',
                  'ligand': 'ligand.pdb', 'obstacles': 'obstacles.pdb',
                  'ensemble_mask': 'ensemble_mask.ccp4' if n_kept else None},
    }
    (out_dir / 'conformer_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(f'Kept {n_kept:,} of {n_tried:,} candidates ({100 * summary["acceptance"]:.3f}%) '
          f'in {summary["runtime_s"]} s -> {out_dir}')
    if n_kept < args.n_conformers:
        print(f'Warning: only {n_kept} of {args.n_conformers} conformers kept before reaching '
              f'the candidate limit ({args.max_candidates:,}: --max-candidates-per-conformer '
              f'{args.max_candidates_per_conformer:,} x {args.n_conformers}).')


if __name__ == '__main__':
    main()
