"""
generate_conformers - samples a null ensemble of ligand placements in one chain's binding pocket.

Each sample is a random torsional conformer of the dataset's ligand, translated and rotated at
random within the chain's pocket volume's minimum enclosing sphere, and kept only if it lies
inside the pocket volume and doesn't clash with the structure's bound state. Samples are drawn
until --n-conformers are kept (or --max-candidates-per-conformer x --n-conformers have been
tried).
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
         distance to --radius-step Å. The centre and max radius come from the pocket ensemble:
         the minimum enclosing sphere of the centroids of every ensemble model's LIG residues in
         this chain (models superposed onto the reference on all CA; centre moved into this
         structure's frame), its radius plus a cushion (default 2 Å). This is a cap, not a
         bound: a pose whose centroid is farther out isn't sampled even if it fits the pocket
         volume (step 3);
       - an orientation: a uniformly random rotation (all three rotational degrees of freedom)
         about the conformer's centroid.
  3. Pocket volume: every heavy atom of a candidate must lie inside ligand_chain's pocket volume
     from the pocket ensemble (--clash-ensemble): every ensemble model's LIG heavy atoms in that
     chain, each
     made a sphere of one fixed radius (default 2 Å, the same for every element - a cushion
     around the observed atom positions, not a physical size), with a probe (default 2 Å) rolled
     over the union of those spheres: the volume the probe can't enter, i.e. the spheres with
     the crevices and gaps between them narrower than the probe filled in, and no growth
     outward; on a 0.25 Å grid in the reference structure's frame.
     Candidate atoms are moved into that frame by superposing this structure onto the reference
     on all CA atoms they share.
  4. Clash check against this structure: every heavy atom of structure_pdb is an obstacle
     except LIG (every altloc), hydrogens, --exclude-resn residues and atoms whose altloc is in
     --ignore-altlocs (default A). For a multiconformer model with the apo state as altloc A
     and the bound state as B, a candidate must fit the bound state (blank-altloc and B atoms)
     and may overlap the apo state. A candidate is dropped if any of its atoms clashes with any
     obstacle: a ligand-obstacle atom pair clashes when closer than --clash-scale x (sum of vdW
     radii), or --polar-clash-scale x for N-N, N-O and O-O pairs. No symmetry mates. Without
     --clash-ensemble the structure alone gives the pocket volume and sampling sphere.

Speed: candidates are generated and checked in numpy batches. The pocket volume is a grid
lookup, done first. Lookup grids around the pocket mark, per ligand element, the points where an
atom certainly clashes with an obstacle (cutoffs shrunk by half a voxel diagonal, so a grid hit
is always real). Ligand atoms are checked core-first and a candidate is dropped at its first
certain hit; survivors get exact KD-tree checks, so the result is exact. It is meant to
run single-threaded (parallelism comes from running pockets side by side): the hot loop avoids
BLAS, and when calling it directly set OPENBLAS_NUM_THREADS=1 (and OMP_/MKL_NUM_THREADS=1), as
generate_null.sh does, so numpy's threaded OpenBLAS doesn't spread other products over every core.

Outputs (in output_dir):
  conformers.pdb            the kept placements, one MODEL each (heavy atoms)
  conformers.csv            per kept placement: torsion conformer, translation polar/azimuth/
                            distance, rotation quaternion (x, y, z, w), centroid
  torsion_library.pdb       the superposed torsion conformers, one MODEL each
  ligand.pdb                the reference ligands (heavy atoms, first altloc), one MODEL each
  obstacles.pdb             the obstacle atoms near the pocket, as they are in structure_pdb
  ensemble_mask.ccp4        1 inside the vdW volume of any kept placement, 0 elsewhere (see below)
  conformer_summary.json    parameters, sampling centre, obstacles (count, altlocs ignored),
                            pocket ensemble, binding site, pocket volume (superposition
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
# Default --max-candidates-per-conformer: candidates tried, at most, per conformer asked for
# (--n-conformers).
MAX_CANDIDATES_PER_CONFORMER = 1_000_000
# Altlocs that are not obstacles by default: the apo state of the multiconformer models.
DEFAULT_IGNORE_ALTLOCS = ['A']
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
                   help='Pocket ensemble file from build_clash_ensemble (built once per run); it '
                        'provides the pocket volumes and sampling spheres. Default: '
                        'structure_pdb alone (its own ligands).')
    p.add_argument('--exclude-resn', nargs='+', default=['HOH', 'DMS'],
                   help='Residue names left out of the obstacles (hydrogens and every LIG '
                        'residue are always left out). Default: HOH DMS.')
    p.add_argument('--ignore-altlocs', nargs='*', default=DEFAULT_IGNORE_ALTLOCS,
                   help='Altloc letters whose atoms are not obstacles (the apo state of a '
                        'multiconformer model); give none to make every altloc an obstacle. '
                        'Default: A.')
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


def first_altloc(lines):
    """lines of the blank altloc plus the first altloc letter present (alphabetical), e.g. the B
    atoms of a ligand modeled only as altloc B."""
    letters = sorted({l[16] for l in lines} - {' '})
    return [l for l in lines if l[16] == ' ' or (letters and l[16] == letters[0])]


def read_ligand(pdb, chain, resi):
    """Heavy-atom lines of the modeled ligand, first alternate conformer only (blank altloc plus
    the first altloc letter present), with the altloc column blanked: RDKit's PDB reader keeps
    only blank or A altlocs, so a ligand modeled as altloc B would otherwise read as no atoms."""
    lines = first_altloc([l for l in open(pdb) if l.startswith(('ATOM', 'HETATM'))
                          and l[17:20].strip() == LIGAND_RESN and l[21] == chain
                          and int(l[22:26]) == resi and element_of(l) != 'H'])
    if not lines:
        raise SystemExit(f'No {LIGAND_RESN} {chain} {resi} heavy atoms in {pdb}')
    return [f'{l[:16]} {l[17:]}' for l in lines]


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
    """{chain: [centroid of each LIG residue]}: heavy atoms, blank plus first altloc (as
    read_ligand)."""
    out = defaultdict(list)
    for l in open(pdb):
        if (l.startswith(('ATOM', 'HETATM')) and l[17:20].strip() == LIGAND_RESN
                and element_of(l) != 'H'):
            out[(l[21], int(l[22:26]))].append(l)
    centroids = defaultdict(list)
    for (chain, _), lines in out.items():
        centroids[chain].append(coords(first_altloc(lines)).mean(axis=0))
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


class ClashEnsemble:
    """The pocket ensemble, built once per run (build_clash_ensemble) from the ensemble models
    (the class keeps its name so the pkl files keep unpickling):
      binding_site: {residue number: models in which it is a binding-site residue} (chain
        agnostic; a heavy atom within binding_site_cutoff of a LIG residue; informational);
      volumes: {chain: pocket volume}, the probe-closed atom-sphere volume (pocket_volume) of
        every model's LIG heavy atoms in that chain, each model superposed onto the reference on
        all CA atoms:
        dict of origin, spacing, grid (bool [x, y, z]) in the reference frame, n_ligand_atoms,
        and the sampling sphere: center (reference frame), observed_radius and max_radius, the
        minimum enclosing sphere of the models' ligand centroids in that chain, its radius plus
        centroid_margin; n_centroids;
      reference_ca: the reference's CA coordinates (to superpose a sampled structure onto it)."""

    def __init__(self, reference_pdb, model_pdbs, spacing=DEFAULT_VOLUME_SPACING,
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
        self.info = {
            'reference': self.reference_pdb,
            'n_models': len(model_pdbs),
            'binding_site_cutoff_A': binding_site_cutoff,
            'n_binding_site_residues': len(self.binding_site),
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


def sample(library, lig_elements, checks, center, args, rng):
    """checks: per-atom tests (xyz, element) -> dropped, applied in order, atom by atom on the
    survivors."""
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

        # Grid passes then exact passes, atom by atom on the shrinking set of survivors.
        alive = np.arange(batch)
        for check in checks:
            for atom in order:
                if not len(alive):
                    break
                xyz = place_atom(library, lib_centroids, center, conf[alive], rot_m[alive],
                                 shift[alive], atom)
                alive = alive[~check(xyz, lig_elements[atom])]

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


def obstacle_lines(pdb, exclude_resn, ignore_altlocs, center, cutoff):
    """Heavy-atom lines of pdb that are clash obstacles - not LIG, exclude_resn, hydrogens or an
    altloc in ignore_altlocs - within cutoff of center."""
    skip = set(exclude_resn) | {LIGAND_RESN}
    lines = [l for l in open(pdb) if l.startswith(('ATOM', 'HETATM'))
             and l[17:20].strip() not in skip and element_of(l) != 'H'
             and l[16] not in ignore_altlocs]
    if not lines:
        return []
    near = np.linalg.norm(coords(lines) - center, axis=1) <= cutoff
    return [l for l, n in zip(lines, near) if n]


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
        ensemble = ClashEnsemble(args.structure_pdb, [args.structure_pdb])
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

    # Obstacles: this structure's atoms but LIG, excluded residues, hydrogens and the ignored
    # altlocs (the apo state).
    t1 = time.time()
    ignore = set(args.ignore_altlocs or [])
    obs_lines = obstacle_lines(args.structure_pdb, args.exclude_resn, ignore, center, region)
    n_ignored = len(obstacle_lines(args.structure_pdb, args.exclude_resn, set(), center,
                                   region)) - len(obs_lines)
    obs_xyz = coords(obs_lines) if obs_lines else np.zeros((0, 3))
    obs_elements = np.array([element_of(l) for l in obs_lines])
    checker = ClashChecker(obs_xyz, obs_elements, lig_elements, args.clash_scale,
                           args.polar_clash_scale, center, half_width, args.grid_spacing)
    setup_s = time.time() - t1
    print(f'Obstacles: {len(obs_lines)} atoms of this structure within {region:.1f} Å of the '
          f'centre (altlocs {sorted(ignore) or "none"} ignored: {n_ignored} atoms); clash grid '
          f'{checker.shape[0]}^3 points at {args.grid_spacing} Å, built in {setup_s:.1f} s')

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
    result, n_tried = sample(library, lig_elements, checks, center, args, rng)
    n_kept = 0 if result is None else len(result['conf'])

    with open(out_dir / 'ligand.pdb', 'w') as f:
        for i, lines in enumerate(references, start=1):
            f.write(f'MODEL     {i:4d}\n')
            f.writelines(f'{l[:16]} {l[17:]}' for l in lines)
            f.write('ENDMDL\n')
        f.write('END\n')
    write_models(out_dir / 'torsion_library.pdb', lig_lines, library)
    (out_dir / 'obstacles.pdb').write_text(''.join(obs_lines) + 'END\n')
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
        'clash_ensemble': {'file': args.clash_ensemble, **ensemble.info},
        'obstacles': {'n_atoms': len(obs_lines), 'ignore_altlocs': sorted(ignore),
                      'n_ignored_atoms': n_ignored, 'within_A': round(region, 2),
                      'setup_s': round(setup_s, 1)},
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
