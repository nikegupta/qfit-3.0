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

# Sidechain-sidechain clash resolution, matching build_final_model.py's convention (same
# constants, same VDW-distance clash rule, same N/O hydrogen-bond exception) - see
# Rotamer_Optimizer._resolveSidechainClashes for why rotamer_optimize needs this too: each
# residue's optimized rotamer is picked independently against the untouched base structure, so
# two residues that were BOTH independently found to improve can still clash with each other.

# fraction of summed VDW radii below which two sidechain atoms are considered clashing
CLASH_VDW_SCALE = 0.75
# looser scale used instead of CLASH_VDW_SCALE when the clashing pair is one N atom and one O
# atom - a real N-H...O or O-H...N hydrogen bond legitimately sits closer than CLASH_VDW_SCALE
# would otherwise tolerate
HBOND_CLASH_VDW_SCALE = 0.6
BACKBONE_ATOM_NAMES = {'N', 'CA', 'C', 'O', 'OXT'}
# safety caps for _resolveSidechainClashes - see _resolveGroup's docstring
MAX_CLASH_GROUP_SIZE = 8  # residues; stop absorbing new neighbors past this
MAX_CLASH_GROUP_EXPANSIONS = 10  # rounds of "resolve, then absorb new external clashes"
CLASH_DOMAIN_TOP_K = 25  # candidates considered per residue during joint solving
CLASH_SOLVE_NODE_BUDGET = 200_000  # branch-and-bound search nodes before falling back to ICM

# Per-residue candidate pool built by Rotamer_Optimizer.run() and consumed by
# _resolveSidechainClashes. coor: (n_candidates, natoms, 3); rscc: (n_candidates,) - both
# indexed identically, index 0 always the true original (untouched) conformation. A residue
# that didn't pass the sampling threshold (base_rscc >= rscc_threshold) is fixed: exactly one
# candidate. A sampled residue carries its full last-chi-angle sampling pool (up to
# Rotamer_Optimizer.trim conformers, from _sample_sidechains' own top-K trim) as candidates
# 1..N, alongside its original as candidate 0 - not just the single best one - so clash
# resolution has real alternatives to pick from instead of only "swap or don't".
_RotamerCandidates = namedtuple(
    'RotamerCandidates', ['coor', 'rscc', 'template', 'sidechain_mask', 'fixed'],
)


class _NodeBudgetExceeded(Exception):
    """Raised internally by _branchAndBound to abort the search once its node budget is
    exhausted; caught by its caller to trigger the ICM fallback."""

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

        self.max_clash_group_size = MAX_CLASH_GROUP_SIZE

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
        self._resolveSidechainClashes(chosen_idx)

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

    # ---- sidechain-sidechain clash resolution -----------------------------
    #
    # Every residue's Pass 1 pick is decided independently, which can leave pairs of residues
    # whose sidechains clash with each other (a residue's top pick is only ever checked against
    # the FIXED, untouched base structure - never against another residue's own pick). The
    # methods below find those pairs, group them by connectivity, and jointly reselect each
    # group to the highest-total-RSCC combination that clashes with nothing - inside the group
    # or out - searching each movable member's FULL last-chi-angle candidate pool, not just its
    # top pick vs. original. Ported from build_final_model.py's own sidechain clash resolution:
    # same clash rule, same grouping/expansion strategy, and (since a pool here can hold ~10+
    # candidates, same as build_final_model.py's own residues) the same top-K domain truncation
    # plus branch-and-bound/ICM solver.

    def _residueReachSpheres(self):
        """Per residue, returns (centroids, reach): reach[key] is the distance from
        centroids[key] to the farthest sidechain atom across every gathered candidate of that
        residue - a conservative bounding sphere. Two residues can only possibly sidechain-clash
        if their reach-spheres overlap (plus a margin covering the VDW clash threshold), which
        lets the clash search skip an exact atom-pairwise check for residue pairs that are
        obviously too far apart. Residues with no sidechain atoms (e.g. glycine) get reach 0."""
        centroids = {}
        reach = {}
        for key, cand in self._candidates.items():
            if not cand.sidechain_mask.any():
                centroids[key] = cand.coor[0].mean(axis=0)
                reach[key] = 0.0
                continue
            pts = cand.coor[:, cand.sidechain_mask, :].reshape(-1, 3)
            centroid = pts.mean(axis=0)
            centroids[key] = centroid
            reach[key] = float(np.max(np.linalg.norm(pts - centroid, axis=1)))
        return centroids, reach

    def _candidatePairsWithinReach(self, keys_a, centroids, reach, keys_b=None, margin=3.0):
        """Yields (key1, key2) pairs - key1 from keys_a, key2 from keys_b (defaults to keys_a
        itself, in which case each unordered pair is yielded once) - whose reach-spheres come
        within `margin` of overlapping. `margin` just needs to conservatively cover the largest
        plausible VDW clash threshold - it does not need to be exact, since this is only a cheap
        prefilter and every pair it yields still gets an exact atom-pairwise check."""
        self_pairs = keys_b is None
        keys_b = keys_a if self_pairs else keys_b
        for i, k1 in enumerate(keys_a):
            others = keys_b[i + 1:] if self_pairs else keys_b
            for k2 in others:
                if k1 == k2:
                    continue
                d = np.linalg.norm(centroids[k1] - centroids[k2])
                if d <= reach[k1] + reach[k2] + margin:
                    yield k1, k2

    def _domainCompatibilityMatrix(self, key1, idx1, key2, idx2):
        """Returns an (len(idx1), len(idx2)) boolean matrix: True where candidate idx1[i] of
        residue key1 does NOT sidechain-clash with candidate idx2[j] of residue key2 (sidechain
        atoms only). The per-atom-pair threshold is CLASH_VDW_SCALE * summed VDW radii, EXCEPT
        for an (N, O) atom pair (either order), which uses HBOND_CLASH_VDW_SCALE instead - a
        real N-H...O or O-H...N hydrogen bond legitimately sits closer than a generic clash
        would tolerate. idx1/idx2 are arrays of candidate indices."""
        cand1, cand2 = self._candidates[key1], self._candidates[key2]
        mask1, mask2 = cand1.sidechain_mask, cand2.sidechain_mask
        if not mask1.any() or not mask2.any():
            return np.ones((len(idx1), len(idx2)), dtype=bool)

        coor1 = cand1.coor[idx1][:, mask1, :]  # (n1, a1, 3)
        coor2 = cand2.coor[idx2][:, mask2, :]  # (n2, a2, 3)
        vdw1 = np.asarray(cand1.template.vdw_radius)[mask1]  # (a1,)
        vdw2 = np.asarray(cand2.template.vdw_radius)[mask2]  # (a2,)
        e1 = np.asarray(cand1.template.e)[mask1]  # (a1,) element symbols
        e2 = np.asarray(cand2.template.e)[mask2]  # (a2,)

        vdw_sum = vdw1[:, None] + vdw2[None, :]  # (a1, a2)
        is_n_o_pair = (
            ((e1 == 'N')[:, None] & (e2 == 'O')[None, :])
            | ((e1 == 'O')[:, None] & (e2 == 'N')[None, :])
        )  # (a1, a2)
        scale = np.where(is_n_o_pair, HBOND_CLASH_VDW_SCALE, CLASH_VDW_SCALE)
        thresh = scale * vdw_sum  # (a1, a2)

        diff = coor1[:, None, :, None, :] - coor2[None, :, None, :, :]  # (n1,n2,a1,a2,3)
        dists = np.linalg.norm(diff, axis=-1)  # (n1,n2,a1,a2)
        clashing = np.any(dists < thresh[None, None, :, :], axis=(2, 3))  # (n1,n2)
        return ~clashing

    def _pairClashes(self, key1, idx1, key2, idx2):
        """Whether residue key1's candidate idx1 sidechain-clashes with residue key2's
        candidate idx2 (both single indices, not arrays)."""
        compat = self._domainCompatibilityMatrix(key1, np.array([idx1]), key2, np.array([idx2]))
        return not compat[0, 0]

    def _findClashingPairs(self, keys, chosen_idx, centroids, reach):
        """Among `keys`' CURRENT choices in chosen_idx, returns every pair that
        sidechain-clashes (after the reach-sphere prefilter)."""
        return [
            (k1, k2) for k1, k2 in self._candidatePairsWithinReach(keys, centroids, reach)
            if self._pairClashes(k1, chosen_idx[k1], k2, chosen_idx[k2])
        ]

    def _externalClashes(self, group, chosen_idx, centroids, reach):
        """Among `group`'s CURRENT choices in chosen_idx, returns every pair that
        sidechain-clashes with a residue outside the group."""
        others = [k for k in self._candidates if k not in group]
        return [
            (k1, k2) for k1, k2 in self._candidatePairsWithinReach(group, centroids, reach, keys_b=others)
            if self._pairClashes(k1, chosen_idx[k1], k2, chosen_idx[k2])
        ]

    def _connectedComponents(self, keys, pairs):
        """Groups `keys` into connected components of the graph formed by `pairs` (undirected
        edges). Keys with no edge at all are omitted."""
        adjacency = {k: set() for k in keys}
        for k1, k2 in pairs:
            adjacency[k1].add(k2)
            adjacency[k2].add(k1)

        seen = set()
        components = []
        for k in keys:
            if k in seen or not adjacency[k]:
                continue
            stack = [k]
            seen.add(k)
            comp = []
            while stack:
                cur = stack.pop()
                comp.append(cur)
                for nb in adjacency[cur]:
                    if nb not in seen:
                        seen.add(nb)
                        stack.append(nb)
            components.append(sorted(comp))
        return components

    def _format_group(self, keys):
        return ', '.join(f'{c}{r}' for c, r in keys)

    def _assignmentClashFree(self, group, assignment):
        """Whether `assignment` ({key: local_candidate_index}, one per member of `group`) has
        zero pairwise sidechain clash among every pair in `group`. `group` is always small
        (capped by max_clash_group_size), so this is a plain O(n^2) check."""
        return all(
            not self._pairClashes(k1, assignment[k1], k2, assignment[k2])
            for i, k1 in enumerate(group) for k2 in group[i + 1:]
        )

    def _domainsFor(self, group, top_k):
        """{key: candidate indices to consider}, cheapest-first (highest RSCC first, since this
        module maximizes rather than minimizes). top_k=None means every candidate (no
        truncation); fixed residues always get their single candidate regardless of top_k."""
        domains = {}
        for key in group:
            cand = self._candidates[key]
            if cand.fixed:
                domains[key] = np.array([0])
            else:
                order = np.argsort(-cand.rscc)
                domains[key] = order if top_k is None else order[:top_k]
        return domains

    def _solveGroupAssignmentOverDomains(self, group, domains, domain_label):
        """Solves the joint RSCC-maximization problem (see _solveGroupAssignment) over exactly
        the given `domains` - no truncation or widening here. Returns ({key:
        global_candidate_index}, resolved) where resolved is False if no combination within
        these domains eliminates every pairwise clash."""
        compat = {}
        for i, key1 in enumerate(group):
            for key2 in group[i + 1:]:
                compat[(key1, key2)] = self._domainCompatibilityMatrix(
                    key1, domains[key1], key2, domains[key2]
                )

        result = self._branchAndBound(group, domains, compat)
        if result is None:
            print(f'  rotamer clash group [{self._format_group(group)}]: exact search over the '
                  f'{domain_label} domain found no fully compatible combination (or exhausted its '
                  f'node budget); falling back to a heuristic (ICM) reassignment.')
            result = self._icmAssignment(group, domains, compat)

        assignment = {key: int(domains[key][local_i]) for key, local_i in result.items()}
        return assignment, self._assignmentClashFree(group, assignment)

    def _solveGroupAssignment(self, group):
        """Returns ({key: chosen_candidate_index}, resolved) for one clash group: the
        combination of candidates (one per residue, searched over each movable member's FULL
        last-chi-angle candidate pool plus its original - not just its top pick vs. original) that
        maximizes total RSCC subject to no pairwise sidechain clash within the group - and
        whether that goal was actually achieved.

        Efficiency: each residue's domain is first truncated to its CLASH_DOMAIN_TOP_K
        highest-RSCC candidates (a low-scoring candidate essentially never wins even when it's
        compatible), then solved exactly via _branchAndBound (DFS, most-constrained-residue-
        first, pruned by an admissible cost bound); if that exceeds its node budget, or finds no
        fully-compatible combination within the truncated domains, _icmAssignment (a fast,
        always-terminating local-search heuristic) is used instead. If even that doesn't find a
        fully compatible combination, this retries ONCE with each residue's FULL (untruncated)
        domain before conceding.

        `resolved` is only False if even the full-domain retry couldn't eliminate every internal
        clash - this shouldn't normally happen, since base_structure (final_model.pdb) was
        already clash-resolved by build_final_model.py, so "every movable residue at its original
        candidate (index 0)" is itself always a valid, clash-free combination as long as that
        resolution fully covered this residue set."""
        top_k_domains = self._domainsFor(group, CLASH_DOMAIN_TOP_K)
        assignment, resolved = self._solveGroupAssignmentOverDomains(
            group, top_k_domains, f'top-{CLASH_DOMAIN_TOP_K}'
        )

        if not resolved:
            full_domains = self._domainsFor(group, top_k=None)
            assignment, resolved = self._solveGroupAssignmentOverDomains(group, full_domains, 'full')

        if not resolved:
            movable = [k for k in group if not self._candidates[k].fixed]
            fallback = {k: 0 for k in group if self._candidates[k].fixed}
            fallback.update({k: 0 for k in movable})
            return fallback, False

        return assignment, True

    def _branchAndBound(self, group, domains, compat, node_budget=CLASH_SOLVE_NODE_BUDGET):
        """Exact DFS branch-and-bound over `domains` (local candidate indices per residue),
        maximizing total RSCC subject to `compat` (pairwise domain-compatibility matrices - see
        _domainCompatibilityMatrix - keyed by (key1, key2) in `group` order). Residues are
        visited most-constrained-first (smallest domain first); within a residue, candidates are
        tried highest-RSCC-first, and a branch is pruned once its partial score plus the best
        possible completion (each remaining residue's own maximum candidate RSCC - an admissible
        upper bound, since it ignores compatibility) can no longer beat the best solution found
        so far.

        Returns {key: local_domain_index} for the optimal assignment, or None if the node budget
        was exhausted before one fully-compatible assignment was found (including the case where
        none exists at all within these domains)."""
        order = sorted(group, key=lambda k: len(domains[k]))
        scores = [self._candidates[key].rscc[domains[key]] for key in order]
        best_first = [np.argsort(-s) for s in scores]

        n = len(order)
        suffix_max = [0.0] * (n + 1)
        for k in range(n - 1, -1, -1):
            suffix_max[k] = suffix_max[k + 1] + float(scores[k].max())

        def get_matrix(k1, k2):
            key1, key2 = order[k1], order[k2]
            if (key1, key2) in compat:
                return compat[(key1, key2)], False
            return compat[(key2, key1)], True

        current = [None] * n
        best = {'assignment': None, 'score': -float('inf')}
        nodes = {'count': 0}

        def compat_ok(k, local_i):
            for prev in range(k):
                m, swapped = get_matrix(prev, k)
                i, j = (current[prev], local_i) if not swapped else (local_i, current[prev])
                if not m[i, j]:
                    return False
            return True

        def dfs(k, score_so_far):
            if score_so_far + suffix_max[k] <= best['score']:
                return
            nodes['count'] += 1
            if nodes['count'] > node_budget:
                raise _NodeBudgetExceeded()
            if k == n:
                best['assignment'] = list(current)
                best['score'] = score_so_far
                return
            for local_i in best_first[k]:
                if not compat_ok(k, local_i):
                    continue
                current[k] = local_i
                dfs(k + 1, score_so_far + float(scores[k][local_i]))
            current[k] = None

        try:
            dfs(0, 0.0)
        except _NodeBudgetExceeded:
            return None

        if best['assignment'] is None:
            return None
        return {key: idx for key, idx in zip(order, best['assignment'])}

    def _icmAssignment(self, group, domains, compat, max_iters=25):
        """Iterated Conditional Modes: a fast, always-terminating heuristic for the same joint
        RSCC-maximization problem _branchAndBound solves exactly. Starting every residue at its
        own highest-RSCC candidate, repeatedly revisits each residue in `group` in turn and
        reassigns it to its highest-RSCC candidate that's compatible with every OTHER residue's
        CURRENT pick, until a full pass changes nothing (or max_iters is hit). May still leave
        residual clashes if even the (already top-K-truncated) domains contain no fully mutually
        compatible combination at all.

        Returns {key: local_domain_index}."""
        def get_matrix(key_a, key_b):
            if (key_a, key_b) in compat:
                return compat[(key_a, key_b)], False
            return compat[(key_b, key_a)], True

        current = {key: 0 for key in group}

        for _ in range(max_iters):
            changed = False
            for key in group:
                scores = self._candidates[key].rscc[domains[key]]
                for local_i in np.argsort(-scores):
                    ok = True
                    for other in group:
                        if other == key:
                            continue
                        m, swapped = get_matrix(key, other)
                        i, j = (local_i, current[other]) if not swapped else (current[other], local_i)
                        if not m[i, j]:
                            ok = False
                            break
                    if ok:
                        if local_i != current[key]:
                            current[key] = int(local_i)
                            changed = True
                        break
            if not changed:
                break

        return current

    def _resolveGroup(self, group, chosen_idx, centroids, reach):
        """Jointly reselects one clash group to the highest-total-RSCC combination of
        candidates with no sidechain clash inside the group (see _solveGroupAssignment) -
        mutating chosen_idx in place for every member. If the new picks clash with a residue
        outside the group, that residue is absorbed into the group and the whole group is
        resolved again, repeating until stable (same expansion strategy as
        build_final_model.py's _resolveGroup)."""
        group = list(group)
        original_group = list(group)
        hit_cap = False
        unresolved = False

        for _round in range(MAX_CLASH_GROUP_EXPANSIONS):
            assignment, resolved = self._solveGroupAssignment(group)
            for key, idx in assignment.items():
                chosen_idx[key] = idx

            if not resolved:
                unresolved = True
                print(f'WARNING: rotamer clash group [{self._format_group(group)}] could NOT be '
                      f'fully resolved - no combination from each member\'s sampled candidate '
                      f'pool eliminates every clash within the group; reverting every movable '
                      f'member of this group to its original conformation.')
                break

            external = self._externalClashes(group, chosen_idx, centroids, reach)
            new_members = sorted({k2 for (_, k2) in external if k2 not in group})
            if not new_members:
                break

            if len(group) + len(new_members) > self.max_clash_group_size:
                hit_cap = True
                print(f'WARNING: rotamer clash group [{self._format_group(group)}] would grow '
                      f'past max_clash_group_size={self.max_clash_group_size} residues after '
                      f'absorbing [{self._format_group(new_members)}]; stopping expansion here. '
                      f'Every movable member of this group is reverted to its original '
                      f'conformation rather than risk leaving a residual clash unresolved.')
                for key in group:
                    if not self._candidates[key].fixed:
                        chosen_idx[key] = 0
                break

            group.extend(new_members)
        else:
            hit_cap = True
            print(f'WARNING: rotamer clash group [{self._format_group(group)}] kept absorbing '
                  f'new neighbors past {MAX_CLASH_GROUP_EXPANSIONS} round(s); reverting every '
                  f'movable member of this group to its original conformation.')
            for key in group:
                if not self._candidates[key].fixed:
                    chosen_idx[key] = 0

        flagged = hit_cap or unresolved
        verb = 'left with a residual clash (reverted to original)' if flagged else 'resolved'
        print(f'rotamer clash group [{self._format_group(group)}] ({len(group)} residue(s), '
              f'{len(original_group)} originally clashing) {verb}.')

        return {'residues': group, 'original_residues': original_group,
                'hit_cap': hit_cap, 'unresolved': unresolved}

    def _resolveSidechainClashes(self, chosen_idx):
        """Finds every sidechain-sidechain clash among the current candidate picks in
        chosen_idx (every residue starts at its Pass 1 decision: its top pick if that
        independently cleared the 0.1 threshold, else its original), groups clashing residues by
        connectivity, and resolves each group via _resolveGroup - mutating chosen_idx in place.
        Returns a list of per-group summary rows."""
        centroids, reach = self._residueReachSpheres()
        keys = list(self._candidates.keys())

        initial_pairs = self._findClashingPairs(keys, chosen_idx, centroids, reach)
        groups = self._connectedComponents(keys, initial_pairs)

        if groups:
            print(f'{len(groups)} sidechain-sidechain clash group(s) found among independently '
                  f'optimized rotamer picks; resolving each jointly.')

        group_rows = []
        settled = set()
        for group in groups:
            if settled.issuperset(group):
                continue
            row = self._resolveGroup(group, chosen_idx, centroids, reach)
            settled.update(row['residues'])
            group_rows.append(row)
        return group_rows

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
    ro = Rotamer_Optimizer(args.dataset, args.model_file, args.output_folder, args.resolution)
    ro.run()

if __name__ == '__main__':
    main()
