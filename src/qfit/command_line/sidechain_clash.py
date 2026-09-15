"""
Shared sidechain-sidechain clash detection + joint-reselection engine, used identically by
build_final_model.py (Stage 6) and rotamer_optimize.py (Stage 7): each residue's best-scoring
candidate conformation is picked independently, so two residues that were BOTH independently
found to score well can still clash with each other. This resolves each connected group of
clashing residues to the best-total-score combination with no internal clash, expanding the
group to absorb any newly-introduced external clash, up to configurable safety caps.

Operates on a generic per-residue candidate pool so both callers share one implementation of the
actual clash math and joint-reselection solver, rather than maintaining two independently-drifting
copies of a ~300-line branch-and-bound algorithm. `candidates` is a {key: candidate_pool} dict,
one entry per residue (key is any hashable, e.g. (chain_id, resi)). Each candidate_pool needs:
  - `.coor`: (n_candidates, natoms, 3)
  - `.template`: an object with `.vdw_radius`/`.e` arrays, atom order matching `.coor`'s atom axis
  - `.sidechain_mask`: (natoms,) bool
  - `.fixed`: bool - True means this residue has exactly one, always-kept candidate
`cost_of(candidate_pool)` must return a (n_candidates,) array where LOWER IS ALWAYS BETTER
(matching build_final_model's own MSE convention) - rotamer_optimize passes -RSCC, since "lower
cost" and "higher RSCC" mean the same thing here; the resolver itself never needs to know which
metric it actually is.
"""
import numpy as np

CLASH_VDW_SCALE = 0.75
HBOND_CLASH_VDW_SCALE = 0.6
MAX_CLASH_GROUP_SIZE = 8
MAX_CLASH_GROUP_EXPANSIONS = 10
CLASH_DOMAIN_TOP_K = 25
CLASH_SOLVE_NODE_BUDGET = 200_000


class NodeBudgetExceeded(Exception):
    """Raised internally by branch_and_bound to abort the search once its node budget is
    exhausted; caught by its caller to trigger the ICM fallback."""


class SidechainClashResolver:
    def __init__(self, candidates, cost_of,
                 clash_vdw_scale=CLASH_VDW_SCALE,
                 hbond_clash_vdw_scale=HBOND_CLASH_VDW_SCALE,
                 max_clash_group_size=MAX_CLASH_GROUP_SIZE,
                 max_clash_group_expansions=MAX_CLASH_GROUP_EXPANSIONS,
                 clash_domain_top_k=CLASH_DOMAIN_TOP_K,
                 clash_solve_node_budget=CLASH_SOLVE_NODE_BUDGET,
                 revert_index=None, group_label='clash group'):
        """revert_index: if not None, every movable member of a group that couldn't be fully
        resolved is reset to this candidate index instead of keeping whatever the solver's best
        (still-clashing) effort was - e.g. rotamer_optimize's index 0 is documented to always be
        the residue's untouched original conformation, so reverting there is always safe;
        build_final_model has no such universally-safe index and leaves this None, keeping the
        solver's best effort instead."""
        self._candidates = candidates
        self._cost = {key: np.asarray(cost_of(cand)) for key, cand in candidates.items()}
        self.clash_vdw_scale = clash_vdw_scale
        self.hbond_clash_vdw_scale = hbond_clash_vdw_scale
        self.max_clash_group_size = max_clash_group_size
        self.max_clash_group_expansions = max_clash_group_expansions
        self.clash_domain_top_k = clash_domain_top_k
        self.clash_solve_node_budget = clash_solve_node_budget
        self.revert_index = revert_index
        self.group_label = group_label

    def residue_reach_spheres(self):
        """Per residue, returns (centroids, reach): reach[key] is the distance from
        centroids[key] to the farthest sidechain atom across EVERY gathered candidate conformer
        of that residue (not just the chosen one) - a conservative bounding sphere. Two residues
        can only possibly sidechain-clash if their reach-spheres overlap (plus a margin covering
        the VDW clash threshold), which lets every clash search below skip an exact
        atom-pairwise check for residue pairs that are obviously too far apart. Residues with no
        sidechain atoms (e.g. glycine) get reach 0 - they can never clash."""
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

    def _candidate_pairs_within_reach(self, keys_a, centroids, reach, keys_b=None, margin=3.0):
        """Yields (key1, key2) pairs - key1 from keys_a, key2 from keys_b (defaults to keys_a
        itself, in which case each unordered pair is yielded once) - whose reach-spheres come
        within `margin` of overlapping. `margin` just needs to conservatively cover the largest
        plausible VDW clash threshold (~2-3 Angstrom for two heavy atoms) - it does not need to
        be exact, since this is only a cheap prefilter and every pair it yields still gets an
        exact atom-pairwise check."""
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

    def domain_compatibility_matrix(self, key1, idx1, key2, idx2):
        """Returns an (len(idx1), len(idx2)) boolean matrix: True where candidate idx1[i] of
        residue key1 does NOT sidechain-clash with candidate idx2[j] of residue key2 (sidechain
        atoms only). The per-atom-pair threshold is self.clash_vdw_scale * summed VDW radii,
        EXCEPT for an (N, O) atom pair (either order) - a real N-H...O or O-H...N hydrogen bond
        legitimately sits closer than a generic clash would tolerate, so those pairs use
        self.hbond_clash_vdw_scale instead. idx1/idx2 are arrays of candidate indices (e.g. a
        truncated top-K domain, or a single index to check one specific pair of candidates)."""
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
        scale = np.where(is_n_o_pair, self.hbond_clash_vdw_scale, self.clash_vdw_scale)
        thresh = scale * vdw_sum  # (a1, a2)

        diff = coor1[:, None, :, None, :] - coor2[None, :, None, :, :]  # (n1,n2,a1,a2,3)
        dists = np.linalg.norm(diff, axis=-1)  # (n1,n2,a1,a2)
        clashing = np.any(dists < thresh[None, None, :, :], axis=(2, 3))  # (n1,n2)
        return ~clashing

    def pair_clashes(self, key1, idx1, key2, idx2):
        """Whether residue key1's candidate idx1 sidechain-clashes with residue key2's candidate
        idx2 (both single indices, not arrays)."""
        compat = self.domain_compatibility_matrix(key1, np.array([idx1]), key2, np.array([idx2]))
        return not compat[0, 0]

    def find_clashing_pairs(self, keys, chosen_idx, centroids, reach):
        """Among `keys`' CURRENT choices in chosen_idx, returns every pair that
        sidechain-clashes (after the reach-sphere prefilter)."""
        return [
            (k1, k2) for k1, k2 in self._candidate_pairs_within_reach(keys, centroids, reach)
            if self.pair_clashes(k1, chosen_idx[k1], k2, chosen_idx[k2])
        ]

    def external_clashes(self, group, chosen_idx, centroids, reach):
        """Among `group`'s CURRENT choices in chosen_idx, returns every pair that
        sidechain-clashes with a residue outside the group."""
        others = [k for k in self._candidates if k not in group]
        return [
            (k1, k2) for k1, k2 in self._candidate_pairs_within_reach(
                group, centroids, reach, keys_b=others)
            if self.pair_clashes(k1, chosen_idx[k1], k2, chosen_idx[k2])
        ]

    def connected_components(self, keys, pairs):
        """Groups `keys` into connected components of the graph formed by `pairs` (undirected
        edges). Keys with no edge at all are omitted - only residues that are actually part of
        some clash end up in a returned component."""
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

    def format_group(self, keys):
        return ', '.join(f'{c}{r}' for c, r in keys)

    def group_cost(self, group, chosen_idx):
        return sum(
            float(self._cost[key][chosen_idx[key]])
            for key in group if not self._candidates[key].fixed
        )

    def assignment_clash_free(self, group, assignment):
        """Whether `assignment` ({key: global_candidate_index}, one per member of `group`) has
        zero pairwise sidechain clash among every pair in `group`. `group` is always small
        (capped by max_clash_group_size), so this is a plain O(n^2) check - no reach-sphere
        prefilter needed."""
        return all(
            not self.pair_clashes(k1, assignment[k1], k2, assignment[k2])
            for i, k1 in enumerate(group) for k2 in group[i + 1:]
        )

    def domains_for(self, group, top_k):
        """{key: candidate indices to consider}, cheapest-first. top_k=None means every
        candidate (no truncation); fixed residues always get their single candidate regardless
        of top_k."""
        domains = {}
        for key in group:
            cand = self._candidates[key]
            if cand.fixed:
                domains[key] = np.array([0])
            else:
                order = np.argsort(self._cost[key])
                domains[key] = order if top_k is None else order[:top_k]
        return domains

    def solve_group_assignment_over_domains(self, group, domains, domain_label):
        """Solves the joint cost-minimization problem (see solve_group_assignment) over exactly
        the given `domains` - no truncation or widening here. Returns ({key:
        global_candidate_index}, resolved) where resolved is False if no combination within
        these domains eliminates every pairwise clash (branch-and-bound found nothing AND the
        ICM fallback also didn't land on a compatible combination) - the returned assignment is
        still the best/cheapest one found, just not clash-free."""
        compat = {}
        for i, key1 in enumerate(group):
            for key2 in group[i + 1:]:
                compat[(key1, key2)] = self.domain_compatibility_matrix(
                    key1, domains[key1], key2, domains[key2]
                )

        result = self.branch_and_bound(group, domains, compat)
        if result is None:
            print(f'  {self.group_label} [{self.format_group(group)}]: exact search over the '
                  f'{domain_label} domain found no fully compatible combination (or exhausted '
                  f'its node budget); falling back to a heuristic (ICM) reassignment.')
            result = self.icm_assignment(group, domains, compat)

        assignment = {key: int(domains[key][local_i]) for key, local_i in result.items()}
        return assignment, self.assignment_clash_free(group, assignment)

    def solve_group_assignment(self, group):
        """Returns ({key: chosen_candidate_index}, resolved) for one clash group: the
        combination of candidates (one per residue) that minimizes total cost subject to no
        pairwise sidechain clash within the group - and whether that goal was actually achieved.

        Efficiency: each residue's domain is first truncated to its clash_domain_top_k
        cheapest candidates - a candidate far down the ranking essentially never wins even when
        it's compatible, so this turns what can be a ~100-300-way domain into a ~25-way one with
        negligible risk of losing the true optimum. Given the truncated domains,
        branch_and_bound does an exact search (DFS, most-constrained-residue-first, pruned by an
        admissible cost bound); if that exceeds its node budget, or finds no fully-compatible
        combination at all within the truncated domains, icm_assignment (a fast,
        always-terminating local-search heuristic) is used instead.

        If even that fails to find a fully compatible combination - which does happen: e.g. a
        residue with no real candidate at all (a fixed, single-candidate "domain") can clash
        with EVERY one of a real neighbor's top-K candidates, or two movable residues' only
        mutually compatible pair can simply rank outside the top-K on both sides - this retries
        ONCE with each residue's FULL (untruncated) candidate domain before conceding, since
        that's cheap and can genuinely find a real, compatible - if costlier - combination the
        truncated search missed. If even the full-domain retry can't eliminate every internal
        clash and self.revert_index is set, every movable member of the group is forced to that
        index instead of trusting whatever best-effort (still-clashing) combination the solver
        found; `resolved` is only False if that also happens with self.revert_index left None,
        in which case the caller (resolve_group) is responsible for surfacing that honestly
        rather than reporting success."""
        top_k_domains = self.domains_for(group, self.clash_domain_top_k)
        assignment, resolved = self.solve_group_assignment_over_domains(
            group, top_k_domains, f'top-{self.clash_domain_top_k}'
        )

        if not resolved:
            full_domains = self.domains_for(group, top_k=None)
            assignment, resolved = self.solve_group_assignment_over_domains(
                group, full_domains, 'full')

        if not resolved and self.revert_index is not None:
            assignment = {key: self.revert_index for key in group}

        return assignment, resolved

    def branch_and_bound(self, group, domains, compat, node_budget=None):
        """Exact DFS branch-and-bound over `domains` (local candidate indices per residue),
        minimizing total cost subject to `compat` (pairwise domain-compatibility matrices - see
        domain_compatibility_matrix - keyed by (key1, key2) in `group` order). Residues are
        visited most-constrained-first (smallest domain first); within a residue, candidates are
        tried cheapest-first, and a branch is pruned once its partial cost plus the cheapest
        possible completion (each remaining residue's own minimum candidate cost - an admissible
        lower bound, since it ignores compatibility) can no longer beat the best solution found
        so far.

        Returns {key: local_domain_index} for the optimal assignment, or None if the node budget
        was exhausted before one fully-compatible assignment was found (including the case where
        none exists at all within these domains)."""
        if node_budget is None:
            node_budget = self.clash_solve_node_budget

        order = sorted(group, key=lambda k: len(domains[k]))
        costs = [self._cost[key][domains[key]] for key in order]
        cheapest_first = [np.argsort(c) for c in costs]

        n = len(order)
        suffix_min = [0.0] * (n + 1)
        for k in range(n - 1, -1, -1):
            suffix_min[k] = suffix_min[k + 1] + float(costs[k].min())

        def get_matrix(k1, k2):
            key1, key2 = order[k1], order[k2]
            if (key1, key2) in compat:
                return compat[(key1, key2)], False
            return compat[(key2, key1)], True

        current = [None] * n
        best = {'assignment': None, 'cost': float('inf')}
        nodes = {'count': 0}

        def compat_ok(k, local_i):
            for prev in range(k):
                m, swapped = get_matrix(prev, k)
                i, j = (current[prev], local_i) if not swapped else (local_i, current[prev])
                if not m[i, j]:
                    return False
            return True

        def dfs(k, cost_so_far):
            if cost_so_far + suffix_min[k] >= best['cost']:
                return
            nodes['count'] += 1
            if nodes['count'] > node_budget:
                raise NodeBudgetExceeded()
            if k == n:
                best['assignment'] = list(current)
                best['cost'] = cost_so_far
                return
            for local_i in cheapest_first[k]:
                if not compat_ok(k, local_i):
                    continue
                current[k] = local_i
                dfs(k + 1, cost_so_far + float(costs[k][local_i]))
            current[k] = None

        try:
            dfs(0, 0.0)
        except NodeBudgetExceeded:
            return None

        if best['assignment'] is None:
            return None
        return {key: idx for key, idx in zip(order, best['assignment'])}

    def icm_assignment(self, group, domains, compat, max_iters=25):
        """Iterated Conditional Modes: a fast, always-terminating heuristic for the same joint
        cost-minimization problem branch_and_bound solves exactly. Starting every residue at its
        own cheapest candidate, repeatedly revisits each residue in `group` in turn and
        reassigns it to its cheapest candidate that's compatible with every OTHER residue's
        CURRENT pick, until a full pass changes nothing (or max_iters is hit). May still leave
        residual clashes if even the (already top-K-truncated) domains contain no fully mutually
        compatible combination at all - callers check for that afterward via external_clashes /
        a subsequent find_clashing_pairs pass on the next expansion round.

        Returns {key: local_domain_index}."""
        def get_matrix(key_a, key_b):
            if (key_a, key_b) in compat:
                return compat[(key_a, key_b)], False
            return compat[(key_b, key_a)], True

        current = {key: 0 for key in group}

        for _ in range(max_iters):
            changed = False
            for key in group:
                costs = self._cost[key][domains[key]]
                for local_i in np.argsort(costs):
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

    def resolve_group(self, group, chosen_idx, centroids, reach):
        """Jointly reselects one clash group to the lowest-total-cost combination of candidates
        with no sidechain clash inside the group (see solve_group_assignment) - mutating
        chosen_idx in place for every member. If the new picks clash with a residue outside the
        group, that residue is absorbed into the group and the whole group is resolved again,
        repeating until stable.

        Two independent ways this can end up NOT fully clash-free, both honestly reported in the
        returned row rather than silently reported as success:
          - unresolved: solve_group_assignment itself could not find any candidate combination
            (even outside the top-K) that eliminates every clash WITHIN the current group, and
            self.revert_index is None (so there was no safe fallback to force).
          - hit_cap: the group WAS fully resolved internally, but doing so introduced (or left)
            a clash against a residue outside the group, and absorbing it would exceed
            max_clash_group_expansions (rounds) or max_clash_group_size (residues) - so
            expansion stops there instead of growing further or forcing convergence.
        """
        group = list(group)
        original_group = list(group)
        original_cost = self.group_cost(group, chosen_idx)
        hit_cap = False
        unresolved = False

        def revert():
            if self.revert_index is None:
                return
            for key in group:
                if not self._candidates[key].fixed:
                    chosen_idx[key] = self.revert_index

        for _round in range(self.max_clash_group_expansions):
            assignment, resolved = self.solve_group_assignment(group)
            for key, idx in assignment.items():
                chosen_idx[key] = idx

            if not resolved:
                unresolved = True
                verb = 'reverting every movable member to its fallback conformation' \
                    if self.revert_index is not None \
                    else 'keeping the lowest-total-cost combination found (still clashing)'
                print(f'WARNING: {self.group_label} [{self.format_group(group)}] could NOT be '
                      f'fully resolved - no combination of candidates (checked up to the full '
                      f'candidate pool of every member) eliminates every clash within the '
                      f'group; {verb} - flagged for manual review.')
                break

            external = self.external_clashes(group, chosen_idx, centroids, reach)
            new_members = sorted({k2 for (_, k2) in external if k2 not in group})
            if not new_members:
                break

            if len(group) + len(new_members) > self.max_clash_group_size:
                hit_cap = True
                print(f'WARNING: {self.group_label} [{self.format_group(group)}] would grow '
                      f'past max_clash_group_size={self.max_clash_group_size} residues after '
                      f'absorbing [{self.format_group(new_members)}]; stopping expansion here. '
                      f'Residual clash(es) against [{self.format_group(new_members)}] are left '
                      f'unresolved.')
                revert()
                break

            group.extend(new_members)
        else:
            hit_cap = True
            print(f'WARNING: {self.group_label} [{self.format_group(group)}] kept absorbing '
                  f'new neighbors past {self.max_clash_group_expansions} round(s); stopping '
                  f'here. Residual clashes may remain.')
            revert()

        final_cost = self.group_cost(group, chosen_idx)
        flagged = hit_cap or unresolved
        verb = 'left with a residual clash' if flagged else 'resolved'
        print(f'{self.group_label} [{self.format_group(group)}] ({len(group)} residue(s), '
              f'{len(original_group)} originally clashing) {verb}: total cost '
              f'{original_cost:.4f} -> {final_cost:.4f}'
              + (' (see warning above)' if flagged else '') + '.')

        return {
            'residues': group,
            'original_residues': original_group,
            'size': len(group),
            'original_size': len(original_group),
            'original_cost': original_cost,
            'final_cost': final_cost,
            'hit_cap': hit_cap,
            'unresolved': unresolved,
        }

    def resolve_sidechain_clashes(self, chosen_idx):
        """Finds every sidechain-sidechain clash among the independently chosen conformers in
        `chosen_idx`, groups clashing residues by connectivity, and resolves each group via
        resolve_group - mutating chosen_idx in place. Returns a list of per-group summary rows.

        `groups` (connected components) are computed once, up front, from the INITIAL
        (pre-resolution) clash graph. But resolve_group's own expansion can grow one group to
        fully absorb the residues of a different, separately-identified initial component - if
        that happens, that other component is skipped rather than redundantly (and wastefully)
        resolved again from scratch."""
        centroids, reach = self.residue_reach_spheres()
        keys = list(self._candidates.keys())

        initial_pairs = self.find_clashing_pairs(keys, chosen_idx, centroids, reach)
        groups = self.connected_components(keys, initial_pairs)

        if groups:
            print(f'{len(groups)} {self.group_label}(s) found among independently chosen '
                  f'conformers; resolving each jointly.')

        group_rows = []
        settled = set()
        for group in groups:
            if settled.issuperset(group):
                continue
            row = self.resolve_group(group, chosen_idx, centroids, reach)
            settled.update(row['residues'])
            group_rows.append(row)
        return group_rows
