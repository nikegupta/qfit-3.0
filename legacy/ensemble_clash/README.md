# Ensemble-based clash strategy (retired 2026-10-29)

This folder keeps the clash strategy that `generate_conformers` used up to and including run_17,
so it can be re-implemented or restored. It was replaced by the multiconformer `_001` rule:
reject any clash with this structure's atoms except LIG, HOH/DMS, hydrogens and altloc A.

The files here are the last working versions, copied unchanged from just before the switch:

| File | What it is |
|---|---|
| `generate_conformers.py` | sampling, plus the side-chain ensemble and the clash checkers |
| `build_clash_ensemble.py` | the once-per-run ensemble build (CLI) |
| `generate_null.sh` | the driver, wired for the ensemble (`--rotamer_tolerance`, stage 1a) |

They compile, but `build_clash_ensemble.py` imports `qfit.command_line.generate_conformers`. To
run them, copy all three back over the current files in `src/qfit/command_line/` and the repo
root. Running only one of them against the current `generate_conformers` won't work.

Line numbers below refer to `generate_conformers.py` in this folder.

---

## 1. The idea

The null asks: where could this ligand plausibly sit in this pocket? The protein is not one
fixed structure across the PXR datasets. Side chains lining the pocket take different rotamers
depending on which fragment is bound. The ensemble rule therefore treats:

- **the backbone as rigid**, always this structure's own backbone; and
- **binding-site side chains as flexible**, drawing on every side-chain conformation observed at
  that residue in any of the 186 datasets.

A placement is rejected only if it clashes with the backbone, or if, **for some residue, it
clashes with every one of that residue's observed conformations**. In words: a pose is allowed if
each residue it touches has at least one observed rotamer that makes room for it.

Two properties follow:

- **It's permissive.** Each residue picks its rotamer independently, and the chosen rotamers are
  never checked against each other.
- **It's dataset-agnostic.** The same ensemble of side chains is grafted onto every dataset's
  backbone.

---

## 2. Build step: `ClashEnsemble` (once per run)

`build_clash_ensemble reference.pdb ensemble_models.txt logs/<run>/clash_ensemble.pkl
[--aligned-dir ...]`. `generate_null.sh` stage 1a writes `ensemble_models.txt` with one
structure per dataset in the full `datasets.txt` (`STRUCTURE_PDB_PATTERN`, then
`{dataset}.pdb`), even when the run uses `--dataset`. Code: `ClashEnsemble.__init__`, line 751.

### 2a. Binding-site residues: `binding_site_resis` (line 320)

- In each model's own frame, a protein residue (ATOM record) is a binding-site residue if any of
  its heavy atoms, in any altloc, is within `--binding-site-cutoff` (default 4.0 Å) of a heavy
  atom of any LIG residue.
- Residues are keyed by **residue number only (chain-agnostic)**. A number counts if it's in the
  binding site in any chain of any model.
- Output: `binding_site = {resi: number of models}`, written to `pocket_residues.csv`.
- In run_17: 186 models gave 52 residue numbers, i.e. 104 residues over chains A and B.

### 2b. Side-chain conformations: `ensemble_residues` (line 468)

For every residue (chain, resi) whose number is in the binding site:

1. **Reference frame.** The residue's backbone frame is the N, CA and C coordinates from
   `reference.pdb`, parsed with `read_protein` and `conformations()[0]`. If the reference lacks
   the residue, the first model that has it is used.
2. **Gather.** In every model, each altloc of the residue is one conformation:
   `conformations()` gives the blank-altloc atoms plus, for each altloc letter, that letter's
   atoms. For each conformation:
   - superpose its own N, CA, C onto the reference frame (Kabsch, `superpose`);
   - keep only the side-chain atoms, i.e. everything except `BACKBONE_ATOMS = (N, CA, C, O,
     OXT)`.

   Because each conformation is fitted on its own backbone, backbone or loop shifts between
   models never count as a different side-chain conformation. Residues with no side-chain atoms
   (glycine) are skipped, as are conformations missing N, CA or C.
3. **Drop incomplete conformations.** Count how often each side-chain atom name appears. The
   "core" names are those present in at least half the conformations. A conformation missing
   any core atom is dropped (if that would drop all of them, all are kept). This removes
   truncated side chains.
4. **De-duplicate.** Greedy leader clustering (`cluster_conformations`, line 454):
   - walk the conformations in order;
   - each joins the first existing representative within `--rotamer-tolerance` (default
     0.3 Å) heavy-atom RMSD over shared atom names (`conformation_rmsd`), or else becomes a new
     representative;
   - count members per representative.
5. **Cap.** If a residue still has more than `MAX_CONFORMATIONS` (128) representatives, the
   tolerance is raised in 0.25 Å steps and the clustering redone. The final tolerance is stored
   per residue. In run_17 none needed raising.
6. **Store.** Representatives are sorted most common first. Each residue becomes a dict:
   - `key` (chain, resi), `resn`;
   - `frame`: 3×3 array of the reference N, CA, C;
   - `lines`: atom name → a PDB line, for element and name lookups;
   - `confs`: a list of (side-chain atom names, xyz in the reference frame);
   - `counts`, `tolerance`.

In run_17 this gave 104 residues with 2,300 conformations in all: a median of 13 per residue, a
maximum of 97, and 5 residues with a single conformation. The build took 4.9 s.

### 2c. Pocket volumes and sampling spheres

These are unchanged in the current code and are documented in the current
`generate_conformers` docstring. In brief:

- **Pocket volume, per LIG chain:** every model is superposed onto `reference.pdb` on all
  shared CA atoms. Its LIG heavy atoms become 2.0 Å spheres, and a 0.5 Å probe is rolled over
  the union (0.25 Å grid).
- **Sampling sphere:** the minimum enclosing sphere of the ligand centroids, plus a 2.0 Å
  margin.

### 2d. Saved object

`clash_ensemble.pkl` is a pickled `ClashEnsemble` with:
- `reference_pdb`, `reference_ca`;
- `binding_site`;
- `residues` (above);
- `volumes` (per chain);
- `info` (parameters and counts).

The pkl from run_17 is still at `logs/run_17/clash_ensemble.pkl`, with `residues` intact. It
unpickles with the current class: the attributes come back, even though the methods that used
them are gone.

---

## 3. Per pocket: building the obstacles (`ClashEnsemble.placed`, line 832)

For the sampled structure (`read_protein`: heavy atoms, no LIG or `--exclude-resn` residues,
keyed (chain, resi) → (resn, {altloc: {name: line}})):

1. **Rigid backbone.** Every `BACKBONE_ATOMS` atom of every residue, in every altloc
   conformation, becomes a rigid obstacle. A dict de-duplicates blank-altloc atoms, which appear
   in every altloc's conformation.
2. **Grafted side chains.** For each residue whose key (chain, resi) is in `ensemble.residues`,
   if the structure's own first conformation has N, CA and C:
   - superpose the ensemble frame (reference N, CA, C) onto **this structure's** N, CA, C,
     giving r and t;
   - transform every stored conformation with `xyz @ r.T + t`.

   The observed rotamers are thereby rebuilt on this dataset's backbone.
3. **Fallback.** Any other residue keeps **its own** side-chain conformations: its altlocs,
   de-duplicated with the same tolerance (`own_side_chains`, line 528).
4. **Distance filter.** Only residues with some conformation atom within `region` of the
   sampling centre are kept, and likewise only backbone atoms within `region`. Here:
   - `region = reach + max_cut + 1 Å`;
   - `reach` = the maximum sampling radius plus the ligand's extent from its centroid;
   - `max_cut` = the largest clash cutoff.

The kept residues are then split (`main`, about line 1105):

- **Rigid:** residues with exactly one conformation. Their atoms join the backbone as plain
  obstacles in `ClashChecker`.
- **Flexible:** residues with two or more conformations, which go to `FlexibleChecker`.

Example from run_17, x02813-1 chain A: 665 backbone atoms, and 161 nearby side chains (112
single-conformation, 49 flexible with 1,268 conformations). Set-up took 6.1 s.

---

## 4. Clash tests

**Pair rule (all checkers).** A ligand atom of element e and an obstacle atom clash if their
distance is below:

    cutoff = clash_scale * (r_vdw(e) + r_vdw(obstacle))    # --clash-scale, default 0.75

or, if both atoms are N or O, below `polar_clash_scale` (default 0.6) times the same sum. The
vdW radii come from `qfit.structure.vdw_radii.vdwRadiiTable`, with carbon's radius as the default
for unknown elements. Hydrogens are never obstacles, and no symmetry mates are used.

### 4a. `ClashChecker` (line 597): backbone and single-conformation side chains

- **Grouping.** Obstacles are grouped by (vdW radius, is-polar), so every atom in a group has
  the same cutoff to a given ligand element. Each group gets a KD-tree.
- **`certain_clash(xyz, e)`.** A boolean lookup grid per ligand element (cube around the centre,
  `--grid-spacing` 0.25 Å). A grid point is marked if it's within `cutoff − pad` of some
  obstacle, with `pad` = half the voxel diagonal. Any position that rounds to a marked point
  therefore really clashes, so a grid hit is never a false positive.
- **`clash(xyz, e)`.** The exact test: a KD-tree query per group with that group's cutoff.

### 4b. `FlexibleChecker` (line 663): the all-conformations rule

A candidate is rejected if **for some flexible residue, it clashes with every one of that
residue's conformations**.

- **Data.** All conformations' atoms are stacked into one KD-tree. Each atom records its
  residue index `res`, its conformation index `conf`, and, for the bitmask, `word = conf // 64`
  and `bit = 1 << (conf % 64)`. Each residue has a `full` mask: a bit set for each of its
  conformations, over `MASK_WORDS = MAX_CONFORMATIONS / 64` 64-bit words.
- **Exact test, `clashes_all(xyz, lig_elements)`,** on whole candidates (candidates × atoms × 3):

      masks = zeros(n_candidates, n_residues, MASK_WORDS) uint64
      for each ligand element e:
          points = ligand atoms of element e, all candidates
          pairs  = points × flexible atoms within max cutoff (sparse_distance_matrix)
          hit    = pair distance < cutoff(e, that flexible atom)
          masks[candidate, res[j], word[j]] |= bit[j]     for every hit pair
      reject = any residue r with masks[candidate, r, :] == full[r]

  That is, a candidate is rejected when it has hit every conformation of at least one residue.
- **Grid speed-up, `add_certain(checker)`.** This writes into `ClashChecker`'s certain-clash
  grids:
  - for each flexible residue, only points within reach of every conformation can clash with
    all of them, so it takes the box common to all conformations' extents, padded by `max_cut`;
  - for each grid point in that box and each element, it marks the point if, **for every
    conformation**, some atom of that conformation is within `cutoff − pad`.

  Such a point clashes with all conformations whatever the rotamer, so the grid stage can drop
  it early.

### 4c. Order in `sample()` (line 892)

For each batch of random candidates (torsion conformer, translation, rotation; see the current
docstring):

1. `VolumeChecker.outside`: pocket volume, atom by atom.
2. `ClashChecker.certain_clash`: rigid plus the flexible "certain" points, atom by atom.
3. `ClashChecker.clash`: exact, rigid obstacles only, atom by atom.
4. `FlexibleChecker.clashes_all`: exact all-conformations test on the surviving whole
   candidates, in chunks of `FLEXIBLE_CHUNK` = 100,000 to bound memory.

Steps 1–3 go atom by atom, core atoms first, on a shrinking set of survivors. Step 4 needs the
whole pose, because a residue's mask fills up from different ligand atoms.

---

## 5. Outputs that reflect the ensemble

- **`obstacles.pdb`** (`obstacle_lines`, line 1024):
  - the rigid backbone atoms;
  - every nearby residue's side-chain conformations, grafted. Flexible residues are written as
    altlocs A, B, …, most common first, up to 52, with occupancy = the share of that residue's
    observed conformations each one represents.
- **`conformer_summary.json` → `clash_ensemble`:**
  - the ensemble `info`: `n_models`, `n_binding_site_residues`, `n_residues`,
    `n_conformations`, `rotamer_tolerance_A`, `max_conformations`, the pocket parameters and
    `build_time_s`;
  - plus `n_backbone_atoms_near`, `n_side_chains_near`, `n_rigid_side_chains`,
    `n_flexible_residues`, `n_conformations_near`, `tolerance_raised` and `setup_s`.

---

## 6. Parameters and defaults

| Option (`generate_null.sh` / Python) | Default | Used in |
|---|---|---|
| `--rotamer_tolerance` / `--rotamer-tolerance` | 0.3 Å | conformation de-duplication (2b) |
| `MAX_CONFORMATIONS` (constant) | 128 | per-residue cap; tolerance raised in 0.25 Å steps |
| `--binding_site_cutoff` / `--binding-site-cutoff` | 4.0 Å | binding-site residues (2a) |
| `--clash_scale` / `--clash-scale` | 0.75 | pair cutoff |
| `--polar_clash_scale` / `--polar-clash-scale` | 0.6 | N/O–N/O pair cutoff |
| `--exclude_resn` / `--exclude-resn` | HOH, DMS | left out of obstacles (and of the ensemble build) |
| `--grid-spacing` | 0.25 Å | lookup grids |
| `FLEXIBLE_CHUNK` (constant) | 100,000 | candidates per `clashes_all` call |
| `--pocket_atom_radius`, `--pocket_probe_radius`, `--centroid_margin` | 2.0, 0.5, 2.0 Å | pocket volume and sphere |

The ligand was read from altloc blank or A only (`read_ligand`, `ligand_centroids`:
`l[16] in ' A'`), which suited the bound-state `{dataset}.pdb` files.

---

## 7. Behaviour and known limitations

These were measured on run_17 nulls in `claude_scripts/null_clash_analysis.py` and
`check_001_clash.py`:

- **Null atoms overlap the deposited side chains.** Because any observed rotamer can make room,
  null poses regularly sit on top of the deposited structure's own side chains, which are what
  EDIA scores against. Across 9 sampled pockets:
  - 33% of null atoms were within 2.5 Å of a protein atom in EDIA's target, 17% within 2 Å and
    7% within 1.5 Å;
  - every one of those contacts was with a side chain, never the backbone.
- **Rotamers aren't checked against each other.** Each residue's rescuing rotamer is chosen on
  its own, so the combination of rotamers that admits a pose may itself be impossible.
- **Residues are matched by number, not chain.** Binding-site residue numbers are pooled over
  chains, and conformations are pooled over every dataset.
- **Effect on the null:**
  - against the `_001` bound states (blank and B altlocs as obstacles), only a median 2.2% of
    run_17's null conformers per pocket would survive;
  - acceptance in run_17 was about 0.4% of random candidates (x02813-1 chain A: 1,000 kept from
    269,000 tried in 17 s).

---

## 8. Re-implementation checklist

1. Restore the code: copy the three files back, or port these pieces:
   - `conformations`, `read_protein`;
   - `conformation_rmsd`, `cluster_conformations`, `ensemble_residues`, `own_side_chains`;
   - `FlexibleChecker`, `ClashEnsemble.placed`;
   - the rotamer and `MAX_CONFORMATIONS` constants;
   - `obstacle_lines(residues)`;
   - the `flexible` argument of `sample()`, and the obstacle assembly in `main()`.
2. `ClashEnsemble.__init__` takes `(reference_pdb, model_pdbs, rotamer_tolerance, exclude_resn,
   spacing, binding_site_cutoff, pocket_atom_radius, pocket_probe_radius, centroid_margin)` and
   builds `residues`. The current class drops the two arguments after `model_pdbs` and the
   residue build.
3. If it's combined with the `_001` multiconformer structures, decide how altlocs A and B of
   **this** structure interact with the ensemble rotamers. As written, a structure's own altlocs
   only matter for non-binding-site residues (fallback) and the backbone.
4. Check it: `claude_scripts/check_ensemble_residues.py` and `coverage_atoms.py` test the
   ensemble and checkers. They import these names and will work again once the code is
   restored.
