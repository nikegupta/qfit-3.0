# snEDIAm: density-based scoring of predicted ligand poses

This branch (`snEDIAm`) turns qfit-3.0 into a toolkit for asking one question: **how well does a
predicted ligand pose fit the experimental electron density, compared with what chance alone
would give?**

A raw density-fit score is hard to interpret. A ligand of a given size, at a given resolution, in
a given pocket, can score moderately well just by sitting in the right place. snEDIAm calibrates
the score:

- **Per-atom EDIA:** density fit is measured as per-atom EDIA (electron density for individual
  atoms, EDIAscorer 1.3), summarised as the **mean atom EDIA** of the ligand.
- **Null distribution:** for every modeled ligand conformation, a local null distribution is
  built. It contains up to ~1000 random, non-clashing placements of the same ligand (random
  torsions and orientation, centroid within 3 Å of the modeled ligand's), each scored against the
  same map exactly as the real ligand is.
- **Percentile:** a predicted pose is scored the same way, and its **percentile in that null
  distribution** is the result. A pose at the 99th percentile fits the density better than 99% of
  random placements in the same site.

Two driver scripts do the work:

| Script | Purpose |
|---|---|
| `generate_null.sh` | Builds and scores the null distribution of every modeled ligand conformation, per dataset |
| `evaluate_structures.sh` | Matches each predicted structure to a reference ligand, scores its ligand and reports its percentile in that ligand's null |

`prep_distribution.sh` (kept in `BASE_DIR`, alongside the data) connects the two. It collects one
`generate_null.sh` run into a self-contained `distribution/` tree, which `evaluate_structures.sh`
reads.

## Installation

### 1. qfit-3.0 (this branch) and its conda environment

```bash
git clone -b snEDIAm https://github.com/nikegupta/qfit-3.0
cd qfit-3.0
conda env create -f environment.yml    # creates the "snEDIAm" environment
conda activate snEDIAm
pip install -e .
```

The install provides the console tools the drivers call (`generate_conformers`, `run_edia`,
`build_clash_ensemble`). `match_ligand` runs as `python -m qfit.command_line.match_ligand`, and
the plotting scripts live in `src/snediam_analysis_scripts/`.

### 2. EDIAscorer 1.3

Get EDIAscorer 1.3 from the ZBH (University of Hamburg) and install its licence, which
EDIAscorer reads from `~/.config/ZBH/`. EDIAscorer 1.3 needs glibc ≥ 2.29, so `run_edia` runs it
inside a rootless **podman** container:

```bash
podman pull registry.access.redhat.com/ubi9/ubi
```

`$HOME` is mounted into the container. EDIAscorer, the maps, the output folders and the licence
must therefore all be under `$HOME`.

### 3. Other requirements

- GNU `parallel` (for `-p` dataset parallelism).
- A conda `conda.sh` init script.

### Configuration

Each driver has a **"User-specified configuration"** block near its top. Edit it to match your
install:

- `BASE_DIR`, `CONDA_SH`, `CONDA_ENV_NULL` (`snEDIAm`);
- `EDIASCORER` (path to the EDIAscorer 1.3 executable), `EDIA_IMAGE`;
- the data paths and file-name patterns described below.

The scripts stop with a clear error if a required file is missing.

## Input layout

Under `BASE_DIR`:

- **`datasets_test.txt`** (or any list set by `DATASETS_FILE`): dataset ids, one per line.
- **`pxr_fragments.csv`** (`CSV_FILE`): one row per dataset with columns
  `dataset,resolution,ligand_name,a,b,c,alpha,beta,gamma,space_group,smiles`.
  - The unit cell, space group and resolution are always taken from here, never from PDB
    CRYST1/REMARK records.
  - The SMILES gives the ligand's bond orders.
- **`datasets/<dataset>/`**, per dataset:
  - `<dataset>_001.pdb`: the reference model (`STRUCTURE_PDB_PATTERN`) with the ligand(s) as
    residue `LIG`. It is a multiconformer model: altloc A is the apo state and is ignored for
    clash checks, but every altloc is kept as density context for EDIA.
  - `<dataset>_refined_2fofc.ccp4`: the density map (`MAP_NAME_PATTERN`), in the model's frame.
- **`test/datasets/<dataset>/`**: the predicted structures to evaluate, e.g.
  `<dataset>_seed_42_sample_1_model.pdb` (`PREDICTION_PATTERN`). They can be numbered and
  chained differently from the reference; matching uses sequence alignment.

## Usage

### 1. Build the null distributions: `generate_null.sh`

```bash
./generate_null.sh <run_name> [-p <num_parallel>] [--dataset <id[,id...]>]
                   [--n_conformers 1000] [--sampling_radius 3] [--edia_workers <n>]
                   [--overwrite] [--replot] [...]
```

Run `./generate_null.sh -h` for every option. Each modeled ligand conformation (each altloc of
each `LIG` residue) gets its own folder, `datasets/<dataset>/<run_name>/LIG_<chain><resi>_<altloc>/`
(e.g. `LIG_A701_B`).

- **Stage 1: `generate_conformers`.**
  - Random placements are generated: a torsion conformer, a centroid drawn uniformly from a 3 Å
    sphere around the modeled ligand's centroid, and a random orientation.
  - Placements are kept if they don't clash with the protein (0.5 × vdW, altloc A ignored).
  - Writes `conformers.pdb` (one MODEL per placement), `ligand.pdb` (the modeled conformation),
    `conformer_summary.json` and `ensemble_mask.ccp4`.
  - Pooled plots of ensemble volume, acceptance and reference coverage go to
    `graphs/<run_name>/`.
- **Stage 2: `run_edia`.**
  - Per-atom EDIA of every placement (`distribution_residue_scores.csv`,
    `distribution_atom_scores.csv`).
  - Per-atom EDIA of the modeled ligand (`reference_residue_scores.csv`,
    `reference_atom_scores.csv`).
  - A histogram of the null's mean atom EDIA.
- **Stage 3: pooled evaluation plots.** The reference ligands' scores, null medians and means,
  and each reference's percentile in its own null.

Every step skips outputs that already exist (`--overwrite` redoes them, `--replot` redoes the
plots). Interrupted EDIA scoring resumes from its finished calls. `--dataset` restricts a run to
specific datasets.

`run_edia` is a standalone, general scorer. It takes any single- or multi-model ligand PDB, a
protein target (LIG/HOH/DMS removed) and a map:

```bash
run_edia ligand.pdb target.pdb map.ccp4 out_dir --cell a b c alpha beta gamma \
         --space-group P212121 --resolution 1.96 [--smiles ...] \
         [--residue-scores scores.csv] [--atom-scores atoms.csv]
```

### 2. Collect a run's distributions: `prep_distribution.sh`

```bash
./prep_distribution.sh <generate_null_run> datasets_test.txt
```

The script copies the reference structure and map once per dataset, and the run's null and
reference score CSVs per run:

```
distribution/<dataset>/<dataset>_001.pdb, <dataset>_refined_2fofc.ccp4
distribution/<dataset>/<generate_null_run>/LIG_<chain><resi>_<altloc>/
    distribution_residue_scores.csv, reference_residue_scores.csv (+ params)
```

Reruns skip files that are already up to date.

### 3. Evaluate predictions: `evaluate_structures.sh`

```bash
./evaluate_structures.sh <distribution_run> <run_name> [-p <num_parallel>] [--dataset <id[,id...]>]
                         [--prediction_pattern '{dataset}_seed_42_sample_*_model.pdb']
                         [--metric mean_atom_edia] [--overwrite] [--replot]
```

`<distribution_run>` is the `generate_null.sh` run collected by `prep_distribution.sh`, and
`<run_name>` names this evaluation. Each prediction gets its own folder,
`test/datasets/<dataset>/<distribution_run>/<run_name>/<prediction>/`.

- **Stage 1: `match_ligand`.**
  - The prediction is CA-superposed onto each reference chain that contains a ligand, by
    sequence alignment.
  - Its ligand is matched to the reference ligand conformation (with a null) whose centroid is
    nearest.
  - Writes `match.txt` (e.g. `LIG_B701_B`), `match.json` and `ligand.pdb` (the predicted ligand
    in the reference frame).
- **Stage 2: `run_edia`.** Per-atom EDIA of the predicted ligand against the reference protein
  and map, scored exactly like the nulls (`prediction_residue_scores.csv`,
  `prediction_atom_scores.csv`).
- **Stage 3: pooled results**, in `test/graphs/<distribution_run>/<run_name>/`:
  - `evaluation.csv`: one row per prediction, with match, fit, score, percentile in the matched
    null, and the reference ligand's score and percentile.
  - A histogram of prediction percentiles.
  - A scatter of prediction against matched-reference mean atom EDIA.
  - Violins of each matched reference ligand's null, with its predictions and reference
    overlaid.

Logs of every run go to `logs/<run_name>/` (`generate_null.sh`) and
`test/logs/<distribution_run>/<run_name>/` (`evaluate_structures.sh`).

## Typical workflow

```bash
./generate_null.sh run_21 -p 8 --edia_workers 4      # build and score the nulls
../prep_distribution.sh run_21 ../datasets_test.txt  # collect them into distribution/
./evaluate_structures.sh run_21 eval_1 -p 4          # score the predictions against them
```
