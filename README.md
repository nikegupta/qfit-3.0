# nikhils_program

` Nikhil's program is a pipeline for automatically fitting ligands and protein conformational changes to PanDDAs maps.
 Given, per dataset, an apo/ground-state protein model, one or more PanDDA event maps, and a candidate ligand
(SMILES + crystal cell/space group), the pipeline places the ligand into density, samples and
refines its conformation (and the surrounding protein sidechains) across several rounds, merges
everything into one composite model, real-space refines it, optionally re-optimizes rotamers near
the ligand, and finally rescores the surviving ligand pose with an independent statistical
potential (DESPOT). Every stage is driven by `program.sh`, a single bash script that wires
together several console tools from a local `qfit-3.0` checkout, an external PLACER install, and
an external DESPOT install.

## Installation

The pipeline spans four separate pieces of software, each with its own conda environment.
`program.sh`'s own "User-specified configuration" block (near the top of the file) hardcodes the
paths and environment names below - edit that block to match wherever you install things.

### 1. qfit-3.0

```bash
git clone https://github.com/nikegupta/qfit-3.0 program_rotamer/qfit-3.0
cd qfit-3.0
conda env create -f environment.yml    # creates "nikhils_program_rotamer" - see name: in the yml
conda activate nikhils_program_rotamer
pip install -e .
```

This environment is used for most steps of the pipeline

### 2. PLACER (protein-ligand conformer sampling)

PLACER is a separate repository, not vendored here:

```bash
git clone https://github.com/baker-laboratory/PLACER.git
cd PLACER
conda env create -f envs/placer_env.yml
conda activate placer_env
```

Point `program.sh`'s `RUN_PLACER_PY` variable at that checkout's `run_PLACER.py`, and
`CONDA_ENV_PLACER` at the `placer_env` environment name. PLACER also needs a CUDA-capable GPU
(cuda-toolkit >= 12.1) - see the PLACER repo's own README for its full requirement list.

### 3. Open Babel (ligand/protein -> mol2 conversion, used ahead of PLACER and DESPOT)

```bash
conda create -n openbabel -c conda-forge openbabel
```

Point `CONDA_ENV_OBABEL` at this environment's name.

### 4. DESPOT (final ligand-pose scoring)

Also a separate repository:

```bash
git clone https://github.com/KUL-LBMD/DESPOT.git
cd DESPOT
conda env create -f environment.yml
conda activate DESPOT
pip install -e .
bash download_data.sh   # downloads the pretrained potentials + metadata (~1.9 GB) into DESPOT/data
```

Point `DESPOT_SCRIPT` at that checkout's `scripts/score_complex.py` and `CONDA_ENV_DESPOT` at the
`DESPOT` environment name. `DESPOT_DATABASE` selects which trained potential to score against
(default: `CROWN`).

### Configuring program.sh

Once all four environments exist, edit the "User-specified configuration" block at the top of
`program.sh`: `BASE_DIR`, `CONDA_SH` (your `conda.sh` init script), the five `CONDA_ENV_*`
variables, `RUN_PLACER_PY`, `DESPOT_SCRIPT`. The
script will refuse to run (with a clear "required file not found" error) if any configured script
path doesn't actually exist, so a misconfiguration is caught immediately rather than partway
through a run.

## Input layout

Before running the pipeline you need, under `BASE_DIR`:

- **`datasets.txt`** - A file listing the datasets to run the pipeline on, one dataset per line.
- **`datasets/<dataset>/`** - one directory per dataset id, containing:
  - `<dataset>-aligned-structure.pdb` - the apo/ground-state protein model.
  - `<dataset>-z_map.native.ccp4` - the zmap for that protein model
  - `<dataset>-event_<N>_1-BDC_<value>_map.native.ccp4` - one or more PanDDA event maps.
- **A CSV with required information about the datasets** (path set by `CSV_FILE`, `pxr_fragments.csv` in the github version) with columns
  `dataset,resolution,ligand_name,a,b,c,alpha,beta,gamma,space_group,smiles` - one row per
  dataset, giving the candidate ligand's SMILES and the crystal's cell/space group 
- **`pdb_final_geometry/`** (path set by `LIG_PDB_DIR`) - contains the pdb structure and cif restraints file for each ligand in the dataset.
- **`reference_set/<dataset>/`** (optional, only needed for `-c`) - a deposited/reference
  structure per dataset, to compare pipeline output against.

## Running the pipeline

```bash
./program.sh <run_name> [placer_run_name [filter_run_name [placer2_run_name [filter2_run_name [final_run_name [rotamer_run_name [despot_run_name]]]]]]]
             [-n <num_placer_confs>] [-n2 <num_placer2_confs>] [-g <gpu_ids>] [-p <num_parallel>]
             [-c] [--overwrite] [--replot] [--dataset <id[,id...]>] [...many per-stage tunables]
```

Only `<run_name>` is required. The eight positional names correspond to the eight nested stages
of the pipeline (see below) - supplying fewer of them simply stops the pipeline after that stage,
and each name becomes a subdirectory nested under the previous one, so re-running with a new name
at any point branches off cleanly without touching earlier results. Every step checks whether its
own output already exists and skips it if so (pass `--overwrite` to force a redo, or `--replot`
to redo just the graphing/analysis steps).

Examples:

```bash
./program.sh run_1 placer_1 filter_1                                    # stages 0-3 only
./program.sh run_1 placer_1 filter_1 placer2_1 filter2_1 final_1        # through build_final + RSR
./program.sh run_1 placer_1 filter_1 placer2_1 filter2_1 final_1 rotamer_1              # + rotamer optimization
./program.sh run_1 placer_1 filter_1 placer2_1 filter2_1 final_1 rotamer_1 despot_1     # + DESPOT scoring
./program.sh run_1 placer_1 filter_1 placer2_1 filter2_1 final_1 rotamer_1 -c            # + compare to reference_set
./program.sh run_1 placer_1 filter_1 --dataset x00001-1,x00002-1        # restrict to specific datasets
```

Key flags: `-n`/`-n2` set how many PLACER conformers to sample in rounds 1 and 2 (default 100
each), these variables most significantly affect run time.
`-g` sets which GPU id(s) PLACER uses; `-p` sets CPU parallelism for every other stage;
`-c` additionally scores everything against `reference_set/` for validation. A large set of
per-stage tunables are passed straight through to the corresponding step; see `./program.sh -h` for the full list.
For best results `--f2_filter_proportion 1` should be added to prevent excessive culling of potential ligands.

## Pipeline stages

### Stage 1: fit_ligand

For each dataset, `fit_ligand` scans the event map(s) for density peaks above a Z-score threshold
and, at each candidate peak and places a copy of the ligand at each peak (or a copy of each possible stereoisomer).
Each accepted peak produces one seeded ligand-in-protein starting structure,
recorded in a per-dataset manifest linking it back to the peak and ligand that produced it.

### Stage 2: PLACER (round 1) + real-space refinement

Each `fit_ligand` seed is handed to **PLACER** (Protein-Ligand Atomistic Conformational Ensemble
Resolver, run out of a separate install), which stochastically resamples the ligand's atomic
positions and nearby protein sidechains to generate an ensemble of plausible conformers for
that seed. Every conformer in the resulting ensemble is then **real-space refined** against the
event map: a coot-headless-scripted refinement (`rsr_placer`, via `src/rsr_scripts/`) that
locally minimizes the ligand and surrounding residues into the density without changing the rest
of the model.

### Stage 3: filter

`filter` clusters the (refined) round-1 conformers spatially, scores every conformer's fit to the
event map (RSCC), and keeps one representative per cluster. 
It then real-space refines the protein backbone around each surviving cluster's
representative (`rsr_backbone`) and computes that backbone's own refined RSCC, giving each
cluster a refined structural context to hand off to round 2.

### Stage 4: PLACER (round 2) + real-space refinement

The same PLACER sampling is run again, once per surviving cluster, this time seeded from that
cluster's real-space-refined backbone (`filter`'s round-1 output) rather than the raw round-1
seed - a second, more targeted round of conformer sampling now that the local protein environment
has already been refined once. Every round-2 conformer is again real-space refined
(`rsr_placer2`).

### Stage 5: filter2

The same `filter` script is run again on the round-2 ensemble, once more clustering and picking a
representative conformer per cluster - this is the final round of pose selection before the
per-dataset model is assembled.

### Stage 6: build_final_model + real-space refinement

`build_final_model` merges every surviving filter2 cluster's representative ligand pose into a
single composite structure built on top of the apo protein, resolving any sidechain-sidechain
clashes that result from combining poses that were each optimized independently. 
The merged model is then real-space refined as a whole (`rsr_final`), and its
per-residue and per-ligand RSCC against the event map(s) is computed.

### Stage 7: rotamer_optimize + real-space refinement

`rotamer_optimize` re-samples chi/aromatic-ring
rotamers for every residue near the ligand whose RSCC (against the pre-refinement final model)
falls below a threshold, keeping a resampled rotamer only if it improves RSCC by a meaningful
margin; it also resolves any new sidechain-sidechain clashes the resampling introduces. The
result is real-space refined again (`rsr_rotamer`), and its RSCC recomputed. Because RSR can
shift things after the fact in ways `rotamer_optimize`'s own acceptance check couldn't see,
`select_optimized_residues` then compares each candidate residue's post-refinement RSCC against
its RSCC in the (pre-rotamer-optimization) final model and reverts any residue whose RSCC didn't
actually improve enough to justify the change.

### Stage 8: DESPOT scoring

Every round-2 PLACER
conformer (not just each cluster's single selected representative) is pooled and the model's
protein is symmetry-expanded around all of them; each conformer is scored against the protein
with DESPOT's `score_complex.py`, an independent statistical potential for protein-ligand
interactions. For each filter2 cluster, `despot_filter` looks at the MSE-vs-DESPOT Pareto front
of that cluster's conformers, computes each front member's own real-space-refined RSCC, and
picks the conformer that best trades off RSCC against normalized DESPOT score - which can differ
from the pose stage 5/6/7 carried forward. That winning pose is kept only if it clears both an
RSCC and a DESPOT-score threshold; otherwise the whole cluster's ligand is dropped from the final
output, and any residue whose only rotamer support came from a dropped ligand is reset back to
its apo conformation.
