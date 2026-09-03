# test_case

A minimal, self-contained end-to-end test of the pipeline: one dataset (`x00407-1`), one
ligand (`DSI_1_G22`), one reference structure. Run `test.sh` to exercise every pipeline
stage (fit_ligand → PLACER → filter → PLACER2 → filter2 → build_final → despot) plus the
`-c` reference-set comparison plots.

## 1. Install the repository (qfit-3.0)

```bash
git clone -b main https://github.com/ExcitedStates/qfit-3.0.git
cd qfit-3.0
mamba env create -f environment.yml     # creates conda env "nikhils_program_exp"
mamba activate nikhils_program_exp
pip install .                           # or `pip install -e .` for a dev install
```

This installs qFit's console scripts (`fit_ligand`, `filter`, `build_final_model`,
`symmetry_expand`, `despot_filter`, `calc_rscc`, etc.) used throughout `test.sh`.

## 2. Install PLACER

```bash
git clone https://github.com/baker-laboratory/PLACER.git
cd PLACER
conda env create -f envs/placer_env.yml   # creates conda env "placer_env"
```

Requires a CUDA-capable GPU (cuda-toolkit >= 12.1). Note the path to `run_PLACER.py`
inside the cloned repo — you'll need it for step 4.

## 3. Install DESPOT

```bash
git clone https://github.com/KUL-LBMD/DESPOT.git
cd DESPOT
conda env create -f environment.yml -n DESPOT   # repo's environment.yml names the env
                                                 # "DESPOT_true"; -n DESPOT overrides that
                                                 # to match what test.sh expects
conda activate DESPOT
pip install -e .
bash download_data.sh                            # downloads metadata + potentials (~1.9 GB)
                                                   # from Zenodo into DESPOT/data/
```

Note the path to `scripts/score_complex.py` inside the cloned repo — needed in step 4.

## 4. An openbabel environment

A small separate conda env providing the `obabel` CLI (used for pdb→mol2 conversion):

```bash
conda create -n openbabel -c conda-forge openbabel
```

## 5. Configure test.sh

Everything under `test_case/` (datasets, ligands, datasets.txt, pxr_fragments.csv,
reference_set) is already wired up — don't need to touch those. Only the
**"User-specified configuration"** block near the top of `test.sh` needs editing for your
machine:

| Variable | Set to |
|---|---|
| `CONDA_SH` | Path to your `conda.sh` (e.g. `~/miniconda3/etc/profile.d/conda.sh`) |
| `CONDA_ENV_QFIT`, `CONDA_ENV_RSR`, `CONDA_ENV_EVAL` | The qfit-3.0 env from step 1 (`nikhils_program_exp`) |
| `CONDA_ENV_PLACER` | The PLACER env from step 2 (`placer_env`) |
| `CONDA_ENV_OBABEL` | The openbabel env from step 4 (`openbabel`) |
| `CONDA_ENV_DESPOT` | The DESPOT env from step 3 (`DESPOT`) |
| `RUN_PLACER_PY` | Absolute path to your PLACER clone's `run_PLACER.py` |
| `DESPOT_SCRIPT` | Absolute path to your DESPOT clone's `scripts/score_complex.py` |
| `DESPOT_DATABASE` | `CROWN` (matches `download_data.sh`'s default archives) |
| `RSR_SCRIPTS_DIR`, `ANALYSIS_SCRIPTS_DIR`, `LIG_SCRIPTS_DIR` | Absolute paths to this qfit-3.0 clone's `rsr_scripts/`, `analysis_scripts/`, `lig_scripts/` |

`BASE_DIR` should already point at this `test_case/` directory — leave it as-is unless
you move the folder.

## 6. Run

```bash
./test.sh
```

`test.sh` ignores any stage-name arguments and always runs the full pipeline
(`run_1 placer_1 filter_1 placer2_1 filter2_1 final_1 despot_1 -c --f2_filter_proportion 1`);
pass-through flags like `-g <gpu_ids>` still work. On success it prints:

```
run_1 test_successful
```
