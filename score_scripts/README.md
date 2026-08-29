# score.sh

Scores a protein-ligand complex against a density map: per-residue RSCC (qfit) plus gnina
CNN docking-pose scoring, for every instance of a given ligand resname. Writes everything into
one output directory.

## Installation

### 1. Conda environments (two required)

**qfit / RSCC** - `nikhils_program_exp`, defined by `../environment.yml`:
```
conda env create -f ../environment.yml
conda activate nikhils_program_exp
pip install -e ..        # installs qfit itself into the env
```

**obabel / meeko (pdbqt prep)** - `openbabel`, defined by `score_env.yml` in this directory:
```
conda env create -f score_env.yml
```

Both env names, and the path to `conda.sh`, are set at the top of `score.sh`
(`CONDA_SH`, `CONDA_ENV_QFIT`, `CONDA_ENV_OBABEL`) - edit those if your setup differs.

### 2. Docker/podman + gnina image

Requires a working `docker` CLI (podman's docker-emulation works fine too). Pull the gnina
image once:
```
docker pull gnina/gnina:latest
```
No GPU is required - gnina falls back to CPU (slower) automatically if none is detected.

## Usage

```
score.sh <map_file> <structure_file> <output_dir> <ligand_resname>
         [-em] [--resolution <float>] [--bfactor <float>] [--label <F,PHI>]
         [--cnn <model>] [--gnina-image <image>]
```

| Arg | Meaning |
|---|---|
| `map_file` | Density map: `.ccp4`/`.mrc`/`.map` or `.mtz` |
| `structure_file` | Protein-ligand complex `.pdb` |
| `output_dir` | Created if missing; every output is written here |
| `ligand_resname` | e.g. `LIG` - every instance/altloc of this resname is scored |
| `-em` | Cryo-EM map (electron scattering factors, static RSCC mask radius) |
| `--resolution <Å>` | Map resolution, for RSCC's mask radius. Optional |
| `--bfactor <float>` | Constant B-factor override for RSCC. Default: use each atom's own |
| `--label <F,PHI>` | MTZ column labels. Default: `FWT,PHWT` |
| `--cnn <model>` | gnina CNN model. Default: `crossdock_default2018` |
| `--gnina-image <image>` | Docker image. Default: `gnina/gnina:latest` |

Example:
```
bash score.sh F02.mrc F02.pdb F02_scores F02 -em --resolution 3.0
```

### Output (`<output_dir>/`)

- `scores.csv` - final merged result, one row per ligand instance: `residue,rscc,affinity_kcal_mol,cnnscore,cnnaffinity`
- `rscc.csv`, `gnina_scores.csv` - the two inputs `scores.csv` is merged from
- `receptor.pdbqt`, `ligand_<label>.pdbqt` - prepared docking inputs (kept for inspection/reuse)
