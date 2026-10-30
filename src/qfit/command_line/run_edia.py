"""
run_edia - per-atom EDIA of every null conformer of one chain pocket and of the pocket's
reference ligands (generate_null.sh stage 2a), with EDIAscorer 1.3: one EDIAscorer call per
molecule, against the model cropped around that molecule.

Inputs, in <dataset_dir>/<conformer_run_name>/chain<X>/ (from generate_conformers):
conformers.pdb (one MODEL per kept placement, in the dataset's crystal frame), ligand.pdb (the
template ligand: atom names/order) and conformer_summary.json (SMILES). Also the dataset's
structure (default <dataset_dir>/<dataset>_001.pdb, the multiconformer model; also the source
of the reference ligands) and a
density map (--map, any CCP4 map in the same frame, e.g. <dataset>_2fofc.ccp4), and the
fragments CSV (--fragments-csv, e.g. pxr_fragments.csv: the dataset's unit cell, space group
and resolution).

Steps:
  1. edia_target.pdb: the structure with every LIG residue and every --exclude-resn residue
     (default HOH DMS) removed - the protein context EDIA scores the ligands in, every altloc
     kept (a multiconformer model's apo state contributes density too) - with a CRYST1 record
     from the fragments CSV.
  2. The molecules: every conformer, named conformer_<n> (n = MODEL number in conformers.pdb,
     1, 2, ...), and the chain's reference ligands from the structure: every LIG residue, one
     molecule per altloc, named reference_<resi> (no altlocs) or reference_<resi>_alt<letter>.
     For each, in edia_atom_work/: <name>.sdf (bond orders from the SMILES, RDKit
     AssignBondOrdersFromTemplate on the template ligand; coordinates in the template's atom
     order, references matched by atom name; heavy atoms only) and <name>.pdb, edia_target.pdb
     cropped to the whole residues with an atom within --crop-radius Å of the molecule (the
     whole target if there are none). The map is never cropped: EDIA scales the density by the
     map's sigma. Atoms beyond the crop radius barely change the scores (checked with
     claude_scripts/edia13_crop_survey.py against the whole model, 100 conformers: 6 Å
     identical for all; the default 5 Å identical for 98, the other 2 within 0.01 per atom,
     EDIAm unchanged; 4 Å up to 0.07 per atom, as density is shared up to ~5 Å).
  3. EDIAscorer -t <name>.pdb -d <map> -i <name>.sdf -k 0 -u <resolution> -o <name>/ per
     molecule (resolution from the fragments CSV). EDIAscorer writes per-atom scores of an -i
     molecule only when it scores that one molecule (-k); its EDIAm equals EDIAscorer 1.1's
     batched score. EDIAscorer 1.3 needs glibc >= 2.29, so all of the pocket's calls run in one
     throwaway rootless podman container of --container-image (from the local image store),
     --workers calls at a time, with $HOME mounted at the same path: EDIAscorer, the map, the
     pocket and the ZBH licence (~/.config/ZBH) must be under $HOME. A finished call keeps only
     its scores (<name>.atoms.csv, <name>.structure.csv) and log.
  4. Each call's 'l' rows: the molecule's EDIAm, min/median/max EDIA and atom problem counts
     (structurescores) and one EDIA per heavy atom (atomscores; EDIA names atoms by element and
     1-based SDF index, e.g. C1, N14, and its 'Infile id' is that index). A call fails if
     EDIAscorer exits non-zero or writes no scores, or if its atoms don't match the template's
     (count, indices, elements). Failures are recorded, not fatal: the molecule gets a row with
     status 'failed' and empty scores, and the call's log, SDF and crop go to
     edia_atom_failed/.
  5. Outputs:
     edia_scores.csv: one row per conformer (the null distribution; references not included):
       model, EDIAm, min/median/max EDIA, atom problem counts, mean_atom_edia (the plain mean of
       the per-atom EDIA), n_atoms, status (ok or failed).
     conformers/conformer_<n>.csv: the conformer's per-atom scores, one row per heavy atom:
       atom_index (SDF order), atom_name (the template's), edia_atom_name, element, edia,
       fault_code (EDIA fault analysis), b_factor, occupancy (as EDIA reports them), x, y, z.
     reference_edia.csv: one row per reference (chain, resi, altloc, name, the same fields);
       reference_atom_scores.csv: their per-atom scores (chain, name, resi, altloc, the same
       atom columns).
     edia_params.json: inputs, settings (mode per_atom, crop radius, workers, image), call
       times, failed molecules and their reasons, logged EDIAscorer errors.
Skipped if edia_scores.csv and reference_edia.csv exist and edia_params.json records per-atom
scoring, unless --overwrite (which clears earlier outputs first). A pocket scored per atom with
the same map, structure, excluded residues and crop radius but without reference_edia.csv gets
only its references scored; a pocket scored by an earlier version (batched EDIAscorer 1.1,
EDIAm only) is rescored in full. An interrupted run resumes: edia_atom_work/ keeps every
finished call, with the settings it was made with, until the outputs are written, and a rerun
with the same settings runs only the calls not yet finished (failed calls are retried).

Usage:
  run_edia <dataset_dir> <conformer_run_name> <chain> --map <map.ccp4>
           --fragments-csv <csv> [--structure <pdb>] [--exclude-resn RESN ...]
           [--ediascorer <path>] [--container-image <image>] [--crop-radius <Å>]
           [--workers <n>] [--overwrite]
"""
import argparse
import csv
import io
import json
import os
import re
import shutil
import subprocess
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Geometry import Point3D
from scipy.spatial import cKDTree

LIGAND_RESN = 'LIG'
# Batched EDIAscorer 1.1 (run_ediascorer; no longer used by run_edia or evaluate_structure).
DEFAULT_EDIASCORER = '/home/ngupta/ediascorer_1.1.0/ediascorer'
# Per-atom scoring (run_edia): EDIAscorer 1.3, run inside the container image.
DEFAULT_EDIASCORER_13 = '/home/ngupta/EDIAscorer_1.3.0/EDIAscorer'
DEFAULT_IMAGE = 'registry.access.redhat.com/ubi9/ubi'
DEFAULT_CROP_RADIUS = 5.0
WORK_DIR = 'edia_atom_work'
FAILED_DIR = 'edia_atom_failed'
CONFORMER_DIR = 'conformers'
NAME_RE = re.compile(r'conformer_(\d+)')
# EDIAscorer 1.1 *structurescores.csv columns (from its --help), used when it writes no header.
STRUCTURE_COLUMNS = ['Structure identifier', 'Name', 'Protein position or molecule ID', 'Chain',
                     'EDIAm', 'Min EDIA', 'Median EDIA', 'Max EDIA',
                     '% degree of neighborhood conformity', '# good clusters',
                     '% atoms in good clusters', '# no problem', '# clash',
                     '# not enough density', '# too much density', 'Mean B Factor',
                     '#occupancy below 1']
SUMMARY_FIELDS = {'EDIAm': 'EDIAm', 'Min EDIA': 'min_edia', 'Median EDIA': 'median_edia',
                  'Max EDIA': 'max_edia', '# no problem': 'n_no_problem', '# clash': 'n_clash',
                  '# not enough density': 'n_not_enough_density',
                  '# too much density': 'n_too_much_density'}
# Per-molecule fields of edia_scores.csv / reference_edia.csv.
SCORE_FIELDS = [*SUMMARY_FIELDS.values(), 'mean_atom_edia', 'n_atoms', 'status']
# EDIAscorer 1.3 *atomscores.csv columns kept per atom, and their names in our files.
ATOM_COLUMNS = {'Atom name': 'edia_atom_name', 'Element': 'element', 'EDIA': 'edia',
                'EDIA fault analysis': 'fault_code', 'B factor': 'b_factor',
                'Occupancy': 'occupancy'}
ATOM_FIELDS = ['atom_index', 'atom_name', *ATOM_COLUMNS.values(), 'x', 'y', 'z']
# Rootless podman notices that don't affect the run.
HARMLESS_PODMAN = re.compile(r'subuid|rootless single mapping|Network file system')

# Runs inside the container (bash -s <EDIAscorer> <map> <resolution> <workers>), in the work
# folder: one job per line of jobs.txt, a molecule name. A job keeps only the two score CSVs of
# a finished call, and writes <name>.time (start, end, exit code) last, so a call with a .time
# file is complete. Elapsed times are computed on the host (the image has no bc).
CONTAINER_SCRIPT = r'''
EDIASCORER=$1 MAP=$2 RES=$3 WORKERS=$4
export EDIASCORER MAP RES
job() {
    local t0=$(date +%s.%N)
    "$EDIASCORER" -t "$1.pdb" -d "$MAP" -i "$1.sdf" -k 0 -u "$RES" -o "$1/" > "$1.log" 2>&1
    local status=$?
    local t1=$(date +%s.%N)
    if [ $status -eq 0 ] && cp "$1"/0/*atomscores.csv "$1.atoms.csv" 2>/dev/null \
            && cp "$1"/0/*structurescores.csv "$1.structure.csv" 2>/dev/null; then
        rm -rf "$1"
    fi
    echo "$1 $t0 $t1 $status" > "$1.time.tmp" && mv "$1.time.tmp" "$1.time"
}
export -f job
xargs -P "$WORKERS" -L 1 bash -c 'job "$@"' _ < jobs.txt
'''


# --- Inputs ----------------------------------------------------------------------------------

def crystal_info(csv_path, dataset):
    """{'cell': (a, b, c, alpha, beta, gamma), 'space_group', 'resolution'} from the fragments
    CSV."""
    with open(csv_path, newline='') as f:
        for row in csv.DictReader(f):
            if row['dataset'] == dataset:
                return {'cell': tuple(float(row[k])
                                      for k in ('a', 'b', 'c', 'alpha', 'beta', 'gamma')),
                        'space_group': row['space_group'], 'resolution': float(row['resolution'])}
    raise SystemExit(f'Dataset {dataset} not found in {csv_path}')


def cryst1(info):
    """CRYST1 record; the space group written in Hermann-Mauguin form (P 21 21 21)."""
    sg = info['space_group']
    if ' ' not in sg:  # e.g. P212121 -> P 21 21 21
        sg = re.sub(r'^([PCIFRH])(\d+)(\d\d)(\d\d)$', r'\1 \2 \3 \4', sg)
    a, b, c, alpha, beta, gamma = info['cell']
    return f'CRYST1{a:9.3f}{b:9.3f}{c:9.3f}{alpha:7.2f}{beta:7.2f}{gamma:7.2f} {sg:<11s}{1:4d}\n'


def write_target(structure, out, info, exclude_resn=()):
    """The structure without LIG and exclude_resn atoms, its CRYST1 replaced by one from the
    fragments CSV (other header records kept)."""
    skip = {LIGAND_RESN, *exclude_resn}
    with open(structure) as f, open(out, 'w') as g:
        g.write(cryst1(info))
        for line in f:
            if line.startswith('CRYST1'):
                continue
            if line.startswith(('ATOM', 'HETATM', 'ANISOU')) and line[17:20].strip() in skip:
                continue
            g.write(line)


def read_models(path):
    """[[(x, y, z) per atom] per MODEL] of a multi-MODEL PDB."""
    models, current = [], None
    for line in open(path):
        if line.startswith('MODEL'):
            current = []
        elif line.startswith(('ATOM', 'HETATM')) and current is not None:
            current.append((float(line[30:38]), float(line[38:46]), float(line[46:54])))
        elif line.startswith('ENDMDL') and current is not None:
            models.append(current)
            current = None
    return models


def template_ligand(pocket_dir):
    """(RDKit mol with bond orders from the SMILES, its atom names): the template ligand, the
    first MODEL of ligand.pdb - the atom order of every conformer in conformers.pdb."""
    summary = json.loads((pocket_dir / 'conformer_summary.json').read_text())
    template_block = ''.join(l for l in open(pocket_dir / 'ligand.pdb')
                             if not l.startswith('MODEL'))
    template_block = template_block.split('ENDMDL')[0]  # first reference ligand only
    names = [l[12:16].strip() for l in template_block.splitlines()
             if l.startswith(('ATOM', 'HETATM'))]
    template = Chem.MolFromPDBBlock(template_block, removeHs=True)
    mol = AllChem.AssignBondOrdersFromTemplate(Chem.MolFromSmiles(summary['ligand']['smiles']),
                                               template)
    if mol.GetNumAtoms() != len(names):
        raise SystemExit(f'{pocket_dir}/ligand.pdb: {len(names)} atoms, {mol.GetNumAtoms()} in '
                         f'the RDKit template')
    return mol, names


def reference_ligands(structure, chain, names, resn=LIGAND_RESN):
    """[(name, resi, altloc, xyz)]: every conformation of every resn residue of the chain in the
    structure (heavy atoms; the blank-altloc atoms plus, for each altloc letter, that altloc's
    atoms - one reference per letter, altloc '' if there are none), xyz in the template's atom
    order. Named reference_<resi> or reference_<resi>_alt<letter> in the SDF. A residue whose
    atom names differ from the template's is skipped with a warning."""
    residues = {}
    for line in open(structure):
        if (not line.startswith(('ATOM', 'HETATM')) or line[17:20].strip() != resn
                or line[21].strip() != chain):
            continue
        element = line[76:78].strip() or line[12:16].strip()[0]
        if element.upper() in ('H', 'D'):
            continue
        resi = line[22:27].strip()
        xyz = (float(line[30:38]), float(line[38:46]), float(line[46:54]))
        residues.setdefault(resi, {}).setdefault(line[16], {})[line[12:16].strip()] = xyz
    out = []
    for resi, altlocs in residues.items():
        blank = altlocs.get(' ', {})
        letters = sorted(a for a in altlocs if a != ' ')
        confs = [(a, {**blank, **altlocs[a]}) for a in letters] or [('', blank)]
        for altloc, atoms in confs:
            if set(atoms) != set(names):
                print(f'Warning: {resn} {chain}{resi}{" alt " + altloc if altloc else ""} '
                      f'atom names differ from the template ligand\'s '
                      f'(missing {sorted(set(names) - set(atoms))}, extra '
                      f'{sorted(set(atoms) - set(names))}); not scored')
                continue
            name = f'reference_{resi}' + (f'_alt{altloc}' if altloc else '')
            out.append((name, resi, altloc, [atoms[n] for n in names]))
    return out


def write_sdf(mol, out, molecules):
    """SDF of molecules [(name, xyz in the template's atom order)], bond orders from mol."""
    writer = Chem.SDWriter(str(out))
    for name, xyz in molecules:
        if len(xyz) != mol.GetNumAtoms():
            raise SystemExit(f'{name} has {len(xyz)} atoms, the template ligand '
                             f'{mol.GetNumAtoms()}')
        m = Chem.Mol(mol)
        conf = m.GetConformer()
        for i, (x, y, z) in enumerate(xyz):
            conf.SetAtomPosition(i, Point3D(x, y, z))
        m.SetProp('_Name', name)
        writer.write(m)
    writer.close()


# --- EDIA output -----------------------------------------------------------------------------

def read_edia_csv(path):
    """(header lines, rows): rows are lists of fields; header lines are EDIA's own lines before
    the data (kept verbatim for the per-conformer files). '#' section lines are dropped from
    the rows but noted on each row as its section."""
    text = Path(path).read_text()
    delimiter = csv.Sniffer().sniff(text[:4096], delimiters=',;\t').delimiter
    header, rows, section = [], [], ''
    for line in text.splitlines():
        if not line.strip():
            continue
        if line.startswith('#'):
            section = line.strip('#').strip().lower()
            continue
        fields = next(csv.reader(io.StringIO(line), delimiter=delimiter))
        if not rows and not header and not _numeric_row(fields):
            header.append(line)
            continue
        rows.append((section, fields))
    return header, rows, delimiter


def _numeric_row(fields):
    return any(re.fullmatch(r'-?\d+(\.\d+)?', f.strip()) for f in fields[1:])


def ligand_groups(rows):
    """{conformer number: [rows]} of the rows EDIA names conformer_<n> (the SDF molecule names;
    the 'l' rows). Other molecules of the target, e.g. DMS ('m' rows), are left out."""
    groups = defaultdict(list)
    for _, fields in rows:
        match = next((m for f in fields if (m := NAME_RE.search(f))), None)
        if match:
            groups[int(match.group(1))].append(fields)
    return groups


def assign(groups, n_conformers, what):
    """groups, checked to hold exactly conformers 1..n_conformers."""
    if sorted(groups) != list(range(1, n_conformers + 1)):
        raise SystemExit(f'{what}: EDIA output names {len(groups)} conformers, '
                         f'{n_conformers} given (see edia_log.txt)')
    return dict(groups)


def reference_groups(rows, names):
    """{reference name: [rows]} of the rows EDIA names after the given reference molecules,
    checked to hold each exactly once."""
    groups = defaultdict(list)
    wanted = set(names)
    for _, fields in rows:
        name = next((f.strip() for f in fields if f.strip() in wanted), None)
        if name:
            groups[name].append(fields)
    if set(groups) != wanted or any(len(v) != 1 for v in groups.values()):
        raise SystemExit(f'EDIA output names {sorted(groups)} of the reference ligands '
                         f'{sorted(wanted)}, each expected once (see the EDIA log)')
    return dict(groups)


def summary_row(fields, columns):
    """SUMMARY_FIELDS of one EDIA structurescores row."""
    return {name: fields[columns.index(column)].strip() if column in columns else ''
            for column, name in SUMMARY_FIELDS.items()}


def run_ediascorer(pocket, sdf, raw, log, density, info, ediascorer, target='edia_target.pdb'):
    """Runs EDIAscorer on sdf (-i) in the context of target (in pocket), or, with sdf None, on
    target's own residues and molecules only; output in pocket/raw/. Returns (command,
    structurescores rows, their columns, logged errors, structurescores file name)."""
    command = [ediascorer, '-t', target, '-d', str(density),
               *(['-i', sdf] if sdf else []), '-u', str(info['resolution']), '-o', f'{raw}/']
    (pocket / raw).mkdir()  # EDIAscorer doesn't create its output folder
    with open(pocket / log, 'w') as f:
        status = subprocess.run(command, cwd=pocket, stdout=f, stderr=subprocess.STDOUT).returncode
    # EDIAscorer logs some harmless 'ERROR' lines (e.g. 'Differing atom name: C1 vs.' for some
    # ligands, with complete output), and can exit 0 after a fatal one (e.g. an unwritable
    # output folder), so failure is judged by the exit code and the output CSVs below; logged
    # errors are kept as warnings.
    log_errors = [l.strip() for l in open(pocket / log) if 'ERROR' in l]
    if status != 0:
        raise SystemExit(f'ediascorer failed (exit code {status}; {log_errors[:3]}; see '
                         f'{pocket / log})')
    if log_errors:
        print(f'EDIAscorer logged (output checked below): {log_errors[:3]}')
    structure_csvs = sorted((pocket / raw).glob('*structurescores.csv'))
    atom_csvs = sorted((pocket / raw).glob('*atomscores.csv'))
    if len(structure_csvs) != 1 or len(atom_csvs) != 1:
        raise SystemExit(f'Expected one *structurescores.csv and one *atomscores.csv in '
                         f'{pocket / raw}, found {[f.name for f in structure_csvs + atom_csvs]} '
                         f'(EDIAscorer logged: {log_errors[:3]})')
    header, rows, delimiter = read_edia_csv(structure_csvs[0])
    columns = (next(csv.reader([header[-1]], delimiter=delimiter)) if header
               else STRUCTURE_COLUMNS)
    return command, rows, [c.strip() for c in columns], log_errors, structure_csvs[0].name



# --- Per-atom scoring (EDIAscorer 1.3, one -k call per molecule) ----------------------------

class Cropper:
    """A target PDB cropped around molecules: the whole residues with an atom within radius of
    any of the molecule's atoms. Other records are kept, TER records dropped."""

    def __init__(self, target, radius):
        self.lines = [l for l in open(target) if not l.startswith('TER')]
        self.radius = radius
        self.keys, xyz = [], []
        for l in self.lines:
            if l.startswith(('ATOM', 'HETATM')):
                self.keys.append((l[21], l[22:27]))
                xyz.append((float(l[30:38]), float(l[38:46]), float(l[46:54])))
        self.tree = cKDTree(np.array(xyz))

    def write(self, xyz, out):
        """Writes the crop around xyz to out; returns the number of residues kept (0: none,
        nothing written)."""
        hits = self.tree.query_ball_point(np.array(xyz), self.radius)
        keep = {self.keys[i] for h in hits for i in h}
        if keep:
            with open(out, 'w') as f:
                for l in self.lines:
                    if (l.startswith(('ATOM', 'HETATM', 'ANISOU'))
                            and (l[21], l[22:27]) not in keep):
                        continue
                    f.write(l)
        return len(keep)


def call_files(work, name):
    """Every file and folder a call of molecule name makes in work."""
    return [work / f'{name}{suffix}' for suffix in
            ('', '.sdf', '.pdb', '.log', '.atoms.csv', '.structure.csv', '.time', '.time.tmp')]


def finished(work, name):
    """True if name's call completed with exit code 0 and kept its score CSVs."""
    time_file = work / f'{name}.time'
    return (time_file.exists() and time_file.read_text().split()[3] == '0'
            and (work / f'{name}.atoms.csv').exists()
            and (work / f'{name}.structure.csv').exists())


def run_container(work, names, ediascorer, image, density, resolution, workers):
    """Runs the calls of names (their .sdf and .pdb written in work) in one podman container,
    workers at a time, printing progress every 10%."""
    home = os.environ['HOME']
    for path in (ediascorer, density, work):
        if not str(Path(path).resolve()).startswith(home.rstrip('/') + '/'):
            raise SystemExit(f'{path} is not under $HOME ({home}), the only folder mounted in '
                             f'the container')
    (work / 'jobs.txt').write_text(''.join(f'{n}\n' for n in names))
    command = ['podman', 'run', '--rm', '-i', '-e', f'HOME={home}', '-v', f'{home}:{home}',
               '-w', str(work), image, 'bash', '-s', str(ediascorer), str(density),
               str(resolution), str(workers)]
    print(f'Running {len(names)} EDIAscorer calls, {workers} at a time, in one {image} '
          f'container', flush=True)
    start, reported = time.time(), 0
    with open(work / 'podman_stderr.txt', 'w') as err:
        proc = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=err, stderr=err,
                                text=True)
        proc.stdin.write(CONTAINER_SCRIPT)
        proc.stdin.close()
        while True:
            try:
                proc.wait(timeout=60)
                break
            except subprocess.TimeoutExpired:
                pass
            done = sum((work / f'{n}.time').exists() for n in names)
            if done * 10 // len(names) > reported:
                reported = done * 10 // len(names)
                elapsed = time.time() - start
                print(f'  {done}/{len(names)} calls in {elapsed / 60:.0f} min, about '
                      f'{elapsed / done * (len(names) - done) / 60:.0f} min left', flush=True)
    problems = [l.strip() for l in open(work / 'podman_stderr.txt')
                if l.strip() and not HARMLESS_PODMAN.search(l)]
    if proc.returncode != 0:
        raise SystemExit(f'podman run failed (exit code {proc.returncode}): {problems[:5]} '
                         f'(see {work / "podman_stderr.txt"})')
    if problems:
        print(f'Container stderr: {problems[:5]}')


def ligand_rows(path):
    """(columns, the 'l' rows) of an EDIAscorer 1.3 score CSV (header line first)."""
    header, rows, delimiter = read_edia_csv(path)
    if not header:
        return [], []
    columns = [c.strip() for c in next(csv.reader([header[-1]], delimiter=delimiter))]
    return columns, [f for _, f in rows if f[0].strip() == 'l']


def parse_call(work, name, names, symbols, xyz):
    """(result, None) or (None, failure reason) of name's finished call. result: {'summary':
    SCORE_FIELDS but status, 'atoms': [ATOM_FIELDS rows], 'seconds'}; names, symbols: the
    template's atom names and element symbols; xyz: the molecule's coordinates."""
    time_file = work / f'{name}.time'
    if not time_file.exists():
        return None, 'not run'
    _, t0, t1, status = time_file.read_text().split()
    if status != '0':
        return None, f'EDIAscorer exit code {status}'
    if not finished(work, name):
        return None, 'no score files'
    s_columns, s_rows = ligand_rows(work / f'{name}.structure.csv')
    a_columns, a_rows = ligand_rows(work / f'{name}.atoms.csv')
    structure = [f for f in s_rows if 'Name' in s_columns
                 and f[s_columns.index('Name')].strip() == name]
    if len(structure) != 1:
        return None, f'{len(structure)} structure score rows'
    missing = [c for c in ('Infile id', 'Substructure name', *ATOM_COLUMNS) if c not in a_columns]
    if missing:
        return None, f'atom score columns missing: {missing}'
    col = {c: a_columns.index(c) for c in ('Infile id', 'Substructure name', *ATOM_COLUMNS)}
    atoms = sorted((f for f in a_rows if f[col['Substructure name']].strip() == name),
                   key=lambda f: int(f[col['Infile id']]))
    if [int(f[col['Infile id']]) for f in atoms] != list(range(1, len(names) + 1)):
        return None, f'{len(atoms)} atom score rows, {len(names)} heavy atoms in the template'
    rows = []
    for i, f in enumerate(atoms):
        row = {field: f[col[column]].strip() for column, field in ATOM_COLUMNS.items()}
        if re.sub(r'\d+$', '', row['edia_atom_name']).upper() != symbols[i].upper():
            return None, (f'atom {i + 1} is {row["edia_atom_name"]} in EDIA, '
                          f'{symbols[i]} in the template')
        if row['edia'] == '-0':
            row['edia'] = '0'
        x, y, z = xyz[i]
        rows.append({'atom_index': i + 1, 'atom_name': names[i], **row,
                     'x': f'{x:.3f}', 'y': f'{y:.3f}', 'z': f'{z:.3f}'})
    summary = summary_row(structure[0], s_columns)
    summary['mean_atom_edia'] = round(float(np.mean([float(r['edia']) for r in rows])), 4)
    summary['n_atoms'] = len(rows)
    return {'summary': summary, 'atoms': rows, 'seconds': float(t1) - float(t0)}, None


def score_molecules(pocket, molecules, mol, names, density, info, args, settings):
    """Per-atom EDIA of molecules [(name, xyz)], one call each (module docstring, steps 2-4),
    resuming from pocket/edia_atom_work/ if it was made with the same settings. Returns
    ({name: result}, {name: failure reason}, call stats); failed calls' files are copied to
    pocket/edia_atom_failed/. The work folder is left for the caller to remove once the outputs
    are written."""
    work = pocket / WORK_DIR
    settings = json.loads(json.dumps(settings))
    settings_file = work / 'settings.json'
    if work.exists() and (not settings_file.exists()
                          or json.loads(settings_file.read_text()) != settings):
        print(f'{work} was made with other settings; starting it over')
        shutil.rmtree(work)
    work.mkdir(exist_ok=True)
    settings_file.write_text(json.dumps(settings, indent=2) + '\n')

    cropper = Cropper(pocket / 'edia_target.pdb', args.crop_radius)
    todo, uncropped = [], 0
    for name, xyz in molecules:
        if finished(work, name):
            continue
        for old in call_files(work, name):
            if old.is_dir():
                shutil.rmtree(old)
            else:
                old.unlink(missing_ok=True)
        write_sdf(mol, work / f'{name}.sdf', [(name, xyz)])
        if not cropper.write(xyz, work / f'{name}.pdb'):
            shutil.copy(pocket / 'edia_target.pdb', work / f'{name}.pdb')
            uncropped += 1
        todo.append(name)
    resumed = len(molecules) - len(todo)
    if resumed:
        print(f'Resuming: {resumed} of {len(molecules)} calls already finished in {work}')
    if uncropped:
        print(f'Warning: {uncropped} molecules have no target atom within {args.crop_radius} Å; '
              f'scored against the whole target')
    if todo:
        run_container(work, todo, args.ediascorer, args.container_image, density,
                      info['resolution'], args.workers)

    symbols = [a.GetSymbol() for a in mol.GetAtoms()]
    results, failures, log_errors = {}, {}, Counter()
    ran = set(todo)
    for name, xyz in molecules:
        result, reason = parse_call(work, name, names, symbols, xyz)
        if (work / f'{name}.log').exists():
            log_errors.update(l.strip() for l in open(work / f'{name}.log') if 'ERROR' in l)
        if result:
            results[name] = result
            continue
        failures[name] = reason
        failed_dir = pocket / FAILED_DIR
        failed_dir.mkdir(exist_ok=True)
        for f in call_files(work, name):
            if f.is_dir():
                shutil.copytree(f, failed_dir / f.name, dirs_exist_ok=True)
            elif f.exists():
                shutil.copy(f, failed_dir / f.name)
    if failures:
        print(f'Warning: {len(failures)} of {len(molecules)} calls failed (kept in '
              f'{pocket / FAILED_DIR}): '
              + '; '.join(f'{n}: {r}' for n, r in list(failures.items())[:5]))
    seconds = [r['seconds'] for n, r in results.items() if n in ran]
    stats = {'n_calls_run': len(todo), 'n_calls_resumed': resumed,
             'n_uncropped': uncropped,
             'call_s_median': round(float(np.median(seconds)), 2) if seconds else None,
             'call_s_total': round(float(np.sum(seconds)), 1),
             'ediascorer_log_errors': dict(log_errors.most_common(10))}
    return results, failures, stats


def score_row(result):
    """SCORE_FIELDS of a molecule: its summary and status ok, or status failed (no result)."""
    return {**result['summary'], 'status': 'ok'} if result else {'status': 'failed'}


def write_atom_csv(path, rows, extra=None):
    """ATOM_FIELDS rows (preceded by the extra fields' keys, if any) to path, written under a
    temporary name and then renamed so a partial file never has the final name."""
    tmp = path.with_name(path.name + '.tmp')
    with open(tmp, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[*(extra or {}), *ATOM_FIELDS])
        writer.writeheader()
        for row in rows:
            writer.writerow({**(extra or {}), **row})
    tmp.replace(path)


def write_reference_outputs(pocket, chain, references, results):
    """reference_edia.csv and reference_atom_scores.csv."""
    with open(pocket / 'reference_edia.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['chain', 'resi', 'altloc', 'name', *SCORE_FIELDS])
        writer.writeheader()
        for name, resi, altloc, _ in references:
            writer.writerow({'chain': chain, 'resi': resi, 'altloc': altloc, 'name': name,
                             **score_row(results.get(name))})
    rows = []
    for name, resi, altloc, _ in references:
        if name in results:
            rows += [{'chain': chain, 'name': name, 'resi': resi, 'altloc': altloc, **a}
                     for a in results[name]['atoms']]
    write_atom_csv(pocket / 'reference_atom_scores.csv', rows,
                   {'chain': '', 'name': '', 'resi': '', 'altloc': ''})


def reference_params(references, failures):
    return {'n_references': len(references),
            'references': [{'name': n, 'resi': r, 'altloc': a} for n, r, a, _ in references],
            'failed_references': sorted(n for n, *_ in references if n in failures)}


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('dataset_dir')
    p.add_argument('conformer_run_name')
    p.add_argument('chain')
    p.add_argument('--map', required=True, help='Density map (CCP4) in the structure\'s frame.')
    p.add_argument('--fragments-csv', required=True,
                   help='CSV with dataset, resolution, a, b, c, alpha, beta, gamma, space_group.')
    p.add_argument('--structure', help='Default: <dataset_dir>/<dataset>_001.pdb.')
    p.add_argument('--exclude-resn', nargs='+', default=['HOH', 'DMS'],
                   help='Residue names removed from the EDIA target besides LIG (always '
                        'removed). Default: HOH DMS.')
    p.add_argument('--ediascorer', default=DEFAULT_EDIASCORER_13,
                   help=f'EDIAscorer 1.3 executable, run in the container. Default: '
                        f'{DEFAULT_EDIASCORER_13}.')
    p.add_argument('--container-image', default=DEFAULT_IMAGE,
                   help=f'Podman image EDIAscorer runs in (glibc >= 2.29), from the local '
                        f'image store. Default: {DEFAULT_IMAGE}.')
    p.add_argument('--crop-radius', type=float, default=DEFAULT_CROP_RADIUS,
                   help=f'Each molecule is scored against the target\'s residues with an atom '
                        f'within this distance (Å) of it. Default: {DEFAULT_CROP_RADIUS:g}.')
    p.add_argument('--workers', type=int, default=1,
                   help='EDIAscorer calls run at once (cores used). Default: 1.')
    p.add_argument('--overwrite', action='store_true')
    args = p.parse_args()

    dataset_dir = Path(args.dataset_dir).resolve()
    dataset = dataset_dir.name
    pocket = dataset_dir / args.conformer_run_name / f'chain{args.chain}'
    structure = Path(args.structure or dataset_dir / f'{dataset}_001.pdb').resolve()
    density = Path(args.map).resolve()
    for f in (pocket / 'conformers.pdb', structure, density, Path(args.ediascorer)):
        if not f.exists():
            raise SystemExit(f'Not found: {f}')
    scores, reference_csv = pocket / 'edia_scores.csv', pocket / 'reference_edia.csv'
    excluded = sorted({LIGAND_RESN, *args.exclude_resn})
    params_file = pocket / 'edia_params.json'
    params = json.loads(params_file.read_text()) if params_file.exists() else {}
    per_atom = params.get('mode') == 'per_atom'
    if scores.exists() and reference_csv.exists() and per_atom and not args.overwrite:
        print(f'Skipping {pocket}: edia_scores.csv and reference_edia.csv exist (pass '
              f'--overwrite to redo)')
        return
    # Conformers already scored per atom with these inputs, but no references: score only the
    # references (each molecule is scored alone, so this matches a full run).
    backfill = (scores.exists() and not args.overwrite and per_atom
                and params.get('map') == str(density)
                and params.get('structure') == str(structure)
                and params.get('excluded_resn') == excluded
                and params.get('crop_radius') == args.crop_radius)
    if scores.exists() and not backfill and not args.overwrite:
        print(f'{pocket}: edia_scores.csv was scored '
              + ('with other inputs' if per_atom else 'by an earlier version (EDIAm only)')
              + '; rescoring everything per atom')
    if subprocess.run(['podman', 'image', 'exists', args.container_image]).returncode != 0:
        raise SystemExit(f'{args.container_image} is not in the local podman image store '
                         f'(podman pull it first)')

    start = time.time()
    info = crystal_info(args.fragments_csv, dataset)
    mol, names = template_ligand(pocket)
    references = reference_ligands(structure, args.chain, names)
    if not references:
        print(f'Warning: no {LIGAND_RESN} residues of chain {args.chain} in {structure.name} '
              f'could be scored; reference_edia.csv will be empty')
    reference_molecules = [(r[0], r[3]) for r in references]
    write_target(structure, pocket / 'edia_target.pdb', info, args.exclude_resn)
    conformers_pdb = pocket / 'conformers.pdb'
    settings = {'map': str(density), 'structure': str(structure), 'excluded_resn': excluded,
                'crop_radius': args.crop_radius, 'ediascorer': str(args.ediascorer),
                'resolution': info['resolution'],
                'conformers_pdb': [conformers_pdb.stat().st_size, conformers_pdb.stat().st_mtime]}
    if args.overwrite and (pocket / WORK_DIR).exists():
        shutil.rmtree(pocket / WORK_DIR)
    command = [str(args.ediascorer), '-t', '<name>.pdb', '-d', str(density), '-i', '<name>.sdf',
               '-k', '0', '-u', str(info['resolution']), '-o', '<name>/']
    run_settings = {'mode': 'per_atom', 'ediascorer': str(args.ediascorer),
                    'container_image': args.container_image, 'crop_radius': args.crop_radius,
                    'workers': args.workers, 'command': command}

    if backfill:
        results, failures, stats = score_molecules(pocket, reference_molecules, mol, names,
                                                   density, info, args, settings)
        write_reference_outputs(pocket, args.chain, references, results)
        params.update(reference_params(references, failures))
        params['reference_calls'] = stats
        params['reference_runtime_s'] = round(time.time() - start, 1)
        params_file.write_text(json.dumps(params, indent=2) + '\n')
        shutil.rmtree(pocket / WORK_DIR)
        print(f'{dataset} chain {args.chain}: per-atom EDIA of {len(references)} reference '
              f'ligands ({len(failures)} failed; conformers already scored) in '
              f'{time.time() - start:.0f} s -> {reference_csv}')
        return

    # Full scoring. Clears earlier outputs first (edia_raw*, conformers.sdf, references.sdf and
    # their logs: the batched EDIAscorer 1.1 version; conformer_<n>/: per-conformer splits of
    # an earlier version), but not the work folder, which resumes an interrupted run.
    for old in [pocket / 'edia_raw', pocket / 'edia_raw_reference', pocket / CONFORMER_DIR,
                pocket / FAILED_DIR,
                *(d for d in pocket.glob('conformer_*') if NAME_RE.fullmatch(d.name))]:
        if old.is_dir():
            shutil.rmtree(old)
    for old in (scores, reference_csv, pocket / 'reference_atom_scores.csv',
                pocket / 'conformers.sdf', pocket / 'references.sdf', pocket / 'edia_log.txt',
                pocket / 'edia_reference_log.txt'):
        old.unlink(missing_ok=True)
    models = read_models(conformers_pdb)
    n_conformers = len(models)
    molecules = ([(f'conformer_{n}', xyz) for n, xyz in enumerate(models, start=1)]
                 + reference_molecules)
    results, failures, stats = score_molecules(pocket, molecules, mol, names, density, info,
                                               args, settings)

    (pocket / CONFORMER_DIR).mkdir()
    with open(scores, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['model', *SCORE_FIELDS])
        writer.writeheader()
        for n in range(1, n_conformers + 1):
            result = results.get(f'conformer_{n}')
            writer.writerow({'model': n, **score_row(result)})
            if result:
                write_atom_csv(pocket / CONFORMER_DIR / f'conformer_{n}.csv', result['atoms'])
    write_reference_outputs(pocket, args.chain, references, results)
    failed_models = sorted(int(NAME_RE.fullmatch(n).group(1)) for n in failures
                           if NAME_RE.fullmatch(n))
    params_file.write_text(json.dumps({
        'dataset': dataset, 'chain': args.chain, 'conformer_run_name': args.conformer_run_name,
        **run_settings, 'map': str(density), 'structure': str(structure),
        'fragments_csv': str(Path(args.fragments_csv).resolve()), **info,
        'excluded_resn': excluded, 'n_conformers': n_conformers,
        'failed_models': failed_models, **reference_params(references, failures),
        'failed': failures, **stats, 'runtime_s': round(time.time() - start, 1)},
        indent=2) + '\n')
    shutil.rmtree(pocket / WORK_DIR)
    print(f'{dataset} chain {args.chain}: per-atom EDIA of {n_conformers} conformers and '
          f'{len(references)} reference ligands ({len(failures)} failed) in '
          f'{time.time() - start:.0f} s -> {scores.name}, {CONFORMER_DIR}/, '
          f'{reference_csv.name}')


if __name__ == '__main__':
    main()
