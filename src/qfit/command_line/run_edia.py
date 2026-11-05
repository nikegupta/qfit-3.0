"""
run_edia - per-atom EDIA (EDIAscorer 1.3) of every ligand in a PDB file - one model or many,
e.g. a null distribution's conformers.pdb or a single reference ligand.pdb - against a protein
target and a density map: one EDIAscorer call per molecule, against the target cropped around
that molecule. Used by generate_null.sh stage 2a, but independent of its folder layout.

Inputs:
  ligand_pdb   The molecules to score. Each MODEL (a file without MODEL records is model 1),
               each residue (chain, resn, resi) and each altloc conformation of it (the
               residue's blank-altloc atoms plus one altloc letter's; just the blank-altloc
               atoms if it has no letters) is one molecule, heavy atoms only. --resn limits
               the residues scored.
  target_pdb   The protein context, e.g. a dataset's <dataset>_001.pdb. Its --exclude-resn
               residues (default LIG HOH DMS) are removed; every other atom and altloc is
               kept (a multiconformer model's apo state contributes density too). Its own
               CRYST1 and REMARK records are not used.
  --cell, --space-group, --resolution
               The unit cell (a b c alpha beta gamma), space group and resolution, always
               given explicitly (generate_null.sh reads them from the fragments CSV), as PDB
               writers are inconsistent about CRYST1 and REMARK 3. The target's CRYST1 is
               replaced by one built from them.
  map          Density map (CCP4) in the target's frame. Never cropped: EDIA scales the density
               by the map's sigma.

Steps:
  1. The stripped target, edia_target.pdb, in the work folder, with a CRYST1 from --cell and
     --space-group.
  2. Each molecule's SDF: bonds from its coordinates (RDKit proximity bonding) and, with
     --smiles, bond orders from the SMILES (RDKit AssignBondOrdersFromTemplate); one template
     per distinct list of atom names, built from the first molecule with it, so every model
     of a conformer file shares one topology and atom order. Atom names need not match across
     files. Next to it, the target cropped to the whole residues with an atom within
     --crop-radius Å of the molecule (the whole target if there are none). Atoms beyond the
     crop radius barely change the scores (claude_scripts/edia13_crop_survey.py, 100
     conformers: 6 Å identical for all; the default 5 Å identical for 98, the other 2 within
     0.01 per atom, EDIAm unchanged).
  3. EDIAscorer -t <name>.pdb -d <map> -i <name>.sdf -k 0 -u <resolution> -o <name>/ per
     molecule. EDIAscorer 1.3 needs glibc >= 2.29, so all calls run in one throwaway rootless
     podman container of --container-image (from the local image store), --workers calls at a
     time, with $HOME mounted at the same path: EDIAscorer, the map, the output folder and the
     ZBH licence (~/.config/ZBH) must be under $HOME. A finished call keeps only its scores and
     log.
  4. Each call's 'l' rows: the molecule's EDIAm, min/median/max EDIA and atom problem counts
     (structurescores) and one EDIA per heavy atom (atomscores; EDIA names atoms by element and
     1-based SDF index, e.g. C1, N14). A call fails if EDIAscorer exits non-zero or writes no
     scores, or if its atoms don't match the template's (count, indices, elements). Failures
     are recorded, not fatal: the molecule gets a row with status 'failed' and empty scores,
     and the call's log, SDF and crop go to the failed folder.

Outputs, in output_dir (names set by --residue-scores, --atom-scores and --params):
  residue scores (default residue_scores.csv): one row per molecule: model, chain, resn, resi,
    altloc, EDIAm, min/median/max EDIA, atom problem counts, mean_atom_edia (the plain mean of
    the per-atom EDIA), n_atoms, status (ok or failed).
  atom scores (default atom_scores.csv): one row per heavy atom of each scored molecule:
    model, chain, resn, resi, altloc, atom_index (SDF order), atom_name, edia_atom_name,
    element, edia, fault_code (EDIA fault analysis), b_factor, occupancy (as EDIA reports
    them), x, y, z.
  params (default <residue scores stem>_params.json): inputs, settings (mode per_atom),
    call times, failed molecules and their reasons, logged EDIAscorer errors.
  <residue scores stem>_edia_failed/: the files of failed calls, if any.
Skipped if both CSVs exist and the params JSON records a per-atom run with the same inputs and
settings, unless --overwrite. An interrupted run resumes: the work folder,
<residue scores stem>_edia_work/, keeps every finished call, with the settings it was made
with, until the outputs are written; a rerun with the same settings runs only the calls not
yet finished (failed calls are retried). Two runs with different residue-score names can share
one output_dir.

Usage:
  run_edia <ligand_pdb> <target_pdb> <map.ccp4> <output_dir>
           --cell <a> <b> <c> <alpha> <beta> <gamma> --space-group <sg> --resolution <Å>
           [--residue-scores <csv>] [--atom-scores <csv>] [--params <json>] [--smiles <smiles>]
           [--resn RESN ...] [--exclude-resn RESN ...]
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
# Batched EDIAscorer 1.1 (run_ediascorer; no longer used by run_edia).
DEFAULT_EDIASCORER = '/home/ngupta/ediascorer_1.1.0/ediascorer'
# Per-atom scoring (run_edia): EDIAscorer 1.3, run inside the container image.
DEFAULT_EDIASCORER_13 = '/home/ngupta/EDIAscorer_1.3.0/EDIAscorer'
DEFAULT_IMAGE = 'registry.access.redhat.com/ubi9/ubi'
DEFAULT_CROP_RADIUS = 5.0
WORK_DIR = 'edia_atom_work'
FAILED_DIR = 'edia_atom_failed'
CONFORMER_DIR = 'conformers'  # per-conformer atom CSVs of earlier versions (cleared on rescoring)
CONFORMER_ATOM_CSV = 'conformer_atom_scores.csv'
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
    first MODEL of conformers.pdb - every conformer has the same atom names and order."""
    summary = json.loads((pocket_dir / 'conformer_summary.json').read_text())
    template_block = ''.join(l for l in open(pocket_dir / 'conformers.pdb')
                             if not l.startswith('MODEL'))
    template_block = template_block.split('ENDMDL')[0]  # first conformer only
    names = [l[12:16].strip() for l in template_block.splitlines()
             if l.startswith(('ATOM', 'HETATM'))]
    template = Chem.MolFromPDBBlock(template_block, removeHs=True)
    mol = AllChem.AssignBondOrdersFromTemplate(Chem.MolFromSmiles(summary['ligand']['smiles']),
                                               template)
    if mol.GetNumAtoms() != len(names):
        raise SystemExit(f'{pocket_dir}/conformers.pdb: {len(names)} atoms, '
                         f'{mol.GetNumAtoms()} in the RDKit template')
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


def score_molecules(pocket, molecules, mol, names, density, info, args, settings, work=None,
                    failed_dir=None, target=None):
    """Per-atom EDIA of molecules [(name, xyz)] or [(name, xyz, mol, names)] (a molecule's own
    template; else mol and names), one call each (module docstring, steps 2-4), resuming from
    work (default pocket/edia_atom_work/) if it was made with the same settings; each molecule
    is scored against target (default pocket/edia_target.pdb) cropped around it. Returns
    ({name: result}, {name: failure reason}, call stats); failed calls' files are copied to
    failed_dir (default pocket/edia_atom_failed/). The work folder is left for the caller to
    remove once the outputs are written."""
    work = Path(work) if work else pocket / WORK_DIR
    failed_dir = Path(failed_dir) if failed_dir else pocket / FAILED_DIR
    target = Path(target) if target else pocket / 'edia_target.pdb'
    molecules = [m if len(m) == 4 else (*m, mol, names) for m in molecules]
    settings = json.loads(json.dumps(settings))
    settings_file = work / 'settings.json'
    if work.exists() and (not settings_file.exists()
                          or json.loads(settings_file.read_text()) != settings):
        print(f'{work} was made with other settings; starting it over')
        shutil.rmtree(work)
    work.mkdir(exist_ok=True)
    settings_file.write_text(json.dumps(settings, indent=2) + '\n')

    cropper = Cropper(target, args.crop_radius)
    todo, uncropped = [], 0
    for name, xyz, m_mol, _ in molecules:
        if finished(work, name):
            continue
        for old in call_files(work, name):
            if old.is_dir():
                shutil.rmtree(old)
            else:
                old.unlink(missing_ok=True)
        write_sdf(m_mol, work / f'{name}.sdf', [(name, xyz)])
        if not cropper.write(xyz, work / f'{name}.pdb'):
            shutil.copy(target, work / f'{name}.pdb')
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

    results, failures, log_errors = {}, {}, Counter()
    ran = set(todo)
    for name, xyz, m_mol, m_names in molecules:
        symbols = [a.GetSymbol() for a in m_mol.GetAtoms()]
        result, reason = parse_call(work, name, m_names, symbols, xyz)
        if (work / f'{name}.log').exists():
            log_errors.update(l.strip() for l in open(work / f'{name}.log') if 'ERROR' in l)
        if result:
            results[name] = result
            continue
        failures[name] = reason
        failed_dir.mkdir(exist_ok=True)
        for f in call_files(work, name):
            if f.is_dir():
                shutil.copytree(f, failed_dir / f.name, dirs_exist_ok=True)
            elif f.exists():
                shutil.copy(f, failed_dir / f.name)
    if failures:
        print(f'Warning: {len(failures)} of {len(molecules)} calls failed (kept in '
              f'{failed_dir}): '
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



# --- General scoring (run_edia's command line) ------------------------------------------------

MOLECULE_FIELDS = ['model', 'chain', 'resn', 'resi', 'altloc']
DEFAULT_EXCLUDE_RESN = [LIGAND_RESN, 'HOH', 'DMS']


def write_stripped_target(target, out, exclude_resn, info):
    """target without its exclude_resn atoms (ATOM, HETATM, ANISOU), its CRYST1 replaced by
    one from info ({'cell', 'space_group'}); every other record kept."""
    skip = set(exclude_resn)
    with open(target) as f, open(out, 'w') as g:
        g.write(cryst1(info))
        for line in f:
            if line.startswith('CRYST1'):
                continue
            if line.startswith(('ATOM', 'HETATM', 'ANISOU')) and line[17:20].strip() in skip:
                continue
            g.write(line)


def ligand_molecules(path, resn=None):
    """The molecules of a ligand PDB, in file order: [{'model', 'chain', 'resn', 'resi',
    'altloc', 'names', 'lines', 'xyz'}], one per MODEL x residue (chain, resn, resi) x altloc
    conformation (the blank-altloc atoms plus one altloc letter's, or the blank-altloc atoms if
    there are no letters), heavy atoms only; lines have the altloc column blanked. A file
    without MODEL records is model 1. resn: residue names to keep (default all)."""
    models, model = {}, None
    n_models = 0
    for line in open(path):
        if line.startswith('MODEL'):
            n_models += 1
            try:
                model = int(line[5:].split()[0])
            except (IndexError, ValueError):
                model = n_models
            continue
        if line.startswith('ENDMDL'):
            model = None
            continue
        if not line.startswith(('ATOM', 'HETATM')):
            continue
        element = line[76:78].strip() or line[12:16].strip()[0]
        if element.upper() in ('H', 'D'):
            continue
        name = line[17:20].strip()
        if resn and name not in resn:
            continue
        key = (line[21].strip(), name, line[22:27].strip())
        residues = models.setdefault(1 if model is None else model, {})
        residues.setdefault(key, {}).setdefault(line[16], []).append(line)
    out = []
    for model, residues in models.items():
        for (chain, name, resi), altlocs in residues.items():
            blank = altlocs.get(' ', [])
            letters = sorted(a for a in altlocs if a != ' ')
            for altloc in letters or ['']:
                atoms = {l[12:16].strip(): f'{l[:16]} {l[17:]}'
                         for l in blank + altlocs.get(altloc, [])}
                lines = list(atoms.values())
                out.append({'model': model, 'chain': chain, 'resn': name, 'resi': resi,
                            'altloc': altloc, 'names': list(atoms), 'lines': lines,
                            'xyz': [(float(l[30:38]), float(l[38:46]), float(l[46:54]))
                                    for l in lines]})
    return out


def molecule_template(lines, smiles=None):
    """RDKit mol of one molecule's PDB lines (heavy atoms, file order): bonds by proximity and,
    with smiles, bond orders from it."""
    mol = Chem.MolFromPDBBlock(''.join(lines) + 'END\n', removeHs=True)
    if mol is None:
        raise ValueError('RDKit could not read the molecule')
    if smiles:
        reference = Chem.MolFromSmiles(smiles)
        if reference is None:
            raise ValueError(f'RDKit could not parse the SMILES {smiles}')
        mol = AllChem.AssignBondOrdersFromTemplate(reference, mol)
    if mol.GetNumAtoms() != len(lines):
        raise ValueError(f'{len(lines)} atoms, {mol.GetNumAtoms()} in the RDKit molecule')
    return mol


def molecule_name(m):
    return (f'm{m["model"]}_{m["chain"]}{m["resi"]}'
            + (f'_alt{m["altloc"]}' if m['altloc'] else ''))


def output_path(output_dir, name):
    return Path(name) if Path(name).is_absolute() else output_dir / name


def file_stamp(path):
    st = path.stat()
    return [st.st_size, st.st_mtime]


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('ligand_pdb', help='Ligand(s) to score: a single- or multi-model PDB.')
    p.add_argument('target_pdb', help='Protein target; its --exclude-resn residues are '
                                      'removed and its CRYST1 replaced.')
    p.add_argument('map', help='Density map (CCP4) in the target\'s frame.')
    p.add_argument('output_dir')
    p.add_argument('--residue-scores', default='residue_scores.csv',
                   help='Per-molecule score CSV, in output_dir unless absolute. Default: '
                        'residue_scores.csv.')
    p.add_argument('--atom-scores', default='atom_scores.csv',
                   help='Per-atom score CSV, in output_dir unless absolute. Default: '
                        'atom_scores.csv.')
    p.add_argument('--params', help='Run parameters JSON. Default: <residue scores '
                                    'stem>_params.json next to the residue scores.')
    p.add_argument('--smiles', help='SMILES giving the ligand\'s bond orders. Default: none '
                                    '(bonds from the coordinates, all single).')
    p.add_argument('--resn', nargs='+', help='Residue names of ligand_pdb to score. Default: '
                                             'all.')
    p.add_argument('--cell', type=float, nargs=6, required=True,
                   metavar=('A', 'B', 'C', 'ALPHA', 'BETA', 'GAMMA'),
                   help='Unit cell (Å, degrees), written to the target\'s CRYST1.')
    p.add_argument('--space-group', required=True,
                   help='Space group, e.g. P212121 or "P 21 21 21", written to the target\'s '
                        'CRYST1.')
    p.add_argument('--resolution', type=float, required=True,
                   help='Resolution (Å) given to EDIA.')
    p.add_argument('--exclude-resn', nargs='+', default=DEFAULT_EXCLUDE_RESN,
                   help=f'Residue names removed from the target. Default: '
                        f'{" ".join(DEFAULT_EXCLUDE_RESN)}.')
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

    ligand_pdb, target_pdb = Path(args.ligand_pdb).resolve(), Path(args.target_pdb).resolve()
    density = Path(args.map).resolve()
    for f in (ligand_pdb, target_pdb, density, Path(args.ediascorer)):
        if not f.exists():
            raise SystemExit(f'Not found: {f}')
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    residue_csv = output_path(out_dir, args.residue_scores)
    atom_csv = output_path(out_dir, args.atom_scores)
    stem = residue_csv.with_suffix('').name
    params_file = (output_path(out_dir, args.params) if args.params
                   else residue_csv.with_name(f'{stem}_params.json'))
    work = residue_csv.with_name(f'{stem}_edia_work')
    failed_dir = residue_csv.with_name(f'{stem}_edia_failed')

    info = {'cell': tuple(args.cell), 'space_group': args.space_group,
            'resolution': args.resolution}
    resolution = args.resolution
    excluded = sorted(set(args.exclude_resn))
    settings = {'ligand_pdb': str(ligand_pdb), 'ligand_pdb_stamp': file_stamp(ligand_pdb),
                'target': str(target_pdb), 'target_stamp': file_stamp(target_pdb),
                'map': str(density), 'cell': list(info['cell']),
                'space_group': info['space_group'], 'resolution': resolution,
                'excluded_resn': excluded, 'resn': sorted(args.resn) if args.resn else None, 'smiles': args.smiles,
                'crop_radius': args.crop_radius, 'ediascorer': str(args.ediascorer)}

    params = json.loads(params_file.read_text()) if params_file.exists() else {}
    if (residue_csv.exists() and atom_csv.exists() and params.get('mode') == 'per_atom'
            and params.get('settings') == json.loads(json.dumps(settings))
            and not args.overwrite):
        print(f'Skipping: {residue_csv.name} and {atom_csv.name} exist, scored with the same '
              f'inputs and settings (pass --overwrite to redo)')
        return
    if subprocess.run(['podman', 'image', 'exists', args.container_image]).returncode != 0:
        raise SystemExit(f'{args.container_image} is not in the local podman image store '
                         f'(podman pull it first)')

    start = time.time()
    found = ligand_molecules(ligand_pdb, set(args.resn) if args.resn else None)
    if not found:
        raise SystemExit(f'No ' + (f'{"/".join(args.resn)} ' if args.resn else '')
                         + f'heavy atoms in {ligand_pdb}')
    # One template per distinct atom-name list, from the first molecule with it.
    templates, molecules, unbuilt = {}, [], {}
    for m in found:
        key = tuple(m['names'])
        if key not in templates:
            try:
                templates[key] = molecule_template(m['lines'], args.smiles)
            except ValueError as e:
                templates[key] = None
                print(f'Warning: no template for {molecule_name(m)} ({e}); molecules with its '
                      f'atom names are not scored')
        if templates[key] is None:
            unbuilt[molecule_name(m)] = 'no template'
            continue
        molecules.append((molecule_name(m), m['xyz'], templates[key], m['names']))
    duplicated = [n for n, c in Counter(molecule_name(m) for m in found).items() if c > 1]
    if duplicated:
        raise SystemExit(f'{ligand_pdb}: more than one molecule named {duplicated[:5]} (same '
                         f'model, chain, resi and altloc)')
    if not molecules:
        raise SystemExit(f'No molecule of {ligand_pdb} could be built')

    # The work folder resumes only if made with the same settings (as score_molecules checks;
    # checked here too, before the target is written into it).
    work_settings = work / 'settings.json'
    if work.exists() and (args.overwrite or not work_settings.exists()
                          or json.loads(work_settings.read_text())
                          != json.loads(json.dumps(settings))):
        shutil.rmtree(work)
    if failed_dir.exists():
        shutil.rmtree(failed_dir)
    for old in (residue_csv, atom_csv):
        old.unlink(missing_ok=True)
    work.mkdir(exist_ok=True)
    # Recorded now, so score_molecules sees a work folder made with these settings and keeps it
    # (and the target written into it) rather than starting it over.
    work_settings.write_text(json.dumps(settings, indent=2) + '\n')
    target = work / 'edia_target.pdb'
    write_stripped_target(target_pdb, target, excluded, info)
    results, failures, stats = score_molecules(out_dir, molecules, None, None, density, info,
                                               args, settings, work=work,
                                               failed_dir=failed_dir, target=target)
    failures.update(unbuilt)

    atom_rows = []
    with open(residue_csv.with_name(residue_csv.name + '.tmp'), 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[*MOLECULE_FIELDS, *SCORE_FIELDS])
        writer.writeheader()
        for m in found:
            ids = {k: m[k] for k in MOLECULE_FIELDS}
            result = results.get(molecule_name(m))
            writer.writerow({**ids, **score_row(result)})
            if result:
                atom_rows += [{**ids, **a} for a in result['atoms']]
    write_atom_csv(atom_csv, atom_rows, {k: '' for k in MOLECULE_FIELDS})
    residue_csv.with_name(residue_csv.name + '.tmp').replace(residue_csv)

    command = [str(args.ediascorer), '-t', '<name>.pdb', '-d', str(density), '-i', '<name>.sdf',
               '-k', '0', '-u', str(resolution), '-o', '<name>/']
    params_file.write_text(json.dumps({
        'mode': 'per_atom', 'ligand_pdb': str(ligand_pdb), 'target': str(target_pdb),
        'map': str(density), 'cell': list(info['cell']), 'space_group': info['space_group'],
        'resolution': resolution, 'excluded_resn': excluded, 'resn': settings['resn'], 'smiles': args.smiles,
        'residue_scores': str(residue_csv), 'atom_scores': str(atom_csv),
        'ediascorer': str(args.ediascorer), 'container_image': args.container_image,
        'crop_radius': args.crop_radius, 'workers': args.workers, 'command': command,
        'n_molecules': len(found), 'n_models': len({m['model'] for m in found}),
        'n_templates': sum(t is not None for t in templates.values()),
        'failed': failures, **stats, 'settings': settings,
        'runtime_s': round(time.time() - start, 1)}, indent=2) + '\n')
    shutil.rmtree(work)
    print(f'{ligand_pdb.name}: per-atom EDIA of {len(found)} molecules ({len(failures)} '
          f'failed) against {density.name} in {time.time() - start:.0f} s -> '
          f'{residue_csv.name}, {atom_csv.name}')


if __name__ == '__main__':
    main()
