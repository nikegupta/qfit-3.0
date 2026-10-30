"""
evaluate_structure - per-atom EDIA of an experimental structure's ligand(s) placed into a
reference structure, and its percentile in a null distribution (driven by
evaluate_structure.sh). Scored exactly as run_edia scores a pocket's conformers and references
(EDIAscorer 1.3, one call per molecule against the model cropped around it), so the ligand's
mean atom EDIA is comparable with the distribution's.

Steps:
  1. Superposes the experimental structure onto the reference on all CA atoms they share,
     matched by chain and residue number (blank or first altloc). The two must have the same
     sequence: a shared position with different residue names is an error.
  2. chimera.pdb: the reference structure without its --ligand-resn (default LIG) and
     --exclude-resn (default HOH DMS) residues, plus every --ligand-resn residue of the
     experimental structure (heavy atoms, every altloc), moved into the reference's frame by
     that superposition. The reference's header (CRYST1 included) is kept; CONECT records are
     dropped and the ligand atoms renumbered after the reference's. edia_target.pdb: the same
     without the ligands, the protein context the ligands are scored in.
  3. The molecules: every experimental ligand residue, one per altloc, named ligand_<chain><resi>
     or ligand_<chain><resi>_alt<letter>, in the atom order and with the bond orders of the
     pocket's template ligand (--pocket: ligand.pdb and conformer_summary.json; default the
     distribution's folder, i.e. the pocket whose edia_scores.csv it is). Atoms are matched by
     name; a residue whose atom names differ from the template's is skipped with a warning.
  4. Per-atom EDIA of each molecule, as run_edia does (see run_edia --help, steps 2-4):
     EDIAscorer 1.3 -k 0 against edia_target.pdb cropped to the whole residues within
     --crop-radius Å (default as run_edia's), the full map, resolution from --resolution or the
     reference's REMARK 3, all calls in one podman container (--container-image), --workers at
     a time. The map must be in the reference's frame. A failed call is recorded, not fatal
     (status failed, its files in edia_atom_failed/), unless every ligand fails.
  5. Each ligand's --metric (default mean_atom_edia, the plain mean of the per-atom EDIA, as in
     generate_null.sh's plots) is ranked in the distribution's column of that name (e.g. a
     pocket's edia_scores.csv from run_edia; empty values, i.e. failed conformers, are left
     out): percentile = 100 * (n_below + 0.5 * n_equal) / n, as in plot_edia_evaluation.py.

Outputs, in <output_dir>:
  chimera.pdb, edia_target.pdb  the reference with the ligands placed, and without them
  ligand_edia.csv        one row per ligand (and altloc): chain, resi, altloc, name, EDIAm,
                         min/median/max EDIA, atom problem counts, mean_atom_edia, n_atoms,
                         status, then the <metric> percentile, its counts and the
                         distribution's median <metric>
  ligand_atom_scores.csv per-atom scores of the ligands: chain, name, resi, altloc, then
                         run_edia's per-atom columns (atom_index, atom_name, edia_atom_name,
                         element, edia, fault_code, b_factor, occupancy, x, y, z)
  edia_atom_failed/      the files of failed calls, if any
  evaluate_params.json   inputs, CA superposition (n, RMSD, rotation, translation), resolution,
                         EDIA settings, call times, failures, logged EDIA errors
Earlier outputs in <output_dir> are replaced.

Usage:
  evaluate_structure <reference_pdb> <experimental_pdb> <map.ccp4> <distribution.csv>
                     <output_dir> [--pocket <dir>] [--metric <column>] [--resolution R]
                     [--ligand-resn LIG] [--exclude-resn RESN ...] [--ediascorer <path>]
                     [--container-image <image>] [--crop-radius <Å>] [--workers <n>]
"""
import argparse
import csv
import json
import re
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np

from qfit.command_line.generate_conformers import ca_superposition, coords, element_of
from qfit.command_line.run_edia import (DEFAULT_CROP_RADIUS, DEFAULT_EDIASCORER_13, DEFAULT_IMAGE,
                                        FAILED_DIR, LIGAND_RESN, SCORE_FIELDS, WORK_DIR,
                                        reference_ligands, score_molecules, score_row,
                                        template_ligand, write_atom_csv)

METRICS = [f for f in SCORE_FIELDS if f != 'status']


def build_argparser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('reference_pdb', help='Structure the ligand is placed into (its protein '
                                         'context and frame; the map\'s frame).')
    p.add_argument('experimental_pdb', help='Structure the ligand comes from; same sequence '
                                            'as the reference.')
    p.add_argument('map', help='Density map (CCP4) in the reference\'s frame.')
    p.add_argument('distribution', help='Null distribution: a CSV with a --metric column (e.g. '
                                        'a pocket\'s edia_scores.csv from run_edia).')
    p.add_argument('output_dir')
    p.add_argument('--pocket', help='Folder with the template ligand (ligand.pdb, '
                                    'conformer_summary.json). Default: the distribution\'s '
                                    'folder.')
    p.add_argument('--metric', default='mean_atom_edia', choices=METRICS,
                   help='Score ranked in the distribution. Default: mean_atom_edia.')
    p.add_argument('--resolution', type=float,
                   help='Resolution (Å) given to EDIA. Default: the reference\'s REMARK 3 '
                        'RESOLUTION RANGE HIGH.')
    p.add_argument('--ligand-resn', default=LIGAND_RESN,
                   help=f'Residue name of the ligand(s). Default: {LIGAND_RESN}.')
    p.add_argument('--exclude-resn', nargs='+', default=['HOH', 'DMS'],
                   help='Residue names removed from the reference besides the ligand\'s. '
                        'Default: HOH DMS.')
    p.add_argument('--ediascorer', default=DEFAULT_EDIASCORER_13,
                   help=f'EDIAscorer 1.3 executable, run in the container. Default: '
                        f'{DEFAULT_EDIASCORER_13}.')
    p.add_argument('--container-image', default=DEFAULT_IMAGE,
                   help=f'Podman image EDIAscorer runs in (glibc >= 2.29), from the local '
                        f'image store. Default: {DEFAULT_IMAGE}.')
    p.add_argument('--crop-radius', type=float, default=DEFAULT_CROP_RADIUS,
                   help=f'Each ligand is scored against the reference\'s residues with an atom '
                        f'within this distance (Å) of it. Default: {DEFAULT_CROP_RADIUS:g}, '
                        f'as run_edia.')
    p.add_argument('--workers', type=int, default=1,
                   help='EDIAscorer calls run at once. Default: 1.')
    return p


def ca_residues(pdb):
    """{(chain, resi): (resn, CA coordinates)} of a PDB file (blank or first altloc)."""
    out = {}
    for l in open(pdb):
        if l.startswith('ATOM') and l[12:16] == ' CA ' and l[16] in ' A':
            out.setdefault((l[21], int(l[22:26])), (l[17:20].strip(), coords([l])[0]))
    return out


def reference_resolution(pdb):
    for l in open(pdb):
        m = re.match(r'REMARK   3   RESOLUTION RANGE HIGH \(ANGSTROMS\)\s*:\s*([\d.]+)', l)
        if m:
            return float(m.group(1))
    return None


def is_atom(line):
    return line.startswith(('ATOM', 'HETATM'))


def write_chimera(reference_pdb, ligand_lines, out, skip_resn):
    """The reference minus skip_resn residues, then the ligand lines (renumbered), then END."""
    kept, serial, residues = [], 0, set()
    for line in open(reference_pdb):
        if line.startswith(('END', 'CONECT', 'MASTER')):
            continue
        if is_atom(line) or line.startswith(('ANISOU', 'TER')):
            if line[17:20].strip() in skip_resn:
                continue
            if is_atom(line):
                residues.add((line[21], line[22:27]))
                serial = max(serial, int(line[6:11]) if line[6:11].strip().isdigit() else 0)
        kept.append(line)
    clashes = sorted({(l[21], l[22:27].strip()) for l in ligand_lines
                      if (l[21], l[22:27]) in residues})
    if clashes:
        raise SystemExit(f'Ligand residue numbers {clashes} are also used by residues kept in '
                         f'the reference; renumber one of them')
    with open(out, 'w') as f:
        f.writelines(kept)
        for i, line in enumerate(ligand_lines, start=serial + 1):
            f.write(f'HETATM{i % 100000:5d}{line[11:]}')
        f.write('END\n')


def read_distribution(path, metric):
    """The finite values of the distribution CSV's metric column (empty ones left out)."""
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        if metric not in (reader.fieldnames or []):
            raise SystemExit(f'{path} has no {metric} column (scored by an earlier run_edia, '
                             f'EDIAm only? Rescore the pocket, or pass --metric EDIAm)')
        null = []
        for row in reader:
            try:
                v = float(row[metric])
            except (TypeError, ValueError):
                continue
            if np.isfinite(v):
                null.append(v)
    if not null:
        raise SystemExit(f'No {metric} values in {path}')
    return np.array(null)


def main():
    args = build_argparser().parse_args()
    start = time.time()
    reference = Path(args.reference_pdb).resolve()
    experimental = Path(args.experimental_pdb).resolve()
    density = Path(args.map).resolve()
    distribution = Path(args.distribution).resolve()
    pocket = Path(args.pocket).resolve() if args.pocket else distribution.parent
    for f in (reference, experimental, density, distribution, Path(args.ediascorer),
              pocket / 'ligand.pdb', pocket / 'conformer_summary.json'):
        if not f.exists():
            raise SystemExit(f'Not found: {f}' + (' (pass --pocket: the pocket folder with the '
                                                  'template ligand)' if f.parent == pocket else ''))
    if not any(l.startswith('CRYST1') for l in open(reference)):
        raise SystemExit(f'{reference} has no CRYST1 record (EDIA needs the unit cell)')
    resolution = args.resolution or reference_resolution(reference)
    if resolution is None:
        raise SystemExit(f'No REMARK 3 resolution in {reference}; pass --resolution')
    null = read_distribution(distribution, args.metric)
    if subprocess.run(['podman', 'image', 'exists', args.container_image]).returncode != 0:
        raise SystemExit(f'{args.container_image} is not in the local podman image store '
                         f'(podman pull it first)')

    # 1. CA superposition, experimental onto reference.
    ref_ca, exp_ca = ca_residues(reference), ca_residues(experimental)
    shared = sorted(set(ref_ca) & set(exp_ca))
    mismatched = [(k, ref_ca[k][0], exp_ca[k][0]) for k in shared if ref_ca[k][0] != exp_ca[k][0]]
    if mismatched:
        raise SystemExit(f'{len(mismatched)} shared positions have different residues (chain, '
                         f'resi, reference, experimental), e.g. {mismatched[:5]}; the '
                         f'structures must have the same sequence')
    fit = ca_superposition({k: v[1] for k, v in ref_ca.items()},
                           {k: v[1] for k, v in exp_ca.items()})
    if fit is None:
        raise SystemExit('Fewer than 3 shared CA atoms (matched by chain and residue number)')
    r, t, n_ca, rmsd = fit
    print(f'Superposed {experimental.name} onto {reference.name} on {n_ca} CA (RMSD '
          f'{rmsd:.3f} Å); {len(ref_ca) - n_ca} reference-only and {len(exp_ca) - n_ca} '
          f'experimental-only CA')

    # 2. The chimera: the reference's protein context with the experimental ligand(s).
    ligand_lines = []
    for line in open(experimental):
        if (is_atom(line) and line[17:20].strip() == args.ligand_resn
                and element_of(line) not in ('H', 'D')):
            x, y, z = coords([line])[0] @ r.T + t
            ligand_lines.append(f'{line[:30]}{x:8.3f}{y:8.3f}{z:8.3f}{line[54:]}')
    if not ligand_lines:
        raise SystemExit(f'No {args.ligand_resn} heavy atoms in {experimental}')
    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    # edia_raw/ and edia_log.txt: the batched EDIAscorer 1.1 version's outputs.
    for old in (out / 'edia_raw', out / WORK_DIR, out / FAILED_DIR):
        if old.exists():
            shutil.rmtree(old)
    for old in (out / 'edia_log.txt', out / 'ligand_edia.csv', out / 'ligand_atom_scores.csv'):
        old.unlink(missing_ok=True)
    skip = {args.ligand_resn, *args.exclude_resn}
    write_chimera(reference, ligand_lines, out / 'chimera.pdb', skip)
    write_chimera(reference, [], out / 'edia_target.pdb', skip)

    # 3. The ligand molecules, in the template's atom order.
    mol, names = template_ligand(pocket)
    ligands = []
    for chain in sorted({l[21].strip() for l in ligand_lines}):
        for name, resi, altloc, xyz in reference_ligands(out / 'chimera.pdb', chain, names,
                                                         args.ligand_resn):
            ligands.append((name.replace('reference_', f'ligand_{chain}', 1), chain, resi,
                            altloc, xyz))
    if not ligands:
        raise SystemExit(f'No {args.ligand_resn} residue of {experimental.name} has the '
                         f'template ligand\'s atom names ({pocket / "ligand.pdb"})')

    # 4. Per-atom EDIA, one call per ligand.
    info = {'resolution': resolution}
    settings = {'map': str(density), 'reference': str(reference),
                'experimental': str(experimental), 'excluded_resn': sorted(skip),
                'crop_radius': args.crop_radius, 'ediascorer': str(args.ediascorer),
                'resolution': resolution}
    results, failures, stats = score_molecules(out, [(l[0], l[4]) for l in ligands], mol, names,
                                               density, info, args, settings)

    # 5. Percentile in the distribution.
    metric = args.metric
    fields = ['chain', 'resi', 'altloc', 'name', *SCORE_FIELDS, f'{metric}_percentile',
              'n_distribution', 'n_distribution_below', 'n_distribution_equal',
              f'distribution_median_{metric}']
    median = round(float(np.median(null)), 4)
    rows, atom_rows = [], []
    for name, chain, resi, altloc, _ in ligands:
        row = {'chain': chain, 'resi': resi, 'altloc': altloc, 'name': name,
               **score_row(results.get(name))}
        if name in results and row.get(metric) not in (None, ''):
            value = float(row[metric])
            below, equal = int((null < value).sum()), int((null == value).sum())
            row.update({f'{metric}_percentile': round(100 * (below + 0.5 * equal) / len(null), 2),
                        'n_distribution': len(null), 'n_distribution_below': below,
                        'n_distribution_equal': equal, f'distribution_median_{metric}': median})
            atom_rows += [{'chain': chain, 'name': name, 'resi': resi, 'altloc': altloc, **a}
                          for a in results[name]['atoms']]
        rows.append(row)
    with open(out / 'ligand_edia.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    write_atom_csv(out / 'ligand_atom_scores.csv', atom_rows,
                   {'chain': '', 'name': '', 'resi': '', 'altloc': ''})
    command = [str(args.ediascorer), '-t', '<name>.pdb', '-d', str(density), '-i', '<name>.sdf',
               '-k', '0', '-u', str(resolution), '-o', '<name>/']
    (out / 'evaluate_params.json').write_text(json.dumps({
        'reference': str(reference), 'experimental': str(experimental), 'map': str(density),
        'distribution': str(distribution), 'pocket': str(pocket), 'metric': metric,
        'resolution': resolution, 'ligand_resn': args.ligand_resn,
        'excluded_resn': sorted(set(args.exclude_resn)),
        'superposition': {'n_ca': n_ca, 'ca_rmsd_A': round(rmsd, 4),
                          'rotation': np.round(r, 6).tolist(),
                          'translation': np.round(t, 4).tolist()},
        'n_ligand_atoms': len(ligand_lines), 'mode': 'per_atom',
        'ediascorer': str(args.ediascorer), 'container_image': args.container_image,
        'crop_radius': args.crop_radius, 'workers': args.workers, 'command': command,
        'ligands': [{'name': n, 'chain': c, 'resi': ri, 'altloc': a}
                    for n, c, ri, a, _ in ligands],
        'failed': failures, **stats, 'runtime_s': round(time.time() - start, 1)},
        indent=2) + '\n')
    shutil.rmtree(out / WORK_DIR)
    for row in rows:
        label = (f'{args.ligand_resn} {row["chain"]}{row["resi"]}'
                 + (f' alt {row["altloc"]}' if row['altloc'] else ''))
        if row['status'] != 'ok':
            print(f'{label}: failed ({failures.get(row["name"])})')
            continue
        print(f'{label}: {metric} {row[metric]} (EDIAm {row["EDIAm"]}), percentile '
              f'{row.get(f"{metric}_percentile", "-")} of {len(null)} (distribution median {median})')
    if len(failures) == len(ligands):
        raise SystemExit(f'Every ligand\'s EDIA call failed (see {out / FAILED_DIR})')
    print(f'Wrote {out / "ligand_edia.csv"} in {time.time() - start:.0f} s')


if __name__ == '__main__':
    main()
