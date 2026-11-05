"""
match_ligand - which reference ligand a predicted structure's ligand corresponds to
(evaluate_structures.sh stage 1), and the prediction's ligand moved into the reference's frame.

Steps:
  1. The prediction's ligand: its one --ligand-resn residue (heavy atoms; one conformation - a
     prediction with several ligand residues or altlocs is an error). Its protein: the CA of
     every residue of every chain (blank or first altloc).
  2. For each chain of the reference with a --ligand-resn residue, and each protein chain of
     the prediction: a global sequence alignment of the prediction chain to the reference chain
     (biotite align_optimal, BLOSUM62, gap penalty (-10, -1), free terminal gaps; residue
     numbers need not correspond - predictions are typically numbered from 1), and a CA
     superposition (Kabsch) of the prediction onto the reference on the identical aligned
     residue pairs.
  3. Candidates: every ligand conformation of that reference chain (each residue, each altloc
     letter: the blank-altloc atoms plus that letter's, as generate_null.sh's nulls), or, with
     --distribution-dir, only those with a null there (<dir>/LIG_<chain><resi>_<altloc>/
     distribution_residue_scores.csv). Each candidate's distance is that between its centroid
     and the prediction ligand's centroid superposed with that chain's fit.
  4. The match: the candidate with the smallest centroid distance.

Outputs, in output_dir:
  match.txt   the match's label, as generate_null.sh's null folders: LIG_<chain><resi>_<altloc>
              (LIG_<chain><resi> for a ligand without altloc letters), e.g. LIG_A701_B
  ligand.pdb  the prediction's ligand superposed with the match's fit (heavy atoms, altloc
              blank, chain and residue number those of the match, other fields kept)
  match.json  the match, every candidate (centroid distance, n_ca, CA RMSD, prediction chain)
              and every fit (rotation, translation)
Exits non-zero (writing nothing) if the reference has no ligand, no fit has 3 aligned CA pairs,
or no candidate has a null in --distribution-dir.

Usage:
  match_ligand <prediction_pdb> <reference_pdb> <output_dir> [--distribution-dir <dir>]
               [--ligand-resn <resn>]
"""
import argparse
import json
from pathlib import Path

import biotite.sequence as seq
import biotite.sequence.align as align
import numpy as np

from qfit.command_line.generate_conformers import superpose
from qfit.command_line.run_edia import LIGAND_RESN, ligand_molecules

GAP_PENALTY = (-10, -1)
DISTRIBUTION_CSV = 'distribution_residue_scores.csv'


def protein_cas(pdb, skip_resn):
    """{chain: [(resn, CA xyz)] in file order}: one CA per residue (chain, resi, insertion code;
    the first altloc seen), residues named in skip_resn left out."""
    chains, seen = {}, set()
    for line in open(pdb):
        if line.startswith('ENDMDL'):
            break  # first model only
        if not line.startswith(('ATOM', 'HETATM')) or line[12:16] != ' CA ':
            continue
        resn = line[17:20].strip()
        key = (line[21], line[22:27])
        if resn in skip_resn or key in seen:
            continue
        seen.add(key)
        chains.setdefault(line[21], []).append(
            (resn, np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])])))
    return chains


def one_letter(resns):
    letters = []
    for r in resns:
        try:
            letters.append(seq.ProteinSequence.convert_letter_3to1(r))
        except KeyError:
            letters.append('X')
    return seq.ProteinSequence(''.join(letters))


def aligned_pairs(mobile, target):
    """[(i, j)]: indices of identical residue pairs of a global alignment of the mobile chain's
    sequence to the target chain's."""
    ali = align.align_optimal(one_letter(r for r, _ in mobile), one_letter(r for r, _ in target),
                              align.SubstitutionMatrix.std_protein_matrix(),
                              gap_penalty=GAP_PENALTY, terminal_penalty=False)[0]
    return [(int(i), int(j)) for i, j in ali.trace
            if i >= 0 and j >= 0 and mobile[i][0] == target[j][0]]


def fit_chain(mobile, target):
    """(r, t, n_ca, rmsd) of the CA superposition of the mobile chain onto the target chain on
    their identical aligned residues, or None if fewer than 3."""
    pairs = aligned_pairs(mobile, target)
    if len(pairs) < 3:
        return None
    m = np.array([mobile[i][1] for i, _ in pairs])
    t_xyz = np.array([target[j][1] for _, j in pairs])
    r, t = superpose(m, t_xyz)
    rmsd = float(np.sqrt(((m @ r.T + t - t_xyz) ** 2).sum(axis=1).mean()))
    return r, t, len(pairs), rmsd


def label(m, resn=LIGAND_RESN):
    """Null folder label of a ligand conformation, e.g. LIG_A701_B."""
    return f'{resn}_{m["chain"]}{m["resi"]}' + (f'_{m["altloc"]}' if m['altloc'] else '')


def match(prediction, reference, resn=LIGAND_RESN, distribution_dir=None):
    """The match (module docstring, steps 1-4): {'match', 'candidates', 'fits', 'ligand'
    (the prediction's ligand molecule), 'xyz' (its superposed coordinates)}. Raises ValueError
    if there is no match."""
    ligands = ligand_molecules(prediction, {resn})
    if len(ligands) != 1:
        raise ValueError(f'{prediction}: {len(ligands)} {resn} conformations (residues x '
                         f'altlocs), expected 1')
    ligand = ligands[0]
    lig_xyz = np.array(ligand['xyz'])
    pred_chains = protein_cas(prediction, {resn})
    ref_chains = protein_cas(reference, {resn})
    references = ligand_molecules(reference, {resn})
    if not references:
        raise ValueError(f'No {resn} residue in {reference}')

    fits, candidates = [], []
    for ref_chain in sorted({m['chain'] for m in references}):
        if ref_chain not in ref_chains:
            continue
        for pred_chain, mobile in sorted(pred_chains.items()):
            fit = fit_chain(mobile, ref_chains[ref_chain])
            if fit is None:
                continue
            r, t, n_ca, rmsd = fit
            fits.append({'reference_chain': ref_chain, 'prediction_chain': pred_chain,
                         'n_ca': n_ca, 'ca_rmsd_A': round(rmsd, 3),
                         'rotation': np.round(r, 6).tolist(),
                         'translation': np.round(t, 4).tolist()})
            centroid = lig_xyz.mean(axis=0) @ r.T + t
            for m in references:
                if m['chain'] != ref_chain:
                    continue
                name = label(m, resn)
                has_null = (distribution_dir is None
                            or (Path(distribution_dir) / name / DISTRIBUTION_CSV).exists())
                candidates.append({
                    'label': name, 'chain': m['chain'], 'resi': m['resi'],
                    'altloc': m['altloc'], 'prediction_chain': pred_chain,
                    'centroid_distance_A': round(float(np.linalg.norm(
                        np.array(m['xyz']).mean(axis=0) - centroid)), 3),
                    'n_ca': n_ca, 'ca_rmsd_A': round(rmsd, 3), 'has_null': has_null,
                    '_fit': (r, t)})
    if not fits:
        raise ValueError(f'No prediction chain of {prediction} aligns to a {resn} chain of '
                         f'{reference} with 3 identical CA pairs')
    eligible = [c for c in candidates if c['has_null']]
    if not eligible:
        raise ValueError(f'None of {sorted({c["label"] for c in candidates})} has a null in '
                         f'{distribution_dir} ({DISTRIBUTION_CSV})')
    best = min(eligible, key=lambda c: c['centroid_distance_A'])
    r, t = best['_fit']
    for c in candidates:
        del c['_fit']
    return {'match': best, 'candidates': candidates, 'fits': fits, 'ligand': ligand,
            'xyz': lig_xyz @ r.T + t}


def write_ligand(path, ligand, xyz, chain, resi):
    """The ligand's lines with new coordinates, chain and residue number (altloc blank)."""
    with open(path, 'w') as f:
        for line, (x, y, z) in zip(ligand['lines'], xyz):
            f.write(f'{line[:16]} {line[17:21]}{chain[:1]:1s}{int(resi):4d}{line[26:30]}'
                    f'{x:8.3f}{y:8.3f}{z:8.3f}{line[54:]}')
        f.write('END\n')


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('prediction_pdb')
    p.add_argument('reference_pdb')
    p.add_argument('output_dir')
    p.add_argument('--distribution-dir',
                   help=f'Only reference ligands with a null here (LIG_*/{DISTRIBUTION_CSV}) '
                        f'are candidates. Default: every reference ligand.')
    p.add_argument('--ligand-resn', default=LIGAND_RESN,
                   help=f'Residue name of the ligand. Default: {LIGAND_RESN}.')
    args = p.parse_args()
    prediction = Path(args.prediction_pdb).resolve()
    reference = Path(args.reference_pdb).resolve()
    for f in (prediction, reference):
        if not f.exists():
            raise SystemExit(f'Not found: {f}')
    try:
        result = match(prediction, reference, args.ligand_resn, args.distribution_dir)
    except ValueError as e:
        raise SystemExit(str(e))
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    best = result['match']
    write_ligand(out / 'ligand.pdb', result['ligand'], result['xyz'], best['chain'],
                 best['resi'])
    (out / 'match.json').write_text(json.dumps({
        'prediction': str(prediction), 'reference': str(reference),
        'distribution_dir': (str(Path(args.distribution_dir).resolve())
                             if args.distribution_dir else None),
        'ligand_resn': args.ligand_resn,
        'prediction_ligand': {k: result['ligand'][k] for k in ('chain', 'resi', 'altloc')}
        | {'n_heavy_atoms': len(result['ligand']['names'])},
        'match': best, 'candidates': result['candidates'], 'fits': result['fits']},
        indent=2) + '\n')
    (out / 'match.txt').write_text(best['label'] + '\n')  # last: marks the stage complete
    print(f'{prediction.name}: {best["label"]} (centroid {best["centroid_distance_A"]:.2f} Å; '
          f'{best["n_ca"]} CA, RMSD {best["ca_rmsd_A"]:.2f} Å, prediction chain '
          f'{best["prediction_chain"]})')


if __name__ == '__main__':
    main()
