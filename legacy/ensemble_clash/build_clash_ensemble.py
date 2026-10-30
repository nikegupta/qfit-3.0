"""
build_clash_ensemble - builds, once per run, the clash ensemble every generate_conformers run of
that run loads (--clash-ensemble): the binding-site residues, their side-chain conformations
across all ensemble models, and each chain's pocket volume and sampling sphere (see
build_argparser's description, and generate_conformers' docstring for how they are used).

The ClashEnsemble class itself, and the helpers it builds with, stay in generate_conformers so
that saved clash_ensemble.pkl files keep unpickling.

Usage:
  build_clash_ensemble <reference_pdb> <ensemble_list> <output.pkl> [--aligned-dir <dir>]
                       [--rotamer-tolerance R] [--exclude-resn RESN ...] [--pocket-margin M]
                       [--centroid-margin M] [--volume-spacing S] [--binding-site-cutoff C]
"""
import argparse
import csv
import json
from pathlib import Path

from qfit.command_line.generate_conformers import (
    ClashEnsemble, write_volume, write_superposed, MAX_CONFORMATIONS, DEFAULT_ROTAMER_TOLERANCE,
    DEFAULT_POCKET_ATOM_RADIUS, DEFAULT_POCKET_PROBE_RADIUS, DEFAULT_CENTROID_MARGIN, DEFAULT_VOLUME_SPACING,
    DEFAULT_BINDING_SITE_CUTOFF)


def build_argparser():
    p = argparse.ArgumentParser(
        prog='build_clash_ensemble',
        description='Builds the clash ensemble for generate_conformers (--clash-ensemble), once '
                    'per run. Binding-site residues: residue numbers (chain agnostic) with a '
                    'heavy atom within --binding-site-cutoff of a LIG residue in any model. '
                    'Their side chains: each one\'s distinct side-chain conformations across '
                    'all models, each superposed onto the residue\'s backbone frame (the '
                    'reference structure\'s N, CA, C) on its own N, CA and C (each altloc is a '
                    'conformation; conformations missing atoms most others have are dropped; '
                    'de-duplicated within --rotamer-tolerance, raised for any residue that would '
                    f'otherwise keep more than {MAX_CONFORMATIONS}). Pocket volumes, one per '
                    'chain: every model is superposed onto the reference on all CA atoms; all '
                    'their LIG heavy atoms in that chain are made spheres of --pocket-atom-radius '
                    'and a probe of --pocket-probe-radius is rolled over them; the volume the '
                    'probe can\'t enter (the spheres plus the gaps between them narrower than '
                    'the probe, no outward growth) is kept on a --volume-spacing grid. '
                    'Sampling spheres, one per chain: the minimum enclosing sphere of the '
                    'superposed models\' ligand centroids in that chain, its radius plus '
                    '--centroid-margin. Next to the output it also writes pocket_residues.csv, '
                    'pocket_volume_<chain>.ccp4 and pocket_spheres.json (each sampling sphere: '
                    'centre in the reference frame, observed and max radius), for viewing.')
    p.add_argument('reference_pdb',
                   help='Structure whose residues define the backbone frames (e.g. reference.pdb).')
    p.add_argument('ensemble', help='File listing the model PDB paths, one per line.')
    p.add_argument('output', help='Clash ensemble file to write (e.g. clash_ensemble.pkl).')
    p.add_argument('--aligned-dir',
                   help='Also write every model superposed onto the reference on all CA atoms '
                        'into this directory, for viewing (not used by the clash check).')
    p.add_argument('--rotamer-tolerance', type=float, default=DEFAULT_ROTAMER_TOLERANCE,
                   help='Conformations of a residue within this heavy-atom RMSD (Å) count as '
                        f'one. Default: {DEFAULT_ROTAMER_TOLERANCE}.')
    p.add_argument('--exclude-resn', nargs='+', default=['HOH', 'DMS'],
                   help='Residue names left out (hydrogens and LIG always are). Default: HOH DMS.')
    p.add_argument('--pocket-atom-radius', type=float, default=DEFAULT_POCKET_ATOM_RADIUS,
                   help='Radius (Å) of the sphere every observed ligand heavy atom is made for '
                        'the pocket volumes (every element alike). Default: '
                        f'{DEFAULT_POCKET_ATOM_RADIUS}.')
    p.add_argument('--pocket-probe-radius', type=float, default=DEFAULT_POCKET_PROBE_RADIUS,
                   help='Radius (Å) of the probe rolled over those spheres; gaps narrower than '
                        f'it are filled (0: just the spheres). Default: '
                        f'{DEFAULT_POCKET_PROBE_RADIUS}.')
    p.add_argument('--centroid-margin', type=float, default=DEFAULT_CENTROID_MARGIN,
                   help='Cushion (Å) added to the radius of the observed ligand centroids\' '
                        'minimum enclosing sphere to give the sampling max radius. Default: '
                        f'{DEFAULT_CENTROID_MARGIN}.')
    p.add_argument('--volume-spacing', type=float, default=DEFAULT_VOLUME_SPACING,
                   help=f'Grid spacing (Å) of the pocket volumes. Default: '
                        f'{DEFAULT_VOLUME_SPACING}.')
    p.add_argument('--binding-site-cutoff', type=float, default=DEFAULT_BINDING_SITE_CUTOFF,
                   help='A residue is a binding-site residue if a heavy atom is within this '
                        'distance (Å) of a LIG residue in some model. Default: '
                        f'{DEFAULT_BINDING_SITE_CUTOFF}.')
    return p


def main():
    args = build_argparser().parse_args()
    model_pdbs = [l.strip() for l in open(args.ensemble) if l.strip()]
    ensemble = ClashEnsemble(args.reference_pdb, model_pdbs, args.rotamer_tolerance,
                             args.exclude_resn, args.volume_spacing,
                             args.binding_site_cutoff, args.pocket_atom_radius,
                             args.pocket_probe_radius, args.centroid_margin)
    out_dir = Path(args.output).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    ensemble.save(args.output)
    with open(out_dir / 'pocket_residues.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['resi', 'n_models'])
        writer.writerows(ensemble.binding_site.items())
    for chain, v in ensemble.volumes.items():
        write_volume(out_dir / f'pocket_volume_{chain}.ccp4', v['origin'], v['spacing'],
                     v['grid'])
    print(f'Binding site: {len(ensemble.binding_site)} residue numbers -> '
          f'{out_dir / "pocket_residues.csv"}')
    print('Pocket volumes: ' + ', '.join(
        f'chain {ch} {vol:.0f} Å^3 ({ensemble.volumes[ch]["n_ligand_atoms"]} ligand atoms)'
        for ch, vol in ensemble.info['pocket_volumes_A3'].items())
        + f' -> {out_dir}/pocket_volume_<chain>.ccp4')
    (out_dir / 'pocket_spheres.json').write_text(
        json.dumps(ensemble.info['pocket_spheres'], indent=2) + '\n')
    print('Sampling spheres (reference frame; pocket_spheres.json): ' + ', '.join(
        f'chain {ch} centre {sph["center"]}, {sph["n_ligands"]} ligand centroids within '
        f'{sph["observed_radius_A"]:.2f} Å, max radius {sph["max_radius_A"]:.2f} Å'
        for ch, sph in ensemble.info['pocket_spheres'].items()))
    if args.aligned_dir:
        rows = write_superposed(args.reference_pdb, model_pdbs, args.aligned_dir)
        rmsd = sorted(r['ca_rmsd_A'] for r in rows)
        print(f'Wrote {len(rows)} models superposed onto the reference on all CA atoms to '
              f'{args.aligned_dir} (CA RMSD median {rmsd[len(rmsd) // 2]:.2f}, max {rmsd[-1]:.2f} Å)')
    info = ensemble.info
    raised = {f'{r["resn"]}{r["key"][0]}{r["key"][1]}': r['tolerance']
              for r in ensemble.residues if r['tolerance'] > args.rotamer_tolerance}
    print(f'{info["n_models"]} of {len(model_pdbs)} models; binding-site side chains: '
          f'{info["n_residues"]}, {info["n_conformations"]} conformations'
          + (f'; tolerance raised for {raised}' if raised else ''))
    print(f'Wrote {args.output} in {info["build_time_s"]} s')


if __name__ == '__main__':
    main()
