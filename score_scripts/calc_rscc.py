"""
score_scripts/ copy of src/qfit/command_line/calc_rscc.py, adapted for score.sh's standalone
use: adds --em (cryo-EM electron scattering factors, threaded into qfit's transformer),
--label (MTZ amplitude/phase column labels, threaded into XMap.fromfile - unused for
CCP4/MRC/MAP maps), and --ligand-resname (restricts scoring to residues with that resname,
e.g. the bound ligand, instead of every residue in the structure_pdb - see
_find_residue_keys). The last one is what lets score.sh run this directly on a whole
protein-ligand complex with no separate split/extract step first: --ligand-resname filters
during enumeration, before any per-residue extraction happens, so nothing about how a residue's
own atoms get pulled out (_extract_residue) or scored changes - it only shrinks which
(chain, resi[, altloc]) keys get visited at all. Everything else (per-residue
extraction/scoring/CSV output) is unchanged from the pipeline's own calc_rscc.py.

resolution is optional here (the pipeline's own calc_rscc.py requires it as a positional arg -
this is one behavioral difference from that script, beyond --em/--label). When omitted, or
whenever --em is given, the mask radius falls back to a static DEFAULT_RMASK (1.5 A) instead of
the resolution-derived '0.5 + resolution/3.0' heuristic - see qfit.py's QFitBase.__init__,
which uses that exact same 1.5 A fallback whenever no resolution is known. --em forces this
fallback even when a resolution *is* given because that heuristic was tuned for crystallographic
resolution; a cryo-EM map's single global (often FSC-derived) resolution figure is a much
weaker proxy for local mask radius than it is for X-ray data (see the -em/local-resolution
discussion this script's design came out of).

A CCP4/MRC/MAP file still needs *some* non-None resolution value to load at all
(XMap.from_mapfile raises otherwise), so when --resolution is omitted, DEFAULT_RMASK is passed
there too as a placeholder (self._map_load_resolution). That placeholder is NOT harmless on its
own, though: qfit.xtal.legacy_transformer.Transformer separately derives a reciprocal-space
frequency cutoff (smax = 1/(2*resolution)) from whatever resolution ends up attached to the
XMap, independent of rmask - if left alone, the placeholder would silently render an
over-sharpened model density (as if resolution were really 1.5 A), which measurably lowers RSCC
against a map that's actually coarser than that (confirmed: with a real map, RSCC went from 0.74
with --resolution omitted to 0.81 with the map's real ~3.0 A resolution given - see the
`simple=(self.resolution is None)` in _score_residue for the fix). So whenever the *true*
resolution is unknown, this script also forces simple=True on the transformer, which bypasses
smax entirely in favor of a closed-form Gaussian atom density - exactly matching what
qfit.py's own QFitBase.__init__ does in the same situation (simple=True unless a resolution is
actually known). --em does not affect this - it only changes the mask radius and the scattering
factor table, never simple/smax.
"""
import argparse
import re
from pathlib import Path

import numpy as np

from qfit import Structure
from qfit import XMap
from qfit.xtal.transformer import get_transformer


# Static mask-radius fallback used when no resolution is known, or when --em is given - see
# qfit.py's QFitBase.__init__, which falls back to this exact value in the same situation.
DEFAULT_RMASK = 1.5

# Matches the BDC value embedded in an event map filename, e.g.
# 'x00407-1-event_1_1-BDC_0.08_map.native.ccp4' -> '0.08'.
BDC_PATTERN = re.compile(r'1-BDC_([\d.]+)_')


def parse_bdc(map_filename):
    """Extracts the BDC value (as a string, so '0.080' and '0.08' aren't
    silently treated as equal) from an event map filename. Returns None if
    the filename doesn't match the expected '1-BDC_<value>_' pattern."""
    m = BDC_PATTERN.search(str(map_filename))
    return m.group(1) if m else None


def build_argparser():
    p = argparse.ArgumentParser(
        description="Calculate per-residue RSCC for a single-model or multimodel PDB "
                    "structure against a density map."
    )
    p.add_argument(
        'structure_pdb',
        type=Path,
        help='Path to a single-model or multimodel PDB structure'
    )
    p.add_argument(
        'map_files',
        type=Path,
        nargs='+',
        help='Path(s) to one or more density map files (CCP4/MRC/MAP or MTZ) to score each '
             'residue against. A residue\'s reported RSCC is the max across all maps given. '
             'Maps whose filename embeds a duplicate 1-BDC_<value>_ (same background-'
             'subtraction fraction as an already-loaded map) are skipped, since they are '
             'identical to that map.'
    )
    p.add_argument(
        'output_csv',
        type=Path,
        help='Path to write the per-residue RSCC csv'
    )
    p.add_argument(
        '--resolution',
        type=float,
        default=None,
        help='Resolution (Å) of the map(s). Optional - if omitted (or if --em is given), the '
             f'mask radius falls back to a static {DEFAULT_RMASK} Å instead of the '
             'resolution-derived 0.5 + resolution/3.0 heuristic (see qfit.py\'s own fallback). '
             'A CCP4/MRC/MAP file still needs *some* resolution value to load, so when omitted '
             f'here, {DEFAULT_RMASK} is also used as that placeholder - it has no effect on the '
             'map\'s actual voxel grid/density values, only on this metadata.'
    )
    p.add_argument(
        '--bfactor',
        type=float,
        default=None,
        help='Constant B-factor to use for every atom when generating each residue\'s model '
             'density. Optional - if omitted (the default), each atom\'s own B-factor from '
             '<structure_pdb> is used instead (a per-atom array, not a single value) - this '
             'matters when one part of an otherwise well-fit residue (e.g. a flexible terminal '
             'group) is genuinely more mobile/disordered than the rest: a single constant '
             'B-factor smooths over that and can under- or over-state RSCC for the residue as '
             'a whole. Pass --bfactor to force every atom to the same value instead.'
    )
    p.add_argument(
        '--em',
        action='store_true',
        help='Treat the map(s) as cryo-EM: use electron (Mott-Bethe) scattering factors '
             'instead of X-ray scattering factors when rendering each residue\'s model '
             'density, and treat the map as a single non-periodic P1 box - see '
             'qfit.xtal.legacy_transformer.Transformer\'s "em" flag.'
    )
    p.add_argument(
        '--label',
        default='FWT,PHWT',
        help='MTZ amplitude/phase column labels - only used when a map file is a .mtz '
             '(default: FWT,PHWT)'
    )
    p.add_argument(
        '--ligand-resname',
        default=None,
        help='Restrict scoring to residues with this resname (e.g. a bound ligand) instead of '
             'every residue in structure_pdb. Optional - if omitted, every residue is scored '
             '(the original calc_rscc.py behavior).'
    )
    return p


class ResidueRSCCCalculator:
    def __init__(self, structure_pdb, map_files, output_csv, resolution=None,
                 bfactor=None, em=False, label='FWT,PHWT', ligand_resname=None):
        self.structure_pdb = Path(structure_pdb)
        self.map_files = [Path(m) for m in map_files]
        self.resolution = resolution
        self.output_csv = Path(output_csv)
        self.bfactor = bfactor
        self.em = em
        self.label = label
        self.ligand_resname = ligand_resname

        # See this module's docstring: --em always uses the static fallback (the
        # resolution-derived heuristic below was tuned for X-ray, not cryo-EM), and a missing
        # resolution has nowhere else to fall back to.
        if em or resolution is None:
            self.rmask = DEFAULT_RMASK
        else:
            self.rmask = 0.5 + resolution / 3.0  # from qfit

        # CCP4/MRC/MAP loading (XMap.from_mapfile) requires *some* non-None resolution value to
        # be passed, even though it's only stored as metadata (never used to build the grid/
        # density - see this module's docstring) - reuse DEFAULT_RMASK as that placeholder when
        # no real resolution was given, so a map can still load with --resolution omitted.
        self._map_load_resolution = resolution if resolution is not None else DEFAULT_RMASK

    def _clean_structure(self, structure):
        """Remove hydrogens and rename OXT->O, matching what qfit's transformer expects."""
        structure = structure.extract("e", "H", "!=")
        rename = structure.extract("name", "OXT", "==")
        rename.name = "O"
        structure = structure.extract("name", "OXT", "!=").combine(rename)
        return structure

    def _get_model_serials(self, pdb_path):
        """
        Parses the structure's own 'MODEL' record lines, in the order they appear in the file,
        and returns the serial number written on each one. A single-model PDB typically has no
        MODEL records at all, in which case this returns an empty list and the caller falls back
        to sequential 1-based indices (so a single-model structure is reported as model 1).
        """
        serials = []
        with open(pdb_path) as f:
            for line in f:
                if line.startswith('MODEL'):
                    parts = line.split()
                    if len(parts) >= 2:
                        try:
                            serials.append(int(parts[1]))
                        except ValueError:
                            print(f"Warning: could not parse MODEL serial from line: {line.strip()!r}")
        return serials

    def _find_residue_keys(self, structure):
        """
        Returns [(chain_id, resi, altloc), ...] for every residue in a single-model structure -
        or, if self.ligand_resname is set, only residue groups that have at least one atom_group
        with that resname (e.g. a bound ligand); this is evaluated directly against
        structure's own iotbx hierarchy, which - unlike a prior .extract() selection - always
        reflects every residue actually in the file, so no separate pre-filtering/splitting
        step is needed first (see this module's docstring).

        A residue group with any non-blank altloc (e.g. 'A', whether or not a second altloc
        competes at that same residue) is split into one key per altloc, since each altloc
        represents a physically distinct conformer that should be scored independently rather
        than collapsed into one residue. altloc is '' only for residues with no altloc at all.
        """
        keys = []
        for chain in structure._pdb_hierarchy.only_model().chains():
            chain_id = chain.id.strip()
            for residue_group in chain.residue_groups():
                atom_groups = residue_group.atom_groups()
                if self.ligand_resname is not None:
                    atom_groups = [ag for ag in atom_groups if ag.resname.strip() == self.ligand_resname]
                    if not atom_groups:
                        continue
                resi = int(residue_group.resseq)
                altlocs = sorted({ag.altloc.strip() for ag in atom_groups})
                non_blank_altlocs = [a for a in altlocs if a != '']
                if non_blank_altlocs:
                    for altloc in non_blank_altlocs:
                        keys.append((chain_id, resi, altloc))
                else:
                    keys.append((chain_id, resi, ''))
        return keys

    def _extract_residue(self, model, chain_id, resi, altloc):
        """
        Extracts a single residue conformer's atoms from a model.

        When altloc is non-empty (the residue has two or more altloc conformers), restricts to
        that altloc's atoms plus any blank-altloc atoms of the same residue (atoms shared across
        all of its conformers), so each altloc yields one complete conformer rather than a mix.
        """
        resi_selstr = f"chain {chain_id} and resi {resi}"
        residue_structure = model.extract(resi_selstr)
        if altloc:
            alt_structure = residue_structure.extract("altloc", altloc, "==")
            blank_structure = residue_structure.extract("altloc", "", "==")
            residue_structure = alt_structure.combine(blank_structure)
        return residue_structure

    def _residue_label(self, chain_id, resi, altloc):
        """Formats a (chain_id, resi, altloc) key as a readable residue label, e.g. 'A103' with
        no altloc disorder, or 'A103-B' for the 'B' altloc conformer."""
        label = f'{chain_id}{resi}'
        if altloc:
            label += f'-{altloc}'
        return label

    def _load_maps(self):
        """
        Loads every map in self.map_files into an {name: XMap} dict, along with a matching
        {name: zeroed-template XMap} dict used later to build each residue's model density.
        Keyed by filename (rather than index) so any per-map warnings can reference the file.

        Event maps sharing the same BDC value are identical (same partial-occupancy
        background subtraction), so scoring more than one of them per residue is
        wasted compute for no new information - only the first map seen for a given
        BDC is loaded/kept; later ones with the same BDC are skipped. A map whose
        filename doesn't match the expected BDC pattern is always kept, since its
        BDC (and therefore whether it duplicates another map) can't be determined.
        """
        maps = {}
        map_models = {}
        seen_bdcs = set()
        for map_file in self.map_files:
            bdc = parse_bdc(map_file.name)
            if bdc is not None:
                if bdc in seen_bdcs:
                    print(f'Skipping map {map_file}: duplicate BDC={bdc} '
                          f'(another event map with this BDC was already loaded).')
                    continue
                seen_bdcs.add(bdc)

            name = map_file.name
            resolution_desc = 'unspecified' if self.resolution is None else self.resolution
            print(f'Loading map {map_file} (resolution={resolution_desc}, rmask={self.rmask}'
                  f'{", cryo-EM" if self.em else ""})')
            maps[name] = XMap.fromfile(str(map_file), resolution=self._map_load_resolution,
                                        label=self.label)
            map_model = maps[name].zeros_like(maps[name])
            map_model.set_space_group("P1")
            map_models[name] = map_model
        return maps, map_models

    def _score_residue(self, residue_structure, coor, maps, map_models):
        """Converts a single residue conformer's coordinates to density and returns its highest
        RSCC across all provided maps, using that conformer's own mask (recomputed per map,
        since each map may have a different grid).

        Model density is generated using either a constant B-factor for every atom (if
        --bfactor was given) or, by default, each atom's own B-factor as already parsed from
        <structure_pdb> (residue_structure.b - a per-atom array; get_conformers_densities
        assigns it straight to self.structure.b, see legacy_transformer.Transformer, so a
        per-atom array here is exactly as well-supported as the single constant this script
        used before)."""
        scaled_bulk_solvent = 0
        coor_set = [coor]
        if self.bfactor is None:
            b_array = residue_structure.b.copy()
        else:
            b_array = np.full(coor.shape[0], self.bfactor)
        bfactor_array = [b_array]

        rsccs = []
        for name in maps:
            # simple=True bypasses Transformer's smax reciprocal-space band-limit entirely (see
            # this module's docstring): without a real resolution, letting Transformer derive
            # smax from self._map_load_resolution (the DEFAULT_RMASK placeholder) would silently
            # render an over-sharpened model density using a fake resolution - matches qfit.py's
            # own QFitBase.__init__, which sets simple=True in this exact same situation.
            transformer = get_transformer("qfit", residue_structure, map_models[name],
                                           em=self.em, simple=(self.resolution is None))
            mask = transformer.get_conformers_mask(coor_set, self.rmask)
            target = maps[name].array[mask]
            for density in transformer.get_conformers_densities(coor_set, bfactor_array):
                model_density = density[mask]
                np.maximum(model_density, scaled_bulk_solvent, out=model_density)
                correlation_matrix = np.corrcoef(model_density, target)
                rsccs.append(correlation_matrix[0, 1])
        return max(rsccs)

    def run(self):
        maps, map_models = self._load_maps()

        model_serials = self._get_model_serials(self.structure_pdb)
        raw_models = Structure.fromfile(str(self.structure_pdb)).split_models()

        if len(raw_models) == 0:
            raise ValueError(f'No models found in {self.structure_pdb}')

        # No MODEL records (single-model PDB) or a serial/model count mismatch: fall back to
        # sequential 1-based indices, so a single-model structure is reported as model 1.
        if not model_serials or len(model_serials) != len(raw_models):
            if model_serials:
                print(f'Warning: found {len(model_serials)} MODEL serial(s) in {self.structure_pdb} '
                      f'but split_models() returned {len(raw_models)} model(s); falling back to '
                      f'sequential indices (1-based) as the model idx.')
            model_serials = list(range(1, len(raw_models) + 1))

        models = [self._clean_structure(m) for m in raw_models]

        results = []
        for model_idx, model in zip(model_serials, models):
            residue_keys = self._find_residue_keys(model)
            for chain_id, resi, altloc in residue_keys:
                residue_structure = self._extract_residue(model, chain_id, resi, altloc)
                coor = residue_structure.coor.copy()
                if coor.shape[0] == 0:
                    continue
                label = self._residue_label(chain_id, resi, altloc)
                try:
                    rscc = self._score_residue(residue_structure, coor, maps, map_models)
                except Exception as e:
                    print(f'Warning: failed to score model {model_idx} residue {label} '
                          f'({type(e).__name__}: {e}); skipping.')
                    continue
                results.append((model_idx, label, rscc))

        self._write_csv(results)

    def _write_csv(self, results):
        with open(self.output_csv, 'w+') as f:
            f.write('model_idx,residue,rscc\n')
            for model_idx, label, rscc in results:
                f.write(f'{model_idx},{label},{rscc}\n')
        print(f'{len(results)} residue RSCC value(s) written to {self.output_csv}')


def main():
    args = build_argparser().parse_args()
    calculator = ResidueRSCCCalculator(
        args.structure_pdb, args.map_files, args.output_csv, resolution=args.resolution,
        bfactor=args.bfactor, em=args.em, label=args.label, ligand_resname=args.ligand_resname,
    )
    calculator.run()


if __name__ == '__main__':
    main()
