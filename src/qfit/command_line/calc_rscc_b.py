import argparse
import re
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from qfit import Structure
from qfit import XMap
from qfit.xtal.transformer import get_transformer


# Default B-factor used when generating model density for each residue conformer.
DEFAULT_BFACTOR = 20

# Matches the BDC value embedded in an event map filename, e.g.
# 'x00407-1-event_1_1-BDC_0.08_map.native.ccp4' -> '0.08'.
BDC_PATTERN = re.compile(r'1-BDC_([\d.]+)_')


def parse_bdc(map_filename):
    """Extracts the BDC value (as a string, so '0.080' and '0.08' aren't
    silently treated as equal) from an event map filename. Returns None if
    the filename doesn't match the expected '1-BDC_<value>_' pattern."""
    m = BDC_PATTERN.search(str(map_filename))
    return m.group(1) if m else None


def parse_bfactors(raw):
    """Parses a --bfactors value ('20' or '20,40,60,80,100') into a sorted
    list of unique floats."""
    return sorted({float(b) for b in str(raw).split(',')})


def format_bfactor(bfactor):
    """Formats a bfactor for use in a csv value, e.g. 20.0 -> '20',
    22.5 -> '22.5', so whole-number bfactors don't get a pointless '.0'
    (same helper as real_space_refine_protein.py's format_bfactor)."""
    if float(bfactor).is_integer():
        return str(int(bfactor))
    return str(bfactor)


def parse_residues_file(path):
    """Reads a headerless file of '{chain}{resnum}' labels, one per line
    (e.g. residues_with_placer_conformers.csv), and returns them as a set.
    Used to restrict the (expensive - every residue x every event map x
    every bfactor) sweep to just these residues. Altloc-insensitive: a
    residue key (chain_id, resi, altloc) is included if its base
    '{chain_id}{resi}' (no altloc suffix) is in this set."""
    residues = set()
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                residues.add(line)
    return residues


def build_argparser():
    p = argparse.ArgumentParser(
        description="Calculate per-residue RSCC for a single-model or multimodel PDB "
                    "structure against one or more density maps, at one or more B-factors, "
                    "restricted to an explicit list of residues (this sweep is expensive)."
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
        help='Path(s) to one or more density map files (e.g. .ccp4 maps) to score each residue '
             'against. Every residue is scored against every map, at every --bfactors value, '
             'and every (map, bfactor) combination is written out as its own row - no maximum '
             'is taken across maps. Maps whose filename embeds a duplicate 1-BDC_<value>_ (same '
             'background-subtraction fraction as an already-loaded map) are skipped, since they '
             'are identical to that map.'
    )
    p.add_argument(
        'resolution',
        type=float,
        help='Resolution (Å) of the map(s), used both for loading the XMap(s) and for the mask radius'
    )
    p.add_argument(
        'output_csv',
        type=Path,
        help='Path to write the per-residue, per-map, per-bfactor RSCC csv'
    )
    p.add_argument(
        'residues_file',
        type=Path,
        help='Path to a headerless file listing residue labels ("{chain}{resnum}", one per '
             'line - e.g. residues_with_placer_conformers.csv) to restrict the calculation to. '
             'Required: this sweep (every residue x every event map x every bfactor) is '
             'expensive enough that running it over every residue in the structure is not '
             'practical.'
    )
    p.add_argument(
        '--bfactors',
        type=str,
        default=str(DEFAULT_BFACTOR),
        help='B-factor(s) used when generating each residue\'s model density: a single value '
             f'(e.g. "20") or a comma-separated list (e.g. "20,40,60,80,100"). RSCC is '
             'calculated separately at every (map, bfactor) combination - a residue with 4 '
             'maps and 5 bfactors gets 20 rows in the output csv. '
             f'(default: {DEFAULT_BFACTOR})',
    )
    return p


class ResidueRSCCCalculator:
    def __init__(self, structure_pdb, map_files, resolution, output_csv, residues,
                 bfactors=(DEFAULT_BFACTOR,)):
        self.structure_pdb = Path(structure_pdb)
        self.map_files = [Path(m) for m in map_files]
        self.resolution = resolution
        self.output_csv = Path(output_csv)
        self.residues = set(residues)
        self.bfactors = list(bfactors)
        self.rmask = 0.5 + resolution / 3.0  # from qfit

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
        Returns [(chain_id, resi, altloc), ...] for every residue in a single-model structure
        that's in self.residues (matched by base '{chain_id}{resi}' label, altloc-insensitive).

        A residue group with any non-blank altloc (e.g. 'A', whether or not a second altloc
        competes at that same residue) is split into one key per altloc, since each altloc
        represents a physically distinct conformer that should be scored independently rather
        than collapsed into one residue. altloc is '' only for residues with no altloc at all.
        """
        keys = []
        for chain in structure._pdb_hierarchy.only_model().chains():
            chain_id = chain.id.strip()
            for residue_group in chain.residue_groups():
                resi = int(residue_group.resseq)
                if f'{chain_id}{resi}' not in self.residues:
                    continue
                altlocs = sorted({ag.altloc.strip() for ag in residue_group.atom_groups()})
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
            print(f'Loading map {map_file} at resolution {self.resolution}')
            maps[name] = XMap.fromfile(str(map_file), resolution=self.resolution)
            map_model = maps[name].zeros_like(maps[name])
            map_model.set_space_group("P1")
            map_models[name] = map_model
        return maps, map_models

    def _score_residue_rows(self, residue_structure, coor, maps, map_models):
        """Computes RSCC for a single residue conformer against every map, at every
        B-factor in self.bfactors. Returns [(map_name, bfactor, rscc, spearmans_rho), ...],
        one row per (map, bfactor) combination - no maximum is taken across maps or bfactors
        here. spearmans_rho is the spearman correlation of bfactor vs rscc within that
        (map, bfactor-sweep) group - the same value on every row for a given map, since it's
        a property of the whole per-map bfactor sweep, not of any one bfactor. Left as None
        (written as an empty csv value) when a map has fewer than 2 distinct bfactor rows,
        since a rank correlation isn't defined for a single point.

        A fresh transformer is created for every (bfactor, map) combination, and
        residue_structure.b is set to the current bfactor immediately before building it -
        qFit's Transformer only computes its radial densities from the structure's B-factor
        once per instance (on first use) and silently reuses them afterward, so reusing a
        transformer across bfactors (or forgetting to update .b first) would silently score
        every bfactor after the first against the first bfactor's density. Same fix applied
        in real_space_refine_protein.py's score_residue_rsccs, which this mirrors."""
        scaled_bulk_solvent = 0
        coor_set = [coor]

        rows_by_map = {}
        for bfactor in self.bfactors:
            for name in maps:
                residue_structure.b = bfactor
                transformer = get_transformer("qfit", residue_structure, map_models[name])
                mask = transformer.get_conformers_mask(coor_set, self.rmask)
                target = maps[name].array[mask]
                for density in transformer.get_conformers_densities(coor_set, [bfactor]):
                    model_density = density[mask]
                    np.maximum(model_density, scaled_bulk_solvent, out=model_density)
                    correlation_matrix = np.corrcoef(model_density, target)
                    rscc = correlation_matrix[0, 1]
                rows_by_map.setdefault(name, []).append((bfactor, rscc))

        rows = []
        for name, bfactor_rscc_pairs in rows_by_map.items():
            bfactors = [b for b, _ in bfactor_rscc_pairs]
            rsccs = [r for _, r in bfactor_rscc_pairs]
            rho = None
            if len(set(bfactors)) >= 2:
                rho, _ = spearmanr(bfactors, rsccs)
            for bfactor, rscc in bfactor_rscc_pairs:
                rows.append((name, bfactor, rscc, rho))
        return rows

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
                    rows = self._score_residue_rows(residue_structure, coor, maps, map_models)
                except Exception as e:
                    print(f'Warning: failed to score model {model_idx} residue {label} '
                          f'({type(e).__name__}: {e}); skipping.')
                    continue
                for map_name, bfactor, rscc, rho in rows:
                    results.append((model_idx, label, map_name, bfactor, rscc, rho))

        self._write_csv(results)

    def _write_csv(self, results):
        with open(self.output_csv, 'w+') as f:
            f.write('model_idx,residue,event_map,bfactor,rscc,spearmans_rho\n')
            for model_idx, label, map_name, bfactor, rscc, rho in results:
                rho_str = '' if rho is None or np.isnan(rho) else str(rho)
                f.write(f'{model_idx},{label},{map_name},{format_bfactor(bfactor)},{rscc},{rho_str}\n')
        print(f'{len(results)} row(s) written to {self.output_csv}')


def main():
    args = build_argparser().parse_args()
    calculator = ResidueRSCCCalculator(
        args.structure_pdb, args.map_files, args.resolution, args.output_csv,
        parse_residues_file(args.residues_file), parse_bfactors(args.bfactors),
    )
    calculator.run()


if __name__ == '__main__':
    main()
