import argparse
from pathlib import Path

import numpy as np

from qfit import Structure
from qfit import XMap
from qfit.xtal.transformer import get_transformer


# Default B-factor used when generating model density for each residue conformer.
DEFAULT_BFACTOR = 20


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
        help='Path(s) to one or more density map files (e.g. .ccp4 maps) to score each residue '
             'against. A residue\'s reported RSCC is the max across all maps given.'
    )
    p.add_argument(
        'resolution',
        type=float,
        help='Resolution (Å) of the map(s), used both for loading the XMap(s) and for the mask radius'
    )
    p.add_argument(
        'output_csv',
        type=Path,
        help='Path to write the per-residue RSCC csv'
    )
    p.add_argument(
        '--bfactor',
        type=float,
        default=DEFAULT_BFACTOR,
        help=f'B-factor used when generating each residue\'s model density (default: {DEFAULT_BFACTOR})',
    )
    return p


class ResidueRSCCCalculator:
    def __init__(self, structure_pdb, map_files, resolution, output_csv, bfactor=DEFAULT_BFACTOR):
        self.structure_pdb = Path(structure_pdb)
        self.map_files = [Path(m) for m in map_files]
        self.resolution = resolution
        self.output_csv = Path(output_csv)
        self.bfactor = bfactor
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
        Returns [(chain_id, resi, altloc), ...] for every residue in a single-model structure.

        A residue group that contains two or more distinct non-blank altlocs (e.g. 'A' and 'B')
        is split into one key per altloc, since each altloc represents a physically distinct
        conformer that should be scored independently rather than collapsed into one residue.
        altloc is '' for residues with no altloc disorder.
        """
        keys = []
        for chain in structure._pdb_hierarchy.only_model().chains():
            chain_id = chain.id.strip()
            for residue_group in chain.residue_groups():
                resi = int(residue_group.resseq)
                altlocs = sorted({ag.altloc.strip() for ag in residue_group.atom_groups()})
                non_blank_altlocs = [a for a in altlocs if a != '']
                if len(non_blank_altlocs) >= 2:
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
        """
        maps = {}
        map_models = {}
        for map_file in self.map_files:
            name = map_file.name
            print(f'Loading map {map_file} at resolution {self.resolution}')
            maps[name] = XMap.fromfile(str(map_file), resolution=self.resolution)
            map_model = maps[name].zeros_like(maps[name])
            map_model.set_space_group("P1")
            map_models[name] = map_model
        return maps, map_models

    def _score_residue(self, residue_structure, coor, maps, map_models):
        """Converts a single residue conformer's coordinates to density and returns its highest
        RSCC across all provided maps, using that conformer's own mask (recomputed per map,
        since each map may have a different grid)."""
        scaled_bulk_solvent = 0
        coor_set = [coor]
        bfactor_array = [self.bfactor]

        rsccs = []
        for name in maps:
            transformer = get_transformer("qfit", residue_structure, map_models[name])
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
        args.structure_pdb, args.map_files, args.resolution, args.output_csv, args.bfactor
    )
    calculator.run()


if __name__ == '__main__':
    main()