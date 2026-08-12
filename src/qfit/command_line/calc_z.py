import argparse
from pathlib import Path

from qfit import Structure
from qfit import XMap
from qfit.xtal.transformer import get_transformer


def build_argparser():
    p = argparse.ArgumentParser(
        description="Calculate per-residue Z-map statistics (max/min/average Z-score) for a "
                    "single-model or multimodel PDB structure, using the same per-residue "
                    "voxel mask as calc_rscc's RSCC calculation (and every other density "
                    "calculation in this pipeline)."
    )
    p.add_argument(
        'structure_pdb',
        type=Path,
        help='Path to a single-model or multimodel PDB structure'
    )
    p.add_argument(
        'zmap_file',
        type=Path,
        help="Path to the dataset's Z-map file (PanDDA's native-frame "
             "'<dataset>-z_map.native.ccp4', same file fit_ligand.py's LigandPlacer loads to "
             "find PLACER peaks). Unlike event maps, there is exactly one Z-map per dataset, "
             "so no deduplication/max-across-maps logic applies here."
    )
    p.add_argument(
        'resolution',
        type=float,
        help='Resolution (Å) of the map, used both for loading the XMap and for the mask radius'
    )
    p.add_argument(
        'output_csv',
        type=Path,
        help='Path to write the per-residue Z-map statistics csv'
    )
    return p


class ResidueZCalculator:
    def __init__(self, structure_pdb, zmap_file, resolution, output_csv):
        self.structure_pdb = Path(structure_pdb)
        self.zmap_file = Path(zmap_file)
        self.resolution = resolution
        self.output_csv = Path(output_csv)
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

    def _load_zmap(self):
        """
        Loads the Z-map (same XMap.fromfile convention as fit_ligand.py's LigandPlacer.zmap),
        along with a zeroed-template XMap on the same grid, used to build each residue's mask -
        same pattern as calc_rscc's map_models, just for a single map instead of a list.
        """
        print(f'Loading Z-map {self.zmap_file} at resolution {self.resolution}')
        zmap = XMap.fromfile(str(self.zmap_file), resolution=self.resolution)
        zmap_model = zmap.zeros_like(zmap)
        zmap_model.set_space_group("P1")
        return zmap, zmap_model

    def _score_residue(self, residue_structure, coor, zmap, zmap_model):
        """
        Masks the Z-map down to a single residue conformer's voxels - the same per-residue
        voxel mask calc_rscc's RSCC calculation (and every other density calculation in this
        pipeline) uses - and returns (max_z, min_z, average_z) over that mask. Unlike RSCC, no
        model density is generated here: a Z-map's values are read directly rather than
        compared against a model, so b-factor plays no role (get_conformers_mask depends only
        on atomic coordinates/radii, not b-factor).
        """
        coor_set = [coor]
        transformer = get_transformer("qfit", residue_structure, zmap_model)
        mask = transformer.get_conformers_mask(coor_set, self.rmask)
        target = zmap.array[mask]
        return float(target.max()), float(target.min()), float(target.mean())

    def run(self):
        zmap, zmap_model = self._load_zmap()

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
                    max_z, min_z, average_z = self._score_residue(
                        residue_structure, coor, zmap, zmap_model
                    )
                except Exception as e:
                    print(f'Warning: failed to score model {model_idx} residue {label} '
                          f'({type(e).__name__}: {e}); skipping.')
                    continue
                results.append((model_idx, label, max_z, min_z, average_z))

        self._write_csv(results)

    def _write_csv(self, results):
        with open(self.output_csv, 'w+') as f:
            f.write('model_idx,residue,max_z,min_z,average_z\n')
            for model_idx, label, max_z, min_z, average_z in results:
                f.write(f'{model_idx},{label},{max_z},{min_z},{average_z}\n')
        print(f'{len(results)} residue Z-map statistics row(s) written to {self.output_csv}')


def main():
    args = build_argparser().parse_args()
    calculator = ResidueZCalculator(
        args.structure_pdb, args.zmap_file, args.resolution, args.output_csv
    )
    calculator.run()


if __name__ == '__main__':
    main()
