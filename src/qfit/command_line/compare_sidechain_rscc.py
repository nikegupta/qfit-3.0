import argparse
import time
from pathlib import Path
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from qfit import Structure
from qfit import XMap
from qfit.xtal.transformer import get_transformer


# Default minimum sidechain RMSD (Å) for two conformers of the same residue, across models of
# a multimodel file, to be treated as distinct. Below this, a conformer is considered a
# duplicate of one already collected and is not scored. Files like rotamer_optimized.pdb repeat
# identical coordinates for a residue across every model, so this collapses to a single scored
# conformer; files like filtered_models.pdb have genuinely different coordinates per model, so
# every distinct rotamer is kept and scored.
DEFAULT_RMSD_CUTOFF = 0.1


# Symmetry-aware sidechain RMSD calc, ported from the rotamer optimizer script.
def _get_coordinate_rmsd(reference_coordinates, new_coordinate_set, atom_names=None):
    reference_coordinates = np.array(reference_coordinates)
    new_coordinate_set = np.array(new_coordinate_set)

    # Build mask to exclude backbone atoms
    backbone_atoms = {"N", "CA", "C", "O"}
    if atom_names is not None:
        sidechain_mask = np.array([name not in backbone_atoms for name in atom_names])
    else:
        sidechain_mask = np.ones(reference_coordinates.shape[0], dtype=bool)

    ref_sc = reference_coordinates[sidechain_mask]
    new_sc = new_coordinate_set[:, sidechain_mask, :]

    delta = new_sc - ref_sc
    rmsds = np.sqrt(np.square(delta).sum(axis=2).sum(axis=1))

    if atom_names is not None:
        atom_names = list(atom_names)
        sc_names = [name for name in atom_names if name not in backbone_atoms]
        flip_pairs = None
        if "CD1" in sc_names and "CD2" in sc_names and "CE1" in sc_names and "CE2" in sc_names:
            flip_pairs = [
                (sc_names.index("CD1"), sc_names.index("CD2")),
                (sc_names.index("CE1"), sc_names.index("CE2")),
            ]
        if flip_pairs is not None:
            flipped = new_sc.copy()
            for i, j in flip_pairs:
                flipped[:, i, :], flipped[:, j, :] = flipped[:, j, :].copy(), flipped[:, i, :].copy()
            delta_flipped = flipped - ref_sc
            rmsds_flipped = np.sqrt(np.square(delta_flipped).sum(axis=2).sum(axis=1))
            rmsds = np.minimum(rmsds, rmsds_flipped)

    return min(rmsds)


def build_argparser():
    p = argparse.ArgumentParser()
    p.add_argument(
        'datasets_dir',
        type=Path,
        help='Path to the directory containing all dataset directories'
    )
    p.add_argument(
        'reference_pdb',
        type=str,
        help="Template path to each dataset's reference PDB. Use '{dataset}' as a placeholder for the "
             "dataset name, e.g. '/home/ngupta/main/reference_set/{dataset}-pandda-model.pdb'"
    )
    p.add_argument(
        'multimodel_pdb',
        type=str,
        help="Template path to each dataset's multimodel test PDB. Use '{dataset}' as a placeholder for "
             "the dataset name, e.g. '/.../datasets/{dataset}/run_1/placer_100/filter_4/filtered_models.pdb'"
    )
    p.add_argument(
        'output_folder',
        type=Path,
        help='Folder to save the pooled RSCC comparison csv and scatterplots to'
    )
    p.add_argument(
        "--csv_file",
        default='/home/ngupta/main/20251002_pxr_datasets.csv',
        type=Path,
        help="CSV file used to look up resolution per dataset",
    )
    p.add_argument(
        "--n_cpus",
        default=24,
        type=int,
        help="Max number of datasets to process in parallel",
    )
    p.add_argument(
        "--rmsd_cutoff",
        default=DEFAULT_RMSD_CUTOFF,
        type=float,
        help="Minimum sidechain RMSD (Å) for two conformers of the same residue to be treated "
             "as distinct across models of the multimodel pdb",
    )
    return p


class RSCC_Comparator():
    def __init__(self, datasets_dir, reference_pdb_pattern, multimodel_pdb_pattern, output_folder,
                 csv_file, n_cpus=24, rmsd_cutoff=DEFAULT_RMSD_CUTOFF):
        self.datasets_dir = datasets_dir
        self.reference_pdb_pattern = reference_pdb_pattern
        self.multimodel_pdb_pattern = multimodel_pdb_pattern
        self.output_path = Path(output_folder)
        os.makedirs(self.output_path, exist_ok=True)
        self.csv_file = csv_file
        self.n_cpus = n_cpus
        self.rmsd_cutoff = rmsd_cutoff

        self.default_bfactor = 20
        self.resolutions = self._load_resolution_lookup()

    def _load_resolution_lookup(self):
        """Parses the datasets CSV into {dataset_name: resolution}."""
        resolutions = {}
        with open(self.csv_file) as f:
            for line in f:
                if 'avd-pxr-' not in line:
                    continue
                fields = line.strip().split(',')
                dataset = fields[1].replace('avd-pxr-', '')
                resolution = float(fields[2])
                resolutions[dataset] = resolution
        return resolutions

    def _load_event_maps(self, dataset_dir, resolution):
        event_maps = {}
        event_maps_models = {}
        event_map_files = sorted(dataset_dir.glob('*-event_*_*-BDC_*_map.native.ccp4'))
        for event_file in event_map_files:
            # Use full filename as key
            event_name = str(event_file).split('/')[-1]  # e.g., "x01325-1-event_1_1-BDC_0.3_map.native.ccp4"
            event_maps[event_name] = XMap.fromfile(str(event_file), resolution=resolution)

            # make copies for density steps
            event_map_model = event_maps[event_name].zeros_like(event_maps[event_name])
            event_map_model.set_space_group("P1")
            event_maps_models[event_name] = event_map_model
        return event_maps, event_maps_models

    def _clean_structure(self, structure):
        """Remove hydrogens and rename OXT->O, matching what qfit's transformer expects."""
        structure = structure.extract("e", "H", "!=")
        rename = structure.extract("name", "OXT", "==")
        rename.name = "O"
        structure = structure.extract("name", "OXT", "!=").combine(rename)
        return structure

    def _get_residue_keys(self, structure):
        """Returns {(chain_id, resi): resn} for every residue in a single-model structure."""
        residues = {}
        for chain in structure._pdb_hierarchy.only_model().chains():
            chain_id = chain.id.strip()
            for residue_group in chain.residue_groups():
                resi = int(residue_group.resseq)
                resn = residue_group.atom_groups()[0].resname.strip()
                residues[(chain_id, resi)] = resn
        return residues

    def _get_unique_residues(self, models):
        """
        Walks a list of single-model Structures (from a split multimodel file) and, for each
        (chain_id, resi), collects every conformer whose sidechain RMSD to all previously
        collected conformers of that residue exceeds self.rmsd_cutoff.

        Files like rotamer_optimized.pdb repeat identical coordinates for a residue across
        every model, so this collapses to a single scored conformer per residue (preserving
        the efficiency of scoring only one copy). Files like filtered_models.pdb have
        genuinely different coordinates per model, so every distinct rotamer is kept.

        Returns {(chain_id, resi): (resn, residue_structure, coor_set)}, where
        residue_structure is a representative single-conformer Structure (used only for its
        atom names/topology when scoring) and coor_set is the list of unique coordinate
        arrays found for that residue.
        """
        residue_structures = {}  # key -> (resn, residue_structure)
        coor_sets = {}  # key -> list of coordinate arrays

        for model in models:
            for chain in model._pdb_hierarchy.only_model().chains():
                chain_id = chain.id.strip()
                for residue_group in chain.residue_groups():
                    resi = int(residue_group.resseq)
                    key = (chain_id, resi)
                    resn = residue_group.atom_groups()[0].resname.strip()
                    resi_selstr = f"chain {chain_id} and resi {resi}"
                    residue_structure = model.extract(resi_selstr)

                    if key not in residue_structures:
                        residue_structures[key] = (resn, residue_structure)
                        coor_sets[key] = [residue_structure.coor.copy()]
                        continue

                    existing_coor_set = coor_sets[key]
                    min_rmsd = _get_coordinate_rmsd(
                        residue_structure.coor, np.array(existing_coor_set), residue_structure.name
                    )
                    if min_rmsd > self.rmsd_cutoff:
                        coor_sets[key].append(residue_structure.coor.copy())

        residues = {}
        for key, (resn, residue_structure) in residue_structures.items():
            residues[key] = (resn, residue_structure, coor_sets[key])
        return residues

    def _score_conformers(self, residue, coor_set, event_maps, event_maps_models, rmask):
        """
        Returns a list, one entry per conformer in coor_set, of that conformer's highest RSCC
        against any event map for this dataset.

        Each conformer is scored on its own (as a coor_set of length 1), rather than batching
        the whole coor_set through the transformer at once. Batching shares a single mask
        across all conformers, and that mask is the union of every conformer's footprint - so
        residues with more/more-divergent conformers get a larger mask than residues scored
        with a single conformer, making RSCC values not directly comparable across residues.
        Scoring one conformer at a time keeps the mask (and therefore the RSCC) comparable
        regardless of how many conformers were collected for a given residue, at the cost of
        redoing the transformer setup per conformer instead of amortizing it across a batch.
        """
        return [
            self._score_single_conformer(residue, coor, event_maps, event_maps_models, rmask)
            for coor in coor_set
        ]

    def _score_single_conformer(self, residue, coor, event_maps, event_maps_models, rmask):
        """Converts a single conformer's coordinates to density and returns its highest RSCC
        against any event map for this dataset."""
        scaled_bulk_solvent = 0  # from qfit, maybe should be different
        coor_set = [coor]
        bfactor_array = [self.default_bfactor]

        rsccs = []
        for event_map_name in list(event_maps.keys()):
            transformer = get_transformer("qfit", residue, event_maps_models[event_map_name])

            mask = transformer.get_conformers_mask(coor_set, rmask)
            target = event_maps[event_map_name].array[mask]
            for density in transformer.get_conformers_densities(coor_set, bfactor_array):
                model = density[mask]
                np.maximum(model, scaled_bulk_solvent, out=model)
                correlation_matrix = np.corrcoef(model, target)
                rscc = correlation_matrix[0, 1]
                rsccs.append(rscc)

        return max(rsccs)

    def _process_dataset(self, dataset, dataset_dir):
        resolution = self.resolutions.get(dataset)
        if resolution is None:
            print(f'Warning: no resolution found for {dataset} in {self.csv_file}, skipping.')
            return [], []

        reference_pdb = Path(self.reference_pdb_pattern.format(dataset=dataset))
        multimodel_pdb = Path(self.multimodel_pdb_pattern.format(dataset=dataset))

        if not reference_pdb.exists():
            print(f'Warning: no reference pdb found for {dataset} at {reference_pdb}, skipping.')
            return [], []
        if not multimodel_pdb.exists():
            print(f'Warning: no multimodel pdb found for {dataset} at {multimodel_pdb}, skipping.')
            return [], []

        print(f'Processing {dataset}: resolution={resolution}')
        time0 = time.time()

        event_maps, event_maps_models = self._load_event_maps(dataset_dir, resolution)
        if not event_maps:
            print(f'Warning: no event maps found for {dataset} in {dataset_dir}, skipping.')
            return [], []
        rmask = 0.5 + resolution / 3.0  # from qfit

        reference_structure = self._clean_structure(Structure.fromfile(str(reference_pdb)))
        reference_keys = self._get_residue_keys(reference_structure)

        multimodel_models = [
            self._clean_structure(model)
            for model in Structure.fromfile(str(multimodel_pdb)).split_models()
        ]
        multimodel_residues = self._get_unique_residues(multimodel_models)

        shared_keys = sorted(set(reference_keys) & set(multimodel_residues.keys()))

        results = []
        mismatches = []
        for key in shared_keys:
            chain_id, resi = key
            ref_resn = reference_keys[key]
            test_resn, multimodel_residue, coor_set = multimodel_residues[key]

            if ref_resn != test_resn:
                print(f'WARNING: mismatched residue identity at {dataset} {chain_id}{resi}: '
                      f'reference={ref_resn}, multimodel={test_resn}. Skipping.')
                mismatches.append((dataset, chain_id, resi, ref_resn, test_resn))
                continue

            reference_residue = reference_structure.extract(f"chain {chain_id} and resi {resi}")

            reference_rscc = self._score_conformers(
                reference_residue, [reference_residue.coor], event_maps, event_maps_models, rmask
            )[0]

            conformer_rsccs = self._score_conformers(
                multimodel_residue, coor_set, event_maps, event_maps_models, rmask
            )
            best_rscc = max(conformer_rsccs)
            median_rscc = float(np.median(conformer_rsccs))

            results.append((
                dataset, chain_id, resi, ref_resn, reference_rscc, best_rscc, median_rscc, len(coor_set)
            ))

        print(f'Completed {dataset}: {len(results)} residues scored ({time.time() - time0:.2f}s)')

        return results, mismatches

    def run(self):
        print(self.__dict__)

        dataset_dirs = sorted(d for d in self.datasets_dir.iterdir() if d.is_dir())
        n_workers = max(1, min(self.n_cpus, os.cpu_count(), len(dataset_dirs)))

        all_results = []
        all_mismatches = []
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {
                executor.submit(self._process_dataset, dataset_dir.name, dataset_dir): dataset_dir.name
                for dataset_dir in dataset_dirs
            }
            for future in as_completed(futures):
                dataset = futures[future]
                try:
                    results, mismatches = future.result()
                except Exception as e:
                    print(f'Error processing {dataset}: {e}')
                    continue
                all_results.extend(results)
                all_mismatches.extend(mismatches)

        print(f'Pooled {len(all_results)} residues across {len(dataset_dirs)} datasets ({n_workers} workers)')

        self._write_csv(all_results, all_mismatches)
        self._plot_scatter(
            [r[4] for r in all_results], [r[5] for r in all_results],
            'Best Multimodel RSCC', 'Reference vs Best Multimodel RSCC', 'rscc_scatter_best.png'
        )
        self._plot_scatter(
            [r[4] for r in all_results], [r[6] for r in all_results],
            'Median Multimodel RSCC', 'Reference vs Median Multimodel RSCC', 'rscc_scatter_median.png'
        )

    def _write_csv(self, results, mismatches):
        rscc_output = self.output_path / 'rscc_comparison.csv'
        with open(rscc_output, 'w+') as f:
            f.write('dataset,residue,resname,reference_rscc,best_multimodel_rscc,median_multimodel_rscc,'
                     'n_conformers,delta_best,delta_median')
            f.write('\n')
            for dataset, chain_id, resi, resn, reference_rscc, best_rscc, median_rscc, n_conformers in results:
                f.write(f'{dataset},{chain_id}{resi},{resn},{reference_rscc},{best_rscc},{median_rscc},'
                         f'{n_conformers},{best_rscc - reference_rscc},{median_rscc - reference_rscc}')
                f.write('\n')

        if mismatches:
            mismatch_output = self.output_path / 'residue_mismatches.csv'
            with open(mismatch_output, 'w+') as f:
                f.write('dataset,residue,reference_resname,multimodel_resname')
                f.write('\n')
                for dataset, chain_id, resi, ref_resn, test_resn in mismatches:
                    f.write(f'{dataset},{chain_id}{resi},{ref_resn},{test_resn}')
                    f.write('\n')
            print(f'{len(mismatches)} mismatched residue(s) written to {mismatch_output}')

    def _plot_scatter(self, reference_rsccs, comparison_rsccs, y_label, title, filename):
        from scipy.stats import gaussian_kde
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        reference_rsccs = np.array(reference_rsccs)
        comparison_rsccs = np.array(comparison_rsccs)

        # Color points by local point density (kde), plotting densest points last so they're on top
        xy = np.vstack([reference_rsccs, comparison_rsccs])
        density = gaussian_kde(xy)(xy)
        order = density.argsort()
        reference_rsccs, comparison_rsccs, density = reference_rsccs[order], comparison_rsccs[order], density[order]

        fig, ax = plt.subplots(figsize=(6, 6))
        points = ax.scatter(reference_rsccs, comparison_rsccs, c=density, cmap='viridis', s=6, edgecolor='none')
        fig.colorbar(points, ax=ax, label='point density')

        lims = [0, 1]
        ax.plot(lims, lims, linestyle='--', color='gray', linewidth=1)
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_xlabel('Reference RSCC')
        ax.set_ylabel(y_label)
        ax.set_title(f'{title} (n={len(reference_rsccs)})')
        ax.set_aspect('equal')

        fig_output = self.output_path / filename
        fig.tight_layout()
        fig.savefig(fig_output, dpi=300)
        plt.close(fig)
        print(f'scatterplot written to {fig_output}')


def main():
    args = build_argparser().parse_args()
    comparator = RSCC_Comparator(
        args.datasets_dir, args.reference_pdb, args.multimodel_pdb, args.output_folder,
        args.csv_file, args.n_cpus, args.rmsd_cutoff
    )
    comparator.run()


if __name__ == '__main__':
    main()