import argparse
import time
from pathlib import Path
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from qfit import Structure
from qfit import XMap
from qfit.xtal.transformer import get_transformer


def build_argparser():
    p = argparse.ArgumentParser()
    p.add_argument(
        'datasets_dir',
        type=Path,
        help='Path to the directory containing all dataset directories'
    )
    p.add_argument(
        'reference_dir',
        type=Path,
        help='Path to the directory containing reference PDBs, named <dataset>-pandda-model.pdb'
    )
    p.add_argument(
        'run_name',
        type=str,
        help='Run name'
    )
    p.add_argument(
        'placer_run_name',
        type=str,
        help='Placer run name'
    )
    p.add_argument(
        'filter_run_name',
        type=str,
        help='Filter run name'
    )
    p.add_argument(
        'rotamer_run_name',
        type=str,
        help='Rotamer-optimization run name'
    )
    p.add_argument(
        'output_folder',
        type=Path,
        help='Folder to save the pooled RSCC comparison csv and scatterplot to'
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
    return p


class RSCC_Comparator():
    def __init__(self, datasets_dir, reference_dir, run_name, placer_run_name, filter_run_name,
                 rotamer_run_name, output_folder, csv_file, n_cpus=24):
        self.datasets_dir = datasets_dir
        self.reference_dir = reference_dir
        self.run_name = run_name
        self.placer_run_name = placer_run_name
        self.filter_run_name = filter_run_name
        self.rotamer_run_name = rotamer_run_name
        self.output_path = Path(output_folder)
        os.makedirs(self.output_path, exist_ok=True)
        self.csv_file = csv_file
        self.n_cpus = n_cpus

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

    def _get_first_unique_residues(self, models):
        """
        Walks a list of single-model Structures (from a split multimodel file) and
        keeps only the first instance found of each (chain_id, resi). Coordinates
        for a given (chain_id, resi) are identical across every model in the
        rotamer-optimized file, so the first instance found is representative.
        """
        residues = {}
        for model in models:
            for chain in model._pdb_hierarchy.only_model().chains():
                chain_id = chain.id.strip()
                for residue_group in chain.residue_groups():
                    resi = int(residue_group.resseq)
                    key = (chain_id, resi)
                    if key in residues:
                        continue
                    resn = residue_group.atom_groups()[0].resname.strip()
                    resi_selstr = f"chain {chain_id} and resi {resi}"
                    residue_structure = model.extract(resi_selstr)
                    residues[key] = (resn, residue_structure)
        return residues

    def _calc_rscc_all_events(self, residue, event_maps, event_maps_models, rmask):
        """
        Converts a single residue's coordinates to density using the qfit
        transformer and returns the highest RSCC against any event map for
        this dataset.
        """
        scaled_bulk_solvent = 0  # from qfit, maybe should be different
        coor_set = [residue.coor]
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

        top_rscc = max(rsccs)

        return top_rscc

    def _process_dataset(self, dataset, dataset_dir):
        resolution = self.resolutions.get(dataset)
        if resolution is None:
            print(f'Warning: no resolution found for {dataset} in {self.csv_file}, skipping.')
            return [], []

        reference_pdb = self.reference_dir / f'{dataset}-pandda-model.pdb'
        rotamer_optimized_pdb = (
            dataset_dir / self.run_name / self.placer_run_name / self.filter_run_name
            / self.rotamer_run_name / 'rotamer_optimized.pdb'
        )

        if not reference_pdb.exists():
            print(f'Warning: no reference pdb found for {dataset} at {reference_pdb}, skipping.')
            return [], []
        if not rotamer_optimized_pdb.exists():
            print(f'Warning: no rotamer-optimized pdb found for {dataset} at {rotamer_optimized_pdb}, skipping.')
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

        rotamer_models = [
            self._clean_structure(model)
            for model in Structure.fromfile(str(rotamer_optimized_pdb)).split_models()
        ]
        rotamer_residues = self._get_first_unique_residues(rotamer_models)

        shared_keys = sorted(set(reference_keys) & set(rotamer_residues.keys()))

        results = []
        mismatches = []
        for key in shared_keys:
            chain_id, resi = key
            ref_resn = reference_keys[key]
            rot_resn, rotamer_residue = rotamer_residues[key]

            if ref_resn != rot_resn:
                print(f'WARNING: mismatched residue identity at {dataset} {chain_id}{resi}: '
                      f'reference={ref_resn}, rotamer_optimized={rot_resn}. Skipping.')
                mismatches.append((dataset, chain_id, resi, ref_resn, rot_resn))
                continue

            reference_residue = reference_structure.extract(f"chain {chain_id} and resi {resi}")

            reference_rscc = self._calc_rscc_all_events(reference_residue, event_maps, event_maps_models, rmask)
            rotamer_rscc = self._calc_rscc_all_events(rotamer_residue, event_maps, event_maps_models, rmask)

            results.append((dataset, chain_id, resi, ref_resn, reference_rscc, rotamer_rscc))

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
        self._plot_scatter(all_results)

    def _write_csv(self, results, mismatches):
        rscc_output = self.output_path / 'rscc_comparison.csv'
        with open(rscc_output, 'w+') as f:
            f.write('dataset,residue,resname,reference_rscc,rotamer_optimized_rscc,delta')
            f.write('\n')
            for dataset, chain_id, resi, resn, reference_rscc, rotamer_rscc in results:
                f.write(f'{dataset},{chain_id}{resi},{resn},{reference_rscc},{rotamer_rscc},'
                         f'{rotamer_rscc - reference_rscc}')
                f.write('\n')

        if mismatches:
            mismatch_output = self.output_path / 'residue_mismatches.csv'
            with open(mismatch_output, 'w+') as f:
                f.write('dataset,residue,reference_resname,rotamer_optimized_resname')
                f.write('\n')
                for dataset, chain_id, resi, ref_resn, rot_resn in mismatches:
                    f.write(f'{dataset},{chain_id}{resi},{ref_resn},{rot_resn}')
                    f.write('\n')
            print(f'{len(mismatches)} mismatched residue(s) written to {mismatch_output}')

    def _plot_scatter(self, results):
        from scipy.stats import gaussian_kde
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        reference_rsccs = np.array([r[4] for r in results])
        rotamer_rsccs = np.array([r[5] for r in results])

        # Color points by local point density (kde), plotting densest points last so they're on top
        xy = np.vstack([reference_rsccs, rotamer_rsccs])
        density = gaussian_kde(xy)(xy)
        order = density.argsort()
        reference_rsccs, rotamer_rsccs, density = reference_rsccs[order], rotamer_rsccs[order], density[order]

        fig, ax = plt.subplots(figsize=(6, 6))
        points = ax.scatter(reference_rsccs, rotamer_rsccs, c=density, cmap='viridis', s=6, edgecolor='none')
        fig.colorbar(points, ax=ax, label='point density')

        lims = [0, 1]
        ax.plot(lims, lims, linestyle='--', color='gray', linewidth=1)
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_xlabel('Reference RSCC')
        ax.set_ylabel('Rotamer-optimized RSCC')
        ax.set_title(f'Reference vs Rotamer-optimized RSCC (n={len(results)})')
        ax.set_aspect('equal')

        fig_output = self.output_path / 'rscc_scatter.png'
        fig.tight_layout()
        fig.savefig(fig_output, dpi=300)
        plt.close(fig)
        print(f'scatterplot written to {fig_output}')


def main():
    args = build_argparser().parse_args()
    comparator = RSCC_Comparator(
        args.datasets_dir, args.reference_dir, args.run_name, args.placer_run_name,
        args.filter_run_name, args.rotamer_run_name, args.output_folder, args.csv_file, args.n_cpus
    )
    comparator.run()


if __name__ == '__main__':
    main()