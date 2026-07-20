import argparse
import time
from pathlib import Path
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from qfit import Structure
from qfit import XMap
from qfit.xtal.transformer import get_transformer


# Default max centroid distance (Å) between a reference LIG and its nearest-by-RMSD multimodel
# LIG for the pair to be treated as a match. Reference LIGs whose best multimodel candidate is
# farther than this are considered 'unmatched' and are excluded from RSCC scoring/plotting, but
# are still counted and reported.
DEFAULT_CENTROID_CUTOFF = 2.0


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
             "the dataset name, e.g. '/.../datasets/{dataset}/run_1/placer_100/filter_4/cluster_rep_models.pdb'"
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
    p.add_argument(
        "--centroid_cutoff",
        default=DEFAULT_CENTROID_CUTOFF,
        type=float,
        help="Max centroid distance (Å) between a reference LIG and its nearest-by-RMSD "
             "multimodel LIG for the pair to count as a match, rather than 'unmatched'",
    )
    return p


class LigRSCC_Comparator():
    def __init__(self, datasets_dir, reference_pdb_pattern, multimodel_pdb_pattern, output_folder,
                 csv_file, n_cpus=24, centroid_cutoff=DEFAULT_CENTROID_CUTOFF):
        self.datasets_dir = datasets_dir
        self.reference_pdb_pattern = reference_pdb_pattern
        self.multimodel_pdb_pattern = multimodel_pdb_pattern
        self.output_path = Path(output_folder)
        os.makedirs(self.output_path, exist_ok=True)
        self.csv_file = csv_file
        self.n_cpus = n_cpus
        self.centroid_cutoff = centroid_cutoff

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
            event_name = str(event_file).split('/')[-1]
            event_maps[event_name] = XMap.fromfile(str(event_file), resolution=resolution)

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

    def _find_lig_keys(self, structure):
        """Returns [(chain_id, resi), ...] for every residue named 'LIG' in a single-model structure."""
        keys = []
        for chain in structure._pdb_hierarchy.only_model().chains():
            chain_id = chain.id.strip()
            for residue_group in chain.residue_groups():
                resn = residue_group.atom_groups()[0].resname.strip()
                if resn == 'LIG':
                    resi = int(residue_group.resseq)
                    keys.append((chain_id, resi))
        return keys

    def _find_all_lig_instances(self, models):
        """
        Walks every model of a split multimodel structure and collects every residue named
        'LIG', regardless of which model or (chain_id, resi) it came from. Unlike the sidechain
        version of this comparator, there's no dedup step here: we want every LIG copy across
        every model available as a candidate match for each reference LIG, and comparing a
        reference LIG against a duplicate candidate is harmless (it just won't win the
        nearest-by-RMSD search over a non-duplicate).

        Returns a list of dicts: {model_idx, chain_id, resi, structure, coor, names, centroid}
        """
        instances = []
        for model_idx, model in enumerate(models):
            for chain_id, resi in self._find_lig_keys(model):
                resi_selstr = f"chain {chain_id} and resi {resi}"
                lig_structure = model.extract(resi_selstr)
                coor = lig_structure.coor.copy()
                names = list(lig_structure.name)
                instances.append({
                    'model_idx': model_idx,
                    'chain_id': chain_id,
                    'resi': resi,
                    'structure': lig_structure,
                    'coor': coor,
                    'names': names,
                    'centroid': coor.mean(axis=0),
                })
        return instances

    def _match_and_rmsd(self, ref_names, ref_coor, cand_names, cand_coor):
        """
        Aligns two ligand conformers by atom name (rather than assuming identical atom order,
        since the reference and multimodel structures may not enumerate atoms the same way) and
        returns the RMSD over the atoms they have in common. Returns None if there's no atom
        name overlap at all (e.g. mismatched ligand identity).
        """
        ref_index = {name: i for i, name in enumerate(ref_names)}
        common = [name for name in cand_names if name in ref_index]
        if not common:
            return None

        ref_idx = [ref_index[name] for name in common]
        cand_idx = [cand_names.index(name) for name in common]
        ref_matched = ref_coor[ref_idx]
        cand_matched = cand_coor[cand_idx]

        delta = cand_matched - ref_matched
        rmsd = float(np.sqrt(np.mean(np.square(delta).sum(axis=1))))
        return rmsd

    def _find_nearest_multimodel_lig(self, ref_names, ref_coor, instances):
        """Returns (best_instance, best_rmsd) for the multimodel LIG instance with the lowest
        RMSD to the reference LIG, or (None, None) if no instance shares any atom names with it."""
        best_instance = None
        best_rmsd = np.inf
        for instance in instances:
            rmsd = self._match_and_rmsd(ref_names, ref_coor, instance['names'], instance['coor'])
            if rmsd is None:
                continue
            if rmsd < best_rmsd:
                best_rmsd = rmsd
                best_instance = instance
        if best_instance is None:
            return None, None
        return best_instance, best_rmsd

    def _score_single_lig(self, lig_structure, coor, event_maps, event_maps_models, rmask):
        """Converts a single LIG conformer's coordinates to density and returns its highest RSCC
        against any event map for this dataset."""
        scaled_bulk_solvent = 0
        coor_set = [coor]
        bfactor_array = [self.default_bfactor]

        rsccs = []
        for event_map_name in list(event_maps.keys()):
            transformer = get_transformer("qfit", lig_structure, event_maps_models[event_map_name])

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
            return [], [], 0

        reference_pdb = Path(self.reference_pdb_pattern.format(dataset=dataset))
        multimodel_pdb = Path(self.multimodel_pdb_pattern.format(dataset=dataset))

        if not reference_pdb.exists():
            print(f'Warning: no reference pdb found for {dataset} at {reference_pdb}, skipping.')
            return [], [], 0
        if not multimodel_pdb.exists():
            print(f'Warning: no multimodel pdb found for {dataset} at {multimodel_pdb}, skipping.')
            return [], [], 0

        print(f'Processing {dataset}: resolution={resolution}')
        time0 = time.time()

        event_maps, event_maps_models = self._load_event_maps(dataset_dir, resolution)
        if not event_maps:
            print(f'Warning: no event maps found for {dataset} in {dataset_dir}, skipping.')
            return [], [], 0
        rmask = 0.5 + resolution / 3.0  # from qfit

        reference_structure = self._clean_structure(Structure.fromfile(str(reference_pdb)))
        ref_lig_keys = self._find_lig_keys(reference_structure)
        if not ref_lig_keys:
            print(f'Warning: no LIG residues found in reference for {dataset}, skipping.')
            return [], [], 0

        multimodel_models = [
            self._clean_structure(model)
            for model in Structure.fromfile(str(multimodel_pdb)).split_models()
        ]
        multimodel_lig_instances = self._find_all_lig_instances(multimodel_models)

        results = []
        unmatched = []
        used_instance_keys = set()
        for chain_id, resi in ref_lig_keys:
            resi_selstr = f"chain {chain_id} and resi {resi}"
            ref_lig_structure = reference_structure.extract(resi_selstr)
            ref_coor = ref_lig_structure.coor.copy()
            ref_names = list(ref_lig_structure.name)
            ref_centroid = ref_coor.mean(axis=0)

            if not multimodel_lig_instances:
                unmatched.append((dataset, chain_id, resi, 'no_candidates', None))
                continue

            best_instance, match_rmsd = self._find_nearest_multimodel_lig(
                ref_names, ref_coor, multimodel_lig_instances
            )
            if best_instance is None:
                unmatched.append((dataset, chain_id, resi, 'no_shared_atoms', None))
                continue

            centroid_dist = float(np.linalg.norm(best_instance['centroid'] - ref_centroid))
            if centroid_dist > self.centroid_cutoff:
                unmatched.append((dataset, chain_id, resi, 'centroid_too_far', centroid_dist))
                continue

            reference_rscc = self._score_single_lig(
                ref_lig_structure, ref_coor, event_maps, event_maps_models, rmask
            )
            multimodel_rscc = self._score_single_lig(
                best_instance['structure'], best_instance['coor'], event_maps, event_maps_models, rmask
            )

            results.append((
                dataset, chain_id, resi, reference_rscc, multimodel_rscc, match_rmsd, centroid_dist
            ))
            used_instance_keys.add(
                (best_instance['model_idx'], best_instance['chain_id'], best_instance['resi'])
            )

        # Excess ligands: multimodel LIG instances (across every model) that were never claimed
        # as the best match for any reference LIG. A single multimodel instance can legitimately
        # be the best match for more than one reference LIG (e.g. duplicated across models), so
        # this counts unique unclaimed instances rather than unmatched attempts.
        excess_ligands = len(multimodel_lig_instances) - len(used_instance_keys)

        print(f'Completed {dataset}: {len(results)} LIG(s) scored, {len(unmatched)} unmatched, '
              f'{excess_ligands} excess multimodel LIG(s) ({time.time() - time0:.2f}s)')

        return results, unmatched, excess_ligands

    def run(self):
        print(self.__dict__)

        dataset_dirs = sorted(d for d in self.datasets_dir.iterdir() if d.is_dir())
        n_workers = max(1, min(self.n_cpus, os.cpu_count(), len(dataset_dirs)))

        all_results = []
        all_unmatched = []
        total_excess_ligands = 0
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {
                executor.submit(self._process_dataset, dataset_dir.name, dataset_dir): dataset_dir.name
                for dataset_dir in dataset_dirs
            }
            for future in as_completed(futures):
                dataset = futures[future]
                try:
                    results, unmatched, excess_ligands = future.result()
                except Exception as e:
                    print(f'Error processing {dataset}: {e}')
                    continue
                all_results.extend(results)
                all_unmatched.extend(unmatched)
                total_excess_ligands += excess_ligands

        print(f'Pooled {len(all_results)} LIG(s) across {len(dataset_dirs)} datasets '
              f'({n_workers} workers); {len(all_unmatched)} unmatched; '
              f'{total_excess_ligands} excess multimodel LIG(s)')

        self._write_csv(all_results, all_unmatched)
        self._plot_scatter(
            [r[3] for r in all_results], [r[4] for r in all_results],
            'Reference LIG RSCC', 'Multimodel LIG RSCC',
            'Reference vs Nearest-Matched Multimodel LIG RSCC',
            len(all_unmatched), total_excess_ligands, 'lig_rscc_scatter.png'
        )

    def _write_csv(self, results, unmatched):
        rscc_output = self.output_path / 'lig_rscc_comparison.csv'
        with open(rscc_output, 'w+') as f:
            f.write('dataset,ligand,reference_rscc,multimodel_rscc,delta,match_rmsd,centroid_dist')
            f.write('\n')
            for dataset, chain_id, resi, ref_rscc, mm_rscc, match_rmsd, centroid_dist in results:
                f.write(f'{dataset},{chain_id}{resi},{ref_rscc},{mm_rscc},{mm_rscc - ref_rscc},'
                         f'{match_rmsd},{centroid_dist}')
                f.write('\n')

        if unmatched:
            unmatched_output = self.output_path / 'unmatched_ligs.csv'
            with open(unmatched_output, 'w+') as f:
                f.write('dataset,ligand,reason,centroid_dist')
                f.write('\n')
                for dataset, chain_id, resi, reason, centroid_dist in unmatched:
                    centroid_str = '' if centroid_dist is None else f'{centroid_dist}'
                    f.write(f'{dataset},{chain_id}{resi},{reason},{centroid_str}')
                    f.write('\n')
            print(f'{len(unmatched)} unmatched LIG(s) written to {unmatched_output}')

    def _plot_scatter(self, x, y, x_label, y_label, title, n_unmatched, n_excess, filename):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        x = np.array(x)
        y = np.array(y)

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(x, y, color='steelblue', s=8, edgecolor='none')

        lims = [0, 1]
        ax.plot(lims, lims, linestyle='--', color='gray', linewidth=1)
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(f'{title} (n={len(x)})')
        ax.set_aspect('equal')

        ax.text(
            0.02, 0.98, f'Unmatched reference LIGs: {n_unmatched}\nExcess multimodel LIGs: {n_excess}',
            transform=ax.transAxes, ha='left', va='top', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray', alpha=0.9)
        )

        fig_output = self.output_path / filename
        fig.tight_layout()
        fig.savefig(fig_output, dpi=300)
        plt.close(fig)
        print(f'joint scatterplot written to {fig_output}')


def main():
    args = build_argparser().parse_args()
    comparator = LigRSCC_Comparator(
        args.datasets_dir, args.reference_pdb, args.multimodel_pdb, args.output_folder,
        args.csv_file, args.n_cpus, args.centroid_cutoff
    )
    comparator.run()


if __name__ == '__main__':
    main()