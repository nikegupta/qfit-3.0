import argparse
import time
import traceback
from pathlib import Path
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
import re
from scipy.optimize import linear_sum_assignment

import numpy as np

from qfit import Structure
from qfit import XMap
from qfit.xtal.transformer import get_transformer


# Default max centroid distance (Å) between a reference LIG and its nearest-by-RMSD multimodel
# LIG for the pair to be treated as a match. Reference LIGs whose best multimodel candidate is
# farther than this are considered 'unmatched' and are excluded from RSCC scoring/plotting, but
# are still counted and reported.
DEFAULT_CENTROID_CUTOFF = 2.0

# Below this sidechain-style RMSD (Å), two multimodel LIG instances are treated as the same
# cluster representative repeated across models, rather than two distinct ligands.
# cluster_rep_models.pdb repeats a given representative once per model to encode its cluster
# size (num_members), so without this dedup step the same physical ligand would be counted
# once per member - inflating the excess-ligand count in proportion to cluster size instead of
# reflecting the number of distinct, unmatched cluster representatives.
DEFAULT_DUPLICATE_RMSD_CUTOFF = 0.1


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

    def _get_model_serials(self, pdb_path):
        """
        Parses the PLACER multimodel PDB file's own 'MODEL' record lines, in the order they
        appear in the file, and returns the serial number written on each one (e.g. [1, 2, 3, ...]).

        Structure.fromfile(...).split_models() gives back a plain Python list in file order but
        does not expose the original MODEL serial numbers, so model_idx (an enumerate() counter)
        previously had no guaranteed relationship to the file's own numbering. Reading the serials
        directly from the file lets every ligand instance be traced back to the exact MODEL block
        it came from in the PLACER output.
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

    def _pdb_has_atoms(self, pdb_path):
        """
        Quickly checks whether a PDB file contains any ATOM/HETATM records at all, without
        doing a full structure parse. Some PLACER filter stages produce a cluster_rep_models.pdb
        that is non-empty in bytes (e.g. it contains only a header, an empty MODEL/ENDMDL
        wrapper, or similar boilerplate with zero atoms) because every candidate was rejected
        for that dataset. Such files pass the plain st_size == 0 check but can still fail deep
        inside split_models()/_clean_structure() with an opaque AssertionError. Catching this
        case up front lets it be treated the same as a genuinely empty file - no multimodel LIG
        candidates - with one clear message instead of a confusing parse failure.
        """
        with open(pdb_path) as f:
            for line in f:
                if line.startswith('ATOM') or line.startswith('HETATM'):
                    return True
        return False

    def _find_all_lig_instances(self, models, model_serials):
        """
        Walks every model of a split multimodel structure and collects every residue named
        'LIG', regardless of which model or (chain_id, resi) it came from. This includes every
        repeat of a cluster representative across models used to encode its cluster size
        (num_members) - those repeats are collapsed later by _get_unique_lig_instances.

        model_serials must be the PLACER-file MODEL serial number for each entry in models
        (same length, same order), as returned by _get_model_serials, so that every instance
        can be traced back to its exact MODEL block in the original multimodel PDB.

        Returns a list of dicts: {model_serial, chain_id, resi, structure, coor, names, centroid}
        """
        instances = []
        for model_serial, model in zip(model_serials, models):
            for chain_id, resi in self._find_lig_keys(model):
                resi_selstr = f"chain {chain_id} and resi {resi}"
                lig_structure = model.extract(resi_selstr)
                coor = lig_structure.coor.copy()
                names = list(lig_structure.name)
                instances.append({
                    'model_serial': model_serial,
                    'chain_id': chain_id,
                    'resi': resi,
                    'structure': lig_structure,
                    'coor': coor,
                    'names': names,
                    'centroid': coor.mean(axis=0),
                })
        return instances

    def _get_unique_lig_instances(self, instances, dedup_cutoff=DEFAULT_DUPLICATE_RMSD_CUTOFF):
        """
        Collapses multimodel LIG instances that are (near-)identical in coordinates down to one
        representative each, so that matching and excess-ligand counting are based on distinct
        cluster representatives rather than on how many models/members each was duplicated
        across. Comparison is pairwise against the unique instances collected so far, which is
        fine given the small number of distinct ligand conformers typically present.

        Note: only the first-encountered instance of each duplicate group is kept, so the
        model_serial retained (and later reported for excess ligands) is whichever MODEL block
        in the file happened to appear first among the duplicates - not necessarily the "primary"
        member of the cluster.
        """
        unique_instances = []
        for instance in instances:
            is_duplicate = False
            for existing in unique_instances:
                rmsd = self._match_and_rmsd(
                    existing['names'], existing['coor'], instance['names'], instance['coor']
                )
                if rmsd is not None and rmsd <= dedup_cutoff:
                    is_duplicate = True
                    break
            if not is_duplicate:
                unique_instances.append(instance)
        return unique_instances

    def _element_of(self, name):
        """Extracts the element symbol from a PDB atom name (e.g. 'C6' -> 'C', 'N4' -> 'N')."""
        m = re.match(r'^[A-Za-z]+', name)
        return m.group(0) if m else name

    def _match_and_rmsd(self, ref_names, ref_coor, cand_names, cand_coor):
        """
        Computes a symmetry-aware RMSD between two ligand conformers, without relying on
        atom names being assigned consistently across files. PLACER-generated poses can be
        rotated (e.g. ~180 degrees) such that pseudo-symmetric ring atoms end up with
        swapped names relative to the reference - this badly inflates a naive name-matched
        RMSD even when the two poses are essentially superimposed.

        For each element type present in both structures, atoms are paired by spatial
        proximity using the Hungarian algorithm (linear_sum_assignment over the pairwise
        squared-distance matrix), rather than by matching names directly. This finds the
        assignment of same-element atoms that minimizes total squared displacement, so it
        is unaffected by any name swap among chemically-equivalent or accidentally
        relabeled atoms. If element counts differ between the two sets, only as many pairs
        as the smaller count are matched per element (linear_sum_assignment's default
        behavior on a non-square cost matrix).

        Returns None if the two atom sets share no element type at all.
        """
        ref_elements = [self._element_of(n) for n in ref_names]
        cand_elements = [self._element_of(n) for n in cand_names]

        ref_by_elem = {}
        for i, el in enumerate(ref_elements):
            ref_by_elem.setdefault(el, []).append(i)
        cand_by_elem = {}
        for i, el in enumerate(cand_elements):
            cand_by_elem.setdefault(el, []).append(i)

        shared_elements = set(ref_by_elem) & set(cand_by_elem)
        if not shared_elements:
            return None

        squared_diffs = []
        for el in shared_elements:
            ref_idx = ref_by_elem[el]
            cand_idx = cand_by_elem[el]
            ref_pts = ref_coor[ref_idx]
            cand_pts = cand_coor[cand_idx]

            diff = ref_pts[:, None, :] - cand_pts[None, :, :]
            dist_sq = np.sum(diff ** 2, axis=2)

            row_idx, col_idx = linear_sum_assignment(dist_sq)
            squared_diffs.extend(dist_sq[row_idx, col_idx])

        if not squared_diffs:
            return None

        return float(np.sqrt(np.mean(squared_diffs)))

    def _find_nearest_multimodel_lig(self, ref_names, ref_coor, ref_centroid, instances):
        """
        Returns (best_instance, best_rmsd, best_centroid_dist) for the multimodel LIG instance
        with the lowest centroid distance to the reference LIG - rather than the lowest RMSD.

        Centroid distance is a coarser measure of "same pose, same location" than atom-matched
        RMSD, but it is inherently robust to atom-naming/labeling issues (e.g. PLACER assigning
        different names to symmetry-equivalent ring atoms after a ~180 degree rotation) since it
        only depends on the mean position of all atoms, not on how individual atoms are paired.
        RMSD is still computed (via the symmetry-aware Hungarian matching in _match_and_rmsd) for
        the selected instance, purely for reporting/diagnostic purposes - it no longer drives
        selection.

        Returns (None, None, None) if no instance shares any element type with the reference
        (i.e. _match_and_rmsd can't be computed for reporting), which in practice only happens if
        there are no candidate instances at all.
        """
        best_instance = None
        best_centroid_dist = np.inf
        for instance in instances:
            centroid_dist = float(np.linalg.norm(instance['centroid'] - ref_centroid))
            if centroid_dist < best_centroid_dist:
                best_centroid_dist = centroid_dist
                best_instance = instance

        if best_instance is None:
            return None, None, None

        best_rmsd = self._match_and_rmsd(ref_names, ref_coor, best_instance['names'], best_instance['coor'])
        return best_instance, best_rmsd, best_centroid_dist

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
            return [], [], []

        reference_pdb = Path(self.reference_pdb_pattern.format(dataset=dataset))
        multimodel_pdb = Path(self.multimodel_pdb_pattern.format(dataset=dataset))

        if not reference_pdb.exists():
            print(f'Warning: no reference pdb found for {dataset} at {reference_pdb}, skipping.')
            return [], [], []
        if not multimodel_pdb.exists():
            print(f'Warning: no multimodel pdb found for {dataset} at {multimodel_pdb}, skipping.')
            return [], [], []

        print(f'Processing {dataset}: resolution={resolution}')
        time0 = time.time()

        event_maps, event_maps_models = self._load_event_maps(dataset_dir, resolution)
        if not event_maps:
            print(f'Warning: no event maps found for {dataset} in {dataset_dir}, skipping.')
            return [], [], []
        rmask = 0.5 + resolution / 3.0  # from qfit

        reference_structure = self._clean_structure(Structure.fromfile(str(reference_pdb)))
        ref_lig_keys = self._find_lig_keys(reference_structure)
        if not ref_lig_keys:
            print(f'Warning: no LIG residues found in reference for {dataset}, skipping.')
            return [], [], []

        # An empty (or otherwise malformed) multimodel pdb - e.g. a cluster_rep_models.pdb with
        # zero MODEL blocks because filtering rejected every candidate for this dataset - would
        # otherwise crash deep inside split_models()/only_model() and take the whole dataset down
        # with it. Treat that as "no multimodel LIG candidates" instead, so every reference LIG
        # for this dataset is still scored and counted as unmatched rather than silently dropped.
        if multimodel_pdb.stat().st_size == 0 or not self._pdb_has_atoms(multimodel_pdb):
            print(f'Warning: multimodel pdb for {dataset} at {multimodel_pdb} is empty (or '
                  f'contains no atoms); treating as having no multimodel LIG candidates.')
            multimodel_lig_instances = []
        else:
            try:
                model_serials = self._get_model_serials(multimodel_pdb)
                multimodel_models_raw = Structure.fromfile(str(multimodel_pdb)).split_models()
                if len(model_serials) != len(multimodel_models_raw):
                    print(f'Warning: found {len(model_serials)} MODEL serial(s) in {multimodel_pdb} '
                          f'but split_models() returned {len(multimodel_models_raw)} model(s); '
                          f'falling back to file-order indices (0-based) as the model serial.')
                    model_serials = list(range(len(multimodel_models_raw)))
                multimodel_models = [self._clean_structure(model) for model in multimodel_models_raw]
                multimodel_lig_instances_all = self._find_all_lig_instances(multimodel_models, model_serials)
                multimodel_lig_instances = self._get_unique_lig_instances(multimodel_lig_instances_all)
            except Exception as e:
                # Defense in depth: covers malformed-but-non-empty files that still fail deep
                # inside parsing/cleaning for reasons other than "no atoms" (e.g. corrupt
                # records, unexpected residue naming). The _pdb_has_atoms() check above already
                # handles the common empty-file case before we ever get here.
                print(f'Warning: failed to parse multimodel pdb for {dataset} at {multimodel_pdb} '
                      f'({type(e).__name__}: {e}); treating as having no multimodel LIG candidates.')
                multimodel_lig_instances = []

        results = []
        unmatched = []
        used_instance_keys = set()
        for chain_id, resi in ref_lig_keys:
            resi_selstr = f"chain {chain_id} and resi {resi}"
            ref_lig_structure = reference_structure.extract(resi_selstr)
            ref_coor = ref_lig_structure.coor.copy()
            ref_names = list(ref_lig_structure.name)
            ref_centroid = ref_coor.mean(axis=0)

            # scored regardless of match outcome, so unmatched reference LIGs can still be
            # included in the unmatched-RSCC histogram
            reference_rscc = self._score_single_lig(
                ref_lig_structure, ref_coor, event_maps, event_maps_models, rmask
            )

            if not multimodel_lig_instances:
                unmatched.append((dataset, chain_id, resi, 'no_candidates', None, reference_rscc))
                continue

            best_instance, match_rmsd, centroid_dist = self._find_nearest_multimodel_lig(
                ref_names, ref_coor, ref_centroid, multimodel_lig_instances
            )
            if best_instance is None:
                unmatched.append((dataset, chain_id, resi, 'no_shared_atoms', None, reference_rscc))
                continue

            if centroid_dist > self.centroid_cutoff:
                unmatched.append((dataset, chain_id, resi, 'centroid_too_far', centroid_dist, reference_rscc))
                continue
            multimodel_rscc = self._score_single_lig(
                best_instance['structure'], best_instance['coor'], event_maps, event_maps_models, rmask
            )

            results.append((
                dataset, chain_id, resi, reference_rscc, multimodel_rscc, match_rmsd, centroid_dist
            ))
            used_instance_keys.add(
                (best_instance['model_serial'], best_instance['chain_id'], best_instance['resi'])
            )

        # Excess ligands: unique multimodel LIG cluster representatives (post-dedup) that were
        # never claimed as the best match for any reference LIG. A single representative can
        # legitimately be the best match for more than one reference LIG, so this counts unique
        # unclaimed representatives rather than unmatched attempts - and, since duplicates from
        # cluster-size repeats were already collapsed, it gives 1 per unmatched representative
        # regardless of its num_members. Each is also scored so its RSCC can go in the
        # excess-ligand histogram, and each carries the model_serial of its exact MODEL block in
        # multimodel_pdb so it can be traced directly back to the PLACER file.
        excess_ligands = []
        for instance in multimodel_lig_instances:
            instance_key = (instance['model_serial'], instance['chain_id'], instance['resi'])
            if instance_key in used_instance_keys:
                continue
            excess_rscc = self._score_single_lig(
                instance['structure'], instance['coor'], event_maps, event_maps_models, rmask
            )
            excess_ligands.append((
                dataset, instance['chain_id'], instance['resi'], instance['model_serial'],
                str(multimodel_pdb), excess_rscc
            ))

        print(f'Completed {dataset}: {len(results)} LIG(s) scored, {len(unmatched)} unmatched, '
              f'{len(excess_ligands)} excess multimodel LIG(s) ({time.time() - time0:.2f}s)')

        return results, unmatched, excess_ligands

    def run(self):

        dataset_dirs = sorted(d for d in self.datasets_dir.iterdir() if d.is_dir())
        n_workers = max(1, min(self.n_cpus, os.cpu_count(), len(dataset_dirs)))

        all_results = []
        all_unmatched = []
        all_excess_ligands = []
        dataset_summaries = []  # (dataset, n_reference_ligs, n_matched, n_unmatched, n_excess)
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {
                executor.submit(self._process_dataset, dataset_dir.name, dataset_dir): dataset_dir.name
                for dataset_dir in dataset_dirs
            }
            for future in as_completed(futures):
                dataset = futures[future]
                try:
                    results, unmatched, excess_ligands = future.result()
                except Exception:
                    print(f'Error processing {dataset}:')
                    traceback.print_exc()
                    continue
                all_results.extend(results)
                all_unmatched.extend(unmatched)
                all_excess_ligands.extend(excess_ligands)
                dataset_summaries.append(
                    (dataset, len(results) + len(unmatched), len(results), len(unmatched), len(excess_ligands))
                )

        total_excess_ligands = len(all_excess_ligands)

        print(f'Pooled {len(all_results)} LIG(s) across {len(dataset_dirs)} datasets '
              f'({n_workers} workers); {len(all_unmatched)} unmatched; '
              f'{total_excess_ligands} excess multimodel LIG(s)')

        self._write_csv(all_results, all_unmatched, all_excess_ligands, dataset_summaries)
        self._plot_scatter(
            [r[3] for r in all_results], [r[4] for r in all_results],
            'Reference LIG RSCC', 'Multimodel LIG RSCC',
            'Reference vs Nearest-Matched Multimodel LIG RSCC',
            len(all_unmatched), total_excess_ligands, 'lig_rscc_scatter.png'
        )
        self._plot_histograms(
            [u[5] for u in all_unmatched], [e[5] for e in all_excess_ligands],
            'unmatched_excess_rscc_histogram.png'
        )

    def _write_csv(self, results, unmatched, excess_ligands, dataset_summaries):
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
                f.write('dataset,ligand,reason,centroid_dist,rscc')
                f.write('\n')
                for dataset, chain_id, resi, reason, centroid_dist, rscc in unmatched:
                    centroid_str = '' if centroid_dist is None else f'{centroid_dist}'
                    f.write(f'{dataset},{chain_id}{resi},{reason},{centroid_str},{rscc}')
                    f.write('\n')
            print(f'{len(unmatched)} unmatched LIG(s) written to {unmatched_output}')

        if excess_ligands:
            excess_output = self.output_path / 'excess_ligs.csv'
            with open(excess_output, 'w+') as f:
                f.write('dataset,ligand,model_serial,source_multimodel_pdb,rscc')
                f.write('\n')
                for dataset, chain_id, resi, model_serial, source_pdb, rscc in excess_ligands:
                    f.write(f'{dataset},{chain_id}{resi},{model_serial},{source_pdb},{rscc}')
                    f.write('\n')
            print(f'{len(excess_ligands)} excess LIG(s) written to {excess_output}')

        summary_output = self.output_path / 'lig_match_summary.csv'
        with open(summary_output, 'w+') as f:
            f.write('dataset,n_reference_ligs,n_matched,n_unmatched,n_excess')
            f.write('\n')
            for dataset, n_reference, n_matched, n_unmatched, n_excess in dataset_summaries:
                f.write(f'{dataset},{n_reference},{n_matched},{n_unmatched},{n_excess}')
                f.write('\n')
        print(f'per-dataset match summary written to {summary_output}')

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

        if len(x) > 0:
            stats_text = (
                f'Reference RSCC: mean={np.mean(x):.3f}, median={np.median(x):.3f}\n'
                f'Multimodel RSCC: mean={np.mean(y):.3f}, median={np.median(y):.3f}'
            )
        else:
            stats_text = 'Reference RSCC: n/a\nMultimodel RSCC: n/a'

        ax.text(
            0.02, 0.98,
            f'Unmatched reference LIGs: {n_unmatched}\n'
            f'Excess multimodel LIGs: {n_excess}\n'
            f'{stats_text}',
            transform=ax.transAxes, ha='left', va='top', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray', alpha=0.2)
        )

        fig_output = self.output_path / filename
        fig.tight_layout()
        fig.savefig(fig_output, dpi=300)
        plt.close(fig)
        print(f'joint scatterplot written to {fig_output}')

    def _plot_histograms(self, unmatched_rsccs, excess_rsccs, filename):
        """Plots RSCC histograms for unmatched reference LIGs and excess multimodel LIGs
        overlaid on the same axes, each at alpha=0.5 so both distributions stay visible."""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        unmatched_rsccs = np.array(unmatched_rsccs, dtype=float)
        excess_rsccs = np.array(excess_rsccs, dtype=float)

        fig, ax = plt.subplots(figsize=(6, 6))
        bins = np.linspace(0, 1, 21)

        if len(unmatched_rsccs) > 0:
            ax.hist(
                unmatched_rsccs, bins=bins, alpha=0.5, color='indianred',
                label=f'Unmatched reference LIGs (n={len(unmatched_rsccs)})'
            )
        if len(excess_rsccs) > 0:
            ax.hist(
                excess_rsccs, bins=bins, alpha=0.5, color='steelblue',
                label=f'Excess multimodel LIGs (n={len(excess_rsccs)})'
            )

        ax.set_xlim(0, 1)
        ax.set_xlabel('RSCC')
        ax.set_ylabel('Count')
        ax.set_title('RSCC of Unmatched Reference vs Excess Multimodel LIGs')
        ax.legend(fontsize=9)

        if len(unmatched_rsccs) > 0:
            unmatched_stats = f'mean={np.mean(unmatched_rsccs):.3f}, median={np.median(unmatched_rsccs):.3f}'
        else:
            unmatched_stats = 'n/a'
        if len(excess_rsccs) > 0:
            excess_stats = f'mean={np.mean(excess_rsccs):.3f}, median={np.median(excess_rsccs):.3f}'
        else:
            excess_stats = 'n/a'

        ax.text(
            0.98, 0.98,
            f'Unmatched reference LIGs: {unmatched_stats}\n'
            f'Excess multimodel LIGs: {excess_stats}',
            transform=ax.transAxes, ha='right', va='top', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray', alpha=0.2)
        )

        fig_output = self.output_path / filename
        fig.tight_layout()
        fig.savefig(fig_output, dpi=300)
        plt.close(fig)
        print(f'unmatched/excess RSCC histogram written to {fig_output}')


def main():
    args = build_argparser().parse_args()
    comparator = LigRSCC_Comparator(
        args.datasets_dir, args.reference_pdb, args.multimodel_pdb, args.output_folder,
        args.csv_file, args.n_cpus, args.centroid_cutoff
    )
    comparator.run()

if __name__ == '__main__':
    main()