"""
Shared helpers for the RSCC/geometry analysis scripts in analysis_scripts/.

Most of these scripts don't recompute RSCC - they only read the per-residue
CSVs already written during the pipeline by calc_rscc (calc_apo_rscc,
calc_backbone_refined_rscc, calc_final_refined_rscc, each producing a
model_idx,residue,rscc csv) and the 'rscc' column already written into
cluster_reps.csv by filter/filter2. A few (centroid_rmsd_all.py,
calc_placer_sampling.py/calc_placer_sampling_unrefined.py) instead compare
raw ligand geometry (centroid distance / symmetry-aware RMSD) against the
reference set, since there's no RSCC value to reuse before RSR has run.
"""
import argparse
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


DEFAULT_DATASETS_DIR = '/home/ngupta/main/program_claude/datasets'
DEFAULT_DATASETS_FILE = '/home/ngupta/main/program_claude/datasets.txt'


def build_common_argparser(description):
    """Argparser shared by every analysis script: the six pipeline run
    names plus the datasets dir/file."""
    p = argparse.ArgumentParser(description=description)
    p.add_argument('run_name')
    p.add_argument('placer_run_name')
    p.add_argument('filter_run_name')
    p.add_argument('placer2_run_name')
    p.add_argument('filter2_run_name')
    p.add_argument('final_run_name')
    p.add_argument('--datasets-dir', default=DEFAULT_DATASETS_DIR,
                    help='Root directory containing per-dataset folders')
    p.add_argument('--datasets-file', default=DEFAULT_DATASETS_FILE,
                    help='Path to newline-delimited list of dataset names')
    return p


def build_ref_argparser(description, positional_names):
    """Argparser for the reference-set comparison scripts. Unlike
    build_common_argparser, the positional run-name arguments vary by stage
    (e.g. only run/placer/filter for the stage-3 scripts), so the caller
    lists exactly which ones it needs."""
    p = argparse.ArgumentParser(description=description)
    for name in positional_names:
        p.add_argument(name)
    p.add_argument('--datasets-dir', default=DEFAULT_DATASETS_DIR,
                    help='Root directory containing per-dataset folders')
    p.add_argument('--datasets-file', default=DEFAULT_DATASETS_FILE,
                    help='Path to newline-delimited list of dataset names')
    p.add_argument('--ref-set', required=True,
                    help='Root directory of the reference set (one subfolder per dataset)')
    p.add_argument('--ref-pdb-pattern', default='{dataset}-pandda-model.pdb',
                    help="Reference structure filename pattern within REF_SET/<dataset>/; "
                         "'{dataset}' is replaced with the dataset name")
    p.add_argument('--graphs-dir', required=True,
                    help='Output directory for the pooled (cross-dataset) plot(s)')
    p.add_argument('--centroid-cutoff', type=float, default=2.0,
                    help='Max centroid distance (A) for a reference/pipeline ligand pair to '
                         'count as matched, same default as qfit compare_lig_rscc (default: 2.0)')
    return p


def build_placer_sampling_argparser(description):
    """Argparser for calc_placer_sampling.py / calc_placer_sampling_unrefined.py.
    Like build_ref_argparser, but run_name/placer_run_name are always
    required while filter_run_name/placer2_run_name are optional and must be
    given together - 2 positional args selects MODE A (score round-1 PLACER
    samples directly under placer_run_name/), 4 selects MODE B (score
    round-2 PLACER samples under .../filter_run_name/placer2_run_name/)."""
    p = argparse.ArgumentParser(description=description)
    p.add_argument('run_name')
    p.add_argument('placer_run_name')
    p.add_argument('filter_run_name', nargs='?', default=None)
    p.add_argument('placer2_run_name', nargs='?', default=None)
    p.add_argument('--datasets-dir', default=DEFAULT_DATASETS_DIR,
                    help='Root directory containing per-dataset folders')
    p.add_argument('--datasets-file', default=DEFAULT_DATASETS_FILE,
                    help='Path to newline-delimited list of dataset names')
    p.add_argument('--ref-set', required=True,
                    help='Root directory of the reference set (one subfolder per dataset)')
    p.add_argument('--ref-pdb-pattern', default='{dataset}-pandda-model.pdb',
                    help="Reference structure filename pattern within REF_SET/<dataset>/; "
                         "'{dataset}' is replaced with the dataset name")
    p.add_argument('--graphs-dir', required=True,
                    help='Output directory for the pooled (cross-dataset) plot')
    p.add_argument('--model-chain', default='C',
                    help="Chain ID of the LIG ligand in sampled model files (default: C)")
    p.add_argument('--model-resi', type=int, default=1,
                    help="Residue number of the LIG ligand in sampled model files (default: 1)")
    return p


def resolve_placer_sampling_mode(args):
    """Validates filter_run_name/placer2_run_name were given together and
    returns (mode_b, run_tag)."""
    has_filter = args.filter_run_name is not None
    has_placer2 = args.placer2_run_name is not None
    if has_filter != has_placer2:
        raise SystemExit('filter_run_name and placer2_run_name must be given together '
                          '(4 positional args), or both omitted (2 positional args).')
    mode_b = has_filter and has_placer2
    if mode_b:
        run_tag = f'{args.run_name}/{args.placer_run_name}/{args.filter_run_name}/{args.placer2_run_name}'
    else:
        run_tag = f'{args.run_name}/{args.placer_run_name}'
    return mode_b, run_tag


def ref_pdb_path(args, dataset):
    pdb_pattern = args.ref_pdb_pattern.replace('{dataset}', dataset)
    return Path(args.ref_set) / dataset / pdb_pattern


def ref_rscc_csv_path(args, dataset):
    """calc_ref_set_rscc writes '{structure%.pdb}_rscc.csv' next to the
    reference structure - same convention as every other calc_rscc output."""
    pdb_path = ref_pdb_path(args, dataset)
    return pdb_path.with_name(pdb_path.stem + '_rscc.csv')


def read_pdb_raw_atoms(pdb_path):
    """Parses every ATOM/HETATM record in a PDB file (ignoring MODEL/ENDMDL
    boundaries - use split_pdb_models for multimodel files) into a list of
    {name, altloc, res_name, chain_id, res_id, coord} dicts. Altloc ' '
    (blank) is normalised to ''."""
    atoms = []
    with open(pdb_path) as fh:
        for line in fh:
            if not line.startswith(('ATOM  ', 'HETATM')):
                continue
            name = line[12:16].strip()
            altloc = line[16].strip()
            res_name = line[17:20].strip()
            chain_id = line[21].strip()
            try:
                res_id = int(line[22:26])
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
            except ValueError:
                continue
            atoms.append({
                'name': name, 'altloc': altloc, 'res_name': res_name,
                'chain_id': chain_id, 'res_id': res_id,
                'coord': np.array([x, y, z], dtype=float),
            })
    return atoms


def split_pdb_models(pdb_path):
    """Splits a multimodel PDB into a list of atom-dict-lists, one per
    MODEL/ENDMDL block, in file order. A file with no MODEL records is
    returned as a single one-element list."""
    blocks = []
    current_lines = []
    in_model = False
    has_model_records = False

    with open(pdb_path) as fh:
        for line in fh:
            if line.startswith('MODEL '):
                has_model_records = True
                in_model = True
                current_lines = []
            elif line.startswith('ENDMDL'):
                blocks.append(current_lines)
                current_lines = []
                in_model = False
            elif has_model_records and in_model:
                current_lines.append(line)
            elif not has_model_records:
                current_lines.append(line)

    if not has_model_records and current_lines:
        blocks.append(current_lines)

    def parse_lines(lines):
        atoms = []
        for line in lines:
            if not line.startswith(('ATOM  ', 'HETATM')):
                continue
            name = line[12:16].strip()
            altloc = line[16].strip()
            res_name = line[17:20].strip()
            chain_id = line[21].strip()
            try:
                res_id = int(line[22:26])
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
            except ValueError:
                continue
            atoms.append({
                'name': name, 'altloc': altloc, 'res_name': res_name,
                'chain_id': chain_id, 'res_id': res_id,
                'coord': np.array([x, y, z], dtype=float),
            })
        return atoms

    return [parse_lines(lines) for lines in blocks]


def lig_conformations(atoms):
    """Groups an atom list's LIG residues into {(chain_id, res_id, altloc):
    centroid}. A LIG residue with two or more distinct non-blank altlocs
    gets one entry per altloc (blank-altloc atoms are shared across all of
    them); otherwise it gets a single altloc='' entry. Mirrors qfit
    compare_lig_rscc's _find_lig_keys, but returns only the centroid since
    no RSCC or RMSD is computed here."""
    lig_atoms = [a for a in atoms if a['res_name'] == 'LIG']
    if not lig_atoms:
        return {}

    by_residue = {}
    for a in lig_atoms:
        by_residue.setdefault((a['chain_id'], a['res_id']), []).append(a)

    conformations = {}
    for (chain_id, res_id), res_atoms in by_residue.items():
        explicit_altlocs = sorted({a['altloc'] for a in res_atoms if a['altloc'] != ''})
        if not explicit_altlocs:
            coords = np.array([a['coord'] for a in res_atoms])
            conformations[(chain_id, res_id, '')] = coords.mean(axis=0)
        else:
            for altloc in explicit_altlocs:
                alt_atoms = [a for a in res_atoms if a['altloc'] in (altloc, '')]
                coords = np.array([a['coord'] for a in alt_atoms])
                conformations[(chain_id, res_id, altloc)] = coords.mean(axis=0)
    return conformations


def _group_lig_residues(atoms, chain_id=None, res_id=None):
    """Shared by lig_conformations_filtered/lig_atom_groups: filters an atom
    list down to LIG residues (optionally restricted to one chain_id/res_id)
    and groups them into {(chain_id, res_id, altloc): [atom dict, ...]},
    same altloc-splitting rule as lig_conformations."""
    lig_atoms = [a for a in atoms if a['res_name'] == 'LIG']
    if chain_id is not None:
        lig_atoms = [a for a in lig_atoms if a['chain_id'] == chain_id]
    if res_id is not None:
        lig_atoms = [a for a in lig_atoms if a['res_id'] == res_id]
    if not lig_atoms:
        return {}

    by_residue = {}
    for a in lig_atoms:
        by_residue.setdefault((a['chain_id'], a['res_id']), []).append(a)

    groups = {}
    for (chain, resi), res_atoms in by_residue.items():
        explicit_altlocs = sorted({a['altloc'] for a in res_atoms if a['altloc'] != ''})
        if not explicit_altlocs:
            groups[(chain, resi, '')] = res_atoms
        else:
            for altloc in explicit_altlocs:
                groups[(chain, resi, altloc)] = [
                    a for a in res_atoms if a['altloc'] in (altloc, '')
                ]
    return groups


def lig_conformations_filtered(atoms, chain_id=None, res_id=None):
    """Like lig_conformations, but restricted to a given chain_id/res_id
    before grouping (used for sampled model files where the ligand's
    chain/resi convention is known, e.g. PLACER's chain C res 1)."""
    return {
        key: np.array([a['coord'] for a in res_atoms]).mean(axis=0)
        for key, res_atoms in _group_lig_residues(atoms, chain_id, res_id).items()
    }


def lig_atom_groups(atoms, chain_id=None, res_id=None):
    """Like lig_conformations_filtered, but returns each conformation's full
    atom list ({'name', 'coord'} dicts) instead of just its centroid - needed
    for atom-level RMSD (compute_rmsd_symmetric) rather than centroid
    distance."""
    return {
        key: [{'name': a['name'], 'coord': a['coord']} for a in res_atoms]
        for key, res_atoms in _group_lig_residues(atoms, chain_id, res_id).items()
    }


def _element_of(name):
    """Extracts the element symbol from a PDB atom name (e.g. 'C6' -> 'C')."""
    m = re.match(r'^[A-Za-z]+', name)
    return m.group(0) if m else name


def compute_rmsd_symmetric(model_atoms, ref_atoms):
    """Symmetry-aware RMSD between two ligand conformers: for each element
    type present in both atom lists, atoms are paired by spatial proximity
    (Hungarian assignment over the pairwise squared-distance matrix) rather
    than by atom name, since PLACER poses can be rotated such that
    pseudo-symmetric atoms end up with names swapped relative to the
    reference - this would badly inflate a naive name-matched RMSD even when
    the pose is essentially correct. If element counts differ between the
    two sets, only as many pairs as the smaller count are matched per
    element. Returns None if the two atom sets share no element type."""
    ref_coor = np.array([a['coord'] for a in ref_atoms])
    cand_coor = np.array([a['coord'] for a in model_atoms])
    ref_elements = [_element_of(a['name']) for a in ref_atoms]
    cand_elements = [_element_of(a['name']) for a in model_atoms]

    ref_by_elem = defaultdict(list)
    for i, el in enumerate(ref_elements):
        ref_by_elem[el].append(i)
    cand_by_elem = defaultdict(list)
    for i, el in enumerate(cand_elements):
        cand_by_elem[el].append(i)

    shared_elements = set(ref_by_elem) & set(cand_by_elem)
    if not shared_elements:
        return None

    squared_diffs = []
    for el in shared_elements:
        ref_pts = ref_coor[ref_by_elem[el]]
        cand_pts = cand_coor[cand_by_elem[el]]
        diff = ref_pts[:, None, :] - cand_pts[None, :, :]
        dist_sq = np.sum(diff ** 2, axis=2)
        row_idx, col_idx = linear_sum_assignment(dist_sq)
        squared_diffs.extend(dist_sq[row_idx, col_idx])

    if not squared_diffs:
        return None
    return float(np.sqrt(np.mean(squared_diffs)))


def process_placer_sampling_dataset(model_dir, ref_path, file_pattern, model_chain, model_resi):
    """For a single dataset, returns one min-RMSD value per reference LIG
    conformation matched: the minimum symmetry-aware RMSD from that
    reference LIG to the closest sampled ligand conformer across every
    model_dir.rglob(file_pattern) file. Each matched file may contain one or
    many MODEL/ENDMDL blocks (e.g. PLACER's own multimodel *_model.pdb
    outputs); every block is scored independently. If a block has no LIG
    under model_chain/model_resi, falls back to any LIG in that block.
    Empty list if model_dir/ref_path don't exist or nothing matches."""
    if not model_dir.exists() or not ref_path.exists():
        return []

    model_files = sorted(model_dir.rglob(file_pattern))
    if not model_files:
        return []

    ref_groups = lig_atom_groups(read_pdb_raw_atoms(ref_path))
    if not ref_groups:
        return []

    min_rmsds = {key: np.inf for key in ref_groups}

    for model_file in model_files:
        try:
            blocks = split_pdb_models(model_file)
        except Exception as e:
            print(f'    Warning: could not read {model_file}: {e}')
            continue

        for atoms in blocks:
            model_groups = lig_atom_groups(atoms, chain_id=model_chain, res_id=model_resi)
            if not model_groups:
                model_groups = lig_atom_groups(atoms)
                if not model_groups:
                    continue

            for ref_key, ref_atom_list in ref_groups.items():
                for model_atom_list in model_groups.values():
                    rmsd = compute_rmsd_symmetric(model_atom_list, ref_atom_list)
                    if rmsd is not None and rmsd < min_rmsds[ref_key]:
                        min_rmsds[ref_key] = rmsd

    return [v for v in min_rmsds.values() if np.isfinite(v)]


def plot_distance_histogram(values, title, xlabel, out_path, bin_width=1.0, color='steelblue'):
    """Histogram for distance/RMSD-style measurements with an unbounded
    (non-[0,1]) x-range, in the style of calc_filter_rmsd.py's
    save_histogram. Unlike plot_rscc_histogram, bins span [0, max(values)]
    at bin_width increments rather than the fixed RSCC [0, 1] range."""
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if len(values) == 0:
        print(f'  Skipping {out_path.name}: no data points.')
        return

    bins = np.arange(0, np.ceil(values.max() / bin_width) * bin_width + bin_width, bin_width)
    plt.figure(figsize=(10, 6))
    plt.hist(values, bins=bins, edgecolor='black', alpha=0.7, color=color)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.title(f'{title} (n={len(values)})', fontsize=13)
    plt.xticks(bins, fontsize=8, rotation=90)
    plt.grid(True, alpha=0.3)

    stats_text = (f'Mean:   {values.mean():.2f} Å\n'
                  f'Median: {np.median(values):.2f} Å\n'
                  f'≤ 2 Å:  {(values <= 2).sum()}/{len(values)}')
    plt.text(0.98, 0.98, stats_text, transform=plt.gca().transAxes,
              va='top', ha='right', fontsize=10,
              bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'  Histogram saved to: {out_path}')


def plot_count_histogram(values, title, xlabel, out_path, color='steelblue'):
    """Histogram of small non-negative integer counts (one value per
    dataset), with integer-centered bins and a mean/median stats box - no
    RSCC/distance-specific axis range or threshold line, unlike
    plot_rscc_histogram/plot_distance_histogram."""
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if len(values) == 0:
        print(f'  Skipping {out_path.name}: no data points.')
        return

    lo, hi = int(values.min()), int(values.max())
    bins = np.arange(lo, hi + 2) - 0.5  # one bin per integer value

    plt.figure(figsize=(8, 6))
    plt.hist(values, bins=bins, edgecolor='black', alpha=0.7, color=color)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel('Number of Datasets', fontsize=12)
    plt.title(f'{title} (n={len(values)} datasets)', fontsize=13)
    plt.grid(True, alpha=0.3)

    stats_text = f'Mean:   {values.mean():.2f}\nMedian: {np.median(values):.2f}'
    plt.text(0.98, 0.98, stats_text, transform=plt.gca().transAxes,
              va='top', ha='right', fontsize=10,
              bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'  Histogram saved to: {out_path}')


def residue_label_from_key(chain_id, res_id, altloc):
    """Same format as calc_rscc.py's residue label: '{chain}{resi}', or
    '{chain}{resi}-{altloc}' when altloc is non-blank."""
    label = f'{chain_id}{res_id}'
    if altloc:
        label += f'-{altloc}'
    return label


def _dataset_lig_vs_ref(dataset, args, run_dir):
    """For one dataset, matches every reference LIG conformation in
    ref_pdb_path(args, dataset) to the nearest cluster-rep ligand pose in
    run_dir/cluster_rep_models.pdb (by centroid distance, no RSCC
    computed here - qfit compare_lig_rscc's matching strategy, minus the
    scoring step), using run_dir/cluster_reps.csv's 'rscc' column (row i ==
    the i-th MODEL block, same pipeline-wide indexing convention used
    everywhere else) and ref_rscc_csv_path(args, dataset)'s per-residue
    RSCC for the already-computed values on each side.

    Returns (matched_ref_rscc, matched_pipeline_rscc, n_unmatched_ref,
    n_excess_pipeline).
    """
    ref_pdb = ref_pdb_path(args, dataset)
    ref_csv = ref_rscc_csv_path(args, dataset)
    cluster_csv = run_dir / 'cluster_reps.csv'
    cluster_pdb = run_dir / 'cluster_rep_models.pdb'

    if not (ref_pdb.exists() and ref_csv.exists() and cluster_csv.exists() and cluster_pdb.exists()):
        print(f'  {dataset}: missing required file(s) for lig-vs-reference comparison '
              f'(ref_pdb={ref_pdb.exists()}, ref_rscc={ref_csv.exists()}, '
              f'cluster_reps={cluster_csv.exists()}, cluster_rep_models={cluster_pdb.exists()}); skipping.')
        return [], [], 0, 0

    ref_ligs = lig_conformations(read_pdb_raw_atoms(ref_pdb))
    if not ref_ligs:
        print(f'  {dataset}: no LIG residue found in reference {ref_pdb}; skipping.')
        return [], [], 0, 0
    ref_rscc_df = read_calc_rscc_csv(ref_csv)
    ref_rscc = dict(zip(ref_rscc_df['residue'], ref_rscc_df['rscc']))

    cluster_df = pd.read_csv(cluster_csv)
    pipeline_rscc_by_row = list(cluster_df['rscc'])

    model_blocks = split_pdb_models(cluster_pdb)
    n_models = min(len(model_blocks), len(pipeline_rscc_by_row))
    if len(model_blocks) != len(pipeline_rscc_by_row):
        print(f'  {dataset}: cluster_rep_models.pdb has {len(model_blocks)} model(s) but '
              f'cluster_reps.csv has {len(pipeline_rscc_by_row)} row(s); using first {n_models}.')

    pipeline_centroids = []
    for i in range(n_models):
        lig_atoms = [a for a in model_blocks[i] if a['res_name'] == 'LIG']
        pipeline_centroids.append(
            np.array([a['coord'] for a in lig_atoms]).mean(axis=0) if lig_atoms else None
        )

    used_rows = set()
    matched_ref, matched_pipeline = [], []
    n_unmatched_ref = 0
    for (chain_id, res_id, altloc), ref_centroid in ref_ligs.items():
        ref_label = residue_label_from_key(chain_id, res_id, altloc)
        ref_val = ref_rscc.get(ref_label)
        if ref_val is None or pd.isna(ref_val):
            n_unmatched_ref += 1
            continue

        best_row, best_dist = None, float('inf')
        for i in range(n_models):
            c = pipeline_centroids[i]
            if c is None:
                continue
            dist = float(np.linalg.norm(c - ref_centroid))
            if dist < best_dist:
                best_dist = dist
                best_row = i

        if best_row is None or best_dist > args.centroid_cutoff:
            n_unmatched_ref += 1
            continue

        matched_ref.append(ref_val)
        matched_pipeline.append(pipeline_rscc_by_row[best_row])
        used_rows.add(best_row)

    n_excess_pipeline = sum(
        1 for i in range(n_models) if pipeline_centroids[i] is not None and i not in used_rows
    )
    return matched_ref, matched_pipeline, n_unmatched_ref, n_excess_pipeline


def plot_lig_vs_ref(args, run_dir_for_dataset, title, out_name):
    """Pools stage-appropriate cluster_reps.csv ligand RSCC vs matched
    reference ligand RSCC across every dataset in datasets.txt into a single
    scatter plot (Reference on x, Pipeline on y - qfit compare_lig_rscc's
    convention), with unmatched-reference/excess-pipeline counts labeled.

    run_dir_for_dataset(dataset) -> Path to the directory holding that
    dataset's cluster_reps.csv + cluster_rep_models.pdb for this stage.
    """
    datasets = read_datasets(args.datasets_file)
    all_ref, all_pipeline = [], []
    total_unmatched, total_excess = 0, 0

    for dataset in datasets:
        run_dir = run_dir_for_dataset(dataset)
        ref_vals, pipeline_vals, n_unmatched, n_excess = _dataset_lig_vs_ref(dataset, args, run_dir)
        all_ref.extend(ref_vals)
        all_pipeline.extend(pipeline_vals)
        total_unmatched += n_unmatched
        total_excess += n_excess
        print(f'  {dataset}: {len(ref_vals)} matched ligand(s), {n_unmatched} unmatched '
              f'reference ligand(s), {n_excess} excess pipeline ligand(s)')

    graphs_dir = Path(args.graphs_dir)
    graphs_dir.mkdir(parents=True, exist_ok=True)
    plot_rscc_scatter(
        all_ref, all_pipeline,
        xlabel='Reference RSCC', ylabel='Pipeline RSCC',
        title=title,
        out_path=graphs_dir / out_name,
        extra_text=(f'Unmatched reference LIGs: {total_unmatched}\n'
                    f'Excess pipeline LIGs: {total_excess}'),
    )


def _dataset_residues_vs_ref(dataset, args, structure_rscc, restrict_labels):
    """Matches a dataset's already-collected {residue_label: rscc}
    structure_rscc dict against that dataset's reference per-residue RSCC
    (ref_rscc_csv_path), by residue_base (chain+resnum, altloc-insensitive).
    Returns (all_pairs, restricted_pairs), each a list of (ref_val,
    structure_val) tuples; restricted_pairs is further limited to residues
    whose base label is in restrict_labels."""
    ref_csv = ref_rscc_csv_path(args, dataset)
    ref_df = read_calc_rscc_csv(ref_csv)
    if ref_df.empty:
        print(f'  {dataset}: reference RSCC csv not found/empty ({ref_csv}); skipping.')
        return [], []

    ref_vals = {}
    for residue, rscc in zip(ref_df['residue'], ref_df['rscc']):
        if pd.isna(rscc):
            continue
        ref_vals[residue_base(residue)] = rscc

    struct_vals = {}
    for residue, rscc in structure_rscc.items():
        if rscc is None or pd.isna(rscc):
            continue
        struct_vals[residue_base(residue)] = rscc

    all_pairs, restricted_pairs = [], []
    for base in set(ref_vals) & set(struct_vals):
        pair = (ref_vals[base], struct_vals[base])
        all_pairs.append(pair)
        if base in restrict_labels:
            restricted_pairs.append(pair)

    return all_pairs, restricted_pairs


def plot_residues_vs_ref(args, collect_structure_rscc, collect_restrict_labels,
                          out_dir, out_prefix, structure_label):
    """Pools a per-residue RSCC comparison (structure vs matched reference
    residue) across every dataset in datasets.txt into two scatter plots:
    all residues, and residues restricted to collect_restrict_labels(dataset).

    collect_structure_rscc(dataset) -> {residue_label: rscc} for that
    dataset's structure (e.g. best-across-cluster-reps backbone_refined, or
    final_model_refined).
    collect_restrict_labels(dataset) -> set of '{chain}{resnum}' labels to
    additionally restrict to (e.g. refined_residues.csv or
    residues_with_placer_conformers.csv).
    """
    datasets = read_datasets(args.datasets_file)
    all_pairs, restricted_pairs = [], []

    for dataset in datasets:
        structure_rscc = collect_structure_rscc(dataset)
        restrict_labels = collect_restrict_labels(dataset)
        pairs, r_pairs = _dataset_residues_vs_ref(dataset, args, structure_rscc, restrict_labels)
        all_pairs.extend(pairs)
        restricted_pairs.extend(r_pairs)
        print(f'  {dataset}: {len(pairs)} residue(s) matched to reference, '
              f'{len(r_pairs)} within the restricted residue set')

    graphs_dir = Path(out_dir)
    graphs_dir.mkdir(parents=True, exist_ok=True)

    for pairs, suffix, title_suffix in [
        (all_pairs, '', ''),
        (restricted_pairs, '_restricted', ' (restricted residues)'),
    ]:
        xs = [p[0] for p in pairs]
        ys = [p[1] for p in pairs]
        plot_rscc_scatter(
            xs, ys,
            xlabel='Reference RSCC', ylabel=f'{structure_label} RSCC',
            title=f'{structure_label} RSCC vs Reference{title_suffix}',
            out_path=graphs_dir / f'{out_prefix}_vs_reference_rscc{suffix}.png',
        )


def dataset_final_dir(datasets_dir, dataset, args):
    """The per-dataset final_run_name directory:
    <datasets_dir>/<dataset>/<run>/<placer>/<filter>/<placer2>/<filter2>/<final>"""
    return (Path(datasets_dir) / dataset / args.run_name / args.placer_run_name /
            args.filter_run_name / args.placer2_run_name / args.filter2_run_name /
            args.final_run_name)


def dataset_graphs_dir(datasets_dir, dataset, args):
    """Every plot is dataset-specific: no values are pooled across datasets.
    Each dataset gets its own graphs/ folder inside its own final_run_name
    directory: .../<final_run_name>/graphs/"""
    graphs_dir = dataset_final_dir(datasets_dir, dataset, args) / 'graphs'
    graphs_dir.mkdir(parents=True, exist_ok=True)
    return graphs_dir


def dataset_csvs_dir(datasets_dir, dataset, args):
    """Sibling of dataset_graphs_dir: each dataset's final_run_name
    directory also gets its own csvs/ folder, holding the exact data
    plotted into the matching file under graphs/ (same basename, .csv
    instead of .png): .../<final_run_name>/csvs/"""
    csvs_dir = dataset_final_dir(datasets_dir, dataset, args) / 'csvs'
    csvs_dir.mkdir(parents=True, exist_ok=True)
    return csvs_dir


def write_plot_csv(csvs_dir, plot_filename, df):
    """Writes the exact data underlying a plot named plot_filename (e.g.
    'foo.png', as saved into dataset_graphs_dir) to csvs_dir/foo.csv - same
    basename as the plot, .csv extension instead of .png."""
    csv_path = Path(csvs_dir) / (Path(plot_filename).stem + '.csv')
    df.to_csv(csv_path, index=False)
    print(f'  Plot data csv saved to: {csv_path}')


def read_datasets(datasets_file):
    with open(datasets_file) as f:
        return [line.strip() for line in f if line.strip()]


def read_calc_rscc_csv(path):
    """Reads a calc_rscc-style csv (model_idx,residue,rscc). Returns an
    empty DataFrame with the right columns if the file doesn't exist."""
    path = Path(path)
    if not path.exists():
        return pd.DataFrame(columns=['model_idx', 'residue', 'rscc'])
    return pd.read_csv(path)


def read_calc_z_csv(path):
    """Reads a calc_z-style csv (model_idx,residue,max_z,min_z,average_z).
    Returns an empty DataFrame with the right columns if the file doesn't
    exist."""
    path = Path(path)
    if not path.exists():
        return pd.DataFrame(columns=['model_idx', 'residue', 'max_z', 'min_z', 'average_z'])
    return pd.read_csv(path)


def read_calc_rscc_b_csv(path):
    """Reads a calc_rscc_b-style csv (model_idx,residue,event_map,bfactor,
    rscc,spearmans_rho - one row per (event_map, bfactor) combination per
    residue, restricted to whatever residue list calc_rscc_b was given).
    Returns an empty DataFrame with the right columns if the file doesn't
    exist."""
    path = Path(path)
    if not path.exists():
        return pd.DataFrame(columns=['model_idx', 'residue', 'event_map', 'bfactor', 'rscc',
                                      'spearmans_rho'])
    return pd.read_csv(path)


def best_rscc_per_residue(csv_paths):
    """Pools one or more calc_rscc csvs and returns {residue_label: max_rscc},
    the highest rscc seen for each residue label across every csv and every
    row in it. Used to collapse the multiple {dataset}_backbone_refined_{i}_rscc.csv
    files (one per cluster-rep model) down to a single best value per residue."""
    best = {}
    for path in csv_paths:
        df = read_calc_rscc_csv(path)
        for residue, rscc in zip(df['residue'], df['rscc']):
            if pd.isna(rscc):
                continue
            if residue not in best or rscc > best[residue]:
                best[residue] = rscc
    return best


def residue_base(label):
    """Strips a calc_rscc residue label's altloc suffix, e.g. 'A103-B' ->
    'A103', so it can be matched against residues_with_placer_conformers.csv
    (which carries no altloc information) and across structures whose altloc
    bookkeeping may differ."""
    if label is None:
        return None
    return str(label).split('-', 1)[0]


def read_residue_conformer_list(path):
    """Reads residues_with_placer_conformers.csv: a headerless list of
    '{chain}{resnum}' labels, one per line. Returns a set (empty if the file
    doesn't exist)."""
    path = Path(path)
    if not path.exists():
        return set()
    with open(path) as f:
        return {line.strip() for line in f if line.strip()}


def find_lig_residue_label(pdb_path):
    """Scans a PDB file's ATOM/HETATM records for the first residue named
    'LIG' and returns its calc_rscc-style label ('{chain}{resnum}'), or None
    if no LIG residue is present (e.g. the apo structure) or the file
    doesn't exist. Plain fixed-width text parsing, matching how get_lig_id
    identifies the ligand residue elsewhere in the pipeline - not an RSCC
    calculation, just enough to tell which calc_rscc row is the ligand."""
    path = Path(pdb_path)
    if not path.exists():
        return None
    with open(path) as f:
        for line in f:
            if not line.startswith(('ATOM  ', 'HETATM')):
                continue
            resname = line[17:20].strip()
            if resname != 'LIG':
                continue
            chain_id = line[21].strip()
            resnum = line[22:26].strip()
            return f'{chain_id}{resnum}'
    return None


def find_all_lig_residues(pdb_path):
    """Scans a PDB file for every distinct residue named 'LIG' and returns
    [(residue_label, resnum), ...] sorted by resnum, ascending - unlike
    find_lig_residue_label (which only returns the first LIG residue found),
    this returns every one. final_model_refined.pdb can contain multiple LIG
    residues, one per surviving ligand pose, numbered sequentially starting
    at 1 - the same position-based indexing convention as filter2_run_name's
    cluster_reps.csv rows (the i-th LIG residue, resid i, corresponds to the
    i-th DATA row of filter2's cluster_reps.csv). A residue with two or more
    altloc conformers yields one (label, resnum) pair per altloc, all
    sharing that residue's resnum (same altloc-splitting rule as
    lig_conformations/_group_lig_residues)."""
    path = Path(pdb_path)
    if not path.exists():
        return []
    groups = _group_lig_residues(read_pdb_raw_atoms(path))
    keys = sorted(groups.keys(), key=lambda k: (k[1], k[0], k[2]))
    return [(residue_label_from_key(*key), key[1]) for key in keys]


def _auto_axis_range(x, y, pad_frac=0.05):
    """Computes a shared (min, max) axis range covering both x and y, with a
    fractional padding on each side - used for unbounded metrics (e.g.
    Z-scores) that don't have RSCC's natural [0, 1] range."""
    values = np.concatenate([np.asarray(x, dtype=float), np.asarray(y, dtype=float)])
    lo, hi = float(values.min()), float(values.max())
    pad = (hi - lo) * pad_frac if hi > lo else 1.0
    return (lo - pad, hi + pad)


def plot_rscc_scatter(x, y, xlabel, ylabel, title, out_path, extra_text=None, axis_range=(0, 1)):
    """Scatter plot in the style of qfit's compare_lig_rscc._plot_scatter:
    unity dashed line, fixed square axes, equal aspect, mean/median stats box.
    extra_text, if given, is appended to the stats box (e.g. a lost-ligand
    count) below the mean/median lines.

    axis_range: (min, max) applied to both axes and the unity line. Defaults
    to RSCC's natural (0, 1) range; pass a wider/data-driven range (e.g.
    _auto_axis_range(x, y)) for unbounded metrics like Z-scores."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if len(x) == 0:
        print(f'  Skipping {out_path.name}: no data points.')
        return

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(x, y, color='steelblue', s=8, edgecolor='none')

    lims = list(axis_range)
    ax.plot(lims, lims, linestyle='--', color='gray', linewidth=1)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(f'{title} (n={len(x)})')
    ax.set_aspect('equal')

    stats_text = (
        f'{xlabel}: mean={np.mean(x):.3f}, median={np.median(x):.3f}\n'
        f'{ylabel}: mean={np.mean(y):.3f}, median={np.median(y):.3f}'
    )
    if extra_text:
        stats_text += f'\n{extra_text}'
    ax.text(
        0.02, 0.98, stats_text,
        transform=ax.transAxes, ha='left', va='top', fontsize=9,
        bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray', alpha=0.2)
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f'  Scatterplot saved to: {out_path}')


def plot_rscc_histogram(values, title, xlabel, out_path, color='steelblue', value_range=(0, 1)):
    """Histogram in the style of calc_filter_rmsd.py's save_histogram, binned
    over value_range (default RSCC's fixed [0, 1] range). Pass
    value_range=None to auto-compute a padded range from the data instead -
    for unbounded metrics like Z-scores that don't have RSCC's natural
    [0, 1] range."""
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if len(values) == 0:
        print(f'  Skipping {out_path.name}: no data points.')
        return

    if value_range is None:
        lo, hi = float(values.min()), float(values.max())
        pad = (hi - lo) * 0.05 if hi > lo else 1.0
        value_range = (lo - pad, hi + pad)

    bins = np.linspace(value_range[0], value_range[1], 21)
    plt.figure(figsize=(8, 6))
    plt.hist(values, bins=bins, edgecolor='black', alpha=0.7, color=color)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.title(f'{title} (n={len(values)})', fontsize=13)
    plt.xlim(value_range)
    plt.grid(True, alpha=0.3)

    stats_text = f'Mean:   {values.mean():.3f}\nMedian: {np.median(values):.3f}'
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes,
              va='top', ha='left', fontsize=10,
              bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'  Histogram saved to: {out_path}')


def _collect_dataset_rscc(datasets_dir, dataset, args, mode):
    """Builds this dataset's residue-level apo/backbone/final RSCC rows,
    restricted per `mode` ('protein' keeps everything but the identified LIG
    residue; 'lig' keeps only it). Only reads csvs already written by
    calc_apo_rscc/calc_backbone_refined_rscc/calc_final_refined_rscc - no
    RSCC values are computed here."""
    dataset_dir = Path(datasets_dir) / dataset

    apo_csv = dataset_dir / f'{dataset}-aligned-structure_rscc.csv'
    apo_df = read_calc_rscc_csv(apo_csv)
    apo_vals = {}
    for residue, rscc in zip(apo_df['residue'], apo_df['rscc']):
        if pd.isna(rscc):
            continue
        apo_vals[residue_base(residue)] = rscc

    run_dir = dataset_dir / args.run_name / args.placer_run_name / args.filter_run_name
    backbone_csvs = sorted(run_dir.glob(f'{dataset}_backbone_refined_*_rscc.csv'))
    backbone_best = best_rscc_per_residue(backbone_csvs)
    backbone_vals = {}
    for residue, rscc in backbone_best.items():
        base = residue_base(residue)
        if base not in backbone_vals or rscc > backbone_vals[base]:
            backbone_vals[base] = rscc

    backbone_lig_base = None
    for pdb_path in sorted(run_dir.glob(f'{dataset}_backbone_refined_*.pdb')):
        lig_label = find_lig_residue_label(pdb_path)
        if lig_label:
            backbone_lig_base = residue_base(lig_label)
            break

    final_dir = dataset_final_dir(datasets_dir, dataset, args)
    final_csv = final_dir / 'final_model_refined_rscc.csv'
    final_df = read_calc_rscc_csv(final_csv)
    final_vals = {}
    for residue, rscc in zip(final_df['residue'], final_df['rscc']):
        if pd.isna(rscc):
            continue
        final_vals[residue_base(residue)] = rscc

    final_lig_label = find_lig_residue_label(final_dir / 'final_model_refined.pdb')
    final_lig_base = residue_base(final_lig_label) if final_lig_label else None

    conformer_residues = read_residue_conformer_list(
        final_dir / 'residues_with_placer_conformers.csv'
    )

    print(f'  {dataset}: apo={len(apo_vals)} backbone={len(backbone_vals)} '
          f'final={len(final_vals)} conformer_residues={len(conformer_residues)} '
          f'lig_base(backbone={backbone_lig_base}, final={final_lig_base})')

    rows = []
    all_bases = set(apo_vals) | set(backbone_vals) | set(final_vals)
    for base in all_bases:
        is_lig = base is not None and (base == backbone_lig_base or base == final_lig_base)
        if mode == 'protein' and is_lig:
            continue
        if mode == 'lig' and not is_lig:
            continue
        rows.append({
            'residue': base,
            'apo': apo_vals.get(base),
            'backbone': backbone_vals.get(base),
            'final': final_vals.get(base),
            'has_conformer': base in conformer_residues,
        })

    return pd.DataFrame(rows, columns=['residue', 'apo', 'backbone', 'final', 'has_conformer'])


def run_rscc_aggregator(args, mode):
    """For every dataset independently (no pooling across datasets), builds
    and saves scatter plots (backbone-vs-apo, final-vs-apo, and - for
    'protein' only - final-vs-backbone), restricted to that dataset's
    residues_with_placer_conformers.csv, into that dataset's own
    .../<final_run_name>/graphs/ folder, plus a csv of each plot's
    underlying data (residue label + both RSCC columns) into the sibling
    .../<final_run_name>/csvs/ folder. The unrestricted (all-residues)
    variant of these plots was dropped - it carried no information beyond
    the placer-conformer-restricted one and only added noise.

    mode: 'protein' keeps every residue except the identified LIG residue;
          'lig' keeps only the identified LIG residue. final-vs-backbone is
          skipped for 'lig' since plot_filter2_vs_filter1_lig_rscc already
          covers that comparison (matched via cluster_reps.csv, not the
          per-residue calc_rscc csvs) and would otherwise be redundant.
    """
    datasets = read_datasets(args.datasets_file)
    label = mode.capitalize()

    comparisons = [
        ('apo', 'backbone', f'{label} RSCC: Backbone-Refined vs Apo', 'backbone_vs_apo'),
        ('apo', 'final', f'{label} RSCC: Final-Refined vs Apo', 'final_vs_apo'),
    ]
    if mode != 'lig':
        comparisons.append(
            ('backbone', 'final', f'{label} RSCC: Final-Refined vs Backbone-Refined', 'final_vs_backbone')
        )

    for dataset in datasets:
        df = _collect_dataset_rscc(args.datasets_dir, dataset, args, mode)
        if df.empty:
            print(f'  {dataset}: no {mode} RSCC data found; skipping plots for this dataset.')
            continue

        graphs_dir = dataset_graphs_dir(args.datasets_dir, dataset, args)
        csvs_dir = dataset_csvs_dir(args.datasets_dir, dataset, args)

        conformer_df = df[df['has_conformer']]
        for xcol, ycol, title, tag in comparisons:
            paired = conformer_df.dropna(subset=[xcol, ycol])
            out_name = f'{mode}_{tag}_rscc_placer_conformers.png'
            if paired.empty:
                # e.g. every 'vs apo' comparison in mode='lig': the apo structure
                # has no ligand, so this pair of columns is always all-NaN. Skip
                # entirely rather than writing an empty plot/csv.
                print(f'  {dataset}: no data points for {out_name}; skipping.')
                continue
            plot_rscc_scatter(
                paired[xcol], paired[ycol],
                xlabel=f'{xcol.capitalize()} RSCC', ylabel=f'{ycol.capitalize()} RSCC',
                title=f'{title} ({dataset}) (placer-conformer residues)',
                out_path=graphs_dir / out_name,
            )
            write_plot_csv(
                csvs_dir, out_name,
                paired[['residue', xcol, ycol]].rename(
                    columns={xcol: f'{xcol}_rscc', ycol: f'{ycol}_rscc'}
                ),
            )


def _dataset_z_values(z_csv):
    """Reads a calc_z csv and collapses altloc variants of the same residue
    down to a single {residue_base: {'max_z', 'min_z', 'average_z'}} dict,
    keeping the row with the highest max_z for a given base (same
    'prefer the more extreme reading' choice used for RSCC's altloc
    collapsing, adapted since a Z-map anomaly is what's being flagged
    here). calc_z writes one row per residue (no per-event-map/bfactor
    rows to pool, unlike the RSCC csvs), so no other aggregation is
    needed."""
    df = read_calc_z_csv(z_csv)
    vals = {}
    for residue, max_z, min_z, average_z in zip(df['residue'], df['max_z'], df['min_z'], df['average_z']):
        if pd.isna(max_z):
            continue
        base = residue_base(residue)
        if base not in vals or max_z > vals[base]['max_z']:
            vals[base] = {'max_z': max_z, 'min_z': min_z, 'average_z': average_z}
    return vals


def _dataset_final_vs_apo_z(dataset, args):
    """Matches every residue in a dataset's final_model_refined_z.csv to the
    same residue_base in its apo structure's _z.csv (both calc_z outputs),
    restricted to the residues listed in final_run_name's
    residues_with_placer_conformers.csv - same restriction as the RSCC
    vs-apo plots. Returns a DataFrame with columns ['residue', 'apo_max_z',
    'final_max_z', 'apo_min_z', 'final_min_z', 'apo_average_z',
    'final_average_z']."""
    dataset_dir = Path(args.datasets_dir) / dataset
    apo_csv = dataset_dir / f'{dataset}-aligned-structure_z.csv'
    final_dir = dataset_final_dir(args.datasets_dir, dataset, args)
    final_csv = final_dir / 'final_model_refined_z.csv'

    apo_vals = _dataset_z_values(apo_csv)
    final_vals = _dataset_z_values(final_csv)
    restrict_labels = read_residue_conformer_list(final_dir / 'residues_with_placer_conformers.csv')

    rows = []
    for base in set(apo_vals) & set(final_vals):
        if base not in restrict_labels:
            continue
        rows.append({
            'residue': base,
            'apo_max_z': apo_vals[base]['max_z'],
            'final_max_z': final_vals[base]['max_z'],
            'apo_min_z': apo_vals[base]['min_z'],
            'final_min_z': final_vals[base]['min_z'],
            'apo_average_z': apo_vals[base]['average_z'],
            'final_average_z': final_vals[base]['average_z'],
        })
    return pd.DataFrame(rows, columns=[
        'residue', 'apo_max_z', 'final_max_z', 'apo_min_z', 'final_min_z',
        'apo_average_z', 'final_average_z',
    ])


def run_z_aggregator(args):
    """For every dataset independently (no pooling across datasets), builds
    and saves scatter plots comparing final_model_refined's per-residue
    Z-map statistics (max_z, min_z, average_z) against the apo structure's
    own Z-map statistics (same Z-map, different structure), restricted to
    the residues listed in final_run_name's residues_with_placer_conformers.csv
    - same restriction as the RSCC vs-apo plots. Plots go into that
    dataset's own .../<final_run_name>/graphs/ folder, plus a matching csv
    of each plot's underlying data into the sibling
    .../<final_run_name>/csvs/ folder.

    Unlike RSCC (bounded to [0, 1]), Z-scores are unbounded and roughly
    centered on 0, so each plot's axis range is computed from its own data
    (see _auto_axis_range) rather than using plot_rscc_scatter's [0, 1]
    default.
    """
    datasets = read_datasets(args.datasets_file)

    comparisons = [
        ('apo_max_z', 'final_max_z', 'Max Z-score: Final-Refined vs Apo', 'max_z'),
        ('apo_min_z', 'final_min_z', 'Min Z-score: Final-Refined vs Apo', 'min_z'),
        ('apo_average_z', 'final_average_z', 'Average Z-score: Final-Refined vs Apo', 'average_z'),
    ]

    for dataset in datasets:
        df = _dataset_final_vs_apo_z(dataset, args)
        if df.empty:
            print(f'  {dataset}: no Z-map data found; skipping plots for this dataset.')
            continue

        graphs_dir = dataset_graphs_dir(args.datasets_dir, dataset, args)
        csvs_dir = dataset_csvs_dir(args.datasets_dir, dataset, args)

        for xcol, ycol, title, tag in comparisons:
            paired = df.dropna(subset=[xcol, ycol])
            out_name = f'final_vs_apo_{tag}_placer_conformers.png'
            if paired.empty:
                print(f'  {dataset}: no data points for {out_name}; skipping.')
                continue
            plot_rscc_scatter(
                paired[xcol], paired[ycol],
                xlabel=f'Apo {tag.replace("_", " ").title()}',
                ylabel=f'Final-Refined {tag.replace("_", " ").title()}',
                title=f'{title} ({dataset}) (placer-conformer residues)',
                out_path=graphs_dir / out_name,
                axis_range=_auto_axis_range(paired[xcol], paired[ycol]),
            )
            write_plot_csv(
                csvs_dir, out_name,
                paired[['residue', xcol, ycol]],
            )


def _final_lig_z_values(dataset, args):
    """For one dataset, returns a DataFrame with columns ['resnum',
    'residue', 'max_z', 'min_z', 'average_z'] - one row per distinct LIG
    residue found by scanning final_model_refined.pdb directly (via
    find_all_lig_residues, not inferred from any csv), matched to its
    row in final_model_refined_z.csv by residue label. resnum is each LIG
    residue's own residue number - the same position-based indexing
    convention as filter2_run_name's cluster_reps.csv rows (see
    find_all_lig_residues)."""
    final_dir = dataset_final_dir(args.datasets_dir, dataset, args)
    final_pdb = final_dir / 'final_model_refined.pdb'
    final_csv = final_dir / 'final_model_refined_z.csv'

    lig_residues = find_all_lig_residues(final_pdb)
    if not lig_residues:
        return pd.DataFrame(columns=['resnum', 'residue', 'max_z', 'min_z', 'average_z'])

    z_df = read_calc_z_csv(final_csv)
    z_by_residue = {
        residue: (max_z, min_z, average_z)
        for residue, max_z, min_z, average_z in zip(
            z_df['residue'], z_df['max_z'], z_df['min_z'], z_df['average_z']
        )
    }

    rows = []
    for label, resnum in lig_residues:
        stats = z_by_residue.get(label)
        if stats is None:
            print(f'  {dataset}: LIG residue {label} found in {final_pdb} but has no row in '
                  f'{final_csv}; skipping.')
            continue
        max_z, min_z, average_z = stats
        rows.append({'resnum': resnum, 'residue': label, 'max_z': max_z, 'min_z': min_z,
                      'average_z': average_z})
    return pd.DataFrame(rows, columns=['resnum', 'residue', 'max_z', 'min_z', 'average_z'])


def run_final_lig_z_histograms(args):
    """For every dataset independently (no pooling across datasets, same as
    plot_cluster_reps_rscc.py's cluster_reps_1/2 histograms), plots
    histograms of max_z/min_z/average_z over every LIG residue in
    final_model_refined.pdb - i.e. every surviving ligand pose in the final
    model, found by scanning the pdb for resn 'LIG' (not inferred from
    residues_with_placer_conformers.csv or any other restriction). Plots go
    into that dataset's own .../<final_run_name>/graphs/ folder, plus a
    matching csv of each plot's underlying data (resnum + residue label +
    that plot's value) into the sibling .../<final_run_name>/csvs/ folder.
    """
    datasets = read_datasets(args.datasets_file)

    comparisons = [
        ('max_z', 'Final Ligand Max Z-score'),
        ('min_z', 'Final Ligand Min Z-score'),
        ('average_z', 'Final Ligand Average Z-score'),
    ]

    for dataset in datasets:
        df = _final_lig_z_values(dataset, args)
        if df.empty:
            print(f'  {dataset}: no final LIG residue(s) with Z-map data found; skipping plots '
                  f'for this dataset.')
            continue

        graphs_dir = dataset_graphs_dir(args.datasets_dir, dataset, args)
        csvs_dir = dataset_csvs_dir(args.datasets_dir, dataset, args)

        for col, title in comparisons:
            values_df = df.dropna(subset=[col])
            out_name = f'final_lig_{col}.png'
            if values_df.empty:
                print(f'  {dataset}: no data points for {out_name}; skipping.')
                continue
            plot_rscc_histogram(
                values_df[col],
                title=f'{title} ({dataset})',
                xlabel=title,
                out_path=graphs_dir / out_name,
                value_range=None,
            )
            write_plot_csv(
                csvs_dir, out_name,
                values_df[['resnum', 'residue', col]],
            )


def _plot_bfactor_lines(lines_data, out_path, title, ylabel):
    """lines_data: list of (bfactors, values, rho) tuples, one per residue -
    bfactors/values are that residue's full bfactor sweep (see
    _dataset_final_rscc_b_lines), rho is that sweep's spearmans_rho (or None
    if undefined). Draws one line per tuple onto a single plot.

    With up to ~200 lines pooled across every dataset, a per-residue legend
    would be unreadable, so lines are colored by rho instead of by residue
    identity: a diverging colormap centered at 0, with a colorbar - this
    turns color into an at-a-glance summary of which lines are
    bfactor-sensitive (rho near 0 or negative) vs not (rho near +1), and
    pairs with the companion spearman-rho histogram. Lines with an
    undefined rho (a canonical event map with only 1 bfactor row - shouldn't
    happen when calc_rscc_b is run with >=2 --bfactors, but guarded anyway)
    are drawn in flat gray. Thin, semi-transparent lines let overlapping
    regions read as density rather than a solid blob."""
    if not lines_data:
        print(f'  Skipping {out_path.name}: no data points.')
        return

    fig, ax = plt.subplots(figsize=(9, 7))
    cmap = plt.get_cmap('coolwarm_r')
    norm = plt.Normalize(vmin=-1, vmax=1)

    for bfactors, values, rho in lines_data:
        color = cmap(norm(rho)) if rho is not None else (0.6, 0.6, 0.6, 1.0)
        ax.plot(bfactors, values, color=color, linewidth=0.8, alpha=0.5)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label='Spearman ρ (bfactor vs RSCC), gray = undefined')

    ax.set_xlabel('B-factor')
    ax.set_ylabel(ylabel)
    ax.set_title(f'{title} (n={len(lines_data)} residues)')
    ax.grid(True, alpha=0.3)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f'  Line plot saved to: {out_path}')


def _dataset_final_rscc_b_lines(dataset, args):
    """For one dataset, returns [(bfactors, rsccs, rho, residue), ...] - one
    entry per residue in final_model_refined_rscc_b.csv, where bfactors/
    rsccs are that residue's full bfactor sweep (sorted ascending by
    bfactor) against whichever event map contains that residue's single
    highest RSCC value across all (event_map, bfactor) rows - the
    'canonical' event map for that residue - and rho is calc_rscc_b's own
    spearmans_rho for that (residue, canonical event map) group (None if it
    was left empty, i.e. fewer than 2 distinct bfactors)."""
    final_dir = dataset_final_dir(args.datasets_dir, dataset, args)
    csv_path = final_dir / 'final_model_refined_rscc_b.csv'
    df = read_calc_rscc_b_csv(csv_path)
    if df.empty:
        return []

    lines = []
    for residue, group in df.groupby('residue'):
        group = group.dropna(subset=['rscc'])
        if group.empty:
            continue
        canonical_map = group.loc[group['rscc'].idxmax(), 'event_map']
        map_group = group[group['event_map'] == canonical_map].sort_values('bfactor')
        bfactors = map_group['bfactor'].to_numpy(dtype=float)
        rsccs = map_group['rscc'].to_numpy(dtype=float)
        rho = map_group['spearmans_rho'].iloc[0]
        rho = None if pd.isna(rho) else float(rho)
        lines.append((bfactors, rsccs, rho, residue))
    return lines


def run_bfactor_sensitivity_plots(args):
    """For every dataset independently (no pooling across datasets, same as
    plot_final_lig_z.py), builds three plots from that dataset's own
    canonical-event-map bfactor sweep (see _dataset_final_rscc_b_lines,
    reading final_model_refined_rscc_b.csv) into that dataset's own
    .../<final_run_name>/graphs/ folder, plus a matching csv of each plot's
    underlying data (same basename, .csv instead of .png) into the sibling
    .../<final_run_name>/csvs/ folder:

      bfactor_sensitivity_lines.png             RSCC vs bfactor, one line
                                                 per residue (its canonical
                                                 event map's full sweep)
      bfactor_sensitivity_lines_normalized.png  same, each line shifted so
                                                 its lowest-bfactor RSCC is 0
      bfactor_sensitivity_spearman_rho_hist.png histogram of each residue's
                                                 canonical spearmans_rho,
                                                 one value per residue

    Not restricted further - every residue calc_rscc_b was run on (i.e.
    residues_with_placer_conformers.csv) contributes one line/value here.
    """
    datasets = read_datasets(args.datasets_file)

    for dataset in datasets:
        lines = _dataset_final_rscc_b_lines(dataset, args)
        if not lines:
            print(f'  {dataset}: no bfactor-sweep data found; skipping bfactor sensitivity plots.')
            continue
        print(f'  {dataset}: {len(lines)} residue(s) with a canonical bfactor sweep')

        graphs_dir = dataset_graphs_dir(args.datasets_dir, dataset, args)
        csvs_dir = dataset_csvs_dir(args.datasets_dir, dataset, args)

        # --- Plot 1: raw RSCC vs bfactor ---
        _plot_bfactor_lines(
            [(bfactors, rsccs, rho) for bfactors, rsccs, rho, _ in lines],
            out_path=graphs_dir / 'bfactor_sensitivity_lines.png',
            title=f'RSCC vs B-factor (canonical event map per residue) ({dataset})',
            ylabel='RSCC',
        )
        raw_csv_rows = [
            {'residue': residue, 'bfactor': b, 'rscc': r}
            for bfactors, rsccs, _, residue in lines
            for b, r in zip(bfactors, rsccs)
        ]
        write_plot_csv(
            csvs_dir, 'bfactor_sensitivity_lines.png',
            pd.DataFrame(raw_csv_rows, columns=['residue', 'bfactor', 'rscc']),
        )

        # --- Plot 2: normalized (each line's lowest-bfactor RSCC subtracted off) ---
        normalized_lines_data = []
        norm_csv_rows = []
        for bfactors, rsccs, rho, residue in lines:
            baseline = rsccs[0]  # bfactors sorted ascending, so index 0 is the lowest bfactor
            normalized = rsccs - baseline
            normalized_lines_data.append((bfactors, normalized, rho))
            for b, r, nr in zip(bfactors, rsccs, normalized):
                norm_csv_rows.append({'residue': residue, 'bfactor': b, 'rscc': r,
                                       'normalized_rscc': nr})
        _plot_bfactor_lines(
            normalized_lines_data,
            out_path=graphs_dir / 'bfactor_sensitivity_lines_normalized.png',
            title=f'RSCC vs B-factor, normalized to lowest bfactor (canonical event map per residue) ({dataset})',
            ylabel='RSCC - RSCC(lowest bfactor)',
        )
        write_plot_csv(
            csvs_dir, 'bfactor_sensitivity_lines_normalized.png',
            pd.DataFrame(norm_csv_rows, columns=['residue', 'bfactor', 'rscc', 'normalized_rscc']),
        )

        # --- Plot 3: histogram of canonical spearmans_rho, one per residue ---
        rho_rows = [
            {'residue': residue, 'spearmans_rho': rho}
            for _, _, rho, residue in lines if rho is not None
        ]
        rho_df = pd.DataFrame(rho_rows, columns=['residue', 'spearmans_rho'])
        plot_rscc_histogram(
            rho_df['spearmans_rho'],
            title=f'Spearman ρ of RSCC vs B-factor (canonical event map per residue) ({dataset})',
            xlabel='Spearman ρ',
            out_path=graphs_dir / 'bfactor_sensitivity_spearman_rho_hist.png',
            value_range=(-1.1, 1.1),
        )
        write_plot_csv(csvs_dir, 'bfactor_sensitivity_spearman_rho_hist.png', rho_df)


def _round1_index_from_placer_file(placer_file, dataset):
    """Extracts the round-1 backbone-refined model index i from a filter2
    cluster_reps.csv placer_file entry. Round-2 placer_files are RSR outputs
    named '{dataset}_backbone_refined_{i}_refined.pdb' (rsr_placer2 derives
    them from PLACER round-2's '{dataset}_backbone_refined_{i}_model.pdb',
    which was itself run on filter_run_name's i-th (1-indexed) cluster rep,
    '{dataset}_backbone_refined_{i}.pdb' - the same indexing convention used
    throughout the pipeline). Returns None if the pattern doesn't match."""
    pattern = re.escape(dataset) + r'_backbone_refined_(\d+)_refined\.pdb$'
    m = re.search(pattern, str(placer_file))
    return int(m.group(1)) if m else None


def plot_filter2_vs_filter1_lig_rscc(args):
    """
    For each dataset, matches every filter2_run_name cluster rep back to the
    filter_run_name cluster rep it originated from, via the round-1
    backbone-refined model index embedded in its placer_file column (see
    _round1_index_from_placer_file), and plots their 'rscc' values against
    each other: filter_2 (y) vs filter_1 (x). Also labels how many of
    filter_run_name's cluster reps never made it into filter2_run_name's
    cluster_reps.csv (i.e. were lost between round 1 and round 2 filtering).

    filter.py keeps at most one cluster rep per input placer_file, so each
    round-1 index maps to at most one filter2_run_name row - the match is
    always 1:1, never many:1.

    Also writes the matched (filter_1_cluster_rep_index, filter_2_cluster_rep_index,
    filter_1_rscc, filter_2_rscc) rows to a csv under .../<final_run_name>/csvs/,
    matching the plot's filename. filter_1_cluster_rep_index is the round-1
    index embedded in the placer_file column (1-based position in
    filter_run_name/cluster_reps.csv); filter_2_cluster_rep_index is that
    matched row's own 1-based position in filter2_run_name/cluster_reps.csv,
    which equals the LIG residue number that row ends up as in final_model.pdb.
    """
    datasets = read_datasets(args.datasets_file)

    for dataset in datasets:
        dataset_dir = Path(args.datasets_dir) / dataset
        run_dir = dataset_dir / args.run_name / args.placer_run_name / args.filter_run_name
        filter1_csv = run_dir / 'cluster_reps.csv'
        filter2_csv = (run_dir / args.placer2_run_name / args.filter2_run_name /
                        'cluster_reps.csv')

        if not filter1_csv.exists() or not filter2_csv.exists():
            print(f'  {dataset}: missing cluster_reps.csv (filter_1={filter1_csv.exists()}, '
                  f'filter_2={filter2_csv.exists()}); skipping filter_2-vs-filter_1 comparison.')
            continue

        filter1_df = pd.read_csv(filter1_csv)
        # The i-th (1-indexed) DATA row of filter_run_name/cluster_reps.csv is
        # '{dataset}_backbone_refined_{i}.pdb', matching the pipeline-wide
        # position-based indexing convention.
        filter1_rscc_by_index = {i + 1: rscc for i, rscc in enumerate(filter1_df['rscc'])}
        n_total_filter1 = len(filter1_rscc_by_index)

        filter2_df = pd.read_csv(filter2_csv)

        matched_x, matched_y = [], []
        matched_filter1_idx, matched_filter2_idx = [], []
        matched_indices = set()
        unmatched_placer_files = []
        # filter2_row_idx is that row's own 1-based position in
        # filter2_run_name/cluster_reps.csv - the pipeline-wide position-based
        # indexing convention means this equals the LIG residue number that
        # row ends up as in final_model.pdb.
        for filter2_row_idx, (placer_file, rscc2) in enumerate(
            zip(filter2_df['placer_file'], filter2_df['rscc']), start=1
        ):
            idx = _round1_index_from_placer_file(placer_file, dataset)
            if idx is None:
                unmatched_placer_files.append(placer_file)
                continue
            rscc1 = filter1_rscc_by_index.get(idx)
            if rscc1 is None:
                print(f"  {dataset}: filter_2 cluster rep references round-1 index {idx}, "
                      f"which has no row in {filter1_csv}; skipping.")
                continue
            matched_indices.add(idx)
            matched_x.append(rscc1)
            matched_y.append(rscc2)
            matched_filter1_idx.append(idx)
            matched_filter2_idx.append(filter2_row_idx)

        if unmatched_placer_files:
            preview = unmatched_placer_files[:3]
            print(f"  {dataset}: {len(unmatched_placer_files)} filter_2 cluster rep(s) had a "
                  f"placer_file that didn't match the expected "
                  f"'{dataset}_backbone_refined_<i>_refined.pdb' pattern; skipped: {preview}"
                  f"{'...' if len(unmatched_placer_files) > 3 else ''}")

        n_lost = n_total_filter1 - len(matched_indices)
        print(f'  {dataset}: {len(matched_x)} matched ligand(s); '
              f'{n_lost}/{n_total_filter1} filter_1 cluster rep(s) lost by filter_2')

        if not matched_x:
            print(f'  {dataset}: no matched ligand(s); skipping lig_filter2_vs_filter1_rscc.')
            continue

        graphs_dir = dataset_graphs_dir(args.datasets_dir, dataset, args)
        out_name = 'lig_filter2_vs_filter1_rscc.png'
        plot_rscc_scatter(
            matched_x, matched_y,
            xlabel='Filter_1 RSCC', ylabel='Filter_2 RSCC',
            title=f'Ligand RSCC: Filter_2 vs Filter_1 ({dataset})',
            out_path=graphs_dir / out_name,
            extra_text=f'Lost filter_1 -> filter_2: {n_lost}/{n_total_filter1}',
        )
        csvs_dir = dataset_csvs_dir(args.datasets_dir, dataset, args)
        write_plot_csv(
            csvs_dir, out_name,
            pd.DataFrame({
                'filter_1_cluster_rep_index': matched_filter1_idx,
                'filter_2_cluster_rep_index': matched_filter2_idx,
                'filter_1_rscc': matched_x,
                'filter_2_rscc': matched_y,
            }),
        )
