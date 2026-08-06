"""
Shared helpers for the RSCC analysis scripts in analysis_scripts/.

None of these scripts recompute RSCC - they only read the per-residue CSVs
already written during the pipeline by calc_rscc (calc_apo_rscc,
calc_backbone_refined_rscc, calc_final_refined_rscc, each producing a
model_idx,residue,rscc csv) and the 'rscc' column already written into
cluster_reps.csv by filter/filter2.
"""
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
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


def plot_rscc_scatter(x, y, xlabel, ylabel, title, out_path, extra_text=None):
    """Scatter plot in the style of qfit's compare_lig_rscc._plot_scatter:
    unity dashed line, [0,1] axes, equal aspect, mean/median stats box.
    extra_text, if given, is appended to the stats box (e.g. a lost-ligand
    count) below the mean/median lines."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if len(x) == 0:
        print(f'  Skipping {out_path.name}: no data points.')
        return

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(x, y, color='steelblue', s=8, edgecolor='none')

    lims = [0, 1]
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


def plot_rscc_histogram(values, title, xlabel, out_path, color='steelblue'):
    """Histogram in the style of calc_filter_rmsd.py's save_histogram, binned
    over the fixed [0, 1] RSCC range."""
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if len(values) == 0:
        print(f'  Skipping {out_path.name}: no data points.')
        return

    bins = np.linspace(0, 1, 21)
    plt.figure(figsize=(8, 6))
    plt.hist(values, bins=bins, edgecolor='black', alpha=0.7, color=color)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.title(f'{title} (n={len(values)})', fontsize=13)
    plt.xlim(0, 1)
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
    'protein' only - final-vs-backbone; each over all residues and again
    restricted to that dataset's residues_with_placer_conformers.csv) into
    that dataset's own .../<final_run_name>/graphs/ folder.

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

        subsets = [
            ('all_residues', df, ''),
            ('placer_conformers', df[df['has_conformer']], ' (placer-conformer residues)'),
        ]

        for subset_name, subset_df, title_suffix in subsets:
            suffix = '' if subset_name == 'all_residues' else '_placer_conformers'
            for xcol, ycol, title, tag in comparisons:
                paired = subset_df.dropna(subset=[xcol, ycol])
                out_path = graphs_dir / f'{mode}_{tag}_rscc{suffix}.png'
                plot_rscc_scatter(
                    paired[xcol], paired[ycol],
                    xlabel=f'{xcol.capitalize()} RSCC', ylabel=f'{ycol.capitalize()} RSCC',
                    title=f'{title} ({dataset}){title_suffix}',
                    out_path=out_path,
                )


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
        matched_indices = set()
        unmatched_placer_files = []
        for placer_file, rscc2 in zip(filter2_df['placer_file'], filter2_df['rscc']):
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

        if unmatched_placer_files:
            preview = unmatched_placer_files[:3]
            print(f"  {dataset}: {len(unmatched_placer_files)} filter_2 cluster rep(s) had a "
                  f"placer_file that didn't match the expected "
                  f"'{dataset}_backbone_refined_<i>_refined.pdb' pattern; skipped: {preview}"
                  f"{'...' if len(unmatched_placer_files) > 3 else ''}")

        n_lost = n_total_filter1 - len(matched_indices)
        print(f'  {dataset}: {len(matched_x)} matched ligand(s); '
              f'{n_lost}/{n_total_filter1} filter_1 cluster rep(s) lost by filter_2')

        graphs_dir = dataset_graphs_dir(args.datasets_dir, dataset, args)
        plot_rscc_scatter(
            matched_x, matched_y,
            xlabel='Filter_1 RSCC', ylabel='Filter_2 RSCC',
            title=f'Ligand RSCC: Filter_2 vs Filter_1 ({dataset})',
            out_path=graphs_dir / 'lig_filter2_vs_filter1_rscc.png',
            extra_text=f'Lost filter_1 -> filter_2: {n_lost}/{n_total_filter1}',
        )
