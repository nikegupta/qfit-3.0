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
from scipy.stats import gaussian_kde
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


def build_pooled_argparser(description):
    """Argparser for pooled (cross-dataset) plots that, unlike
    build_ref_argparser's reference-set comparisons, don't need the
    reference set and aren't gated behind -c: the six pipeline run names,
    the datasets dir/file, and --graphs-dir for the pooled plot output
    location (same GRAPHS_DIR/<run>/.../<final_run_name> nesting the -c
    pooled plots use)."""
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
    p.add_argument('--graphs-dir', required=True,
                    help='Output directory for the pooled (cross-dataset) plot(s)')
    return p


def build_despot_argparser(description):
    """Argparser for the per-dataset DESPOT energy-score plot: the six
    pipeline run names plus despot_run_name (needed to locate
    <despot_run_name>/<dataset>_DESPOT.csv), plus the datasets dir/file."""
    p = argparse.ArgumentParser(description=description)
    p.add_argument('run_name')
    p.add_argument('placer_run_name')
    p.add_argument('filter_run_name')
    p.add_argument('placer2_run_name')
    p.add_argument('filter2_run_name')
    p.add_argument('final_run_name')
    p.add_argument('despot_run_name')
    p.add_argument('--datasets-dir', default=DEFAULT_DATASETS_DIR,
                    help='Root directory containing per-dataset folders')
    p.add_argument('--datasets-file', default=DEFAULT_DATASETS_FILE,
                    help='Path to newline-delimited list of dataset names')
    return p


def build_despot_pooled_argparser(description):
    """Argparser for the pooled (cross-dataset) DESPOT energy-score plot:
    same seven positional run names as build_despot_argparser, plus
    --graphs-dir for the pooled plot output location."""
    p = argparse.ArgumentParser(description=description)
    p.add_argument('run_name')
    p.add_argument('placer_run_name')
    p.add_argument('filter_run_name')
    p.add_argument('placer2_run_name')
    p.add_argument('filter2_run_name')
    p.add_argument('final_run_name')
    p.add_argument('despot_run_name')
    p.add_argument('--datasets-dir', default=DEFAULT_DATASETS_DIR,
                    help='Root directory containing per-dataset folders')
    p.add_argument('--datasets-file', default=DEFAULT_DATASETS_FILE,
                    help='Path to newline-delimited list of dataset names')
    p.add_argument('--graphs-dir', required=True,
                    help='Output directory for the pooled (cross-dataset) plot')
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
    Empty list if model_dir/ref_path don't exist or nothing matches.

    Returns a list of {ref_chain, ref_resi, ref_altloc, rmsd, placer_file,
    model_idx} dicts, one per matched reference LIG conformation.
    placer_file/model_idx identify the source file and, 0-based, the
    MODEL/ENDMDL block within it (file order) that produced the minimum-RMSD
    conformer - the same placer_file/index convention filter.py's
    cluster_reps.csv uses."""
    if not model_dir.exists() or not ref_path.exists():
        return []

    model_files = sorted(model_dir.rglob(file_pattern))
    if not model_files:
        return []

    ref_groups = lig_atom_groups(read_pdb_raw_atoms(ref_path))
    if not ref_groups:
        return []

    min_rmsds = {key: np.inf for key in ref_groups}
    best_source = {key: (None, None) for key in ref_groups}

    for model_file in model_files:
        try:
            blocks = split_pdb_models(model_file)
        except Exception as e:
            print(f'    Warning: could not read {model_file}: {e}')
            continue

        for model_idx, atoms in enumerate(blocks):
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
                        best_source[ref_key] = (str(model_file), model_idx)

    return [
        {'ref_chain': key[0], 'ref_resi': key[1], 'ref_altloc': key[2], 'rmsd': v,
         'placer_file': best_source[key][0], 'model_idx': best_source[key][1]}
        for key, v in min_rmsds.items() if np.isfinite(v)
    ]


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


def _dataset_lig_vs_ref(dataset, args, run_dir, alive_rows=None, cluster_csv_override=None,
                         rscc_column='rscc'):
    """For one dataset, matches every reference LIG conformation in
    ref_pdb_path(args, dataset) to the nearest cluster-rep ligand pose in
    run_dir/cluster_rep_models.pdb (by centroid distance, no RSCC
    computed here - qfit compare_lig_rscc's matching strategy, minus the
    scoring step), using run_dir/cluster_reps.csv's 'rscc' column (row i ==
    the i-th MODEL block, same pipeline-wide indexing convention used
    everywhere else) and ref_rscc_csv_path(args, dataset)'s per-residue
    RSCC for the already-computed values on each side.

    cluster_csv_override/rscc_column: optional - read the pipeline-side RSCC values from
    cluster_csv_override's rscc_column column instead of run_dir/cluster_reps.csv's own 'rscc'
    column (still by row position, same indexing). despot_filter.py can reselect a cluster's
    final pose to a DIFFERENT conformer than filter2's own representative, so
    plot_lig_vs_ref_despot.py passes despot_run_dir/cluster_reps.csv (despot_filter.py's own
    traceability output - see its docstring) and rscc_column='despot_rscc', the reselected
    winner's real RSCC, rather than the stale original-representative value filter2's
    cluster_reps.csv would otherwise give. Every other caller (plot_lig_vs_ref_filter1.py/
    plot_lig_vs_ref_filter2.py, where no reselection ever happens) leaves both at their
    defaults and is unaffected.

    alive_rows: optional set of 1-indexed cluster_reps.csv/cluster_rep_models.pdb
    row numbers to restrict matching to (e.g. despot_filter.py survivors - see
    plot_lig_vs_ref_despot.py). A row not in this set is treated exactly like
    a MODEL block with no LIG atoms: never matched, and not counted as excess
    either (it isn't "extra" pipeline output, it was deliberately filtered
    out). None (default) means every row is eligible, the original behavior.

    Returns (matched_ref_rscc, matched_pipeline_rscc, n_unmatched_ref,
    n_excess_pipeline, matched_rows, unmatched_ref_rows, excess_pipeline_rows).

    matched_rows has one dict per matched pair - {ref_chain, ref_resi,
    ref_altloc, ref_rscc, cluster_rep_index, pipeline_chain, pipeline_rscc} -
    for callers that want to write out the full match, not just the pooled
    RSCC values. cluster_rep_index/pipeline_chain identify the matched
    cluster_reps.csv row (1-indexed) and that row's ligand's chain id in
    cluster_rep_models.pdb - the same (chain, resi=cluster_rep_index) pair
    that row's ligand keeps all the way through final_model(_refined).pdb
    and, if it survives, despot_filtered.pdb (build_final_model.py/
    despot_filter.py never renumber it).

    unmatched_ref_rows has one dict per reference LIG conformation that
    never became part of a matched pair - {ref_chain, ref_resi, ref_altloc,
    ref_rscc, reason} - reason is 'no_reference_rscc' (ref_rscc_csv_path had
    no value for this label - see residue_label_from_key) or
    'no_pipeline_match_within_cutoff' (every eligible cluster-rep ligand was
    farther than args.centroid_cutoff, or none were eligible at all);
    ref_rscc is the reference RSCC when known, else None.

    excess_pipeline_rows has one dict per eligible cluster-rep ligand that
    never became any reference ligand's nearest match - {cluster_rep_index,
    pipeline_chain, pipeline_rscc, placer_file, index} - placer_file/index
    are that row's own cluster_reps.csv columns (the filter.py placer_file/
    index convention - the same placer_file/model-index pair the row's pose
    came from), so an excess pipeline ligand can be traced back to its
    source PLACER model.
    """
    ref_pdb = ref_pdb_path(args, dataset)
    ref_csv = ref_rscc_csv_path(args, dataset)
    cluster_csv = run_dir / 'cluster_reps.csv'
    cluster_pdb = run_dir / 'cluster_rep_models.pdb'

    if not (ref_pdb.exists() and ref_csv.exists() and cluster_csv.exists() and cluster_pdb.exists()):
        print(f'  {dataset}: missing required file(s) for lig-vs-reference comparison '
              f'(ref_pdb={ref_pdb.exists()}, ref_rscc={ref_csv.exists()}, '
              f'cluster_reps={cluster_csv.exists()}, cluster_rep_models={cluster_pdb.exists()}); skipping.')
        return [], [], 0, 0, [], [], []

    ref_ligs = lig_conformations(read_pdb_raw_atoms(ref_pdb))
    if not ref_ligs:
        print(f'  {dataset}: no LIG residue found in reference {ref_pdb}; skipping.')
        return [], [], 0, 0, [], [], []
    ref_rscc_df = read_calc_rscc_csv(ref_csv)
    ref_rscc = dict(zip(ref_rscc_df['residue'], ref_rscc_df['rscc']))

    rscc_csv_path = cluster_csv_override if cluster_csv_override is not None else cluster_csv
    if cluster_csv_override is not None and not cluster_csv_override.exists():
        print(f'  {dataset}: missing {cluster_csv_override} for lig-vs-reference comparison; '
              f'skipping.')
        return [], [], 0, 0, [], [], []
    cluster_df = pd.read_csv(rscc_csv_path)
    pipeline_rscc_by_row = list(cluster_df[rscc_column])
    placer_file_by_row = list(cluster_df['placer_file'])
    index_by_row = list(cluster_df['index'])

    model_blocks = split_pdb_models(cluster_pdb)
    n_models = min(len(model_blocks), len(pipeline_rscc_by_row))
    if len(model_blocks) != len(pipeline_rscc_by_row):
        print(f'  {dataset}: cluster_rep_models.pdb has {len(model_blocks)} model(s) but '
              f'cluster_reps.csv has {len(pipeline_rscc_by_row)} row(s); using first {n_models}.')

    pipeline_centroids = []
    pipeline_chains = []
    for i in range(n_models):
        if alive_rows is not None and (i + 1) not in alive_rows:
            pipeline_centroids.append(None)
            pipeline_chains.append(None)
            continue
        lig_atoms = [a for a in model_blocks[i] if a['res_name'] == 'LIG']
        pipeline_centroids.append(
            np.array([a['coord'] for a in lig_atoms]).mean(axis=0) if lig_atoms else None
        )
        pipeline_chains.append(lig_atoms[0]['chain_id'] if lig_atoms else None)

    used_rows = set()
    matched_ref, matched_pipeline, matched_rows = [], [], []
    unmatched_ref_rows = []
    for (chain_id, res_id, altloc), ref_centroid in ref_ligs.items():
        ref_label = residue_label_from_key(chain_id, res_id, altloc)
        ref_val = ref_rscc.get(ref_label)
        if ref_val is None or pd.isna(ref_val):
            unmatched_ref_rows.append({
                'ref_chain': chain_id, 'ref_resi': res_id, 'ref_altloc': altloc,
                'ref_rscc': None, 'reason': 'no_reference_rscc',
            })
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
            unmatched_ref_rows.append({
                'ref_chain': chain_id, 'ref_resi': res_id, 'ref_altloc': altloc,
                'ref_rscc': ref_val, 'reason': 'no_pipeline_match_within_cutoff',
            })
            continue

        pipeline_val = pipeline_rscc_by_row[best_row]
        matched_ref.append(ref_val)
        matched_pipeline.append(pipeline_val)
        matched_rows.append({
            'ref_chain': chain_id, 'ref_resi': res_id, 'ref_altloc': altloc, 'ref_rscc': ref_val,
            'cluster_rep_index': best_row + 1, 'pipeline_chain': pipeline_chains[best_row],
            'pipeline_rscc': pipeline_val,
        })
        used_rows.add(best_row)

    excess_pipeline_rows = [
        {
            'cluster_rep_index': i + 1, 'pipeline_chain': pipeline_chains[i],
            'pipeline_rscc': pipeline_rscc_by_row[i],
            'placer_file': placer_file_by_row[i], 'index': index_by_row[i],
        }
        for i in range(n_models) if pipeline_centroids[i] is not None and i not in used_rows
    ]
    n_excess_pipeline = len(excess_pipeline_rows)
    n_unmatched_ref = len(unmatched_ref_rows)
    return (matched_ref, matched_pipeline, n_unmatched_ref, n_excess_pipeline, matched_rows,
            unmatched_ref_rows, excess_pipeline_rows)


def plot_lig_vs_ref(args, run_dir_for_dataset, title, out_name, alive_rows_for_dataset=None,
                     resi_col_name='cluster_rep_index', chain_col_name='pipeline_chain',
                     cluster_csv_override_for_dataset=None, rscc_column='rscc'):
    """Pools stage-appropriate cluster_reps.csv ligand RSCC vs matched
    reference ligand RSCC across every dataset in datasets.txt into a single
    scatter plot (Reference on x, Pipeline on y - qfit compare_lig_rscc's
    convention), with unmatched-reference/excess-pipeline counts labeled -
    plus three csvs written alongside the plot in <graphs_dir>:
      <out_name stem>.csv - the exact matched pairs (dataset, reference
        ligand chain/resi/altloc/rscc, matched cluster_reps.csv row/chain,
        pipeline rscc).
      <out_name stem>_unmatched_ref.csv - every reference LIG conformation
        that didn't become part of a matched pair (dataset, chain/resi/
        altloc, ref_rscc, reason - see _dataset_lig_vs_ref).
      <out_name stem>_excess_pipeline.csv - every eligible cluster-rep
        ligand that was never any reference ligand's nearest match (dataset,
        cluster_rep_index/chain/rscc, and the placer_file/index it came
        from - see _dataset_lig_vs_ref).

    run_dir_for_dataset(dataset) -> Path to the directory holding that
    dataset's cluster_reps.csv + cluster_rep_models.pdb for this stage.

    alive_rows_for_dataset(dataset) -> optional; if given, restricts that
    dataset's cluster_reps.csv rows to this set (see _dataset_lig_vs_ref's
    alive_rows) instead of considering every row eligible.

    resi_col_name/chain_col_name: column names for the matched row's
    (cluster_rep_index, pipeline_chain) pair in the output csv - override
    these when that pair means something more specific to the caller (e.g.
    plot_lig_vs_ref_despot.py uses despot_filtered_resi/despot_filtered_chain,
    since for despot's survivors that pair is exactly the ligand's residue
    number and chain in despot_filtered.pdb - see _dataset_lig_vs_ref).

    cluster_csv_override_for_dataset(dataset)/rscc_column: optional; threaded straight through
    to _dataset_lig_vs_ref's cluster_csv_override/rscc_column - see that function's docstring.
    """
    datasets = read_datasets(args.datasets_file)
    all_ref, all_pipeline, all_rows = [], [], []
    all_unmatched_ref_rows, all_excess_pipeline_rows = [], []
    total_unmatched, total_excess = 0, 0

    for dataset in datasets:
        run_dir = run_dir_for_dataset(dataset)
        alive_rows = alive_rows_for_dataset(dataset) if alive_rows_for_dataset else None
        cluster_csv_override = (
            cluster_csv_override_for_dataset(dataset) if cluster_csv_override_for_dataset
            else None
        )
        (ref_vals, pipeline_vals, n_unmatched, n_excess, matched_rows,
         unmatched_ref_rows, excess_pipeline_rows) = _dataset_lig_vs_ref(
            dataset, args, run_dir, alive_rows=alive_rows,
            cluster_csv_override=cluster_csv_override, rscc_column=rscc_column,
        )
        all_ref.extend(ref_vals)
        all_pipeline.extend(pipeline_vals)
        total_unmatched += n_unmatched
        total_excess += n_excess
        for row in matched_rows:
            row_out = dict(row)
            row_out['dataset'] = dataset
            all_rows.append(row_out)
        for row in unmatched_ref_rows:
            row_out = dict(row)
            row_out['dataset'] = dataset
            all_unmatched_ref_rows.append(row_out)
        for row in excess_pipeline_rows:
            row_out = dict(row)
            row_out['dataset'] = dataset
            all_excess_pipeline_rows.append(row_out)
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

    if all_rows:
        rows_df = pd.DataFrame(all_rows).rename(
            columns={'cluster_rep_index': resi_col_name, 'pipeline_chain': chain_col_name}
        )[['dataset', 'ref_chain', 'ref_resi', 'ref_altloc', 'ref_rscc',
           chain_col_name, resi_col_name, 'pipeline_rscc']]
        write_plot_csv(graphs_dir, out_name, rows_df)

    out_stem = Path(out_name).stem
    if all_unmatched_ref_rows:
        unmatched_df = pd.DataFrame(all_unmatched_ref_rows)[
            ['dataset', 'ref_chain', 'ref_resi', 'ref_altloc', 'ref_rscc', 'reason']
        ]
        write_plot_csv(graphs_dir, f'{out_stem}_unmatched_ref.png', unmatched_df)

    if all_excess_pipeline_rows:
        excess_df = pd.DataFrame(all_excess_pipeline_rows).rename(
            columns={'cluster_rep_index': resi_col_name, 'pipeline_chain': chain_col_name}
        )[['dataset', chain_col_name, resi_col_name, 'pipeline_rscc', 'placer_file', 'index']]
        write_plot_csv(graphs_dir, f'{out_stem}_excess_pipeline.png', excess_df)


def _dataset_residues_vs_ref(dataset, args, structure_rscc, restrict_labels):
    """Matches a dataset's already-collected {residue_label: rscc}
    structure_rscc dict against that dataset's reference per-residue RSCC
    (ref_rscc_csv_path), by residue_base (chain+resnum, altloc-insensitive).
    Returns (all_pairs, restricted_pairs), each a list of {residue, ref_rscc,
    structure_rscc} dicts; restricted_pairs is further limited to residues
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
        pair = {'residue': base, 'ref_rscc': ref_vals[base], 'structure_rscc': struct_vals[base]}
        all_pairs.append(pair)
        if base in restrict_labels:
            restricted_pairs.append(pair)

    return all_pairs, restricted_pairs


def plot_residues_vs_ref(args, collect_structure_rscc, collect_restrict_labels,
                          out_dir, out_prefix, structure_label, outlier_min_diff=None):
    """Pools a per-residue RSCC comparison (structure vs matched reference
    residue) across every dataset in datasets.txt into two scatter plots:
    all residues, and residues restricted to collect_restrict_labels(dataset).

    collect_structure_rscc(dataset) -> {residue_label: rscc} for that
    dataset's structure (e.g. best-across-cluster-reps backbone_refined, or
    final_model_refined).
    collect_restrict_labels(dataset) -> set of '{chain}{resnum}' labels to
    additionally restrict to (e.g. refined_residues.csv or
    residues_with_placer_conformers.csv).

    Pooling every dataset's residues into one plot can put tens of thousands
    of points on it, so - like the pooled protein_final_vs_apo-style plots -
    these are colored by point density (plot_rscc_scatter's
    color_by_density) rather than a flat color.

    If outlier_min_diff is given (not None), also writes
    {out_prefix}_vs_reference_rscc_outliers.csv to out_dir: every RESTRICTED
    residue (i.e. from collect_restrict_labels - the residues actually
    modeled via PLACER, not just carried over from the apo/backbone
    structure untouched) where ref_rscc - structure_rscc >= outlier_min_diff
    - candidate cases where the pipeline picked a worse-fitting rotamer than
    the reference structure has - sorted by that difference, biggest first.
    """
    datasets = read_datasets(args.datasets_file)
    all_pairs, restricted_pairs = [], []

    for dataset in datasets:
        structure_rscc = collect_structure_rscc(dataset)
        restrict_labels = collect_restrict_labels(dataset)
        pairs, r_pairs = _dataset_residues_vs_ref(dataset, args, structure_rscc, restrict_labels)
        for pair in pairs:
            pair['dataset'] = dataset
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
        xs = [p['ref_rscc'] for p in pairs]
        ys = [p['structure_rscc'] for p in pairs]
        out_name = f'{out_prefix}_vs_reference_rscc{suffix}.png'
        plot_rscc_scatter(
            xs, ys,
            xlabel='Reference RSCC', ylabel=f'{structure_label} RSCC',
            title=f'{structure_label} RSCC vs Reference{title_suffix}',
            out_path=graphs_dir / out_name,
            color_by_density=True,
        )
        if pairs:
            write_plot_csv(graphs_dir, out_name,
                            pd.DataFrame(pairs)[['dataset', 'residue', 'ref_rscc', 'structure_rscc']])

    if outlier_min_diff is not None:
        outliers_df = pd.DataFrame(restricted_pairs,
                                    columns=['dataset', 'residue', 'ref_rscc', 'structure_rscc'])
        outliers_df['rscc_diff'] = outliers_df['ref_rscc'] - outliers_df['structure_rscc']
        outliers_df = outliers_df[outliers_df['rscc_diff'] >= outlier_min_diff]
        outliers_df.sort_values('rscc_diff', ascending=False, inplace=True)

        out_path = graphs_dir / f'{out_prefix}_vs_reference_rscc_outliers.csv'
        outliers_df.to_csv(out_path, index=False)
        print(f'  {len(outliers_df)} residue(s) with ref_rscc - structure_rscc >= '
              f'{outlier_min_diff} out of {len(restricted_pairs)} restricted residue(s); '
              f'written to {out_path}')


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


def write_plot_csv(graphs_dir, plot_filename, df):
    """Writes the exact data underlying a plot named plot_filename (e.g.
    'foo.png') to graphs_dir/foo.csv - the same directory the plot itself was
    saved into, same basename, .csv extension instead of .png."""
    csv_path = Path(graphs_dir) / (Path(plot_filename).stem + '.csv')
    df.to_csv(csv_path, index=False)
    print(f'  Plot data csv saved to: {csv_path}')


def read_despot_csv(path):
    """Reads a DESPOT score_complex.py output csv (ligand,score). Returns an
    empty DataFrame with the right columns if the file doesn't exist."""
    path = Path(path)
    if not path.exists():
        return pd.DataFrame(columns=['ligand', 'score'])
    return pd.read_csv(path)


def run_despot_energies_single(args):
    """For every dataset independently (no pooling across datasets), plots a
    histogram of heavy-atom-normalized DESPOT ligand binding-energy scores
    already written by despot_filter.py into
    .../<final_run_name>/<despot_run_name>/despot_filtered_scores.csv - one
    row per ligand instance that got a DESPOT score (kept or not; instances
    with no matching score, normalized_score == NaN, are excluded here, same
    as everywhere else that reads this csv). despot_filter.py already did
    the heavy-atom normalization (see its own docstring/_heavy_atom_count) -
    no score is computed or normalized here, and the raw (un-normalized)
    <dataset>_DESPOT.csv score is not plotted anywhere in this pipeline
    (raw scores of different-sized ligands aren't comparable to each other).
    Written into that dataset's existing .../<final_run_name>/graphs/ folder
    (not nested under despot_run_name), the same per-dataset location every
    other analysis plot in this file uses:
      despot_energies.png
    """
    datasets = read_datasets(args.datasets_file)

    for dataset in datasets:
        csv_path = despot_filtered_scores_csv_path(args.datasets_dir, dataset, args)
        df = read_despot_filtered_scores_csv(csv_path)
        df = df[df['normalized_score'].notna()]
        if df.empty:
            print(f'  {dataset}: no DESPOT scores found ({csv_path}); skipping.')
            continue

        graphs_dir = dataset_graphs_dir(args.datasets_dir, dataset, args)
        out_name = 'despot_energies.png'

        plot_rscc_histogram(
            df['normalized_score'],
            title=f'DESPOT Ligand Binding Energies, normalized ({dataset})',
            xlabel='Normalized DESPOT Score (per heavy atom)', out_path=graphs_dir / out_name,
            value_range=None,
        )
        write_plot_csv(graphs_dir, out_name,
                        df[['ligand', 'chain', 'resi', 'icode', 'normalized_score', 'kept']])


def run_despot_energies_pooled(args):
    """Pooled (across every dataset in datasets.txt) counterpart of
    run_despot_energies_single: every dataset's heavy-atom-normalized DESPOT
    scores (despot_filtered_scores.csv) combined into one histogram, into
    args.graphs_dir - program.sh points this at
    GRAPHS_DIR/.../<final_run_name>/<despot_run_name>/, nested under
    despot_run_name (unlike the other stage 7/8 pooled plots), since the
    scores themselves are specific to one despot_run_name - with a matching
    csv (now including a 'dataset' column) saved alongside it in
    args.graphs_dir:
      ligand_energies.png
    """
    datasets = read_datasets(args.datasets_file)

    pooled_rows = []
    for dataset in datasets:
        csv_path = despot_filtered_scores_csv_path(args.datasets_dir, dataset, args)
        df = read_despot_filtered_scores_csv(csv_path)
        df = df[df['normalized_score'].notna()].copy()
        if df.empty:
            print(f'  {dataset}: no DESPOT scores found ({csv_path}); skipping.')
            continue
        df['dataset'] = dataset
        pooled_rows.append(df)

    if not pooled_rows:
        print('  No DESPOT scores found for any dataset; skipping pooled plot.')
        return

    pooled_df = pd.concat(pooled_rows, ignore_index=True)

    graphs_dir = Path(args.graphs_dir)
    graphs_dir.mkdir(parents=True, exist_ok=True)

    out_name = 'ligand_energies.png'
    plot_rscc_histogram(
        pooled_df['normalized_score'],
        title='DESPOT Ligand Binding Energies, normalized (pooled)',
        xlabel='Normalized DESPOT Score (per heavy atom)', out_path=graphs_dir / out_name,
        value_range=None,
    )
    write_plot_csv(graphs_dir, out_name,
                    pooled_df[['dataset', 'ligand', 'chain', 'resi', 'icode',
                               'normalized_score', 'kept']])


def despot_filtered_scores_csv_path(datasets_dir, dataset, args):
    """<final_run_name>/<despot_run_name>/despot_filtered_scores.csv - written
    by despot_filter.py alongside despot_filtered.pdb: one row per ligand
    instance (ligand, chain, resi, icode, raw_score, normalized_score, kept) -
    'kept' is True iff that instance survived into despot_filtered.pdb."""
    return dataset_final_dir(datasets_dir, dataset, args) / args.despot_run_name / 'despot_filtered_scores.csv'


def read_despot_filtered_scores_csv(path):
    """Reads despot_filter.py's per-instance score csv. Returns an empty
    DataFrame with the right columns if the file doesn't exist."""
    path = Path(path)
    cols = ['ligand', 'chain', 'resi', 'icode', 'raw_score', 'normalized_score', 'kept']
    if not path.exists():
        return pd.DataFrame(columns=cols)
    return pd.read_csv(path)


def plot_ligand_summary_scatter(x, y, out_path, labels=None,
                                 xlabel='Normalized DESPOT Score (per heavy atom)',
                                 ylabel='Ligand RSCC', title='Ligand Summary'):
    """Flat-color scatter of an arbitrary x metric against RSCC on y. Unlike
    plot_rscc_scatter, x and y here aren't the same kind of value (one's a
    DESPOT score, unbounded; the other's an RSCC), so there's no unity line,
    shared axis range, or forced-equal aspect - each axis is scaled to its
    own data independently.

    x axis is reversed (high -> low, left -> right): a more negative DESPOT
    score is more favorable, so the more-favorable direction reads left-to-
    right like the rest of the plot's positive-is-better convention.

    labels: optional list of per-point text labels (same length/order as
    x/y, e.g. 'C1' for chain C, residue 1), annotated in red just above/right
    of each point. Meant for plots with few enough points to label
    individually (e.g. one dataset's own surviving ligands) - leave as None
    for pooled, cross-dataset plots, where many overlapping labels would be
    unreadable."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if len(x) == 0:
        print(f'  Skipping {out_path.name}: no data points.')
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(x, y, color='steelblue', s=12, edgecolor='none')
    if labels is not None:
        for xi, yi, label in zip(x, y, labels):
            ax.annotate(label, (xi, yi), color='red', fontsize=9,
                        xytext=(4, 4), textcoords='offset points')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(f'{title} (n={len(x)})')
    ax.grid(True, alpha=0.3)
    ax.invert_xaxis()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f'  Scatterplot saved to: {out_path}')


def run_despot_ligand_summary(args):
    """Pooled (across every dataset in datasets.txt) scatter of every
    surviving ligand's (i.e. present in despot_run_name/despot_filtered.pdb -
    see despot_filtered_scores.csv's 'kept' column) heavy-atom-normalized
    DESPOT score against its filter2_run_name/cluster_reps.csv RSCC. No RSCC
    is recomputed here - it's read straight off cluster_reps.csv, matched by
    the same resi-as-row-index convention plot_lig_vs_ref_despot.py uses
    (despot_filtered_scores.csv already records each instance's resi
    directly, so no pdb parsing is needed to make that match). Into
    args.graphs_dir (program.sh points this at
    GRAPHS_DIR/.../<final_run_name>/<despot_run_name>/, nested under
    despot_run_name like the other despot-specific pooled plots):
      ligand_summary.png
    with the exact plotted data saved alongside it, in args.graphs_dir/ligand_summary.csv.
    Doesn't need -c/the reference set - RSCC here comes from cluster_reps.csv,
    not a reference structure.
    """
    datasets = read_datasets(args.datasets_file)
    pooled_rows = []

    for dataset in datasets:
        scores_csv = despot_filtered_scores_csv_path(args.datasets_dir, dataset, args)
        scores_df = read_despot_filtered_scores_csv(scores_csv)
        kept_df = scores_df[scores_df['kept'] == True]  # noqa: E712 (pandas boolean-column filter)
        if kept_df.empty:
            print(f'  {dataset}: no surviving (despot_filter-kept) ligand(s) ({scores_csv}); skipping.')
            continue

        cluster_csv = (Path(args.datasets_dir) / dataset / args.run_name / args.placer_run_name /
                        args.filter_run_name / args.placer2_run_name / args.filter2_run_name /
                        'cluster_reps.csv')
        if not cluster_csv.exists():
            print(f'  {dataset}: {cluster_csv} not found; skipping.')
            continue
        cluster_rscc_by_row = list(pd.read_csv(cluster_csv)['rscc'])

        n_matched = 0
        for _, row in kept_df.iterrows():
            row_idx = int(row['resi']) - 1
            if row_idx < 0 or row_idx >= len(cluster_rscc_by_row):
                print(f'  {dataset}: {row["ligand"]} (resi {row["resi"]}) has no matching '
                      f'cluster_reps.csv row; skipping.')
                continue
            pooled_rows.append({
                'dataset': dataset, 'ligand': row['ligand'],
                'normalized_score': row['normalized_score'],
                'rscc': cluster_rscc_by_row[row_idx],
            })
            n_matched += 1
        print(f'  {dataset}: {n_matched} surviving ligand(s) matched to cluster_reps.csv RSCC')

    graphs_dir = Path(args.graphs_dir)
    graphs_dir.mkdir(parents=True, exist_ok=True)
    out_name = 'ligand_summary.png'

    if not pooled_rows:
        print('  No surviving ligand(s) found for any dataset; skipping ligand summary plot.')
        return

    pooled_df = pd.DataFrame(pooled_rows)
    plot_ligand_summary_scatter(
        pooled_df['normalized_score'], pooled_df['rscc'], graphs_dir / out_name,
    )
    write_plot_csv(graphs_dir, out_name, pooled_df)


def run_despot_ligand_summary_single(args):
    """For every dataset independently (no pooling across datasets),
    scatters that dataset's own surviving (despot_filter-kept) ligands'
    heavy-atom-normalized DESPOT score against their filter2_run_name/
    cluster_reps.csv RSCC - same matching run_despot_ligand_summary (the
    pooled counterpart) uses, just restricted to one dataset. Written
    directly into that dataset's own .../<final_run_name>/<despot_run_name>/
    directory (not graphs_dir - despot_run_name is already a per-dataset
    location, unlike every other pooled plot in this file, so there's no
    separate cross-dataset destination needed for a single-dataset plot):
      ligand_summary.png
    with a matching csv (ligand, chain, resi, icode, normalized_score, rscc)
    saved alongside it. A single dataset typically has only a handful of
    surviving ligands, so - unlike the pooled plot - each point is labeled
    in red with its chain+resi (e.g. 'C1' for chain C, residue 1).
    Doesn't need -c/the reference set - RSCC here comes from cluster_reps.csv,
    not a reference structure.
    """
    datasets = read_datasets(args.datasets_file)

    for dataset in datasets:
        scores_csv = despot_filtered_scores_csv_path(args.datasets_dir, dataset, args)
        scores_df = read_despot_filtered_scores_csv(scores_csv)
        kept_df = scores_df[scores_df['kept'] == True]  # noqa: E712 (pandas boolean-column filter)
        if kept_df.empty:
            print(f'  {dataset}: no surviving (despot_filter-kept) ligand(s) ({scores_csv}); skipping.')
            continue

        cluster_csv = (Path(args.datasets_dir) / dataset / args.run_name / args.placer_run_name /
                        args.filter_run_name / args.placer2_run_name / args.filter2_run_name /
                        'cluster_reps.csv')
        if not cluster_csv.exists():
            print(f'  {dataset}: {cluster_csv} not found; skipping.')
            continue
        cluster_rscc_by_row = list(pd.read_csv(cluster_csv)['rscc'])

        rows = []
        for _, row in kept_df.iterrows():
            row_idx = int(row['resi']) - 1
            if row_idx < 0 or row_idx >= len(cluster_rscc_by_row):
                print(f'  {dataset}: {row["ligand"]} (resi {row["resi"]}) has no matching '
                      f'cluster_reps.csv row; skipping.')
                continue
            rows.append({
                'ligand': row['ligand'], 'chain': row['chain'], 'resi': row['resi'],
                'icode': row['icode'], 'normalized_score': row['normalized_score'],
                'rscc': cluster_rscc_by_row[row_idx],
            })

        if not rows:
            print(f'  {dataset}: no surviving ligand(s) matched to cluster_reps.csv RSCC; skipping.')
            continue
        print(f'  {dataset}: {len(rows)} surviving ligand(s) matched to cluster_reps.csv RSCC')

        despot_dir = dataset_final_dir(args.datasets_dir, dataset, args) / args.despot_run_name
        despot_dir.mkdir(parents=True, exist_ok=True)
        out_name = 'ligand_summary.png'

        df = pd.DataFrame(rows)
        labels = [f'{chain}{resi}' for chain, resi in zip(df['chain'], df['resi'])]
        plot_ligand_summary_scatter(
            df['normalized_score'], df['rscc'], despot_dir / out_name,
            labels=labels, title=f'Ligand Summary ({dataset})',
        )
        write_plot_csv(despot_dir, out_name, df)


def ref_despot_csv_path(args, dataset):
    """program.sh's Stage 0d (ref_set_despot) writes REF_SET/<dataset>/
    <dataset>_DESPOT.csv directly alongside the reference structure - same
    sibling-file convention as ref_rscc_csv_path."""
    return ref_pdb_path(args, dataset).parent / f'{dataset}_DESPOT.csv'


def _ref_despot_conformations(dataset, args):
    """Returns {(chain, resi, altloc): (centroid, normalized_score)} for
    every LIG conformation in this dataset's reference structure that has a
    matching DESPOT score, normalizing ref_despot_csv_path's raw per-instance
    score by that instance's own heavy-atom count (non-'H'-element atom
    count, element inferred from atom name via _element_of) - the same
    normalization despot_filter.py applies on the pipeline side (see
    despot_filter.py's _heavy_atom_count). Instance labels are built the same
    way symmetry_expand.py's split_ligand_instances/assign_bond_orders.py
    named them: f'lig{chain}{resi}', or f'lig{chain}{resi}-{altloc}' for a
    residue with any non-blank altloc (icode is assumed blank - reference
    structures carry no insertion codes in this pipeline). A conformation
    with zero heavy atoms, or no matching score in ref_despot_csv_path, is
    left out. Empty dict if the reference structure or its DESPOT csv is
    missing, or has no LIG atoms."""
    ref_pdb = ref_pdb_path(args, dataset)
    despot_csv = ref_despot_csv_path(args, dataset)
    if not (ref_pdb.exists() and despot_csv.exists()):
        return {}

    groups = lig_atom_groups(read_pdb_raw_atoms(ref_pdb))
    if not groups:
        return {}

    scores_df = read_despot_csv(despot_csv)
    scores_by_label = dict(zip(scores_df['ligand'], scores_df['score']))

    result = {}
    for (chain_id, resi, altloc), atom_list in groups.items():
        n_heavy = sum(1 for a in atom_list if _element_of(a['name']) != 'H')
        if n_heavy == 0:
            continue
        label = f'lig{chain_id}{resi}' + (f'-{altloc}' if altloc else '')
        raw_score = scores_by_label.get(label)
        if raw_score is None:
            continue
        centroid = np.array([a['coord'] for a in atom_list]).mean(axis=0)
        result[(chain_id, resi, altloc)] = (centroid, raw_score / n_heavy)
    return result


def _dataset_despot_vs_ref(dataset, args, run_dir, alive_rows=None):
    """For one dataset, matches every reference LIG conformation with a
    heavy-atom-normalized DESPOT score (_ref_despot_conformations) to the
    nearest cluster-rep ligand pose in run_dir/cluster_rep_models.pdb (by
    centroid distance) - the exact same matching strategy _dataset_lig_vs_ref
    uses for RSCC, just gated on DESPOT-score presence instead of RSCC
    presence. The matched row's own normalized DESPOT score is then read
    from despot_run_name/despot_filtered_scores.csv, keyed by resi ==
    cluster_rep_index (the same position-based indexing convention used
    throughout the pipeline - see despot_filter.py). A matched row with no
    despot_filtered_scores.csv entry (e.g. despot never scored it) is
    treated like a reference ligand with no eligible pipeline match -
    counted as unmatched, not plotted.

    alive_rows: optional set of 1-indexed cluster_reps.csv/cluster_rep_models.pdb
    row numbers to restrict matching to - same meaning and same set
    plot_lig_vs_ref_despot.py passes to _dataset_lig_vs_ref (despot_filter.py
    survivors, from despot_filtered.pdb), so despot_vs_reference.png matches
    the exact same reference-to-pipeline pairs as lig_vs_reference_rscc.png
    in the same despot_run_name folder. None (default) means every row is
    eligible.

    Returns (matched_ref, matched_pipeline, n_unmatched_ref, n_excess_pipeline,
    matched_rows, unmatched_ref_rows, excess_pipeline_rows) - same shape and
    same field conventions as _dataset_lig_vs_ref. matched_rows has one dict
    per matched pair: {ref_chain, ref_resi, ref_altloc, ref_despot_normalized,
    cluster_rep_index, pipeline_chain, pipeline_despot_normalized}.
    unmatched_ref_rows has one dict per reference LIG conformation that never
    matched - {ref_chain, ref_resi, ref_altloc, ref_despot_normalized,
    reason} - reason is 'no_pipeline_match_within_cutoff' or
    'no_pipeline_despot_score' (matched a cluster-rep pose, but that row has
    no despot_filtered_scores.csv entry). excess_pipeline_rows has one dict
    per eligible cluster-rep ligand that was never any reference ligand's
    nearest match - {cluster_rep_index, pipeline_chain, placer_file, index} -
    placer_file/index are that row's own cluster_reps.csv columns.
    """
    ref_conformations = _ref_despot_conformations(dataset, args)
    if not ref_conformations:
        print(f'  {dataset}: no reference DESPOT score(s) found; skipping.')
        return [], [], 0, 0, [], [], []

    cluster_csv = run_dir / 'cluster_reps.csv'
    cluster_pdb = run_dir / 'cluster_rep_models.pdb'
    scores_csv = despot_filtered_scores_csv_path(args.datasets_dir, dataset, args)

    if not (cluster_csv.exists() and cluster_pdb.exists() and scores_csv.exists()):
        print(f'  {dataset}: missing required file(s) for despot-vs-reference comparison '
              f'(cluster_reps={cluster_csv.exists()}, cluster_rep_models={cluster_pdb.exists()}, '
              f'despot_filtered_scores={scores_csv.exists()}); skipping.')
        return [], [], 0, 0, [], [], []

    pipeline_scores_df = read_despot_filtered_scores_csv(scores_csv)
    pipeline_normalized_by_resi = dict(zip(pipeline_scores_df['resi'], pipeline_scores_df['normalized_score']))

    cluster_df = pd.read_csv(cluster_csv)
    placer_file_by_row = list(cluster_df['placer_file'])
    index_by_row = list(cluster_df['index'])

    model_blocks = split_pdb_models(cluster_pdb)
    n_models = min(len(model_blocks), len(placer_file_by_row))
    if len(model_blocks) != len(placer_file_by_row):
        print(f'  {dataset}: cluster_rep_models.pdb has {len(model_blocks)} model(s) but '
              f'cluster_reps.csv has {len(placer_file_by_row)} row(s); using first {n_models}.')

    pipeline_centroids = []
    pipeline_chains = []
    for i in range(n_models):
        if alive_rows is not None and (i + 1) not in alive_rows:
            pipeline_centroids.append(None)
            pipeline_chains.append(None)
            continue
        lig_atoms = [a for a in model_blocks[i] if a['res_name'] == 'LIG']
        pipeline_centroids.append(
            np.array([a['coord'] for a in lig_atoms]).mean(axis=0) if lig_atoms else None
        )
        pipeline_chains.append(lig_atoms[0]['chain_id'] if lig_atoms else None)

    used_rows = set()
    matched_ref, matched_pipeline, matched_rows = [], [], []
    unmatched_ref_rows = []
    for (chain_id, res_id, altloc), (ref_centroid, ref_val) in ref_conformations.items():
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
            unmatched_ref_rows.append({
                'ref_chain': chain_id, 'ref_resi': res_id, 'ref_altloc': altloc,
                'ref_despot_normalized': ref_val, 'reason': 'no_pipeline_match_within_cutoff',
            })
            continue

        pipeline_val = pipeline_normalized_by_resi.get(best_row + 1)
        if pipeline_val is None or pd.isna(pipeline_val):
            unmatched_ref_rows.append({
                'ref_chain': chain_id, 'ref_resi': res_id, 'ref_altloc': altloc,
                'ref_despot_normalized': ref_val, 'reason': 'no_pipeline_despot_score',
            })
            continue

        matched_ref.append(ref_val)
        matched_pipeline.append(pipeline_val)
        matched_rows.append({
            'ref_chain': chain_id, 'ref_resi': res_id, 'ref_altloc': altloc,
            'ref_despot_normalized': ref_val,
            'cluster_rep_index': best_row + 1, 'pipeline_chain': pipeline_chains[best_row],
            'pipeline_despot_normalized': pipeline_val,
        })
        used_rows.add(best_row)

    excess_pipeline_rows = [
        {
            'cluster_rep_index': i + 1, 'pipeline_chain': pipeline_chains[i],
            'placer_file': placer_file_by_row[i], 'index': index_by_row[i],
        }
        for i in range(n_models) if pipeline_centroids[i] is not None and i not in used_rows
    ]
    n_excess_pipeline = len(excess_pipeline_rows)
    n_unmatched_ref = len(unmatched_ref_rows)
    return (matched_ref, matched_pipeline, n_unmatched_ref, n_excess_pipeline, matched_rows,
            unmatched_ref_rows, excess_pipeline_rows)


def plot_despot_vs_ref(args, run_dir_for_dataset, title, out_name, alive_rows_for_dataset=None):
    """Pools per-dataset heavy-atom-normalized DESPOT score comparisons
    (reference structure vs matched pipeline cluster-rep ligand - same
    centroid-distance matching plot_lig_vs_ref uses for RSCC, see
    _dataset_despot_vs_ref) across every dataset in datasets.txt into a
    single scatter plot (Reference on x, Pipeline on y), with unmatched-
    reference/excess-pipeline counts labeled - plus three csvs written
    alongside the plot in <graphs_dir> (same convention as plot_lig_vs_ref):
      <out_name stem>.csv - the exact matched pairs (dataset, reference
        ligand chain/resi/altloc/normalized score, matched cluster_reps.csv
        row/chain, pipeline normalized score). Only written if at least one
        dataset had a matched pair.
      <out_name stem>_unmatched_ref.csv / _excess_pipeline.csv - written
        whenever there's at least one such row, even if no dataset had any
        matched pair at all (see _dataset_despot_vs_ref).

    Unlike plot_lig_vs_ref, the axis range is computed from the data
    (_auto_axis_range) rather than plot_rscc_scatter's [x, 1]-ish RSCC
    default - a DESPOT score is an unbounded energy, not a bounded
    correlation coefficient.

    run_dir_for_dataset(dataset) -> Path to the directory holding that
    dataset's cluster_reps.csv + cluster_rep_models.pdb (filter2_run_name).

    alive_rows_for_dataset(dataset) -> optional; if given, restricts that
    dataset's cluster_reps.csv rows to this set (see _dataset_despot_vs_ref's
    alive_rows) instead of considering every row eligible. Pass the same
    despot_filtered.pdb-derived set plot_lig_vs_ref_despot.py uses so this
    plot's matched pairs - and its unmatched/excess counts - agree exactly
    with lig_vs_reference_rscc.png in the same despot_run_name folder.
    """
    datasets = read_datasets(args.datasets_file)
    all_ref, all_pipeline, all_rows = [], [], []
    all_unmatched_ref_rows, all_excess_pipeline_rows = [], []
    total_unmatched, total_excess = 0, 0

    for dataset in datasets:
        run_dir = run_dir_for_dataset(dataset)
        alive_rows = alive_rows_for_dataset(dataset) if alive_rows_for_dataset else None
        (ref_vals, pipeline_vals, n_unmatched, n_excess, matched_rows,
         unmatched_ref_rows, excess_pipeline_rows) = _dataset_despot_vs_ref(
            dataset, args, run_dir, alive_rows=alive_rows
        )
        all_ref.extend(ref_vals)
        all_pipeline.extend(pipeline_vals)
        total_unmatched += n_unmatched
        total_excess += n_excess
        for row in matched_rows:
            row_out = dict(row)
            row_out['dataset'] = dataset
            all_rows.append(row_out)
        for row in unmatched_ref_rows:
            row_out = dict(row)
            row_out['dataset'] = dataset
            all_unmatched_ref_rows.append(row_out)
        for row in excess_pipeline_rows:
            row_out = dict(row)
            row_out['dataset'] = dataset
            all_excess_pipeline_rows.append(row_out)
        print(f'  {dataset}: {len(ref_vals)} matched ligand(s), {n_unmatched} unmatched '
              f'reference ligand(s), {n_excess} excess pipeline ligand(s)')

    graphs_dir = Path(args.graphs_dir)
    graphs_dir.mkdir(parents=True, exist_ok=True)

    out_stem = Path(out_name).stem
    if all_unmatched_ref_rows:
        unmatched_df = pd.DataFrame(all_unmatched_ref_rows)[
            ['dataset', 'ref_chain', 'ref_resi', 'ref_altloc', 'ref_despot_normalized', 'reason']
        ]
        write_plot_csv(graphs_dir, f'{out_stem}_unmatched_ref.png', unmatched_df)
    if all_excess_pipeline_rows:
        excess_df = pd.DataFrame(all_excess_pipeline_rows)[
            ['dataset', 'pipeline_chain', 'cluster_rep_index', 'placer_file', 'index']
        ]
        write_plot_csv(graphs_dir, f'{out_stem}_excess_pipeline.png', excess_df)

    if not all_ref:
        print(f'  No matched ligand(s) found for any dataset; skipping {out_name}.')
        return

    plot_rscc_scatter(
        all_ref, all_pipeline,
        xlabel='Reference DESPOT Score (normalized)',
        ylabel='Pipeline DESPOT Score (normalized)',
        title=title,
        out_path=graphs_dir / out_name,
        extra_text=(f'Unmatched reference LIGs: {total_unmatched}\n'
                    f'Excess pipeline LIGs: {total_excess}'),
        axis_range=_auto_axis_range(all_ref, all_pipeline),
    )

    rows_df = pd.DataFrame(all_rows)[
        ['dataset', 'ref_chain', 'ref_resi', 'ref_altloc', 'ref_despot_normalized',
         'pipeline_chain', 'cluster_rep_index', 'pipeline_despot_normalized']
    ]
    write_plot_csv(graphs_dir, out_name, rows_df)


def _dataset_rscc_despot_tradeoff(dataset, args, run_dir, alive_rows, cluster_csv_override):
    """Joins _dataset_lig_vs_ref's (with cluster_csv_override/rscc_column='despot_rscc' - the
    despot_filter.py-reselected winner's real, individually-computed RSCC, not filter2's stale
    original-representative value) and _dataset_despot_vs_ref's matched pairs for one dataset,
    by (ref_chain, ref_resi, ref_altloc) - both already use the identical centroid-distance
    matching against the identical cluster_rep_models.pdb/alive_rows, so this reuses their work
    rather than re-implementing a third matching pass. A reference ligand present on only one
    side (e.g. has a DESPOT score but no RSCC) is simply not joined.

    Returns one dict per joined pair: ref_chain, ref_resi, ref_altloc, cluster_rep_index,
    pipeline_chain, ref_rscc, pipeline_rscc, ref_despot_normalized, pipeline_despot_normalized,
    rscc_delta (= pipeline_rscc - ref_rscc), despot_delta (= ref_despot_normalized -
    pipeline_despot_normalized)."""
    _, _, _, _, rscc_rows, _, _ = _dataset_lig_vs_ref(
        dataset, args, run_dir, alive_rows=alive_rows,
        cluster_csv_override=cluster_csv_override, rscc_column='despot_rscc',
    )
    _, _, _, _, despot_rows, _, _ = _dataset_despot_vs_ref(dataset, args, run_dir, alive_rows=alive_rows)

    rscc_by_key = {(r['ref_chain'], r['ref_resi'], r['ref_altloc']): r for r in rscc_rows}
    joined = []
    for d in despot_rows:
        r = rscc_by_key.get((d['ref_chain'], d['ref_resi'], d['ref_altloc']))
        if r is None:
            continue
        joined.append({
            'ref_chain': d['ref_chain'], 'ref_resi': d['ref_resi'], 'ref_altloc': d['ref_altloc'],
            'cluster_rep_index': d['cluster_rep_index'], 'pipeline_chain': d['pipeline_chain'],
            'ref_rscc': r['ref_rscc'], 'pipeline_rscc': r['pipeline_rscc'],
            'ref_despot_normalized': d['ref_despot_normalized'],
            'pipeline_despot_normalized': d['pipeline_despot_normalized'],
            'rscc_delta': r['pipeline_rscc'] - r['ref_rscc'],
            'despot_delta': d['ref_despot_normalized'] - d['pipeline_despot_normalized'],
        })
    return joined


def plot_rscc_despot_tradeoff(args, run_dir_for_dataset, title, out_name, alive_rows_for_dataset,
                               cluster_csv_override_for_dataset):
    """Pooled (across every dataset in datasets.txt) scatter of despot_filter.py's RSCC/DESPOT
    reselection tradeoff, restricted to despot_filter survivors (same alive_rows_for_dataset as
    plot_lig_vs_ref_despot.py/plot_despot_vs_ref.py, so the matched pairs agree with those two
    plots): y = pipeline RSCC - reference RSCC, x = reference DESPOT - pipeline DESPOT (both
    normalized) - see _dataset_rscc_despot_tradeoff. Written to <graphs_dir>/<out_name>, with the
    joined per-ligand data alongside via write_plot_csv.

    run_dir_for_dataset(dataset)/alive_rows_for_dataset(dataset)/cluster_csv_override_for_dataset
    (dataset): same meaning as the equivalent arguments to plot_lig_vs_ref/plot_despot_vs_ref -
    pass the exact same functions plot_lig_vs_ref_despot.py uses, for consistency.
    """
    datasets = read_datasets(args.datasets_file)
    all_rows = []

    for dataset in datasets:
        run_dir = run_dir_for_dataset(dataset)
        alive_rows = alive_rows_for_dataset(dataset)
        cluster_csv_override = cluster_csv_override_for_dataset(dataset)
        joined = _dataset_rscc_despot_tradeoff(
            dataset, args, run_dir, alive_rows, cluster_csv_override)
        for row in joined:
            row_out = dict(row)
            row_out['dataset'] = dataset
            all_rows.append(row_out)
        print(f'  {dataset}: {len(joined)} matched ligand(s) with both RSCC and DESPOT deltas')

    graphs_dir = Path(args.graphs_dir)
    graphs_dir.mkdir(parents=True, exist_ok=True)

    if not all_rows:
        print(f'  No matched ligand(s) found for any dataset; skipping {out_name}.')
        return

    xs = [r['despot_delta'] for r in all_rows]
    ys = [r['rscc_delta'] for r in all_rows]
    median_x, mean_x = float(np.median(xs)), float(np.mean(xs))
    median_y, mean_y = float(np.median(ys)), float(np.mean(ys))

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.axhline(0, color='gray', linewidth=0.8, zorder=0)
    ax.axvline(0, color='gray', linewidth=0.8, zorder=0)
    ax.axvline(median_x, color='tab:orange', linestyle='--', linewidth=1.2, zorder=1,
               label=f'median Δdespot = {median_x:.3f}')
    ax.axvline(mean_x, color='tab:green', linestyle=':', linewidth=1.2, zorder=1,
               label=f'mean Δdespot = {mean_x:.3f}')
    ax.axhline(median_y, color='tab:orange', linestyle='--', linewidth=1.2, zorder=1,
               label=f'median Δrscc = {median_y:.3f}')
    ax.axhline(mean_y, color='tab:green', linestyle=':', linewidth=1.2, zorder=1,
               label=f'mean Δrscc = {mean_y:.3f}')
    ax.scatter(xs, ys, s=18, alpha=0.7, edgecolors='none', color='steelblue', zorder=2)
    ax.set_xlabel('Reference - Pipeline normalized DESPOT score\n'
                  '(positive = pipeline pose is more favorable)')
    ax.set_ylabel('Pipeline - Reference RSCC\n(positive = pipeline pose fits the density better)')
    ax.set_title(f'{title} (n={len(all_rows)})')
    ax.legend(loc='best', fontsize=7)
    fig.tight_layout()
    out_path = graphs_dir / out_name
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f'  Scatterplot saved to: {out_path}')

    rows_df = pd.DataFrame(all_rows)[
        ['dataset', 'ref_chain', 'ref_resi', 'ref_altloc', 'pipeline_chain', 'cluster_rep_index',
         'ref_rscc', 'pipeline_rscc', 'rscc_delta', 'ref_despot_normalized',
         'pipeline_despot_normalized', 'despot_delta']
    ]
    write_plot_csv(graphs_dir, out_name, rows_df)


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


def cluster_rep_rscc_values(csv_path):
    """Returns a DataFrame with columns ['cluster_rep_index', 'rscc'] - one
    row per data row of csv_path's cluster_reps.csv, with cluster_rep_index
    set to that row's 1-based position in the file (dropped rows, e.g. a
    missing rscc, keep the position numbering of the rows that remain)."""
    empty = pd.DataFrame(columns=['cluster_rep_index', 'rscc'])
    if not csv_path.exists():
        print(f'  Warning: cluster_reps.csv not found: {csv_path}')
        return empty
    df = pd.read_csv(csv_path)
    if 'rscc' not in df.columns:
        print(f'  Warning: no rscc column in {csv_path}')
        return empty
    df = df.reset_index(drop=True)
    df['cluster_rep_index'] = df.index + 1
    return df[['cluster_rep_index', 'rscc']].dropna(subset=['rscc'])


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


# Sentinel default for plot_rscc_scatter's axis_range / plot_rscc_histogram's
# value_range: means "not explicitly given" (distinct from a caller passing
# value_range=None, which has its own separate meaning on the histogram
# side - see plot_rscc_histogram). RSCC is a correlation coefficient bounded
# to [-1, 1], but real values are seldom that negative, so a fixed [0, 1] or
# [-1, 1] axis either clips negative RSCCs or wastes a lot of space; when
# left at this default, both functions instead compute the range from the
# data itself: plot_rscc_scatter uses (min(x, y) - 0.1, 1) (a small pad
# below the lowest plotted point, not the whole [-1, 1] worth of headroom -
# a flat "- 1" swallowed most of the plot in empty space below any real
# data); plot_rscc_histogram still uses (min(values) - 1, 1).
_RSCC_RANGE_DEFAULT = object()


def _point_density(x, y):
    """Gaussian-KDE point density for each (x[i], y[i]), for a
    density-colored scatter. Returns None (rather than raising) whenever
    density estimation isn't meaningful/possible: fewer than 3 points, or
    degenerate data (e.g. every point identical) that makes the KDE's
    covariance matrix singular."""
    if len(x) < 3:
        return None
    try:
        xy = np.vstack([x, y])
        return gaussian_kde(xy)(xy)
    except (np.linalg.LinAlgError, ValueError):
        return None


def plot_rscc_scatter(x, y, xlabel, ylabel, title, out_path, extra_text=None,
                       axis_range=_RSCC_RANGE_DEFAULT, color_by_density=False):
    """Scatter plot in the style of qfit's compare_lig_rscc._plot_scatter:
    unity dashed line, fixed square axes, equal aspect, mean/median stats box.
    extra_text, if given, is appended to the stats box (e.g. a lost-ligand
    count) below the mean/median lines.

    axis_range: (min, max) applied to both axes and the unity line, so the
    two axes always share one scale and stay directly visually comparable.
    Left at its default, computed from the data as (min(x, y) - 0.1, 1) -
    see _RSCC_RANGE_DEFAULT. Pass an explicit (min, max) tuple (e.g.
    _auto_axis_range(x, y)) for unbounded metrics like Z-scores.

    color_by_density: colors each point by its local (gaussian-KDE) point
    density instead of a flat color, with a colorbar - meant for pooled
    (cross-dataset) plots, where far more overplotting makes a flat color
    much less informative than in any single dataset's plot. Falls back to
    the flat color if density estimation isn't possible (see
    _point_density)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if len(x) == 0:
        print(f'  Skipping {out_path.name}: no data points.')
        return

    if axis_range is _RSCC_RANGE_DEFAULT:
        axis_range = (float(np.concatenate([x, y]).min()) - 0.1, 1)

    fig, ax = plt.subplots(figsize=(6, 6))
    density = _point_density(x, y) if color_by_density else None
    if density is not None:
        order = np.argsort(density)  # draw densest points last, on top
        sc = ax.scatter(x[order], y[order], c=density[order], cmap='viridis', s=8, edgecolor='none')
        fig.colorbar(sc, ax=ax, label='Point density', shrink=0.8)
    else:
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


def plot_rscc_histogram(values, title, xlabel, out_path, color='steelblue',
                         value_range=_RSCC_RANGE_DEFAULT):
    """Histogram in the style of calc_filter_rmsd.py's save_histogram, binned
    over value_range. Left at its default, computed from the data as
    (min(values) - 1, 1) - see _RSCC_RANGE_DEFAULT. Pass value_range=None to
    instead auto-compute a symmetrically padded range from the data - for
    unbounded metrics like Z-scores that don't have RSCC's natural [0, 1]
    range. Pass an explicit (min, max) tuple to fully override (e.g. a fixed
    [-1.1, 1.1] for a correlation-coefficient-like metric)."""
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if len(values) == 0:
        print(f'  Skipping {out_path.name}: no data points.')
        return

    if value_range is _RSCC_RANGE_DEFAULT:
        value_range = (float(values.min()) - 1, 1)
    elif value_range is None:
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


def _collect_dataset_rscc(datasets_dir, dataset, args):
    """Builds this dataset's residue-level apo/backbone/final RSCC rows,
    restricted to protein residues (everything but the identified LIG
    residue - a ligand-vs-apo RSCC comparison never has data, since the apo
    structure has no ligand; see aggregate_lig_rscc.py for the ligand
    comparison that does make sense). Only reads csvs already written by
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
        if is_lig:
            continue
        rows.append({
            'residue': base,
            'apo': apo_vals.get(base),
            'backbone': backbone_vals.get(base),
            'final': final_vals.get(base),
            'has_conformer': base in conformer_residues,
        })

    return pd.DataFrame(rows, columns=['residue', 'apo', 'backbone', 'final', 'has_conformer'])


def run_rscc_aggregator(args):
    """For every dataset independently (no pooling across datasets), builds
    and saves protein-residue scatter plots (backbone-vs-apo, final-vs-apo,
    final-vs-backbone), restricted to that dataset's
    residues_with_placer_conformers.csv, into that dataset's own
    .../<final_run_name>/graphs/ folder, plus a csv of each plot's
    underlying data (residue label + both RSCC columns) saved alongside it
    in that same folder. The unrestricted (all-residues) variant of these
    plots was dropped - it carried no information beyond the
    placer-conformer-restricted one and only added noise.
    """
    datasets = read_datasets(args.datasets_file)
    label = 'Protein'

    comparisons = [
        ('apo', 'backbone', f'{label} RSCC: Backbone-Refined vs Apo', 'backbone_vs_apo'),
        ('apo', 'final', f'{label} RSCC: Final-Refined vs Apo', 'final_vs_apo'),
        ('backbone', 'final', f'{label} RSCC: Final-Refined vs Backbone-Refined', 'final_vs_backbone'),
    ]

    for dataset in datasets:
        df = _collect_dataset_rscc(args.datasets_dir, dataset, args)
        if df.empty:
            print(f'  {dataset}: no protein RSCC data found; skipping plots for this dataset.')
            continue

        graphs_dir = dataset_graphs_dir(args.datasets_dir, dataset, args)

        conformer_df = df[df['has_conformer']]
        for xcol, ycol, title, tag in comparisons:
            paired = conformer_df.dropna(subset=[xcol, ycol])
            out_name = f'protein_{tag}_rscc_placer_conformers.png'
            if paired.empty:
                print(f'  {dataset}: no data points for {out_name}; skipping.')
                continue
            plot_rscc_scatter(
                paired[xcol], paired[ycol],
                xlabel=f'{xcol.capitalize()} RSCC', ylabel=f'{ycol.capitalize()} RSCC',
                title=f'{title} ({dataset}) (placer-conformer residues)',
                out_path=graphs_dir / out_name,
            )
            write_plot_csv(
                graphs_dir, out_name,
                paired[['residue', xcol, ycol]].rename(
                    columns={xcol: f'{xcol}_rscc', ycol: f'{ycol}_rscc'}
                ),
            )


def run_rscc_aggregator_pooled(args):
    """Pooled (across every dataset in datasets.txt) counterpart of
    run_rscc_aggregator: the same per-residue protein apo/backbone/final RSCC
    comparisons (restricted to residues_with_placer_conformers.csv),
    combined into a single scatter plot per comparison instead of one per
    dataset, colored by point density (plot_rscc_scatter's
    color_by_density) since pooling every dataset's residues creates far
    more overplotting than any single dataset's plot has. Plots go into
    args.graphs_dir, with a matching csv of each plot's underlying data
    (now including a 'dataset' column, since rows are pooled from many)
    saved alongside it in that same folder.
    """
    datasets = read_datasets(args.datasets_file)
    label = 'Protein'

    comparisons = [
        ('apo', 'backbone', f'{label} RSCC: Backbone-Refined vs Apo (pooled)', 'backbone_vs_apo'),
        ('apo', 'final', f'{label} RSCC: Final-Refined vs Apo (pooled)', 'final_vs_apo'),
        ('backbone', 'final', f'{label} RSCC: Final-Refined vs Backbone-Refined (pooled)',
         'final_vs_backbone'),
    ]

    pooled_rows = []
    for dataset in datasets:
        df = _collect_dataset_rscc(args.datasets_dir, dataset, args)
        if df.empty:
            print(f'  {dataset}: no protein RSCC data found; skipping.')
            continue
        df = df[df['has_conformer']].copy()
        df['dataset'] = dataset
        pooled_rows.append(df)

    if not pooled_rows:
        print('  No protein RSCC data found for any dataset; skipping pooled plots.')
        return
    pooled_df = pd.concat(pooled_rows, ignore_index=True)

    graphs_dir = Path(args.graphs_dir)
    graphs_dir.mkdir(parents=True, exist_ok=True)

    for xcol, ycol, title, tag in comparisons:
        paired = pooled_df.dropna(subset=[xcol, ycol])
        out_name = f'protein_{tag}_rscc_placer_conformers_pooled.png'
        if paired.empty:
            print(f'  No data points for {out_name}; skipping.')
            continue
        plot_rscc_scatter(
            paired[xcol], paired[ycol],
            xlabel=f'{xcol.capitalize()} RSCC', ylabel=f'{ycol.capitalize()} RSCC',
            title=f'{title} (placer-conformer residues)',
            out_path=graphs_dir / out_name,
            color_by_density=True,
        )
        write_plot_csv(
            graphs_dir, out_name,
            paired[['dataset', 'residue', xcol, ycol]].rename(
                columns={xcol: f'{xcol}_rscc', ycol: f'{ycol}_rscc'}
            ),
        )


CLASH_GROUPS_CSV_COLUMNS = ['dataset', 'residues', 'original_residues', 'size', 'original_size',
                            'original_mse', 'final_mse', 'hit_cap', 'unresolved']


def run_clash_groups_aggregator(args):
    """Pooled (cross-dataset) aggregation of the sidechain_clash_groups.csv
    files build_final_model already writes into every dataset's own
    .../<final_run_name>/ directory (one row per resolved sidechain-sidechain
    clash group - see build_final_model.py's _write_clash_groups_csv). No new
    clash detection or resolution happens here - this just concatenates what
    build_final already wrote, with a 'dataset' column prepended so a
    run-wide row still identifies which dataset it came from. A dataset with
    no sidechain_clash_groups.csv (build_final_model never ran for it, e.g.
    filter2 rejected every candidate) or an empty one (no clash groups found)
    contributes no rows. Written to
    args.graphs_dir/sidechain_clash_groups_combined.csv.
    """
    datasets = read_datasets(args.datasets_file)

    pooled_rows = []
    for dataset in datasets:
        csv_path = dataset_final_dir(args.datasets_dir, dataset, args) / 'sidechain_clash_groups.csv'
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path)
        if df.empty:
            continue
        df.insert(0, 'dataset', dataset)
        pooled_rows.append(df)

    graphs_dir = Path(args.graphs_dir)
    graphs_dir.mkdir(parents=True, exist_ok=True)
    out_path = graphs_dir / 'sidechain_clash_groups_combined.csv'

    if not pooled_rows:
        print('  No sidechain_clash_groups.csv data found for any dataset; writing empty combined csv.')
        pd.DataFrame(columns=CLASH_GROUPS_CSV_COLUMNS).to_csv(out_path, index=False)
        return

    pooled_df = pd.concat(pooled_rows, ignore_index=True)
    pooled_df.to_csv(out_path, index=False)
    print(f'  {len(pooled_df)} clash-group row(s) from {len(pooled_rows)} dataset(s) written to: {out_path}')


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
    of each plot's underlying data saved alongside it in that same folder.

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
                graphs_dir, out_name,
                paired[['residue', xcol, ycol]],
            )


def run_z_aggregator_pooled(args):
    """Pooled (across every dataset in datasets.txt) counterpart of
    run_z_aggregator: the same final-vs-apo Z-map statistic comparisons
    (max/min/average, restricted to residues_with_placer_conformers.csv),
    combined into a single scatter plot per statistic instead of one per
    dataset, colored by point density (see run_rscc_aggregator_pooled).
    Plots go into args.graphs_dir, with a matching csv (now including a
    'dataset' column) saved alongside it in that same folder.
    """
    datasets = read_datasets(args.datasets_file)

    comparisons = [
        ('apo_max_z', 'final_max_z', 'Max Z-score: Final-Refined vs Apo (pooled)', 'max_z'),
        ('apo_min_z', 'final_min_z', 'Min Z-score: Final-Refined vs Apo (pooled)', 'min_z'),
        ('apo_average_z', 'final_average_z', 'Average Z-score: Final-Refined vs Apo (pooled)', 'average_z'),
    ]

    pooled_rows = []
    for dataset in datasets:
        df = _dataset_final_vs_apo_z(dataset, args)
        if df.empty:
            print(f'  {dataset}: no Z-map data found; skipping.')
            continue
        df = df.copy()
        df['dataset'] = dataset
        pooled_rows.append(df)

    if not pooled_rows:
        print('  No Z-map data found for any dataset; skipping pooled plots.')
        return
    pooled_df = pd.concat(pooled_rows, ignore_index=True)

    graphs_dir = Path(args.graphs_dir)
    graphs_dir.mkdir(parents=True, exist_ok=True)

    for xcol, ycol, title, tag in comparisons:
        paired = pooled_df.dropna(subset=[xcol, ycol])
        out_name = f'final_vs_apo_{tag}_placer_conformers_pooled.png'
        if paired.empty:
            print(f'  No data points for {out_name}; skipping.')
            continue
        plot_rscc_scatter(
            paired[xcol], paired[ycol],
            xlabel=f'Apo {tag.replace("_", " ").title()}',
            ylabel=f'Final-Refined {tag.replace("_", " ").title()}',
            title=f'{title} (placer-conformer residues)',
            out_path=graphs_dir / out_name,
            axis_range=_auto_axis_range(paired[xcol], paired[ycol]),
            color_by_density=True,
        )
        write_plot_csv(graphs_dir, out_name, paired[['dataset', 'residue', xcol, ycol]])


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
    that plot's value) saved alongside it in that same folder.
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
                graphs_dir, out_name,
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
    underlying data (same basename, .csv instead of .png) saved alongside
    it in that same folder:

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
            graphs_dir, 'bfactor_sensitivity_lines.png',
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
            graphs_dir, 'bfactor_sensitivity_lines_normalized.png',
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
        write_plot_csv(graphs_dir, 'bfactor_sensitivity_spearman_rho_hist.png', rho_df)


def run_bfactor_rho_pooled(args):
    """Pooled (across every dataset in datasets.txt) counterpart of
    run_bfactor_sensitivity_plots' spearman-rho histogram only - not the
    two RSCC-vs-bfactor line plots, which don't pool meaningfully since
    each line is already one residue's full bfactor sweep (drawing
    ~200-per-dataset lines from every dataset on one plot would just be
    noise, unlike a rho value which condenses each residue down to a single
    number). Combines every dataset's canonical per-residue spearmans_rho
    (see _dataset_final_rscc_b_lines) into one histogram, with a matching
    csv (now including a 'dataset' column) saved alongside it in args.graphs_dir.
    """
    datasets = read_datasets(args.datasets_file)

    rows = []
    for dataset in datasets:
        lines = _dataset_final_rscc_b_lines(dataset, args)
        if not lines:
            print(f'  {dataset}: no bfactor-sweep data found; skipping.')
            continue
        for _, _, rho, residue in lines:
            if rho is not None:
                rows.append({'dataset': dataset, 'residue': residue, 'spearmans_rho': rho})

    if not rows:
        print('  No bfactor-sweep data found for any dataset; skipping pooled plot.')
        return
    rho_df = pd.DataFrame(rows, columns=['dataset', 'residue', 'spearmans_rho'])

    graphs_dir = Path(args.graphs_dir)
    graphs_dir.mkdir(parents=True, exist_ok=True)

    out_name = 'bfactor_sensitivity_spearman_rho_hist_pooled.png'
    plot_rscc_histogram(
        rho_df['spearmans_rho'],
        title='Spearman ρ of RSCC vs B-factor (canonical event map per residue) (pooled)',
        xlabel='Spearman ρ',
        out_path=graphs_dir / out_name,
        value_range=(-1.1, 1.1),
    )
    write_plot_csv(graphs_dir, out_name, rho_df)


def run_cluster_reps_pooled(args):
    """Pooled (across every dataset in datasets.txt) counterpart of
    plot_cluster_reps_rscc.py's cluster_reps_1/cluster_reps_2 histograms:
    combines every dataset's cluster_reps.csv 'rscc' column (see
    cluster_rep_rscc_values) into one histogram per stage
    (filter_run_name / filter2_run_name) instead of one per dataset. Not a
    scatter plot, so no density coloring applies here - a histogram's bar
    heights already are the density. Plots go into args.graphs_dir, with a
    matching csv (now including a 'dataset' column) saved alongside it in
    that same folder.
    """
    datasets = read_datasets(args.datasets_file)

    stage1_rows, stage2_rows = [], []
    for dataset in datasets:
        dataset_dir = Path(args.datasets_dir) / dataset

        csv1 = (dataset_dir / args.run_name / args.placer_run_name /
                args.filter_run_name / 'cluster_reps.csv')
        values1_df = cluster_rep_rscc_values(csv1)
        if not values1_df.empty:
            values1_df = values1_df.copy()
            values1_df['dataset'] = dataset
            stage1_rows.append(values1_df)

        csv2 = (dataset_dir / args.run_name / args.placer_run_name / args.filter_run_name /
                args.placer2_run_name / args.filter2_run_name / 'cluster_reps.csv')
        values2_df = cluster_rep_rscc_values(csv2)
        if not values2_df.empty:
            values2_df = values2_df.copy()
            values2_df['dataset'] = dataset
            stage2_rows.append(values2_df)

    graphs_dir = Path(args.graphs_dir)
    graphs_dir.mkdir(parents=True, exist_ok=True)

    for rows, out_name, stage_label in [
        (stage1_rows, 'cluster_reps_1_pooled.png', args.filter_run_name),
        (stage2_rows, 'cluster_reps_2_pooled.png', args.filter2_run_name),
    ]:
        if not rows:
            print(f'  No data found for {out_name}; skipping.')
            continue
        pooled_df = pd.concat(rows, ignore_index=True)
        plot_rscc_histogram(
            pooled_df['rscc'],
            title=f'Cluster-Rep RSCC ({stage_label}) (pooled)',
            xlabel='RSCC',
            out_path=graphs_dir / out_name,
        )
        write_plot_csv(graphs_dir, out_name, pooled_df[['dataset', 'cluster_rep_index', 'rscc']])


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
    filter_1_rscc, filter_2_rscc) rows to a csv alongside the plot itself in
    that dataset's .../<final_run_name>/graphs/ folder, matching the plot's
    filename. filter_1_cluster_rep_index is the round-1
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
        write_plot_csv(
            graphs_dir, out_name,
            pd.DataFrame({
                'filter_1_cluster_rep_index': matched_filter1_idx,
                'filter_2_cluster_rep_index': matched_filter2_idx,
                'filter_1_rscc': matched_x,
                'filter_2_rscc': matched_y,
            }),
        )
