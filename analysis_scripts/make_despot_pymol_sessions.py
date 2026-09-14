#!/usr/bin/env python3
"""
Per-dataset PyMOL session (.pse) for visually QC-ing one despot_run_name's despot_filtered.pdb
against the reference structure: loads despot_filtered.pdb (rotamer_run_name/despot_run_name's
output), the matched reference structure, and that dataset's first event map (lowest-numbered
-event_N_... file), then shows sticks and carves an isomesh of the event map around every outlier
residue's conformation (the same dataset:residue rows Stage 8g's
rotamer_refined_despot_vs_reference_rscc_outliers.csv lists for that despot_run_name - i.e. every
modified_residues.csv residue whose RSCC is >= 0.1 worse than the reference's). One isomesh per
structure (mesh_pipeline/mesh_reference), each carved using only that structure's own atoms -
carving around a selection spanning both objects breaks down (empty/garbage mesh, or PyMOL
picking coordinates from the wrong object), since the same chain+resi selects two spatially
distinct copies of that residue at once - see pymol_sessions/make_pymol_sessions.py, which uses
the same one-mesh-per-object pattern.

Colored by util.cbag (green carbons, pipeline) / util.cbac (cyan carbons, reference); reference's
HOH/DMS hidden and its cartoon set to 50% transparent so it doesn't obscure the pipeline
structure. Also applies the same style/ray-tracing SETTINGS_BLOCK, cartoon loop representation,
and mesh coloring (density_blue) that pymol_sessions/make_pymol_sessions.py uses, for visual
consistency across this project's pymol sessions.

Must be run with a PyMOL-enabled Python (pandas/qfit are NOT available/needed here - this script
is intentionally dependency-free besides pymol and the standard library), e.g.:
  /home/ngupta/miniconda3/envs/pymol/bin/python3 make_despot_pymol_sessions.py \\
      <run_name> <placer_run_name> <filter_run_name> <placer2_run_name> <filter2_run_name> \\
      <final_run_name> <rotamer_run_name> <despot_run_name> \\
      --datasets-dir <dir> --datasets-file <file> --ref-set <dir> --graphs-dir <dir>

--graphs-dir must point at the Stage 8g output directory for this exact despot_run_name (i.e.
GRAPHS_DIR/<run>/.../<final_run_name>/<rotamer_run_name>/<despot_run_name>/), the same directory
plot_residues_vs_ref_despot.py wrote rotamer_refined_despot_vs_reference_rscc_outliers.csv into.

Writes pymol_sessions/<rotamer_run_name>/<dataset>.pse for each dataset (skipping/warning on any
dataset missing despot_filtered.pdb, its reference structure, an event map, or any outlier rows) -
--pymol-sessions-dir overrides the default (<datasets_dir>/../pymol_sessions).
"""
import argparse
import csv
import sys
from pathlib import Path

import pymol
pymol.finish_launching(['pymol', '-qc'])
from pymol import cmd, util

# Å radius the event-map isomesh is carved down to around the outlier residues' atoms (in
# either structure) - not specified by the user, chosen as a typical local-density carve radius.
CARVE_RADIUS = 1.6
# Event map contour level (sigma) the isomesh is computed at - not specified by the user, 1.0 is
# a conventional default for viewing PanDDA event map density.
MAP_LEVEL = 1.0

# Same style/ray-tracing settings block as pymol_sessions/make_pymol_sessions.py, applied
# verbatim (one `cmd.do(line)` per line) for visual consistency across sessions.
SETTINGS_BLOCK = """
bg_color white
space cmyk
set orthoscopic, on
set valence, off
set cartoon_side_chain_helper, on
set cartoon_fancy_helices, on

set ray_trace_mode, 0
set ray_shadow, off
set light_count, 8
set ambient, 0.3
set reflect, 0.4
set direct, 0.8
set specular, 0
set ambient_occlusion_mode, 1
set ambient_occlusion_smooth, 10
set ambient_occlusion_scale, 15

set cartoon_rect_length, 1.0
set cartoon_oval_length, 1.0
set stick_radius, 0.2
set solvent_radius, 1.6
set sphere_scale, 0.15
set dash_gap, 0.25
set dash_color, black
set mesh_width, 0.5

set_color density_blue, [0.4, 0.6, 0.8]

set_color teal, [0, 0.5, 0.5]
set_color gold, [1.0, 0.843, 0.0]
set_color plum, [0.568, 0.239, 0.527]
set_color cool_blue, [0.3412, 0.4587, 0.8833]
set_color cool_red, [0.8654, 0.4671, 0.3216]
set_color cool_green, [0.4569, 0.6412, 0.5725]
"""


def build_argparser():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('run_name')
    p.add_argument('placer_run_name')
    p.add_argument('filter_run_name')
    p.add_argument('placer2_run_name')
    p.add_argument('filter2_run_name')
    p.add_argument('final_run_name')
    p.add_argument('rotamer_run_name')
    p.add_argument('despot_run_name')
    p.add_argument('--datasets-dir', required=True)
    p.add_argument('--datasets-file', required=True)
    p.add_argument('--ref-set', required=True)
    p.add_argument('--ref-pdb-pattern', default='{dataset}-pandda-model.pdb')
    p.add_argument('--graphs-dir', required=True,
                    help='Stage 8g graphs dir for this despot_run_name, containing '
                         'rotamer_refined_despot_vs_reference_rscc_outliers.csv')
    p.add_argument('--pymol-sessions-dir', default=None,
                    help='Defaults to <datasets_dir>/../pymol_sessions (i.e. '
                         'program_rotamer/pymol_sessions, the normal layout). Sessions are '
                         'written to <this>/<rotamer_run_name>/<dataset>.pse.')
    return p


def read_datasets(datasets_file):
    with open(datasets_file) as f:
        return [line.strip() for line in f if line.strip()]


def read_outliers_by_dataset(outliers_csv):
    by_dataset = {}
    with open(outliers_csv) as f:
        for row in csv.DictReader(f):
            by_dataset.setdefault(row['dataset'], []).append(row['residue'])
    return by_dataset


def residue_selection(object_name, labels):
    """('{object_name} and chain X and resi N') or ... for every "{chain}{resi}" label."""
    parts = [f'({object_name} and chain {label[0]} and resi {label[1:]})' for label in labels]
    return ' or '.join(parts)


def main():
    args = build_argparser().parse_args()
    datasets = read_datasets(args.datasets_file)

    outliers_csv = Path(args.graphs_dir) / 'rotamer_refined_despot_vs_reference_rscc_outliers.csv'
    if not outliers_csv.is_file():
        sys.exit(f'Error: outliers csv not found: {outliers_csv}')
    outliers_by_dataset = read_outliers_by_dataset(outliers_csv)

    pymol_sessions_dir = (Path(args.pymol_sessions_dir) if args.pymol_sessions_dir
                           else Path(args.datasets_dir).parent / 'pymol_sessions')
    out_dir = pymol_sessions_dir / args.rotamer_run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    n_written = 0
    for dataset in datasets:
        final_dir = (Path(args.datasets_dir) / dataset / args.run_name / args.placer_run_name /
                     args.filter_run_name / args.placer2_run_name / args.filter2_run_name /
                     args.final_run_name)
        despot_dir = final_dir / args.rotamer_run_name / args.despot_run_name
        pipeline_pdb = despot_dir / 'despot_filtered.pdb'
        ref_pdb = Path(args.ref_set) / dataset / args.ref_pdb_pattern.format(dataset=dataset)
        dataset_dir = Path(args.datasets_dir) / dataset
        event_maps = sorted(dataset_dir.glob(f'{dataset}-event_*.ccp4'))

        if not pipeline_pdb.is_file():
            print(f'Skipping {dataset}: {pipeline_pdb} not found.')
            continue
        if not ref_pdb.is_file():
            print(f'Skipping {dataset}: reference structure not found: {ref_pdb}')
            continue
        if not event_maps:
            print(f'Skipping {dataset}: no event map matching {dataset_dir}/{dataset}-event_*.ccp4')
            continue
        first_event_map = event_maps[0]

        labels = outliers_by_dataset.get(dataset, [])
        if not labels:
            print(f'Skipping {dataset}: no outlier residues for it in {outliers_csv}')
            continue

        cmd.reinitialize()
        cmd.load(str(pipeline_pdb), 'pipeline')
        cmd.load(str(ref_pdb), 'reference')
        cmd.load(str(first_event_map), 'event_map')

        util.cbag('pipeline')
        util.cbac('reference')

        for line in SETTINGS_BLOCK.strip().splitlines():
            line = line.strip()
            if line:
                cmd.do(line)
        cmd.do('cartoon loop')

        cmd.hide('everything', 'reference and resn HOH')
        cmd.hide('everything', 'reference and resn DMS')
        cmd.set('cartoon_transparency', 0.5, 'reference')

        pipeline_sele = residue_selection('pipeline', labels)
        reference_sele = residue_selection('reference', labels)

        cmd.select('_sticks_sel', f'({pipeline_sele}) or ({reference_sele})')
        cmd.show('sticks', '_sticks_sel')
        cmd.delete('_sticks_sel')

        # One mesh per object, each carved using only that object's own atoms - carving around a
        # selection spanning both objects breaks down (empty/garbage mesh, or PyMOL picking
        # coordinates from the wrong object), since the same chain+resi selects two spatially
        # distinct copies of that residue at once (see pymol_sessions/make_pymol_sessions.py).
        #
        # mesh_color must be set as a GLOBAL setting before isomesh runs - cmd.color(), and even
        # a per-object `set mesh_color, ..., mesh_pipeline`, are silently ignored for a mesh once
        # its source map is deleted (isomesh bakes it down to static CGO geometry at that point,
        # using whatever the *global* mesh_color setting was at creation time - confirmed
        # empirically; every other ordering/API left the mesh white).
        cmd.set('mesh_color', 'density_blue')
        cmd.isomesh('mesh_pipeline', 'event_map', MAP_LEVEL, pipeline_sele, carve=CARVE_RADIUS)
        cmd.isomesh('mesh_reference', 'event_map', MAP_LEVEL, reference_sele, carve=CARVE_RADIUS)
        # Drop the raw map object once both meshes are carved - each mesh is small, derived
        # geometry; keeping the full ccp4 grid around bloats the saved session by tens of MB for
        # no benefit.
        cmd.delete('event_map')

        cmd.orient('pipeline or reference')

        output_pse = out_dir / f'{dataset}.pse'
        cmd.save(str(output_pse))
        print(f'{dataset}: {len(labels)} outlier residue(s), wrote {output_pse}')
        n_written += 1

    print(f'Wrote {n_written}/{len(datasets)} session(s).')


if __name__ == '__main__':
    main()
