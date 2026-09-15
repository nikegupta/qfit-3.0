#!/usr/bin/env python3
"""
Pooled (across every dataset in datasets.txt) per-residue RSCC comparison:
filter_run_name's backbone-refined structures vs the reference set,
matched by residue label (chain+resnum). No RSCC is computed here - both
sides are read from the calc_rscc csvs already on disk (calc_backbone_refined_rscc
and calc_ref_set_rscc). For a residue with multiple backbone_refined cluster-rep
csvs, the highest RSCC across them is used (same as aggregate_protein_rscc.py).

Produces two plots: one over every matched residue, and one restricted to
the residues listed in filter_run_name/refined_residues.csv (the residues
RSR actually refined).

Run at the end of stage 3, only when -c (compare to reference set) is given.

Usage:
  plot_residues_vs_ref_backbone.py <run_name> <placer_run_name> <filter_run_name> \\
      --ref-set <dir> --graphs-dir <dir> [options]
"""
from pathlib import Path

from rscc_common import (
    build_ref_argparser, plot_residues_vs_ref, best_rscc_per_residue, read_residue_conformer_list,
)


def main():
    args = build_ref_argparser(
        __doc__, ['run_name', 'placer_run_name', 'filter_run_name']
    ).parse_args()

    def run_dir(dataset):
        return (Path(args.datasets_dir) / dataset / args.run_name /
                args.placer_run_name / args.filter_run_name)

    def collect_structure_rscc(dataset):
        csvs = sorted(run_dir(dataset).glob(f'{dataset}_backbone_refined_*_rscc.csv'))
        return best_rscc_per_residue(csvs)

    def collect_restrict_labels(dataset):
        return read_residue_conformer_list(run_dir(dataset) / 'refined_residues.csv')

    plot_residues_vs_ref(
        args, collect_structure_rscc, collect_restrict_labels,
        out_dir=args.graphs_dir, out_prefix='backbone_refined', structure_label='Backbone-Refined',
    )


if __name__ == '__main__':
    main()
