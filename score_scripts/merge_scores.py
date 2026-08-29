#!/usr/bin/env python3
"""
Joins calc_rscc.py's rscc.csv (columns model_idx,residue,rscc) with run_gnina_score.sh's
gnina_scores.csv (columns residue,affinity_kcal_mol,cnnscore,cnnaffinity) into one final
scores.csv, one row per residue label - both are keyed by the same residue label convention
('<chain><resi>' or '<chain><resi>-<altloc>' for a split altloc, see calc_rscc.py's
_residue_label / split_complex_pdbqt.py's matching convention), so this is a straight-across join
on 'residue' - no intermediate instance map needed.

A residue label present in one csv but not the other gets an empty value for that csv's columns
rather than being dropped - every label seen in either input is preserved in the output.

Usage:
  merge_scores.py <rscc_csv> <gnina_scores_csv> <output_csv>
"""
import argparse
import csv


def build_argparser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('rscc_csv')
    p.add_argument('gnina_scores_csv')
    p.add_argument('output_csv')
    return p


def read_csv_rows(path):
    with open(path, newline='') as f:
        return list(csv.DictReader(f))


def main():
    args = build_argparser().parse_args()

    rscc_by_label = {row['residue']: row['rscc'] for row in read_csv_rows(args.rscc_csv)}
    gnina_by_label = {row['residue']: row for row in read_csv_rows(args.gnina_scores_csv)}

    labels = sorted(set(rscc_by_label) | set(gnina_by_label))
    fieldnames = ['residue', 'rscc', 'affinity_kcal_mol', 'cnnscore', 'cnnaffinity']

    out_rows = []
    for label in labels:
        rscc = rscc_by_label.get(label, '')
        gnina_row = gnina_by_label.get(label, {})
        if rscc == '':
            print(f'  Warning: no RSCC value found for residue {label} (missing from {args.rscc_csv}).')
        if not gnina_row:
            print(f'  Warning: no gnina score found for residue {label} (missing from {args.gnina_scores_csv}).')
        out_rows.append({
            'residue': label,
            'rscc': rscc,
            'affinity_kcal_mol': gnina_row.get('affinity_kcal_mol', ''),
            'cnnscore': gnina_row.get('cnnscore', ''),
            'cnnaffinity': gnina_row.get('cnnaffinity', ''),
        })

    with open(args.output_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)
    print(f'{len(out_rows)} residue score row(s) written to {args.output_csv}')


if __name__ == '__main__':
    main()
