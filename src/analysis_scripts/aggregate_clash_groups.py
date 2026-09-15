#!/usr/bin/env python3
"""
Sidechain clash-group aggregator.

Concatenates the sidechain_clash_groups.csv already written by
build_final_model into every dataset's own .../<final_run_name>/ directory
(one row per resolved sidechain-sidechain clash group - see
build_final_model.py's _write_clash_groups_csv) into a single run-wide csv,
with a 'dataset' column identifying which dataset each row came from. No new
clash detection happens here.

Usage:
  aggregate_clash_groups.py <run_name> <placer_run_name> <filter_run_name> \\
      <placer2_run_name> <filter2_run_name> <final_run_name> --graphs-dir DIR [options]
"""
from rscc_common import build_pooled_argparser, run_clash_groups_aggregator


def main():
    args = build_pooled_argparser(__doc__).parse_args()
    run_clash_groups_aggregator(args)


if __name__ == '__main__':
    main()
