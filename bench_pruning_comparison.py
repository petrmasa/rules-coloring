"""
bench_pruning_comparison.py - Table 3: rule counts after pruning for all methods.

Mines each of the four datasets, then applies all comparison algorithms and
the CLM two-phase pruning variants. Prints the combined count table.

Usage:
    python bench_pruning_comparison.py              # all datasets
    python bench_pruning_comparison.py titanic      # single dataset
"""

import sys
import os
import io
import copy
import contextlib
import pandas as pd
from sklearn.impute import SimpleImputer

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(REPO, 'lib'))
sys.path.insert(0, REPO)

from cleverminer import cleverminer
from pruning_by_colors import prune_by_colors, derive_and_prune
from datasets import datasets
from task_configs import DATASETS, DATASET_ORDER

from algorithms import (
    qcba_literal, pep_attr, ripper_metric,
    delta_tar, hamalainen_significance, whole_rule_chi2,
)

ALGO_LABELS = [
    'CLM Phase 1 red',
    'CLM Phase 1 red+yellow',
    'CLM Phase 2 red',
    'CLM Phase 2 red+yellow',
    'QCBA literal',
    'PEP per attribute',
    'RIPPER metric',
    'delta-TAR d=0',
    'delta-TAR d=0.05',
    'delta-TAR d=0.10',
    'Hamalainen a=0.05',
    'Hamalainen a=0.01',
    'Whole-rule chi2 a=0.05',
    'Whole-rule chi2 a=0.01',
]


def mine(ds_name):
    cfg = DATASETS[ds_name]
    ds = datasets()
    df = ds.load_dataset(ds_name)
    if cfg['impute']:
        df = pd.DataFrame(
            SimpleImputer(strategy='most_frequent').fit_transform(df),
            columns=df.columns,
        )
    return cleverminer(df=df, proc='4ftMiner',
                       quantifiers=cfg['quantifiers'],
                       ante=cfg['ante'],
                       succ=cfg['succ'])


def run_all_algos(clm):
    n = clm.get_rulecount()
    results = {}

    # CLM Phase 1
    results['CLM Phase 1 red'] = n - len(prune_by_colors(clm, prune_yellow=False))
    results['CLM Phase 1 red+yellow'] = n - len(prune_by_colors(clm, prune_yellow=True))

    # CLM Phase 2 - mutates clm.result['rules'], restore after each run
    saved = copy.deepcopy(clm.result['rules'])
    with contextlib.redirect_stdout(io.StringIO()):
        p1, p2, derived = derive_and_prune(clm, prune_yellow=False)
    results['CLM Phase 2 red'] = clm.get_rulecount() - len(p1 | p2)
    clm.result['rules'] = copy.deepcopy(saved)

    with contextlib.redirect_stdout(io.StringIO()):
        p1, p2, derived = derive_and_prune(clm, prune_yellow=True)
    results['CLM Phase 2 red+yellow'] = clm.get_rulecount() - len(p1 | p2)
    clm.result['rules'] = saved  # restore to original

    # Simplification algorithms
    results['QCBA literal'] = qcba_literal.run(clm)['n_output']
    results['PEP per attribute'] = pep_attr.run(clm)['n_output']
    results['RIPPER metric'] = ripper_metric.run(clm)['n_output']

    # delta-TAR variants
    for delta, label in [(0.0, 'delta-TAR d=0'), (0.05, 'delta-TAR d=0.05'), (0.10, 'delta-TAR d=0.10')]:
        results[label] = delta_tar.run(clm, delta=delta)['n_output']

    # Statistical significance filters
    for alpha, label in [(0.05, 'Hamalainen a=0.05'), (0.01, 'Hamalainen a=0.01')]:
        results[label] = hamalainen_significance.run(clm, alpha=alpha)['n_output']
    for alpha, label in [(0.05, 'Whole-rule chi2 a=0.05'), (0.01, 'Whole-rule chi2 a=0.01')]:
        results[label] = whole_rule_chi2.run(clm, alpha=alpha)['n_output']

    return results


def main():
    targets = sys.argv[1:] if len(sys.argv) > 1 else DATASET_ORDER
    unknown = [t for t in targets if t not in DATASETS]
    if unknown:
        print(f"Unknown dataset(s): {unknown}. Valid: {DATASET_ORDER}")
        sys.exit(1)

    ds_results = {}
    ds_n_input = {}

    for ds_name in targets:
        print(f"Processing {ds_name}...", flush=True)
        clm = mine(ds_name)
        ds_n_input[ds_name] = clm.get_rulecount()
        ds_results[ds_name] = run_all_algos(clm)
        print(f"  done ({ds_n_input[ds_name]} input rules)")

    # Print table
    col_w = 20
    ds_labels = [f"{d.capitalize()} ({ds_n_input[d]})" for d in targets]
    print()
    print('Table 3: Number of rules after pruning')
    print()
    header = f"{'Method':<{col_w}}" + ''.join(f"{lbl:>18}" for lbl in ds_labels)
    print(header)
    print('-' * len(header))
    for algo in ALGO_LABELS:
        row = f"{algo:<{col_w}}"
        for ds_name in targets:
            row += f"{ds_results[ds_name][algo]:>18}"
        print(row)
    print('-' * len(header))


if __name__ == '__main__':
    main()
