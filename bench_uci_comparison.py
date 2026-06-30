"""
bench_uci_comparison.py - Table 4: rule counts on UCI ML datasets.

Loads each UCI dataset via OpenML, mines 4ft-Miner rules, applies all
comparison algorithms, and prints the count table.

Requires internet access for first run (datasets cached locally by sklearn after first download).

Usage:
    python bench_uci_comparison.py
    python bench_uci_comparison.py uci_iris uci_glass    # subset
"""

import sys
import os
import io
import copy
import contextlib

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(REPO, 'lib'))
sys.path.insert(0, REPO)

from cleverminer import cleverminer
from pruning_by_colors import prune_by_colors, derive_and_prune
from uci_datasets import get_all, load_dataset

from algorithms import (
    qcba_literal, pep_attr, ripper_metric,
    delta_tar, hamalainen_significance, whole_rule_chi2,
)

ALGO_LABELS = [
    ('CLM Ph1 red',    None),
    ('CLM Ph1 r+y',    None),
    ('CLM Ph2 red',    None),
    ('CLM Ph2 r+y',    None),
    ('QCBA',           None),
    ('PEP',            None),
    ('RIPPER',         None),
    ('delta-TAR 0',    None),
    ('delta-TAR .05',  None),
    ('delta-TAR .10',  None),
    ('Ham .05',        None),
    ('Ham .01',        None),
    ('chi2 .05',       None),
    ('chi2 .01',       None),
]


def _build_ante_attrs(df, seq_cols, sub_cols):
    attrs = []
    for col in df.columns:
        if col in seq_cols:
            n_bins = df[col].nunique()
            attrs.append({'name': col, 'type': 'seq', 'minlen': 1, 'maxlen': min(n_bins, 5)})
        elif col in sub_cols:
            attrs.append({'name': col, 'type': 'subset', 'minlen': 1, 'maxlen': 1})
    return attrs


def run_all_algos(clm):
    n = clm.get_rulecount()
    res = {}

    res['CLM Ph1 red'] = n - len(prune_by_colors(clm, prune_yellow=False))
    res['CLM Ph1 r+y'] = n - len(prune_by_colors(clm, prune_yellow=True))

    saved = copy.deepcopy(clm.result['rules'])
    with contextlib.redirect_stdout(io.StringIO()):
        p1, p2, _ = derive_and_prune(clm, prune_yellow=False)
    res['CLM Ph2 red'] = clm.get_rulecount() - len(p1 | p2)
    clm.result['rules'] = copy.deepcopy(saved)

    with contextlib.redirect_stdout(io.StringIO()):
        p1, p2, _ = derive_and_prune(clm, prune_yellow=True)
    res['CLM Ph2 r+y'] = clm.get_rulecount() - len(p1 | p2)
    clm.result['rules'] = saved

    res['QCBA']   = qcba_literal.run(clm)['n_output']
    res['PEP']    = pep_attr.run(clm)['n_output']
    res['RIPPER'] = ripper_metric.run(clm)['n_output']

    res['delta-TAR 0']   = delta_tar.run(clm, delta=0.0)['n_output']
    res['delta-TAR .05'] = delta_tar.run(clm, delta=0.05)['n_output']
    res['delta-TAR .10'] = delta_tar.run(clm, delta=0.10)['n_output']

    res['Ham .05'] = hamalainen_significance.run(clm, alpha=0.05)['n_output']
    res['Ham .01'] = hamalainen_significance.run(clm, alpha=0.01)['n_output']
    res['chi2 .05']  = whole_rule_chi2.run(clm, alpha=0.05)['n_output']
    res['chi2 .01']  = whole_rule_chi2.run(clm, alpha=0.01)['n_output']

    return res


def main():
    registry = get_all()
    if len(sys.argv) > 1:
        only = set(sys.argv[1:])
        registry = [(n, o, v) for n, o, v in registry if n in only]

    col_labels = [lbl for lbl, _ in ALGO_LABELS]
    col_w = 9

    header = f"{'Dataset':<18} {'input':>5}  " + ''.join(f"{c:>{col_w}}" for c in col_labels)
    print()
    print('Table 4: Rule counts on UCI ML datasets')
    print()
    print(header)
    print('-' * len(header))

    for ds_key, openml_name, version in registry:
        try:
            df, seq_cols, sub_cols, target_col = load_dataset(openml_name, version=version)
        except Exception as e:
            print(f"{ds_key:<18}  LOAD ERROR: {e}")
            continue

        n_rows = len(df)
        base = max(10, n_rows // 10)
        ante_attrs = _build_ante_attrs(df, seq_cols, sub_cols)

        try:
            clm = cleverminer(
                df=df, proc='4ftMiner',
                quantifiers={'Base': base, 'conf': 0.8},
                ante={'attributes': ante_attrs, 'minlen': 1, 'maxlen': 2, 'type': 'con'},
                succ={'attributes': [{'name': target_col, 'type': 'subset',
                                      'minlen': 1, 'maxlen': 1}],
                      'minlen': 1, 'maxlen': 1, 'type': 'con'},
            )
        except Exception as e:
            print(f"{ds_key:<18}  MINE ERROR: {e}")
            continue

        n_in = clm.get_rulecount()
        try:
            res = run_all_algos(clm)
        except Exception as e:
            print(f"{ds_key:<18} {n_in:>5}  ERROR: {e}")
            continue

        row = f"{ds_key:<18} {n_in:>5}  " + ''.join(f"{res[lbl]:>{col_w}}" for lbl in col_labels)
        print(row)

    print('-' * len(header))


if __name__ == '__main__':
    main()
