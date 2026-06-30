"""
bench_inert_analysis.py - rules with at least one inert/useless (red) item.

For each dataset, mines the full rule set and the minimal rule set
(after filter_cleverminer_rules), then counts how many rules contain at least
one RED attribute or category.

Usage:
    python bench_inert_analysis.py              # all datasets
    python bench_inert_analysis.py titanic      # single dataset
"""

import sys
import os
import io
import contextlib
import pandas as pd
from sklearn.impute import SimpleImputer

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(REPO, 'lib'))
sys.path.insert(0, REPO)

from cleverminer import cleverminer
from action_literals import action_literals
from datasets import datasets
from minimal_rules import filter_cleverminer_rules
from task_configs import DATASETS, DATASET_ORDER


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


def count_inert(clm, ignore_rules=None):
    """Count rules with at least one red eligible category or red attribute."""
    if ignore_rules is None:
        ignore_rules = set()
    al = action_literals(clm)
    n_total = 0
    n_inert_cat = 0
    n_inert_lit = 0
    for rule_id in range(1, clm.get_rulecount() + 1):
        if rule_id in ignore_rules:
            continue
        n_total += 1
        with contextlib.redirect_stdout(io.StringIO()):
            _, stats = al._get_literal_importance(
                rule_id=rule_id, print_rule=False, print_details=False, get_stats=True
            )
        if stats['has_ignorable_category']:
            n_inert_cat += 1
        if stats['has_ignorable_literal']:
            n_inert_lit += 1
    return n_total, n_inert_cat, n_inert_lit


def main():
    targets = sys.argv[1:] if len(sys.argv) > 1 else DATASET_ORDER
    unknown = [t for t in targets if t not in DATASETS]
    if unknown:
        print(f"Unknown dataset(s): {unknown}. Valid: {DATASET_ORDER}")
        sys.exit(1)

    rows = []
    for ds_name in targets:
        print(f"Processing {ds_name}...", flush=True)
        clm = mine(ds_name)
        n_full, n_full_inert_cat, n_full_inert_lit = count_inert(clm)
        ignore_minimal = filter_cleverminer_rules(clm)
        n_min, n_min_inert_cat, n_min_inert_lit = count_inert(clm, ignore_rules=ignore_minimal)
        rows.append((ds_name.capitalize(), n_full, n_full_inert_cat, n_full_inert_lit,
                     n_min, n_min_inert_cat, n_min_inert_lit))

    print()
    print('Table 1: Comparison of Full Rule Sets and Pruned Minimal Rule Sets')
    print()
    hdr = (f"{'Dataset':<12}  {'Rules':>6}  {'Cat':>6}  {'%':>5}  {'Lit':>6}  {'%':>5}"
           f"  ||  {'Rules':>6}  {'Cat':>6}  {'%':>5}  {'Lit':>6}  {'%':>5}")
    sep = '-' * len(hdr)
    print(f"{'':12}  {'Full Rule Set':^30}  ||  {'Minimal Rule Set':^30}")
    print(hdr)
    print(sep)
    #compute both useless (inert) categories and inert (useless) literals. 
    for ds, nf, nfc, nfl, nm, nmc, nml in rows:
        pfc = 100 * nfc / nf if nf else 0
        pfl = 100 * nfl / nf if nf else 0
        pmc = 100 * nmc / nm if nm else 0
        pml = 100 * nml / nm if nm else 0
        print(f"{ds:<12}  {nf:>6}  {nfc:>6}  {pfc:>4.0f}%  {nfl:>6}  {pfl:>4.0f}%"
              f"  ||  {nm:>6}  {nmc:>6}  {pmc:>4.0f}%  {nml:>6}  {pml:>4.0f}%")
    print(sep)


if __name__ == '__main__':
    main()
