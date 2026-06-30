"""
show_coloring.py - mine rules and show color-coded item importance.

Runs the 4ft-Miner task for each of the four basic datasets and prints full colored rule list.

Usage:
    python show_coloring.py                        # all four datasets
    python show_coloring.py accidents              # single dataset
    python show_coloring.py titanic diabetes       # multiple datasets
    python show_coloring.py --tex                  # TeX color output (all datasets) #helper for preparing the article
    python show_coloring.py titanic --tex          # TeX color output (single dataset) #helper for preparing the article
    python show_coloring.py accidents --rule 1     # detailed analysis for rule 1
    python show_coloring.py accidents --rule 1 --tex  # same with TeX formatting #helper for preparing the article
"""
# TODO: add also other color schemas/formatting, e.g. for B/W print

import sys
import os
import pandas as pd
from sklearn.impute import SimpleImputer

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(REPO, 'lib'))
sys.path.insert(0, REPO)

from cleverminer import cleverminer
from action_literals import action_literals
from datasets import datasets
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


def run(ds_name, color_style=None, rule_id=None):
    print(f"\n{'='*70}")
    print(f"  {ds_name.upper()}  -  colored rule list")
    print(f"{'='*70}")
    clm = mine(ds_name)
    print(f"Mined {clm.get_rulecount()} rules.")
    al = action_literals(clm, color_style=color_style)
    al.print_rulelist()
    if rule_id is not None:
        al.print_rule_literal_importance(rule_id)


if __name__ == '__main__':
    args = sys.argv[1:]
    tex_mode = '--tex' in args
    args = [a for a in args if a != '--tex']
    color_style = 'TeX' if tex_mode else None

    rule_id = None
    if '--rule' in args:
        idx = args.index('--rule')
        if idx + 1 >= len(args):
            print("Error: --rule requires a rule number")
            sys.exit(1)
        try:
            rule_id = int(args[idx + 1])
        except ValueError:
            print(f"Error: --rule argument must be an integer, got '{args[idx + 1]}'")
            sys.exit(1)
        args = args[:idx] + args[idx + 2:]

    targets = args if args else DATASET_ORDER
    unknown = [t for t in targets if t not in DATASETS]
    if unknown:
        print(f"Unknown dataset(s): {unknown}")
        print(f"Valid: {DATASET_ORDER}")
        sys.exit(1)
    for name in targets:
        run(name, color_style=color_style, rule_id=rule_id)
