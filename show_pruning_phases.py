"""
show_pruning_phases.py - show rule lists before and after each pruning phase.

Mines one dataset and prints the full ruleset, after Phase 1, and after Phase 2
(with derived rules), for both red-only and red+yellow variants.

Usage:
    python show_pruning_phases.py                  # all four datasets
    python show_pruning_phases.py titanic          # single dataset
    python show_pruning_phases.py titanic loans    # multiple datasets
"""

import sys
import os
import copy
import pandas as pd
from sklearn.impute import SimpleImputer

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(REPO, 'lib'))
sys.path.insert(0, REPO)

from cleverminer import cleverminer
from action_literals import action_literals
from pruning_by_colors import prune_by_colors, derive_and_prune
from datasets import datasets
from task_configs import DATASETS, DATASET_ORDER


def run_dataset(name):
    cfg = DATASETS[name]
    ds = datasets()
    df = ds.load_dataset(name)

    if cfg.get('impute'):
        imp = SimpleImputer(strategy='most_frequent')
        df = pd.DataFrame(imp.fit_transform(df), columns=df.columns)

    clm = cleverminer(df=df, proc='4ftMiner',
                      quantifiers=cfg['quantifiers'],
                      ante=cfg['ante'],
                      succ=cfg['succ'])

    al = action_literals(clm)
    total = clm.get_rulecount()
    w = "=" * 70

    # Phase 1 (no injection)
    ignore_red     = prune_by_colors(clm, prune_yellow=False)
    ignore_red_yel = prune_by_colors(clm, prune_yellow=True)

    print(f"\n{w}")
    print(f"  {name.upper()}  -FULL RULESET  ({total} rules)")
    print(w)
    al.print_rulelist()

    print(f"\n{w}")
    print(f"  {name.upper()}  -AFTER PHASE 1 RED  ({total - len(ignore_red)} rules)")
    print(w)
    al.print_rulelist(ignore_rules=ignore_red, print_stats=True)

    print(f"\n{w}")
    print(f"  {name.upper()}  -AFTER PHASE 1 RED+YELLOW  ({total - len(ignore_red_yel)} rules)")
    print(w)
    al.print_rulelist(ignore_rules=ignore_red_yel, print_stats=True)

    # Phase 2 (with rule injection) — snapshot/restore so both variants are independent
    saved = copy.deepcopy(clm.result['rules'])

    print(f"\n{w}")
    print(f"  {name.upper()}  -TWO-PHASE PRUNING  RED ONLY")
    print(w)
    p1_red, p2_red, derived_red = derive_and_prune(clm, prune_yellow=False)
    shown_red = clm.get_rulecount() - len(p1_red | p2_red)
    clm.result['rules'] = copy.deepcopy(saved)

    print(f"\n{w}")
    print(f"  {name.upper()}  -TWO-PHASE PRUNING  RED+YELLOW")
    print(w)
    p1_ryel, p2_ryel, derived_ryel = derive_and_prune(clm, prune_yellow=True)
    shown_ryel = clm.get_rulecount() - len(p1_ryel | p2_ryel)

    print(f"\n{w}")
    print(f"  {name.upper()}  -FINAL LIST (Phase 2 red+yellow)  ({shown_ryel} rules)")
    print(w)
    al.print_rulelist(ignore_rules=p1_ryel | p2_ryel, print_stats=True)
    clm.result['rules'] = copy.deepcopy(saved)

    print(f"\n{w}")
    print(f"  {name.upper()}  -SUMMARY")
    print(w)
    print(f"{'':35} {'rules':>6}")
    print(f"{'Original':35} {total:>6}")
    print(f"{'After Phase 1 red':35} {total - len(ignore_red):>6}")
    print(f"{'After Phase 1 red+yellow':35} {total - len(ignore_red_yel):>6}")
    print(f"{'After Phase 2 red (incl. derived)':35} {shown_red:>6}")
    print(f"{'After Phase 2 red+yellow (incl. derived)':35} {shown_ryel:>6}")


if __name__ == '__main__':
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    names = args if args else DATASET_ORDER
    for name in names:
        if name not in DATASETS:
            print(f"Unknown dataset: {name}. Choose from: {', '.join(DATASET_ORDER)}")
            sys.exit(1)
        run_dataset(name)
