"""
bench_coloring_speed.py - rule re-evaluation speed for item coloring.

For each dataset, measures the number and speed of _get_rule_ff calls made
by action_literals coloring (one call per rule + one per eligible attribute/value).

Usage:
    python bench_coloring_speed.py              # all datasets
    python bench_coloring_speed.py titanic      # single dataset
"""

import sys
import os
import time
import timeit
import statistics
import platform
import pandas as pd
from sklearn.impute import SimpleImputer

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(REPO, 'lib'))
sys.path.insert(0, REPO)

from cleverminer import cleverminer
from action_literals import action_literals
from datasets import datasets
from task_configs import DATASETS, DATASET_ORDER

N_REPEAT = 5


def _color_all(clm):
    al = action_literals(clm)
    for rule_id in range(1, clm.get_rulecount() + 1):
        al._get_literal_importance(rule_id=rule_id, print_rule=False, print_details=False)


def _count_ff_calls(clm):
    """Count total _get_rule_ff calls during coloring of all rules."""
    orig = action_literals._get_rule_ff
    count = [0]

    def counting(self, *a, **kw):
        count[0] += 1
        return orig(self, *a, **kw)

    action_literals._get_rule_ff = counting
    _color_all(clm)
    action_literals._get_rule_ff = orig
    return count[0]


def benchmark(ds_name, clm):
    n_rules = clm.get_rulecount()
    n_rows = clm.data['rows_count']

    n_calls = _count_ff_calls(clm)

    # Per-call cost
    al = action_literals(clm)
    ff_times = []
    for rule_id in range(1, n_rules + 1):
        ante = clm.get_rule_variables(rule_id, 'ante', get_names=True)
        t0 = time.perf_counter()
        al._get_rule_ff(rule_id=rule_id)
        ff_times.append(time.perf_counter() - t0)
        for attr in ante:
            t0 = time.perf_counter()
            al._get_rule_ff(rule_id=rule_id, cedent='ante', ignore_var=attr)
            ff_times.append(time.perf_counter() - t0)
    us_per_call = statistics.mean(ff_times) * 1e6

    # Total coloring time
    times = []
    for _ in range(N_REPEAT):
        t0 = time.perf_counter()
        _color_all(clm)
        times.append(time.perf_counter() - t0)
    ms_total = statistics.mean(times) * 1e3

    print(f"  {ds_name:<12}  rows={n_rows:>8,}  rules={n_rules:>4}  "
          f"re-evals={n_calls:>5}  ({n_calls/n_rules:.1f}/rule)  "
          f"time/call={us_per_call:>6.0f}µs  "
          f"total={ms_total:>7.0f}ms  "
          f"rules/s={n_rules/ms_total*1e3:>7,.0f}  "
          f"re-evals/s={n_calls/ms_total*1e3:>7,.0f}")


def main():
    N = 10_000_000
    t = timeit.timeit("x += 1", setup="x = 0", number=N)
    print(f"Python {sys.version.split()[0]} | {platform.system()} {platform.release()}")
    print(f"Integer loop benchmark: {N/t/1e6:.0f} Mops/s")
    print()
    print('Table 2: Rule re-evaluation speed (coloring)')
    print()

    targets = sys.argv[1:] if len(sys.argv) > 1 else DATASET_ORDER
    unknown = [t for t in targets if t not in DATASETS]
    if unknown:
        print(f"Unknown dataset(s): {unknown}. Valid: {DATASET_ORDER}")
        sys.exit(1)

    ds_obj = datasets()
    for ds_name in targets:
        cfg = DATASETS[ds_name]
        df = ds_obj.load_dataset(ds_name)
        if cfg['impute']:
            df = pd.DataFrame(
                SimpleImputer(strategy='most_frequent').fit_transform(df),
                columns=df.columns,
            )
        clm = cleverminer(df=df, proc='4ftMiner',
                          quantifiers=cfg['quantifiers'],
                          ante=cfg['ante'],
                          succ=cfg['succ'])
        benchmark(ds_name, clm)


if __name__ == '__main__':
    main()
