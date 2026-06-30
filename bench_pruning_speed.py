"""
bench_pruning_speed.py - Table 5: pruning re-evaluation speed.

Measures Phase 1 and Phase 2 pruning speed on the four paper datasets.
Phase 1 = prune_by_colors (single pass, in-ruleset suppression).
Phase 2 = derive_and_prune (loop which inserts/replaces rules).

Usage:
    python bench_pruning_speed.py              # all datasets
    python bench_pruning_speed.py titanic      # single dataset
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
import pruning_by_colors as _pcmod
from task_configs import DATASETS, DATASET_ORDER

N_REPEAT = 5
THRESHOLD = 1.96


def _run_phase1(clm, al):
    ignore = set()
    for i in range(clm.get_rulecount()):
        rule_no = i + 1
        ante = clm.get_rule_variables(rule_no, 'ante', get_names=True)
        succ = clm.get_rule_variables(rule_no, 'succ', get_names=True)
        ff_full = al._get_rule_ff(rule_id=rule_no)
        for attr in ante:
            ff_w = al._get_rule_ff(rule_id=rule_no, cedent='ante', ignore_var=attr)
            z, _, _ = al.calc_z_score(ff_full, ff_w)
            if abs(z) <= THRESHOLD:
                if _pcmod._simpler_rule_exists(clm, rule_no, ante, succ, attr):
                    ignore.add(rule_no)
                    break
    return ignore


def _run_phase2(clm, p1_ignore, al):
    p2_ignore = set()
    while True:
        changed = False
        all_suppress = p1_ignore | p2_ignore
        for i in range(clm.get_rulecount()):
            rule_no = i + 1
            if rule_no in all_suppress:
                continue
            ante = clm.get_rule_variables(rule_no, 'ante', get_names=True)
            succ = clm.get_rule_variables(rule_no, 'succ', get_names=True)
            if len(ante) <= 1:
                continue
            red_attrs = _pcmod._find_red_attrs(al, rule_no, ante, THRESHOLD)
            if not red_attrs:
                continue
            suppressed = False
            for attr in red_attrs:
                sid = _pcmod._find_simpler_rule(clm, rule_no, ante, succ, attr,
                                                exclude_ids=all_suppress)
                if sid is not None:
                    p2_ignore.add(rule_no)
                    all_suppress.add(rule_no)
                    changed = True
                    suppressed = True
                    break
            if suppressed:
                continue
            first_red = red_attrs[0]
            _pcmod._inject_derived_rule(clm, rule_no, first_red)
            p2_ignore.add(rule_no)
            all_suppress.add(rule_no)
            changed = True
        if not changed:
            break
    return p2_ignore


def _reset_derived(clm, n_orig):
    del clm.result['rules'][n_orig:]


def _count_ff(clm, phase_fn, *args):
    orig = action_literals._get_rule_ff
    count = [0]

    def counting(self, *a, **kw):
        count[0] += 1
        return orig(self, *a, **kw)

    action_literals._get_rule_ff = counting
    result = phase_fn(*args)
    action_literals._get_rule_ff = orig
    return result, count[0]


def benchmark(ds_name, clm):
    n_rules = clm.get_rulecount()
    n_rows = clm.data['rows_count']

    # Count ff calls and get phase outputs
    al = action_literals(clm)
    p1_ignore, n_ff_p1 = _count_ff(clm, _run_phase1, clm, al)
    _, n_ff_p2 = _count_ff(clm, _run_phase2, clm, p1_ignore, al)
    n_derived = clm.get_rulecount() - n_rules
    _reset_derived(clm, n_rules)

    # Per-call timing probe
    al2 = action_literals(clm)
    ff_times = []
    for rule_id in range(1, n_rules + 1):
        ante = clm.get_rule_variables(rule_id, 'ante', get_names=True)
        t0 = time.perf_counter()
        al2._get_rule_ff(rule_id=rule_id)
        ff_times.append(time.perf_counter() - t0)
        for attr in ante:
            t0 = time.perf_counter()
            al2._get_rule_ff(rule_id=rule_id, cedent='ante', ignore_var=attr)
            ff_times.append(time.perf_counter() - t0)
    us_per_call = statistics.mean(ff_times) * 1e6

    # Phase 1 timing
    times_p1 = []
    for _ in range(N_REPEAT):
        al3 = action_literals(clm)
        t0 = time.perf_counter()
        _run_phase1(clm, al3)
        times_p1.append(time.perf_counter() - t0)
    ms_p1 = statistics.mean(times_p1) * 1e3

    # Phase 2 timing (Phase 1 runs as setup, untimed)
    times_p2 = []
    for _ in range(N_REPEAT):
        al3 = action_literals(clm)
        p1 = _run_phase1(clm, al3)
        t0 = time.perf_counter()
        _run_phase2(clm, p1, al3)
        times_p2.append(time.perf_counter() - t0)
        _reset_derived(clm, n_rules)
    ms_p2 = statistics.mean(times_p2) * 1e3
    ms_total = ms_p1 + ms_p2

    n_p1_supp = len(p1_ignore)
    n_p2_rules = n_rules - n_p1_supp

    print(f"\n  {ds_name.capitalize()}  (rows={n_rows:,}  rules={n_rules})")
    print(f"    per-call cost:      {us_per_call:>8.0f} µs")
    print(f"    Phase 1:  {n_ff_p1:>4} ff calls ({n_ff_p1/n_rules:.1f}/rule)  "
          f"{ms_p1:>7.1f} ms  {n_rules/ms_p1*1e3:>6,.0f} rules/s  "
          f"{n_ff_p1/ms_p1*1e3:>7,.0f} re-evals/s")
    print(f"    Phase 2:  {n_ff_p2:>4} ff calls ({n_ff_p2/max(n_p2_rules,1):.1f}/entering rule)  "
          f"{ms_p2:>7.1f} ms  derived={n_derived}")
    print(f"    Total:    {n_ff_p1+n_ff_p2:>4} ff calls  "
          f"{ms_total:>7.1f} ms  {n_rules/ms_total*1e3:>6,.0f} rules/s  "
          f"×{(ms_total/max(ms_p1,0.001) if ms_p1 > 0 else 0):.1f}× vs P1 alone")


def main():
    N = 10_000_000
    t = timeit.timeit("x += 1", setup="x = 0", number=N)
    print(f"Python {sys.version.split()[0]} | {platform.system()} {platform.release()}")
    print(f"Integer loop benchmark: {N/t/1e6:.0f} Mops/s")
    print()
    print('Table 5: Pruning re-evaluation speed')

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

    print()


if __name__ == '__main__':
    main()
