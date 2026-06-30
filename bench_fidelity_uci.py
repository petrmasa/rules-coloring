"""
bench_fidelity_uci.py - Fidelity benchmark for UCI datasets.

Same metrics as bench_fidelity.py but iterates over all UCI OpenML datasets
using the same mining config as bench_uci_comparison.py (conf>=0.8, base>=N//10,
maxlen=2). Datasets that will get 0 rules will be skipped with a note.

Usage:
    python bench_fidelity_uci.py               # all UCI datasets
    python bench_fidelity_uci.py iris          # single dataset key
"""

import sys
import os
import io
import contextlib
import statistics

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(REPO, 'lib'))
sys.path.insert(0, REPO)

from cleverminer import cleverminer
from pruning_by_colors import derive_and_prune
from uci_datasets import get_all, load_dataset

VARIANTS = [
    (False, 'Red'),
    (True,  'Red+Yel'),
]


def _ante_bitmask(clm, rule_id):
    rule = clm.result['rules'][rule_id - 1]
    maxval = 2 ** clm.data['rows_count'] - 1
    mask = maxval
    for order, idx in enumerate(rule['trace_cedent_dataorder']['ante']):
        varval = 0
        for val_idx in rule['traces']['ante'][order]:
            varval |= clm.data['dm'][idx][val_idx]
        mask &= varval
    return mask


def union_coverage(clm, rule_ids):
    union = 0
    for rid in rule_ids:
        union |= _ante_bitmask(clm, rid)
    return union.bit_count()


def rule_stats(clm, rule_id):
    ff = clm.result['rules'][rule_id - 1]['params']['fourfold']
    base = ff[0]
    conf = ff[0] / (ff[0] + ff[1]) if (ff[0] + ff[1]) > 0 else 0.0
    return base, conf


def _reset_derived(clm, n_orig):
    del clm.result['rules'][n_orig:]


def _build_ante_attrs(df, seq_cols, sub_cols, target_col):
    attrs = []
    for col in df.columns:
        if col == target_col:
            continue
        if col in seq_cols:
            attrs.append({'name': col, 'type': 'seq', 'minlen': 1,
                          'maxlen': min(int(df[col].nunique()), 5)})
        elif col in sub_cols:
            attrs.append({'name': col, 'type': 'subset', 'minlen': 1, 'maxlen': 1})
    return attrs


def main():
    filter_keys = set(a.lower() for a in sys.argv[1:])

    all_datasets = list(get_all())
    if filter_keys:
        all_datasets = [(k, n, v) for k, n, v in all_datasets if k.lower() in filter_keys]
        if not all_datasets:
            valid = [k for k, _, _ in get_all()]
            print(f"Unknown dataset key(s): {sys.argv[1:]}. Valid: {valid}")
            sys.exit(1)

    rows = []

    for ds_key, openml_name, version in all_datasets:
        print(f"Processing {ds_key}...", flush=True)

        try:
            df, seq_cols, sub_cols, target_col = load_dataset(openml_name, version=version)
        except Exception as e:
            print(f"  LOAD ERROR: {e}")
            continue

        n_rows = len(df)
        base = max(10, n_rows // 10)
        ante_attrs = _build_ante_attrs(df, seq_cols, sub_cols, target_col)

        try:
            with contextlib.redirect_stdout(io.StringIO()):
                clm = cleverminer(
                    df=df, proc='4ftMiner',
                    quantifiers={'Base': base, 'conf': 0.8},
                    ante={'attributes': ante_attrs, 'minlen': 1, 'maxlen': 2, 'type': 'con'},
                    succ={'attributes': [{'name': target_col, 'type': 'subset',
                                          'minlen': 1, 'maxlen': 1}],
                          'minlen': 1, 'maxlen': 1, 'type': 'con'},
                )
        except Exception as e:
            print(f"  MINE ERROR: {e}")
            continue

        n_orig = clm.get_rulecount()
        if n_orig == 0:
            print(f"  0 rules mined, skipping")
            continue

        print(f"  {n_orig} rules mined")

        orig_ids = set(range(1, n_orig + 1))
        orig_stats_list = [rule_stats(clm, rid) for rid in sorted(orig_ids)]
        orig_bases = [s[0] for s in orig_stats_list]
        orig_confs = [s[1] for s in orig_stats_list]
        orig_cov   = union_coverage(clm, orig_ids)

        for prune_yellow, label in VARIANTS:
            _reset_derived(clm, n_orig)

            with contextlib.redirect_stdout(io.StringIO()):
                p1_ignore, p2_ignore, derived_ids = derive_and_prune(
                    clm, prune_yellow=prune_yellow)

            all_ids       = orig_ids | set(derived_ids)
            surviving_ids = all_ids - p1_ignore - p2_ignore

            max_depth = max(
                (len(clm.result['rules'][did - 1]['params']['extensions']['derived_by_removing'])
                 for did in derived_ids),
                default=0,
            )

            new_cov = union_coverage(clm, surviving_ids)
            surv_stats_list = [rule_stats(clm, rid) for rid in sorted(surviving_ids)]
            surv_bases = [s[0] for s in surv_stats_list] if surv_stats_list else [0.0]
            surv_confs = [s[1] for s in surv_stats_list] if surv_stats_list else [0.0]

            rows.append(dict(
                name          = ds_key,
                variant       = label,
                n_rows        = n_rows,
                n_orig        = n_orig,
                n_after_p1    = n_orig - len(p1_ignore),
                n_after_p2    = len(surviving_ids),
                p2_discard    = len(p2_ignore),
                p2_intro      = len(derived_ids),
                max_depth     = max_depth,
                orig_cov      = orig_cov,
                new_cov       = new_cov,
                orig_avg_conf = statistics.mean(orig_confs),
                new_avg_conf  = statistics.mean(surv_confs),
                orig_avg_base = statistics.mean(orig_bases),
                new_avg_base  = statistics.mean(surv_bases),
            ))

        _reset_derived(clm, n_orig)

    W = 175
    print()
    print('=' * W)
    print('FIDELITY BENCHMARK (UCI) - Phase 1+2 pruning quality metrics (Red vs Red+Yellow)')
    print('=' * W)
    HDR = (f"{'Dataset':<20}  {'Variant':<8}  {'N rows':>8}  "
           f"{'Init':>5}  {'P1':>5}  {'P2':>5}  "
           f"{'P2-disc':>7}  {'P2-intr':>7}  {'Depth':>5}  "
           f"{'Cov orig':>9}  {'Cov new':>9}  "
           f"{'Conf orig':>9}  {'Conf new':>9}  "
           f"{'Base orig':>9}  {'Base new':>9}")
    print(HDR)
    print('-' * W)
    prev_name = None
    for r in rows:
        if prev_name and r['name'] != prev_name:
            print()
        prev_name = r['name']
        print(
            f"{r['name']:<20}  {r['variant']:<8}  {r['n_rows']:>8,}  "
            f"{r['n_orig']:>5}  {r['n_after_p1']:>5}  {r['n_after_p2']:>5}  "
            f"{r['p2_discard']:>7}  {r['p2_intro']:>7}  {r['max_depth']:>5}  "
            f"{r['orig_cov']:>9,}  {r['new_cov']:>9,}  "
            f"{r['orig_avg_conf']:>9.3f}  {r['new_avg_conf']:>9.3f}  "
            f"{r['orig_avg_base']:>9.1f}  {r['new_avg_base']:>9.1f}"
        )
    print('=' * W)
    print()
    print('Coverage  = unique instances whose antecedent matches >=1 rule.')
    print('Conf/Base = average over all rules for the respective set.')
    print('Depth     = max removals in a single derivation chain (0 = no derived rules needed).')
    print('P2-disc   = rules suppressed in Phase 2 only (excludes Phase 1 suppressions).')
    print('P2-intr   = derived rules injected in Phase 2 (some may be subsequently superseded).')


if __name__ == '__main__':
    main()
