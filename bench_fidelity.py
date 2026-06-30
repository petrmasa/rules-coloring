"""
bench_fidelity.py - Fidelity benchmark: Phase 1+2 pruning quality metrics.

For each of the five datasets (four main); mines rules and calculate measures
how well pruned rule set keeps coverage, confidence and base rate.
Note that (from the principle of the statistical method), some decrease is allowed
(like in train-test set for classificators).

Usage:
    python bench_fidelity.py              # all datasets
    python bench_fidelity.py titanic      # single dataset
"""

import sys
import os
import io
import contextlib
import statistics
import pandas as pd
from sklearn.impute import SimpleImputer

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(REPO, 'lib'))
sys.path.insert(0, REPO)

from cleverminer import cleverminer
from action_literals import action_literals
from datasets import datasets
from pruning_by_colors import derive_and_prune
from uci_datasets import load_dataset as load_openml

ds_loader = datasets()


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


FIDELITY_DATASETS = [
    {
        'name': 'Accidents',
        'loader': 'accidents',
        'impute': True,
        'quantifiers': {'Base': 1000, 'conf': 0.045},
        'ante_attrs': [
            {'name': 'Driver_Age_Band', 'type': 'seq',    'minlen': 1, 'maxlen': 4},
            {'name': 'Speed_limit',     'type': 'seq',    'minlen': 1, 'maxlen': 3},
            {'name': 'Light',           'type': 'subset', 'minlen': 1, 'maxlen': 1},
            {'name': 'Journey',         'type': 'subset', 'minlen': 1, 'maxlen': 1},
            {'name': 'Road_Type',       'type': 'subset', 'minlen': 1, 'maxlen': 1},
            {'name': 'Sex',             'type': 'subset', 'minlen': 1, 'maxlen': 1},
        ],
        'ante_maxlen': 4,
        'succ_attr': {'name': 'Severity', 'type': 'lcut'},
    },
    {
        'name': 'Diabetes',
        'loader': 'diabetes',
        'impute': False,
        'quantifiers': {'Base': 50, 'conf': 0.05},
        'ante_attrs': [
            {'name': 'Pregnancies',   'type': 'seq',    'minlen': 1, 'maxlen': 1},
            {'name': 'Glucose',       'type': 'seq',    'minlen': 1, 'maxlen': 4},
            {'name': 'BloodPressure', 'type': 'seq',    'minlen': 1, 'maxlen': 3},
            {'name': 'SkinThickness', 'type': 'subset', 'minlen': 1, 'maxlen': 1},
            {'name': 'Insulin',       'type': 'subset', 'minlen': 1, 'maxlen': 1},
            {'name': 'BMI',           'type': 'seq',    'minlen': 1, 'maxlen': 2},
        ],
        'ante_maxlen': 4,
        'succ_attr': {'name': 'Outcome', 'type': 'rcut'},
    },
    {
        'name': 'Loans',
        'loader': 'loans',
        'impute': False,
        'quantifiers': {'Base': 50, 'conf': 0.8},
        'ante_attrs': [
            {'name': ' no_of_dependents',        'type': 'seq',    'minlen': 1, 'maxlen': 1},
            {'name': ' income_annum',             'type': 'seq',    'minlen': 1, 'maxlen': 4},
            {'name': ' loan_amount',              'type': 'seq',    'minlen': 1, 'maxlen': 3},
            {'name': ' loan_term',                'type': 'subset', 'minlen': 1, 'maxlen': 1},
            {'name': ' education',                'type': 'subset', 'minlen': 1, 'maxlen': 1},
            {'name': ' residential_assets_value', 'type': 'seq',    'minlen': 1, 'maxlen': 2},
        ],
        'ante_maxlen': 4,
        'succ_attr': {'name': ' loan_status', 'type': 'lcut'},
    },
    {
        'name': 'Titanic',
        'loader': 'titanic',
        'impute': False,
        'quantifiers': {'Base': 50, 'conf': 0.8},
        'ante_attrs': [
            {'name': 'Pclass',   'type': 'seq',    'minlen': 1, 'maxlen': 1},
            {'name': 'Age',      'type': 'seq',    'minlen': 1, 'maxlen': 4},
            {'name': 'SibSp',    'type': 'seq',    'minlen': 1, 'maxlen': 3},
            {'name': 'Parch',    'type': 'subset', 'minlen': 1, 'maxlen': 1},
            {'name': 'Sex',      'type': 'subset', 'minlen': 1, 'maxlen': 1},
            {'name': 'Fare',     'type': 'seq',    'minlen': 1, 'maxlen': 2},
            {'name': 'Embarked', 'type': 'seq',    'minlen': 1, 'maxlen': 2},
        ],
        'ante_maxlen': 4,
        'succ_attr': {'name': 'Survived', 'type': 'rcut'},
    },
]

DATASET_NAMES = [cfg['name'] for cfg in FIDELITY_DATASETS]

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


def rule_stats(al, rule_id):
    ff = al._get_rule_ff(rule_id=rule_id)
    base = ff[0]
    conf = ff[0] / (ff[0] + ff[1]) if (ff[0] + ff[1]) > 0 else 0.0
    return base, conf


def _reset_derived(clm, n_orig):
    del clm.result['rules'][n_orig:]


def main():
    args = sys.argv[1:]
    targets_lower = [a.lower() for a in args]
    cfgs = [cfg for cfg in FIDELITY_DATASETS
            if not args or cfg['name'].lower() in targets_lower]
    if not cfgs:
        print(f"Unknown dataset(s): {args}. Valid: {DATASET_NAMES}")
        sys.exit(1)

    rows = []

    for cfg in cfgs:
        print(f"Processing {cfg['name']}...", flush=True)

        if cfg['loader'] == 'openml':
            df, seq_cols, sub_cols, target_col = load_openml(
                cfg['openml_name'], version=cfg.get('openml_version'))
            n_rows = len(df)
            ante_attrs = _build_ante_attrs(df, seq_cols, sub_cols, target_col)
            quantifiers = {'Base': max(10, n_rows // 10), 'conf': 0.8}
            ante_maxlen = 2
            succ_attr = {'name': target_col, 'type': 'subset', 'minlen': 1, 'maxlen': 1}
        else:
            df = ds_loader.load_dataset(cfg['loader'])
            if cfg['impute']:
                imp = SimpleImputer(strategy='most_frequent')
                df = pd.DataFrame(imp.fit_transform(df), columns=df.columns)
            ante_attrs = cfg['ante_attrs']
            quantifiers = cfg['quantifiers']
            ante_maxlen = cfg['ante_maxlen']
            succ_attr = dict(**cfg['succ_attr'], minlen=1, maxlen=1)

        clm = cleverminer(
            df=df, proc='4ftMiner',
            quantifiers=quantifiers,
            ante={'attributes': ante_attrs, 'minlen': 1, 'maxlen': ante_maxlen, 'type': 'con'},
            succ={'attributes': [succ_attr], 'minlen': 1, 'maxlen': 1, 'type': 'con'},
        )

        al = action_literals(clm)
        n_orig = clm.get_rulecount()
        orig_ids = set(range(1, n_orig + 1))

        orig_stats_list = [rule_stats(al, rid) for rid in sorted(orig_ids)]
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

            new_cov    = union_coverage(clm, surviving_ids)
            surv_stats = [rule_stats(al, rid) for rid in sorted(surviving_ids)]
            surv_bases = [s[0] for s in surv_stats]
            surv_confs = [s[1] for s in surv_stats]

            rows.append(dict(
                name          = cfg['name'],
                variant       = label,
                n_rows        = clm.data['rows_count'],
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

    W = 158
    print()
    print('=' * W)
    print('FIDELITY BENCHMARK - Phase 1+2 pruning quality metrics (Red vs Red+Yellow)')
    print('=' * W)
    HDR = (f"{'Dataset':<12}  {'Variant':<8}  {'N rows':>8}  "
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
            f"{r['name']:<12}  {r['variant']:<8}  {r['n_rows']:>8,}  "
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
