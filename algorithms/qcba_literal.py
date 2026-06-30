"""
QCBA-style literal (attribute) pruning applied to 4ft-Miner rules.

Reference: Kliegr & Izquierdo (2023) : QCBA: Improving Rule Classifiers Learned from Quantitative Data by
Recovering Information Lost by Discretisation, Applied Intelligence, section 3.3 literal pruning

Criterion: remove attribute A from rule R's antecedent if conf(R without A) >= conf(R).
Loop until not removed
Deduplication is also needed.
"""


def _compute_ff(clm_output, rule_id, ignore_attrs=None):
    if ignore_attrs is None:
        ignore_attrs = frozenset()
    rule_dict = clm_output.result['rules'][rule_id - 1]
    datalabels = clm_output.result['datalabels']
    row_count = clm_output.data['rows_count']
    maxval = 2 ** row_count - 1

    cedents = {}
    for cedent_name, indices in rule_dict['trace_cedent_dataorder'].items():
        val_lists = rule_dict['traces'][cedent_name]
        cedentval = maxval
        for order, idx in enumerate(indices):
            varname = datalabels['varname'][idx]
            if cedent_name == 'ante' and varname in ignore_attrs:
                continue
            varval = 0
            cnt = 0
            for val_idx in val_lists[order]:
                varval |= clm_output.data['dm'][idx][val_idx]
                cnt += 1
            if cnt > 0:
                cedentval &= varval
        cedents[cedent_name] = cedentval

    return [
        (cedents['ante'] & cedents['succ'] & cedents['cond']).bit_count(),
        (cedents['ante'] & (maxval - cedents['succ']) & cedents['cond']).bit_count(),
        ((maxval - cedents['ante']) & cedents['succ'] & cedents['cond']).bit_count(),
        ((maxval - cedents['ante']) & (maxval - cedents['succ']) & cedents['cond']).bit_count(),
    ]


def _conf(ff):
    n = ff[0] + ff[1]
    return ff[0] / n if n > 0 else 0.0


def _render_ante_str(clm_output, rule_id, exclude_attrs=None):
    if exclude_attrs is None:
        exclude_attrs = frozenset()
    rule_dict = clm_output.result['rules'][rule_id - 1]
    datalabels = clm_output.result['datalabels']
    indices = rule_dict['trace_cedent_dataorder']['ante']
    val_lists = rule_dict['traces']['ante']
    parts = []
    for order, idx in enumerate(indices):
        vn = datalabels['varname'][idx]
        if vn in exclude_attrs:
            continue
        cats = [str(datalabels['catnames'][idx][vi]) for vi in val_lists[order]]
        parts.append(f"{vn}({' '.join(cats)})")
    return ' & '.join(parts) if parts else '---'


def run(clm_output):
    """
    Apply QCBA-style literal pruning to all rules in clm_output.

    Returns dict with n_input, n_rules_simplified, n_output, rules.
    """
    rule_count = clm_output.get_rulecount()
    simplified = []

    for rule_no in range(1, rule_count + 1):
        ante = list(clm_output.get_rule_variables(rule_no, 'ante', get_names=True))
        succ_str = clm_output.result['rules'][rule_no - 1]['cedents_str']['succ']

        conf_current = _conf(_compute_ff(clm_output, rule_no))
        removed = set()

        changed = True
        while changed and (len(ante) - len(removed)) > 1:
            changed = False
            for attr in [a for a in ante if a not in removed]:
                ff_try = _compute_ff(clm_output, rule_no, removed | {attr})
                if _conf(ff_try) >= conf_current:
                    removed.add(attr)
                    conf_current = _conf(ff_try)
                    changed = True
                    break

        ff_result = _compute_ff(clm_output, rule_no, removed)
        simplified.append({
            'original_rule_id': rule_no,
            'ante_str': _render_ante_str(clm_output, rule_no, removed),
            'succ_str': succ_str,
            'attrs_removed': sorted(removed),
            'base': ff_result[0],
            'conf': round(_conf(ff_result), 3),
        })

    seen = set()
    deduped = []
    for r in simplified:
        key = (r['ante_str'], r['succ_str'])
        if key not in seen:
            seen.add(key)
            deduped.append(r)

    return {
        'n_input': rule_count,
        'n_rules_simplified': sum(1 for r in simplified if r['attrs_removed']),
        'n_output': len(deduped),
        'rules': deduped,
    }
