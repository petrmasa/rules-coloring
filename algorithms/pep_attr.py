"""
Pessimistic Error Pruning (PEP) per attribute, adapted to 4ft-Miner rules.

Reference: Liu 1998 - CBA: Integrating classification and association rule mining
(e.g. https://dl.acm.org/doi/10.5555/3000292.3000305,https://sci2s.ugr.es/keel/pdf/algorithm/congreso/1998-Liu-CBA.pdf)

Idea - remove attribute A if pep_error does not raise.

Criterion: for each antecedent attribute A of rule R, compute pessimistic error
    pep_error = (FP + 0.5) / (TP + FP)    
for R and for R without A.  For us, TP = ff[0] (or "a" in fourfold table, as written in the article) and FP = ff[1] (or "b" in classical ff table)
If R\A's error <= R's error, remove A. Keep at least 1 attribute.
Also, dedulpication is needed.
"""


def _pep_error(ff):
    denom = ff[0] + ff[1]
    return (ff[1] + 0.5) / denom if denom > 0 else 1.0


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
    Apply PEP per attribute to all rules in clm_output.

    Returns dict with n_input, n_rules_simplified, n_output, rules.
    """
    rule_count = clm_output.get_rulecount()
    simplified = []

    for rule_no in range(1, rule_count + 1):
        ante = list(clm_output.get_rule_variables(rule_no, 'ante', get_names=True))
        succ_str = clm_output.result['rules'][rule_no - 1]['cedents_str']['succ']

        ff_orig = _compute_ff(clm_output, rule_no)
        err_current = _pep_error(ff_orig)
        removed = set()

        changed = True
        while changed and (len(ante) - len(removed)) > 1:
            changed = False
            for attr in [a for a in ante if a not in removed]:
                ff_try = _compute_ff(clm_output, rule_no, removed | {attr})
                if _pep_error(ff_try) <= err_current:
                    removed.add(attr)
                    err_current = _pep_error(ff_try)
                    changed = True
                    break

        ff_result = _compute_ff(clm_output, rule_no, removed)
        simplified.append({
            'original_rule_id': rule_no,
            'ante_str': _render_ante_str(clm_output, rule_no, removed),
            'succ_str': succ_str,
            'attrs_removed': sorted(removed),
            'base': ff_result[0],
            'conf': round(ff_result[0] / (ff_result[0] + ff_result[1]), 3)
                    if (ff_result[0] + ff_result[1]) > 0 else 0.0,
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
