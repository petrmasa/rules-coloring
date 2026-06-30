"""
RIPPER pruning metric per attribute, adapted to 4ft-Miner association rules.

Reference: Cohen : Fast Effective Rule Induction. ICML 1995, https://dl.acm.org/doi/10.5555/3091622.3091637
https://crystal.uta.edu/~gonzalez/ml/Ripper.pdf

Criterion: suppress rule R if there exists an antecedent attribute A such that
    v*(R\A) >= v*(R)   where v* = (TP - FP) / (TP + FP) 
    [!!! ALSO: there is (p-n)/(p+n) where p is positive and n is negative (not probability and number of observations like in statistical tests!!!)]    
    AND a mined rule with antecedent R.ante\{A} and same succedent/categories already exists in the ruleset.

TP and FP - see ff in PEP_ATTR.PY (ff[0] and ff[1])

v* is called metric in the following code    

Note that this metric was used for on-the-fly deletion during learning, so it was adapted to post-hoc pruning.

Note: metric(R) = (a-b)/(a+b) = conf(R)-b/(a+b) = conf(R)+(a-(a+b))/(a+b) = 2*conf(R)-1

Also note that simpler rule must exists.
"""


def _metric(ff):
    denom = ff[0] + ff[1]
    return (ff[0] - ff[1]) / denom if denom > 0 else 0.0


def _get_ff_without(clm_output, rule_id, ignore_attr):
    rule_dict = clm_output.result['rules'][rule_id - 1]
    datalabels = clm_output.result['datalabels']
    row_count = clm_output.data['rows_count']
    maxval = 2 ** row_count - 1

    cedents = {}
    for cedent_name, indices in rule_dict['trace_cedent_dataorder'].items():
        val_lists = rule_dict['traces'][cedent_name]
        cedentval = maxval
        for order, idx in enumerate(indices):
            if cedent_name == 'ante' and datalabels['varname'][idx] == ignore_attr:
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


def _cats_match(clm_output, rule_no, cand_no, cedent, variables):
    for var in variables:
        cats1 = set(clm_output.get_rule_categories(rule_no, cedent, var, get_names=True))
        cats2 = set(clm_output.get_rule_categories(cand_no, cedent, var, get_names=True))
        if cats1 != cats2:
            return False
    return True


def _find_simpler(clm_output, rule_no, ante, succ, removed_attr):
    expected_ante = [a for a in ante if a != removed_attr]
    for j in range(clm_output.get_rulecount()):
        cand = j + 1
        if cand == rule_no:
            continue
        cand_ante = list(clm_output.get_rule_variables(cand, 'ante', get_names=True))
        cand_succ = list(clm_output.get_rule_variables(cand, 'succ', get_names=True))
        if set(cand_ante) != set(expected_ante) or set(cand_succ) != set(succ):
            continue
        if not _cats_match(clm_output, rule_no, cand, 'succ', succ):
            continue
        if not _cats_match(clm_output, rule_no, cand, 'ante', expected_ante):
            continue
        return cand
    return None


def run(clm_output):
    """
    Apply RIPPER-style pruning metric per attribute to 4ft-Miner rules.

    Returns dict with n_input, n_output, n_suppressed, rules.
    """
    rule_count = clm_output.get_rulecount()
    suppressed = set()

    for rule_no in range(1, rule_count + 1):
        ante = list(clm_output.get_rule_variables(rule_no, 'ante', get_names=True))
        succ = list(clm_output.get_rule_variables(rule_no, 'succ', get_names=True))
        m_full = _metric(clm_output.result['rules'][rule_no - 1]['params']['fourfold'])

        for attr in ante:
            if _metric(_get_ff_without(clm_output, rule_no, attr)) >= m_full:
                if _find_simpler(clm_output, rule_no, ante, succ, attr) is not None:
                    suppressed.add(rule_no)
                    break

    kept = [i for i in range(1, rule_count + 1) if i not in suppressed]
    rules = []
    for rid in kept:
        rd = clm_output.result['rules'][rid - 1]
        ff = rd['params']['fourfold']
        conf = round(ff[0] / (ff[0] + ff[1]), 3) if (ff[0] + ff[1]) > 0 else 0.0
        rules.append({
            'rule_id': rid,
            'ante_str': rd['cedents_str']['ante'],
            'succ_str': rd['cedents_str']['succ'],
            'base': ff[0],
            'conf': conf,
        })

    return {
        'n_input': rule_count,
        'n_output': len(kept),
        'n_suppressed': len(suppressed),
        'rules': rules,
    }
