"""
delta-Tolerance Association Rules (delta-TARs) pruning.

Reference: Cheng, Ke & Ng (2008) — "Effective Elimination of Redundant
Association Rules." Data Mining and Knowledge Discovery 16, 221–249.

Criterion (Definition 1 in the article - section 4): suppress rule R : A => c if there exists a mined
rule R_spec : A_spec => c with A_spec superset of A (same succedent/categories,
same categories on shared antecedent attributes) such that
    base(R_spec) >= (1 - delta) * base(R).

It is called delta-tolerance in the article

Goes for more specific rule (is base does not decreases too much), somehow different direction that we are going. But worth to compare. 
More rules can be supressed by one specific
"""


def _cats_match(clm_output, rule_no, cand_no, cedent, variables):
    for var in variables:
        cats1 = set(clm_output.get_rule_categories(rule_no, cedent, var, get_names=True))
        cats2 = set(clm_output.get_rule_categories(cand_no, cedent, var, get_names=True))
        if cats1 != cats2:
            return False
    return True


def _find_most_specific_superset(clm_output, rule_no, ante, succ):
    """Return (cand_id, cand_base) for the mined rule with the highest base
    whose antecedent is a proper superset of ante (same succedent/categories),
    or (None, -1) if none exists."""
    ante_set = set(ante)
    succ_set = set(succ)
    best_id = None
    best_base = -1

    for j in range(clm_output.get_rulecount()):
        cand = j + 1
        if cand == rule_no:
            continue
        cand_ante = set(clm_output.get_rule_variables(cand, 'ante', get_names=True))
        cand_succ = set(clm_output.get_rule_variables(cand, 'succ', get_names=True))
        if not (cand_ante > ante_set):
            continue
        if cand_succ != succ_set:
            continue
        if not _cats_match(clm_output, rule_no, cand, 'succ', succ):
            continue
        if not _cats_match(clm_output, rule_no, cand, 'ante', list(ante_set)):
            continue
        base = clm_output.result['rules'][cand - 1]['params']['fourfold'][0]
        if base > best_base:
            best_base = base
            best_id = cand

    return best_id, best_base


def run(clm_output, delta=0.0):
    """
    Apply delta-TAR pruning to all rules in clm_output.

    Parameters
    ----------
    delta : float
        Support tolerance (0 <= delta <= 1). Typical values: 0.0, 0.05, 0.10.

    Returns dict with n_input, n_output, n_suppressed, delta, rules.
    """
    rule_count = clm_output.get_rulecount()
    suppressed = set()

    for rule_no in range(1, rule_count + 1):
        ante = list(clm_output.get_rule_variables(rule_no, 'ante', get_names=True))
        succ = list(clm_output.get_rule_variables(rule_no, 'succ', get_names=True))
        base_r = clm_output.result['rules'][rule_no - 1]['params']['fourfold'][0]
        if base_r <= 0:
            continue
        spec_id, spec_base = _find_most_specific_superset(clm_output, rule_no, ante, succ)
        if spec_id is not None and spec_base >= (1.0 - delta) * base_r:
            suppressed.add(rule_no)

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
        'delta': delta,
        'rules': rules,
    }
