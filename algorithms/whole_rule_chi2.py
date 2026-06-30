"""
Whole-rule chi-sq significance filter for association rules.

Reference: first test used in Liu, Hsu & Ma (1999) Pruning and Summarizing the Discovered Associations. ACM SIGKDD 99

Criterion: suppress rule R if the chi-sq test on its fourfold table [[a,b],[c,d]] is not statistically significant (p > alpha). 

"""

from scipy.stats import chi2_contingency as _chi2


def _chi2_p(ff):
    table = [[ff[0], ff[1]], [ff[2], ff[3]]]
    if ff[0] + ff[1] == 0 or ff[2] + ff[3] == 0:
        return 1.0
    _, p, _, _ = _chi2(table, correction=False)
    return p


def run(clm_output, alpha=0.05):
    """
    Suppress rules whose chi-sq test gives p > alpha.

    Parameters
    ----------
    alpha : float
        Significance threshold. p > alpha → suppressed.

    Returns dict with n_input, n_output, n_suppressed, alpha, rules.
    """
    rule_count = clm_output.get_rulecount()
    suppressed = set()

    for rule_no in range(1, rule_count + 1):
        ff = clm_output.result['rules'][rule_no - 1]['params']['fourfold']
        if _chi2_p(ff) > alpha:
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
        'alpha': alpha,
        'rules': rules,
    }
