"""
Hämäläinen (2014) statistical significance filter for association rules.

Reference: Hämäläinen & Nykänen (2014) — "Assessing the Statistical
Significance of Association Rules." arXiv:1405.1360.

Criterion: suppress rule R: X->Y if the upper-tail binomial p-value
    p = P(M >= base(R)),  M ~ Bin(n, P(X)·P(Y))
exceeds alpha. Rules with p <= alpha are retained as statistically significant.

Whole rule test only.

"""

from scipy.stats import binom as _binom


def _binom_pvalue(ff):
    """P(M >= a) under M ~ Bin(n, P(X)*P(Y)) (Eq. 2 in Hämäläinen 2014)."""
    a, b, c, d = ff[0], ff[1], ff[2], ff[3]
    n = a + b + c + d
    if n == 0 or a == 0:
        return 1.0
    p_x = (a + b) / n
    p_y = (a + c) / n
    p_indep = p_x * p_y
    if p_indep <= 0.0:
        return 1.0
    return float(_binom.sf(a - 1, n, p_indep))


def run(clm_output, alpha=0.05):
    """
    Suppress rules that fail the whole-rule binomial significance test.

    Parameters
    ----------
    alpha : float
        Significance level. p > alpha → suppressed.

    Returns dict with n_input, n_output, n_suppressed, alpha, rules.
    """
    rule_count = clm_output.get_rulecount()
    suppressed = set()

    for rule_no in range(1, rule_count + 1):
        ff = clm_output.result['rules'][rule_no - 1]['params']['fourfold']
        if _binom_pvalue(ff) > alpha:
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
