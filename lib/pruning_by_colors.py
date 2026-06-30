import sys
import os
import copy
sys.path.insert(0, os.path.dirname(__file__))

from action_literals import action_literals

red_threshold = action_literals.red_threshold
yellow_threshold = action_literals.yellow_threshold



def prune_by_colors(clm_output, prune_yellow=False):
    """
    Returns a set of rule IDs to hide (Phase 1 only).

    A rule is hidden when it has at least one antecedent attribute whose color
    is within the pruning threshold AND a simpler rule already exists in the
    result set that covers the same succedent without that attribute (making
    the complex rule redundant).

    prune_yellow=False (default): prune only RED attributes
    prune_yellow=True:            also prune YELLOW attributes

    Intended to be passed as the ignore_rules argument of print_rulelist().
    """
    threshold = yellow_threshold if prune_yellow else red_threshold

    al = action_literals(clm_output)
    rule_count = clm_output.get_rulecount()
    ignore_rule_ids = set()

    for i in range(rule_count):
        rule_no = i + 1
        ante = clm_output.get_rule_variables(rule_no, 'ante', get_names=True)
        succ = clm_output.get_rule_variables(rule_no, 'succ', get_names=True)

        ff_full = al._get_rule_ff(rule_id=rule_no)
        for attr in ante:
            ff_without = al._get_rule_ff(rule_id=rule_no, cedent='ante', ignore_var=attr)
            z_score, _, _ = al.calc_z_score(ff_full, ff_without)

            if abs(z_score) <= threshold:
                if _simpler_rule_exists(clm_output, rule_no, ante, succ, attr):
                    ignore_rule_ids.add(rule_no)
                    break

    return ignore_rule_ids


def prune_by_colors_detailed(clm_output, prune_yellow=False):
    """
    Returns a dict mapping every rule_id to (suppressed_by, suppressed_by_attr).

    Kept rules         : rule_id -> (None, None)
    Suppressed rules   : rule_id -> (simpler_rule_id, attr_name)

    suppressed_by_attr is the antecedent attribute that was RED/YELLOW and
    (by settings) whose removal matched an existing simpler rule.
    """
    threshold = yellow_threshold if prune_yellow else red_threshold
    al = action_literals(clm_output)
    rule_count = clm_output.get_rulecount()
    result = {}

    for i in range(rule_count):
        rule_no = i + 1
        ante = clm_output.get_rule_variables(rule_no, 'ante', get_names=True)
        succ = clm_output.get_rule_variables(rule_no, 'succ', get_names=True)
        suppressed_by = None
        suppressed_by_attr = None

        ff_full = al._get_rule_ff(rule_id=rule_no)
        for attr in ante:
            ff_without = al._get_rule_ff(rule_id=rule_no, cedent='ante', ignore_var=attr)
            z_score, _, _ = al.calc_z_score(ff_full, ff_without)

            if abs(z_score) <= threshold:
                simpler_id = _find_simpler_rule(clm_output, rule_no, ante, succ, attr)
                if simpler_id is not None:
                    suppressed_by = simpler_id
                    suppressed_by_attr = attr
                    #stop at the first such attribute found
                    break

        result[rule_no] = (suppressed_by, suppressed_by_attr)

    return result


def derive_and_prune(clm_output, prune_yellow=False):
    """
    Two-phase pruning with rule injection. 

    Phase 1: suppress mined rules where a simpler mined rule already exists.
    Phase 2: for a rule's red attribute, inject a derived rule by removing that
             attribute, then repeat until no change is achieved.

    Prints the rulelist at stages for comparison:
      1. Original (all mined rules)
      2. After Phase 1 (in-ruleset suppression only)
      3. After Phase 2 (with derived rules, full suppression)

    Returns (phase1_ignore, phase2_ignore, derived_ids).
    """
    threshold = yellow_threshold if prune_yellow else red_threshold
    al = action_literals(clm_output)
    original_count = clm_output.get_rulecount()

    # Print all mined rules
    print("\n" + "=" * 60)
    print("ORIGINAL (all mined rules)")
    print("=" * 60)
    al.print_rulelist()

    # Phase 1: in-ruleset hiding only
    phase1_ignore = set()
    phase1_detail = {}  # rule_no -> (simpler_id, attr)

    for i in range(original_count):
        rule_no = i + 1
        ante = clm_output.get_rule_variables(rule_no, 'ante', get_names=True)
        succ = clm_output.get_rule_variables(rule_no, 'succ', get_names=True)

        ff_full = al._get_rule_ff(rule_id=rule_no)
        for attr in ante:
            ff_without = al._get_rule_ff(rule_id=rule_no, cedent='ante', ignore_var=attr)
            z_score, _, _ = al.calc_z_score(ff_full, ff_without)

            if abs(z_score) <= threshold:
                simpler_id = _find_simpler_rule(clm_output, rule_no, ante, succ, attr)
                if simpler_id is not None:
                    phase1_ignore.add(rule_no)
                    phase1_detail[rule_no] = (simpler_id, attr)
                    break

    print("\n" + "=" * 60)
    print("AFTER PHASE 1 (in-ruleset pruning only)")
    print("=" * 60)
    al.print_rulelist(ignore_rules=phase1_ignore)
    if phase1_detail:
        print("Suppressed in Phase 1:")
        for rno, (by_id, attr) in sorted(phase1_detail.items()):
            print(f"  Rule {rno:>4} suppressed by rule {by_id:>4} (removed: {attr})")

    # Phase 2: out of ruleset pruning
    phase2_ignore = set()
    phase2_detail = {}  # rule_no -> (by_id, attr)
    derived_ids = []

    while True:
        changed = False
        current_count = clm_output.get_rulecount()
        all_suppress = phase1_ignore | phase2_ignore

        for i in range(current_count):
            rule_no = i + 1
            if rule_no in all_suppress:
                continue

            ante = clm_output.get_rule_variables(rule_no, 'ante', get_names=True)
            succ = clm_output.get_rule_variables(rule_no, 'succ', get_names=True)

            if len(ante) <= 1:
                continue  # cannot simplify a single-attribute antecedent

            red_attrs = _find_red_attrs(al, rule_no, ante, threshold)
            if not red_attrs:
                continue

            # Step 1: prefer an existing non-suppressed simpler form, so we need to look whether it exists
            suppressed = False
            for attr in red_attrs:
                simpler_id = _find_simpler_rule(
                    clm_output, rule_no, ante, succ, attr,
                    exclude_ids=all_suppress
                )
                if simpler_id is not None:
                    phase2_ignore.add(rule_no)
                    phase2_detail[rule_no] = (simpler_id, attr)
                    all_suppress.add(rule_no)
                    changed = True
                    suppressed = True
                    break

            if suppressed:
                continue

            # Step 2: no existing simpler form — so we need to construct it 
            first_red = red_attrs[0]
            # Re-check in case that another rule that was derived during this pass already covers it
            simpler_id = _find_simpler_rule(
                clm_output, rule_no, ante, succ, first_red,
                exclude_ids=all_suppress
            )
            if simpler_id is not None:
                phase2_ignore.add(rule_no)
                phase2_detail[rule_no] = (simpler_id, first_red)
                all_suppress.add(rule_no)
            else:
                new_id = _inject_derived_rule(clm_output, rule_no, first_red)
                derived_ids.append(new_id)
                phase2_ignore.add(rule_no)
                phase2_detail[rule_no] = (new_id, first_red)
                all_suppress.add(rule_no)
            changed = True

        if not changed:
            break

    print("\n" + "=" * 60)
    print("AFTER PHASE 2 (with derived rules, fixed-point)")
    print("=" * 60)
    final_ignore = phase1_ignore | phase2_ignore
    al.print_rulelist(ignore_rules=final_ignore)

    # Show which listed rules are derived (to compute statistisc)
    final_derived = [did for did in derived_ids if did not in final_ignore]
    if final_derived:
        print("Derived rules visible in the listing above (injected, not mined):")
        for did in final_derived:
            rule = clm_output.result['rules'][did - 1]
            ext = rule['params'].get('extensions', {})
            print(f"  Rule {did:>4}: derived from rule "
                  f"{ext.get('derived_from', '?')} "
                  f"by removing {ext.get('derived_by_removing', [])}")

    if phase2_detail:
        print("\nPhase 2 suppression/insertion chain:")
        for rno, (by_id, attr) in sorted(phase2_detail.items()):
            origin = clm_output.result['rules'][by_id - 1]['params'].get('origin', 'mined')
            print(f"  Rule {rno:>4} -> rule {by_id:>4} [{origin}] (removed: {attr})")

    if derived_ids:
        print("\nAll derived rules inserted (including intermediate):")
        for did in derived_ids:
            rule = clm_output.result['rules'][did - 1]
            ante_str = rule['cedents_str']['ante']
            succ_str = rule['cedents_str']['succ']
            ext = rule['params'].get('extensions', {})
            if did in final_ignore:
                status = f"superseded by rule {phase2_detail.get(did, ('?',))[0]}"
            else:
                status = "final (shown above)"
            print(f"  Rule {did:>4}: {ante_str} => {succ_str}  [{status}]")

    return phase1_ignore, phase2_ignore, derived_ids


# ------------------------------------------------------------------ #
# Helpers                                                              #
# ------------------------------------------------------------------ #

def _find_red_attrs(al, rule_no, ante, threshold):
    """Returns antecedent attributes with abs(z) <= threshold, in antecedent order."""
    ff_full = al._get_rule_ff(rule_id=rule_no)
    red = []
    for attr in ante:
        ff_without = al._get_rule_ff(rule_id=rule_no, cedent='ante', ignore_var=attr)
        z_score, _, _ = al.calc_z_score(ff_full, ff_without)
        if abs(z_score) <= threshold:
            red.append(attr)
    return red


def _render_cedent_str(clm_output, rule_dict, cedent):
    """Tools for CleverMiner dict processing/insertion. Re-render cedents_str for one cedent from traces + datalabels."""
    datalabels = clm_output.result['datalabels']
    indices = rule_dict['trace_cedent_dataorder'][cedent]
    val_lists = rule_dict['traces'][cedent]

    if not indices:
        return '---'

    parts = []
    for order, idx in enumerate(indices):
        vn = datalabels['varname'][idx]
        cat_names = [str(datalabels['catnames'][idx][vi]) for vi in val_lists[order]]
        parts.append(vn + '(' + ' '.join(cat_names) + ')')

    return ' & '.join(parts)


def _inject_derived_rule(clm_output, parent_rule_id, remove_attr):
    """
    Tools for CleverMiner dict processing/insertion. Injects a new rule equal to parent_rule minus remove_attr in ante.

    Sets params['origin'] = 'derived' and params['extensions'] for traceability. The derived_by_removing list accumulates the full
    chain of removed attributes if the parent is itself derived. This is stored in supplementary fields of CleverMiner library.
    Will be replaced by CleverMiner native support in the future.

    Returns the new rule_id.
    """
    datalabels = clm_output.result['datalabels']
    parent = clm_output.result['rules'][parent_rule_id - 1]

    ante_indices = parent['trace_cedent_dataorder']['ante']
    attr_pos = None
    for pos, idx in enumerate(ante_indices):
        if datalabels['varname'][idx] == remove_attr:
            attr_pos = pos
            break

    if attr_pos is None:
        raise ValueError(
            f"Attribute '{remove_attr}' not found in ante of rule {parent_rule_id}"
        )

    new_rule = copy.deepcopy(parent)

    new_rule['trace_cedent_dataorder']['ante'] = [
        idx for pos, idx in enumerate(ante_indices) if pos != attr_pos
    ]
    new_rule['traces']['ante'] = [
        vl for pos, vl in enumerate(parent['traces']['ante']) if pos != attr_pos
    ]

    new_rule['cedents_str']['ante'] = _render_cedent_str(clm_output, new_rule, 'ante')

    # Recompute fourfold table via parent traces with remove_attr ignored
    al = action_literals(clm_output)
    ff = al._get_rule_ff(rule_id=parent_rule_id, cedent='ante', ignore_var=remove_attr)

    total = sum(ff)
    base = ff[0]
    conf = ff[0] / (ff[0] + ff[1]) if (ff[0] + ff[1]) > 0 else 0.0
    p_succ = (ff[0] + ff[2]) / total if total > 0 else 0.0
    aad = (conf / p_succ - 1) if (p_succ > 0 and base > 0) else 0.0
    p_not_ante = ff[2] / (ff[2] + ff[3]) if (ff[2] + ff[3]) > 0 else 0.0
    bad = (p_not_ante / p_succ - 1) if p_succ > 0 else 0.0
    rel_base = base / clm_output.data['rows_count'] if clm_output.data['rows_count'] > 0 else 0.0

    new_rule['params']['fourfold'] = ff
    new_rule['params']['base'] = base
    new_rule['params']['conf'] = conf
    new_rule['params']['aad'] = aad
    new_rule['params']['bad'] = bad
    new_rule['params']['rel_base'] = rel_base

    # rule_id = position-based (len before append + 1)
    new_id = len(clm_output.result['rules']) + 1
    new_rule['rule_id'] = new_id

    # Build removal chain from parent's chain - we are kkeping the entire chain for easy traceability
    parent_chain = list(
        parent['params'].get('extensions', {}).get('derived_by_removing', [])
    )
    parent_chain.append(remove_attr)

    new_rule['params']['origin'] = 'derived'
    new_rule['params']['extensions'] = {
        'derived_from': parent_rule_id,
        'derived_by_removing': parent_chain,
    }

    clm_output.result['rules'].append(new_rule)
    return new_id


def _find_simpler_rule(clm_output, rule_no, ante, succ, removed_attr,
                       exclude_ids=None):
    """
    Returns the rule_id of the first non-excluded rule with the same succedent and antecedent equal to ante minus removed_attr (values identical), or return None if not such rule.
    """
    if exclude_ids is None:
        exclude_ids = set()
    expected_ante = [a for a in ante if a != removed_attr]
    rule_count = clm_output.get_rulecount()

    for j in range(rule_count):
        candidate_no = j + 1
        if candidate_no == rule_no or candidate_no in exclude_ids:
            continue

        cand_succ = clm_output.get_rule_variables(candidate_no, 'succ', get_names=True)
        cand_ante = clm_output.get_rule_variables(candidate_no, 'ante', get_names=True)

        if set(cand_succ) != set(succ):
            continue
        if set(cand_ante) != set(expected_ante):
            continue
        if not _categories_match(clm_output, rule_no, candidate_no, 'succ', succ):
            continue
        if not _categories_match(clm_output, rule_no, candidate_no, 'ante', expected_ante):
            continue

        return candidate_no

    return None


def _simpler_rule_exists(clm_output, rule_no, ante, succ, removed_attr):
    """Returns True if there is a simpler rule (used by prune_by_colors)."""
    return _find_simpler_rule(clm_output, rule_no, ante, succ, removed_attr) is not None


def _categories_match(clm_output, rule_no, candidate_no, cedent, variables):
    for var in variables:
        cats1 = clm_output.get_rule_categories(rule_no, cedent, var, get_names=True)
        cats2 = clm_output.get_rule_categories(candidate_no, cedent, var, get_names=True)
        if set(cats1) != set(cats2):
            return False
    return True

