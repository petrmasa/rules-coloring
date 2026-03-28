def filter_cleverminer_rules(clm_output):
    """
    Filters rules from a cleverminer result set.
    A rule is 'ignored' if there is a simpler rule (subset) that covers it.
    """
    # Extract the list of rules from the cleverminer object
    # In newer versions, this is usually clm_output.rules
    rule_count = clm_output.get_rulecount()
    
    # We will store the IDs of rules that should be ignored
    ignored_rule_ids = set()

    for i in range(rule_count):
        rule_no = i+1

        #print(" ")
        #print(f"Cedents: {clm_output.get_rule_cedent_list(rule_no)}")

        succ = clm_output.get_rule_variables(rule_no, 'succ', get_names=True)
        ante = clm_output.get_rule_variables(rule_no, 'ante', get_names=True)

        for j in range(rule_count):
            if not(i==j):
                rule2_no = j+1
                succ2 = clm_output.get_rule_variables(rule2_no, 'succ', get_names=True)
                ante2 = clm_output.get_rule_variables(rule2_no, 'ante', get_names=True)

                if not(succ2==succ):
                    pass #print(f"Rules {rule_no} vs {rule2_no} : succedents are different - {succ} vs {succ2}")
                else:
                    succ_same = True
                    for k in succ:
                        vals1=clm_output.get_rule_categories(rule_no, 'succ', k, get_names=True)
                        vals2 = clm_output.get_rule_categories(rule2_no, 'succ', k, get_names=True)
                        if not(set(vals1)==set(vals2)):
                            #print(f"Values of var #{k} are different - {vals1} vs {vals2}")
                            succ_same = False
                    if succ_same:
                        #print(f"Rules {rule_no} vs {rule2_no} : succedents are THE SAME - {succ} vs {succ2}")
                        if set(ante2) <= set(ante):# and (len(ante)==len(ante2)+1):
                            #print(f"..will check individual values")
                            ante_ss = True
                            for l in ante2: #ante2 is shorter and our subject of interest!
                                vals1 = clm_output.get_rule_categories(rule_no, 'ante', l, get_names=True)
                                vals2 = clm_output.get_rule_categories(rule2_no, 'ante', l, get_names=True)
                                if not (set(vals1) <= set(vals2)):
                                    pass#print(f"Values of var #{l} are different  & rule1 HAS SOME EXTRA -> ... {vals1} vs {vals2}")
                                    ante_ss = False
                                else:
                                    pass#print(f" ... passed for: Values of var #{l}  rule1 HAS NOTHING EXTRA -> ... {vals1} vs {vals2}")
                            if ante_ss:
                                #print(f"BINGO: rule {rule_no} can be thrown out due to {rule2_no}")
                                #print("TODO: CHECK CONFIDENCES")
                                conf1 = clm_output.get_quantifiers(rule_no)['conf']
                                conf2 = clm_output.get_quantifiers(rule2_no)['conf']
                                if conf2>conf1:
                                    print(f"BINGO: rule {rule_no} can be thrown out due to {rule2_no}")
                                    ignored_rule_ids.add(rule_no)
                                else:
                                    pass#print(f"CONFIDENCE LOWER")


                        else:
                            pass#print(f"Rules {rule_no} vs {rule2_no} : antecedent #1 is not a superset of antecedent #2 - {ante} vs {ante2}")
    return ignored_rule_ids

    if 1>0:


        # Extract antecedents and succedents for rule A
        # 'cedent_struct' contains [{'attribute': 'Name', 'categories': [...]}, ...]
        ant_a = rule_a['antecedent_struct']
        suc_a = rule_a['succedent_struct']

        for j, rule_b in enumerate(all_rules):
            if i == j: continue  # Don't compare a rule to itself
            
            ant_b = rule_b['antecedent_struct']
            suc_b = rule_b['succedent_struct']

            # 1. Succedents must be identical
            if suc_a != suc_b:
                continue

            # 2. Compare Antecedents
            # Convert antecedents to dictionaries for easier attribute comparison: {attr: [cats]}
            dict_a = {item['attribute']: set(item['categories']) for item in ant_a}
            dict_b = {item['attribute']: set(item['categories']) for item in ant_b}

            # Check if B is a "simpler" version of A:
            # - B must have same attributes as A, except potentially one missing
            attrs_a = set(dict_a.keys())
            attrs_b = set(dict_b.keys())

            is_subset_attr = attrs_b.issubset(attrs_a) and (len(attrs_a) - len(attrs_b) <= 1)
            
            if is_subset_attr:
                # 3. Check Categories
                # All same attributes must have same categories (or B can have extra categories)
                match = True
                for attr in attrs_b:
                    # Logic: B is simpler if it covers the same ground or more (extra categories)
                    # than the complex rule A.
                    if not dict_a[attr].issubset(dict_b[attr]):
                        match = False
                        break
                
                if match:
                    # Rule A is redundant because Rule B is simpler/more general
                    ignored_rule_ids.add(rule_a['rule_id'])
                    break # No need to check other rules for rule_a

    # Mark rules as ignored in the dataset
    for rule in all_rules:
        rule['ignored'] = rule['rule_id'] in ignored_rule_ids

    return all_rules

# Example usage:
# clm = cleverminer(df=my_df, target='Target', ...)
# filtered_rules = filter_cleverminer_rules(clm)
# for r in filtered_rules:
#     if not r['ignored']:
#         print(f"Kept Rule: {r['rule_text']}")