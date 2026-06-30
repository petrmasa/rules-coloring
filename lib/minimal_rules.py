def filter_cleverminer_rules(clm_output):
    """
    Filters rules from a cleverminer result set.
    A rule is 'ignored' if there is a simpler rule (subset) that covers it.
    """
    # Extract the list of rules from the cleverminer object
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
                                #print("TODO: CHECK CONFIDENCES") #DONE
                                conf1 = clm_output.get_quantifiers(rule_no)['conf']
                                conf2 = clm_output.get_quantifiers(rule2_no)['conf']
                                if conf2>conf1:
                                    print(f"SUCCESS: rule {rule_no} can be thrown out due to {rule2_no}")
                                    ignored_rule_ids.add(rule_no)
                                else:
                                    pass#print(f"CONFIDENCE LOWER")


                        else:
                            pass#print(f"Rules {rule_no} vs {rule2_no} : antecedent #1 is not a superset of antecedent #2 - {ante} vs {ante2}")
    return ignored_rule_ids


# Example usage:
# clm = cleverminer(df=my_df, target='Target', ...)
# filtered_rules = filter_cleverminer_rules(clm)
# for r in filtered_rules:
#     if not r['ignored']:
#         print(f"Kept Rule: {r['rule_text']}")