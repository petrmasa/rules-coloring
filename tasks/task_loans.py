from lib.action_literals import action_literals
from lib.datasets import datasets

from cleverminer import cleverminer

ds = datasets()

df = ds.load_dataset('loans')

clm = cleverminer(df=df,proc='4ftMiner',
               quantifiers= {'Base':50, 'conf':0.8},
               ante ={
                    'attributes':[
                        {'name': ' no_of_dependents', 'type': 'seq', 'minlen': 1, 'maxlen': 1},
                        {'name': ' income_annum', 'type': 'seq', 'minlen': 1, 'maxlen': 4},
                        {'name': ' loan_amount', 'type': 'seq', 'minlen': 1, 'maxlen': 3},
                        {'name': ' loan_term', 'type': 'subset', 'minlen': 1, 'maxlen': 1},
                        {'name': ' education', 'type': 'subset', 'minlen': 1, 'maxlen': 1},
                        {'name': ' residential_assets_value', 'type': 'seq', 'minlen': 1, 'maxlen': 2}
                    ], 'minlen':1, 'maxlen':4, 'type':'con'},
               succ ={
                    'attributes':[
                        {'name': ' loan_status', 'type': 'lcut', 'minlen': 1, 'maxlen': 1}
                    ], 'minlen':1, 'maxlen':1 , 'type':'con'}
               )


clm.print_data_definition()

clm2 = clm



clm2.print_rulelist()
clm2.print_summary()
clm2.print_rule(53)




al = action_literals(clm2)
rule_id=53
clm2.print_rule(rule_id)
al.print_rule_literal_importance(rule_id)
al.print_rulelist()










