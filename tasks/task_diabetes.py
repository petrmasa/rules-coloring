
from lib.action_literals import action_literals
from lib.datasets import datasets

from cleverminer import cleverminer


ds = datasets()

df = ds.load_dataset('diabetes')

clm = cleverminer(df=df,proc='4ftMiner',
               quantifiers= {'Base':50, 'conf':0.05},
               ante ={
                    'attributes':[
                        {'name': 'Pregnancies', 'type': 'seq', 'minlen': 1, 'maxlen': 1},
                        {'name': 'Glucose', 'type': 'seq', 'minlen': 1, 'maxlen': 4},
                        {'name': 'BloodPressure', 'type': 'seq', 'minlen': 1, 'maxlen': 3},
                        {'name': 'SkinThickness', 'type': 'subset', 'minlen': 1, 'maxlen': 1},
                        {'name': 'Insulin', 'type': 'subset', 'minlen': 1, 'maxlen': 1},
                        {'name': 'BMI', 'type': 'seq', 'minlen': 1, 'maxlen': 2}
                    ], 'minlen':1, 'maxlen':4, 'type':'con'},
               succ ={
                    'attributes':[
                        {'name': 'Outcome', 'type': 'rcut', 'minlen': 1, 'maxlen': 1}
                    ], 'minlen':1, 'maxlen':1 , 'type':'con'}
               )



clm2 = clm



clm2.print_rulelist()
clm2.print_summary()
clm2.print_rule(62)



al = action_literals(clm2)
rule_id=62
clm2.print_rule(rule_id)
al.print_rule_literal_importance(rule_id)
al.print_rulelist()









