from lib.action_literals import action_literals
from lib.datasets import datasets

from cleverminer import cleverminer

ds = datasets()

df = ds.load_dataset('titanic')

clm = cleverminer(df=df,proc='4ftMiner',
               quantifiers= {'Base':50, 'conf':0.8},
               ante ={
                    'attributes':[
                        {'name': 'Pclass', 'type': 'seq', 'minlen': 1, 'maxlen': 1},
                        {'name': 'Age', 'type': 'seq', 'minlen': 1, 'maxlen': 4},
                        {'name': 'SibSp', 'type': 'seq', 'minlen': 1, 'maxlen': 3},
                        {'name': 'Parch', 'type': 'subset', 'minlen': 1, 'maxlen': 1},
                        {'name': 'Sex', 'type': 'subset', 'minlen': 1, 'maxlen': 1},
                        {'name': 'Fare', 'type': 'seq', 'minlen': 1, 'maxlen': 2},
                        {'name': 'Embarked', 'type': 'seq', 'minlen': 1, 'maxlen': 2}
                    ], 'minlen':1, 'maxlen':4, 'type':'con'},
               succ ={
                    'attributes':[
                        {'name': 'Survived', 'type': 'rcut', 'minlen': 1, 'maxlen': 1}
                    ], 'minlen':1, 'maxlen':1 , 'type':'con'}
               )


clm.print_data_definition()

clm2 = clm



clm2.print_rulelist()
clm2.print_summary()
clm2.print_rule(62)




al = action_literals(clm2)
rule_id=62
clm2.print_rule(rule_id)
al.print_rule_literal_importance(rule_id)
al.print_rulelist()









