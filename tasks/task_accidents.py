


from lib.action_literals import action_literals
from lib.datasets import datasets

import pandas as pd

from sklearn.impute import SimpleImputer

from cleverminer import cleverminer

ds = datasets()

df = ds.load_dataset('accidents')


imputer = SimpleImputer(strategy="most_frequent")
df = pd.DataFrame(imputer.fit_transform(df),columns = df.columns)

clm = cleverminer(df=df,proc='4ftMiner',
               quantifiers= {'Base':1000, 'conf':0.045},
               ante ={
                    'attributes':[
                        {'name': 'Driver_Age_Band', 'type': 'seq', 'minlen': 1, 'maxlen': 4},
                        {'name': 'Speed_limit', 'type': 'seq', 'minlen': 1, 'maxlen': 3},
                        {'name': 'Light', 'type': 'subset', 'minlen': 1, 'maxlen': 1},
                        {'name': 'Journey', 'type': 'subset', 'minlen': 1, 'maxlen': 1},
                        {'name': 'Road_Type', 'type': 'subset', 'minlen': 1, 'maxlen': 1},
                        {'name': 'Sex', 'type': 'subset', 'minlen': 1, 'maxlen': 1}
                    ], 'minlen':1, 'maxlen':4, 'type':'con'},
               succ ={
                    'attributes':[
                        {'name': 'Severity', 'type': 'lcut', 'minlen': 1, 'maxlen': 1}
                    ], 'minlen':1, 'maxlen':1 , 'type':'con'}
               )



clm2 = clm


#print original rules and selected rule
clm2.print_rulelist()
clm2.print_summary()
clm2.print_rule(1)


#now, show action literals / attribute and value importance
al = action_literals(clm2)
rule_id=1
clm2.print_rule(rule_id)
al.print_rule_literal_importance(rule_id)
al.print_rulelist()









