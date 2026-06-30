"""
CLM 4ft-Miner task configurations for the four main paper datasets.

Each entry contains the mining quantifiers, antecedent/succedent spec, and
whether the dataset needs imputation before mining.
"""

DATASETS = {
    'accidents': {
        'quantifiers': {'Base': 1000, 'conf': 0.045},
        'ante': {
            'attributes': [
                {'name': 'Driver_Age_Band', 'type': 'seq',    'minlen': 1, 'maxlen': 4},
                {'name': 'Speed_limit',     'type': 'seq',    'minlen': 1, 'maxlen': 3},
                {'name': 'Light',           'type': 'subset', 'minlen': 1, 'maxlen': 1},
                {'name': 'Journey',         'type': 'subset', 'minlen': 1, 'maxlen': 1},
                {'name': 'Road_Type',       'type': 'subset', 'minlen': 1, 'maxlen': 1},
                {'name': 'Sex',             'type': 'subset', 'minlen': 1, 'maxlen': 1},
            ],
            'minlen': 1, 'maxlen': 4, 'type': 'con',
        },
        'succ': {
            'attributes': [{'name': 'Severity', 'type': 'lcut', 'minlen': 1, 'maxlen': 1}],
            'minlen': 1, 'maxlen': 1, 'type': 'con',
        },
        'impute': True,
    },
    'titanic': {
        'quantifiers': {'Base': 50, 'conf': 0.8},
        'ante': {
            'attributes': [
                {'name': 'Pclass',   'type': 'seq',    'minlen': 1, 'maxlen': 1},
                {'name': 'Age',      'type': 'seq',    'minlen': 1, 'maxlen': 4},
                {'name': 'SibSp',    'type': 'seq',    'minlen': 1, 'maxlen': 3},
                {'name': 'Parch',    'type': 'subset', 'minlen': 1, 'maxlen': 1},
                {'name': 'Sex',      'type': 'subset', 'minlen': 1, 'maxlen': 1},
                {'name': 'Fare',     'type': 'seq',    'minlen': 1, 'maxlen': 2},
                {'name': 'Embarked', 'type': 'seq',    'minlen': 1, 'maxlen': 2},
            ],
            'minlen': 1, 'maxlen': 4, 'type': 'con',
        },
        'succ': {
            'attributes': [{'name': 'Survived', 'type': 'rcut', 'minlen': 1, 'maxlen': 1}],
            'minlen': 1, 'maxlen': 1, 'type': 'con',
        },
        'impute': False,
    },
    'diabetes': {
        'quantifiers': {'Base': 50, 'conf': 0.05},
        'ante': {
            'attributes': [
                {'name': 'Pregnancies',   'type': 'seq',    'minlen': 1, 'maxlen': 1},
                {'name': 'Glucose',       'type': 'seq',    'minlen': 1, 'maxlen': 4},
                {'name': 'BloodPressure', 'type': 'seq',    'minlen': 1, 'maxlen': 3},
                {'name': 'SkinThickness', 'type': 'subset', 'minlen': 1, 'maxlen': 1},
                {'name': 'Insulin',       'type': 'subset', 'minlen': 1, 'maxlen': 1},
                {'name': 'BMI',           'type': 'seq',    'minlen': 1, 'maxlen': 2},
            ],
            'minlen': 1, 'maxlen': 4, 'type': 'con',
        },
        'succ': {
            'attributes': [{'name': 'Outcome', 'type': 'rcut', 'minlen': 1, 'maxlen': 1}],
            'minlen': 1, 'maxlen': 1, 'type': 'con',
        },
        'impute': False,
    },
    'loans': {
        'quantifiers': {'Base': 50, 'conf': 0.8},
        'ante': {
            'attributes': [
                {'name': ' no_of_dependents',        'type': 'seq',    'minlen': 1, 'maxlen': 1},
                {'name': ' income_annum',             'type': 'seq',    'minlen': 1, 'maxlen': 4},
                {'name': ' loan_amount',              'type': 'seq',    'minlen': 1, 'maxlen': 3},
                {'name': ' loan_term',                'type': 'subset', 'minlen': 1, 'maxlen': 1},
                {'name': ' education',                'type': 'subset', 'minlen': 1, 'maxlen': 1},
                {'name': ' residential_assets_value', 'type': 'seq',    'minlen': 1, 'maxlen': 2},
            ],
            'minlen': 1, 'maxlen': 4, 'type': 'con',
        },
        'succ': {
            'attributes': [{'name': ' loan_status', 'type': 'lcut', 'minlen': 1, 'maxlen': 1}],
            'minlen': 1, 'maxlen': 1, 'type': 'con',
        },
        'impute': False,
    },
}

DATASET_ORDER = ['accidents', 'titanic', 'diabetes', 'loans']
