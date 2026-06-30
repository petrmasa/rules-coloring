"""
Loader for the 22 UCI datasets used in QCBA (Kliegr & Izquierdo 2023) Table 4.
Source: OpenML (mirrors UCI) via sklearn.datasets.fetch_openml.

Each dataset is fetched, imputed, and numeric columns are qcut into 5 bins so
that 4ft-Miner can receive the binned DataFrame directly.
"""

import warnings
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.impute import SimpleImputer

# Int columns with more unique values than this threshold get qcut (seq type);
# those with <= threshold unique values are kept as categorical (subset type).
_SEQ_INT_THRESHOLD = 10

# (comparison name, OpenML dataset name, OpenML version or None for default)
_DATASET_REGISTRY = [
    ('uci_anneal',        'anneal',          1),
    ('uci_australian',    'Australian',      2),   
    ('uci_autos',         'autos',           1),
    ('uci_breast_w',      'breast-w',        1),
    ('uci_colic',         'colic',           1),
    ('uci_credit_a',      'credit-approval', None), 
    ('uci_credit_g',      'credit-g',        1),
    ('uci_diabetes',      'diabetes',        1),
    ('uci_glass',         'glass',           1),
    ('uci_heart_statlog', 'heart-statlog',   1),
    ('uci_hepatitis',     'hepatitis',       1),
    ('uci_hypothyroid',   'hypothyroid',     1),
    ('uci_ionosphere',    'ionosphere',      1),
    ('uci_iris',          'iris',            1),
    ('uci_labor',         'labor',           1),
    ('uci_letter',        'letter',          1),
    ('uci_lymph',         'lymph',           1),
    ('uci_segment',       'segment',         1),
    ('uci_sonar',         'sonar',           1),
    ('uci_spambase',      'spambase',        1),
    ('uci_vehicle',       'vehicle',         1),
    ('uci_vowel',         'vowel',           1),
]


def get_all():
    """Return list of (result_name, openml_name, version) tuples."""
    return list(_DATASET_REGISTRY)


def load_dataset(openml_name, version=None):
    """
    Download, impute and pre-discretise one QCBA benchmark dataset.

    Returns
    -------
    df          : pre-processed DataFrame (all columns in one)
    seq_cols    : feature column names to declare as seq in 4ft-Miner
    sub_cols    : feature column names to declare as subset in 4ft-Miner
    target_col  : name of the target column
    """
    kw = {'version': version} if version is not None else {}
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        bundle = fetch_openml(name=openml_name, as_frame=True, parser='auto', **kw)

    X = bundle.data.copy()
    y = bundle.target.copy()
    target_col = y.name

    # --- Categorise feature columns ---
    # float64 -> seq; int64 with many values -> seq; everything else -> subset
    potential_seq, potential_sub = [], []
    for col in X.columns:
        if pd.api.types.is_float_dtype(X[col]):
            potential_seq.append(col)
        elif pd.api.types.is_integer_dtype(X[col]):
            if X[col].nunique() > _SEQ_INT_THRESHOLD:
                potential_seq.append(col)
            else:
                potential_sub.append(col)
        else:
            potential_sub.append(col)

    # --- Convert non-numeric features and target to str (before imputation) ---
    for col in potential_sub:
        X[col] = X[col].astype(str).replace('nan', np.nan)
    y = y.astype(str).replace('nan', np.nan)

    # Drop feature columns that are entirely NaN (SimpleImputer would drop them
    # silently, causing a shape mismatch when storing into the DataFrame).
    potential_seq = [c for c in potential_seq if X[c].notna().any()]
    potential_sub = [c for c in potential_sub if X[c].notna().any()]

    df = pd.concat([X[potential_seq + potential_sub], y], axis=1)

    # --- Impute missing values ---
    if potential_seq and df[potential_seq].isnull().any().any():
        imputed = SimpleImputer(strategy='mean').fit_transform(df[potential_seq])
        df[potential_seq] = pd.DataFrame(imputed, columns=potential_seq, index=df.index)

    cat_all = potential_sub + [target_col]
    if any(df[c].isnull().any() for c in cat_all):
        imputed = SimpleImputer(strategy='most_frequent').fit_transform(df[cat_all])
        df[cat_all] = pd.DataFrame(imputed, columns=cat_all, index=df.index)

    # --- qcut numeric columns into 5 bins ---
    seq_cols, sub_cols = [], list(potential_sub)
    for col in potential_seq:
        n_uniq = int(df[col].nunique())
        if n_uniq < 2:
            df[col] = df[col].astype(str)
            sub_cols.append(col)
            continue
        try:
            df[col] = pd.qcut(df[col], q=min(5, n_uniq), duplicates='drop')
            seq_cols.append(col)
        except Exception:
            df[col] = df[col].astype(str)
            sub_cols.append(col)

    return df, seq_cols, sub_cols, target_col
