import pandas as pd
import plotly.express as px
import numpy as np
from collections import Counter


def null_analysis(df):
    assert isinstance(df, pd.DataFrame)
    
    null_cnt = df.isna().sum()
    
    if 'CellType' in df.columns:
        question_mark_count = (df['CellType'] == '???').sum()
        null_cnt['CellType'] = null_cnt['CellType'] + question_mark_count
    
    null_cnt = null_cnt[null_cnt != 0]
    
    null_percent = null_cnt / len(df) * 100
    
    null_table = pd.concat([pd.DataFrame(null_cnt), pd.DataFrame(null_percent)], axis=1)
    null_table.columns = ['counts', 'percentage']
    null_table.sort_values('counts', ascending=False, inplace=True)
    
    return null_table

def boxplot_outlier_removal(X, q1=0.25, q3=0.75, exclude=['']):
    before = len(X)

    # iterate each column
    for col in X.columns:
        if col not in exclude:
            # get Q1, Q3 & Interquantile Range
            Q1 = X[col].quantile(q1)
            Q3 = X[col].quantile(q3)
            IQR = Q3 - Q1
            # define outliers and remove them
            filter_ = (X[col] > Q1 - 1.5 * IQR) & (X[col] < Q3 + 1.5 *IQR)
            X = X[filter_]

    after = len(X)
    diff = before-after
    percent = diff/before*100
    print('{} ({:.2f}%) outliers removed'.format(diff, percent))
    return X

def replace_question_marks_with_nan(df, column):
    df = df.copy()
    df[column] = df[column].replace('???', np.nan)
    return df


def one_hot_encode_column(df, column_name, prefix=None, drop_original=True, handle_nan=True):
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame")
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in DataFrame")
    
    df_encoded = df.copy()
    
    prefix = column_name if prefix is None else prefix
    
    encoded_cols = pd.get_dummies(df_encoded[column_name], 
                                 prefix=prefix, 
                                 dtype=int)
    
    df_encoded = pd.concat([df_encoded, encoded_cols], axis=1)
    
    if drop_original:
        df_encoded = df_encoded.drop(columns=[column_name])
    
    return df_encoded

def convert_to_binary(data, column_name):

    df = data.copy()
    

    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in the dataset.")
    
    mapping = {'Presnt': 1, 'Absnt': 0}
    
    df[column_name] = df[column_name].map(mapping)
    
    if df[column_name].isna().any():
        raise ValueError(f"Column '{column_name}' contains values other than 'Presnt' or 'Absnt'.")
    
    return df

def normalize_columns(data, exclude_cols=None, stats=None):
    assert isinstance(data, pd.DataFrame)
    df = data.copy()
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    
    if exclude_cols:
        numerical_cols = [col for col in numerical_cols if col not in exclude_cols]
    
    computed_stats = {}
    
    for col in numerical_cols:
        if stats is not None and col in stats:
            mu = stats[col]['mu']
            sigma = stats[col]['sigma']
            if sigma == 0:
                raise ValueError(f"Provided standard deviation for column '{col}' is zero.")
        else:
            mu = df[col].mean()
            sigma = df[col].std()
            if sigma == 0:
                raise ValueError(f"Column '{col}' has zero standard deviation.")
            computed_stats[col] = {'mu': mu, 'sigma': sigma}
        
        df[col] = (df[col] - mu) / sigma
    
    return df, computed_stats

import pandas as pd
import numpy as np
from collections import Counter

def knn_impute_all_optimized(df, k=3):
    df = df.copy()
    
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    encoders, decoders = {}, {}
    for col in cat_cols:
        unique = df[col].dropna().unique()
        encoder = {val: i for i, val in enumerate(unique)}
        decoder = {i: val for val, i in encoder.items()}
        df[col] = df[col].map(encoder)
        encoders[col], decoders[col] = encoder, decoder

    data_np = df.to_numpy()
    n_rows, n_cols = data_np.shape
    col_names = df.columns.tolist()

    col_index_map = {col: i for i, col in enumerate(col_names)}
    cat_col_indices = [col_index_map[c] for c in cat_cols]
    nan_cols = np.any(pd.isna(df), axis=0)

    for target_idx, is_nan_col in enumerate(nan_cols):
        if not is_nan_col:
            continue
        
        is_cat = target_idx in cat_col_indices

        missing_rows = np.isnan(data_np[:, target_idx])
        known_rows = ~missing_rows

        X_known = data_np[known_rows][:, [i for i in range(n_cols) if i != target_idx]]
        y_known = data_np[known_rows, target_idx]
        X_missing = data_np[missing_rows][:, [i for i in range(n_cols) if i != target_idx]]
        missing_indices = np.where(missing_rows)[0]

        feat_types = [
            'categorical' if i in cat_col_indices else 'numerical'
            for i in range(n_cols) if i != target_idx
        ]

        for i, row in enumerate(X_missing):
            mask = ~np.isnan(row)
            if not np.any(mask):
                continue

            diffs = X_known[:, mask] - row[mask]
            dists = np.zeros(len(X_known))
            for j, col_type in enumerate(np.array(feat_types)[mask]):
                if col_type == 'numerical':
                    dists += diffs[:, j] ** 2
                else:  # categorical
                    dists += diffs[:, j] != 0
            dists = np.sqrt(dists)

            k_idx = np.argpartition(dists, k)[:k]
            neighbors = y_known[k_idx]

            if is_cat:
                imputed = Counter(neighbors[~np.isnan(neighbors)]).most_common(1)
                if imputed:
                    data_np[missing_indices[i], target_idx] = imputed[0][0]
            else:
                valid_vals = neighbors[~np.isnan(neighbors)]
                if len(valid_vals) > 0:
                    data_np[missing_indices[i], target_idx] = np.mean(valid_vals)

    df_imputed = pd.DataFrame(data_np, columns=col_names)

    for col in cat_cols:
        decoder = decoders[col]
        df_imputed[col] = df_imputed[col].round().astype(int).map(decoder)

    return df_imputed