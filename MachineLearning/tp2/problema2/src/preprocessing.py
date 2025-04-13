import pandas as pd
import plotly.express as px
import math
import numpy as np
from collections import Counter
import random

def null_analysis(df):
    '''
    desc: get nulls for each column in counts & percentages, treating '???' as missing in CellType
    arg: dataframe
    return: dataframe
    '''
    assert isinstance(df, pd.DataFrame)
    
    # Calculate null counts for all columns using isna()
    null_cnt = df.isna().sum()
    
    # Specifically for 'CellType', add the count of '???' to the null count
    if 'CellType' in df.columns:
        # Count '???' occurrences in CellType
        question_mark_count = (df['CellType'] == '???').sum()
        # Add to the existing NaN count for CellType
        null_cnt['CellType'] = null_cnt['CellType'] + question_mark_count
    
    # Remove columns with no nulls
    null_cnt = null_cnt[null_cnt != 0]
    
    # Calculate null percentages
    null_percent = null_cnt / len(df) * 100
    
    # Create the result table
    null_table = pd.concat([pd.DataFrame(null_cnt), pd.DataFrame(null_percent)], axis=1)
    null_table.columns = ['counts', 'percentage']
    null_table.sort_values('counts', ascending=False, inplace=True)
    
    return null_table

def boxplot_outlier_removal(X, q1=0.25, q3=0.75, exclude=['']):
    '''
    remove outliers detected by boxplot (Q1/Q3 -/+ IQR*1.5)

    Parameters
    ----------
    X : dataframe
      dataset to remove outliers from
    exclude : list of str
      column names to exclude from outlier removal

    Returns
    -------
    X : dataframe
      dataset with outliers removed
    '''
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
    """
    Performs one-hot encoding on a specified column in a DataFrame using pandas.
    
    Parameters:
    df (pandas.DataFrame): Input DataFrame
    column_name (str): Name of the column to one-hot encode
    prefix (str, optional): Prefix for the new column names (default: column_name)
    drop_original (bool, optional): Whether to drop the original column (default: True)
    handle_nan (bool, optional): Whether to include a column for NaN values (default: True)
    
    Returns:
    pandas.DataFrame: DataFrame with the specified column one-hot encoded
    """
    # Validate input
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame")
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in DataFrame")
    
    # Create a copy of the DataFrame to avoid modifying the original
    df_encoded = df.copy()
    
    # Set prefix for new columns
    prefix = column_name if prefix is None else prefix
    
    # Perform one-hot encoding with pandas.get_dummies
    # dummy_na=True includes a column for NaN if handle_nan is True
    encoded_cols = pd.get_dummies(df_encoded[column_name], 
                                 prefix=prefix, 
                                #  dummy_na=handle_nan, 
                                 dtype=int)
    
    # Concatenate the encoded columns with the original DataFrame
    df_encoded = pd.concat([df_encoded, encoded_cols], axis=1)
    
    # Drop the original column if requested
    if drop_original:
        df_encoded = df_encoded.drop(columns=[column_name])
    
    return df_encoded

def convert_to_binary(data, column_name):
    """
    Convert a categorical column with 'Presnt' and 'Absnt' to a binary column (1 and 0).
    
    Parameters:
    - data (pd.DataFrame): Input dataset containing the column to convert.
    - column_name (str): Name of the column to convert.
    
    Returns:
    - pd.DataFrame: Dataset with the specified column converted to binary.
    """
    
    # Make a copy of the dataset to avoid modifying the original
    df = data.copy()
    
    # Check if the column exists
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in the dataset.")
    
    # Mapping dictionary for conversion
    mapping = {'Presnt': 1, 'Absnt': 0}
    
    # Convert the column using the mapping
    df[column_name] = df[column_name].map(mapping)
    
    # Check for any values that weren't mapped (e.g., typos or unexpected values)
    if df[column_name].isna().any():
        raise ValueError(f"Column '{column_name}' contains values other than 'Presnt' or 'Absnt'.")
    
    return df

def normalize_columns(data, exclude_cols=None, stats=None):
    """
    Normalize numerical columns in a DataFrame using either provided stats or computed ones.
    
    Parameters:
    data (pandas.DataFrame): Input DataFrame to normalize
    exclude_cols (list, optional): Columns to exclude from normalization
    stats (dict, optional): Dictionary with column names as keys and {'mu': mean, 'sigma': std} as values
    
    Returns:
    tuple: (normalized DataFrame, stats dictionary with mean and std for each column)
           If stats is provided, the returned stats will be empty unless computed internally.
    """
    assert isinstance(data, pd.DataFrame)
    df = data.copy()
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    
    if exclude_cols:
        numerical_cols = [col for col in numerical_cols if col not in exclude_cols]
    
    computed_stats = {}
    
    # If stats is provided, use it; otherwise, compute the stats
    for col in numerical_cols:
        if stats is not None and col in stats:
            mu = stats[col]['mu']
            sigma = stats[col]['sigma']
            if sigma == 0:
                raise ValueError(f"Provided standard deviation for column '{col}' is zero.")
        else:
            # Compute mean and std if not provided
            mu = df[col].mean()
            sigma = df[col].std()
            if sigma == 0:
                raise ValueError(f"Column '{col}' has zero standard deviation.")
            computed_stats[col] = {'mu': mu, 'sigma': sigma}
        
        # Normalize the column
        df[col] = (df[col] - mu) / sigma
    
    return df, computed_stats

import pandas as pd
import numpy as np
from collections import Counter

def knn_impute_all_optimized(df, k=3):
    df = df.copy()
    
    # Identify categorical columns
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    # Encode categorical columns
    encoders, decoders = {}, {}
    for col in cat_cols:
        unique = df[col].dropna().unique()
        encoder = {val: i for i, val in enumerate(unique)}
        decoder = {i: val for val, i in encoder.items()}
        df[col] = df[col].map(encoder)
        encoders[col], decoders[col] = encoder, decoder

    # Convert DataFrame to NumPy array for fast computation
    data_np = df.to_numpy()
    n_rows, n_cols = data_np.shape
    col_names = df.columns.tolist()

    # Get column indices
    col_index_map = {col: i for i, col in enumerate(col_names)}
    cat_col_indices = [col_index_map[c] for c in cat_cols]
    nan_cols = np.any(pd.isna(df), axis=0)

    for target_idx, is_nan_col in enumerate(nan_cols):
        if not is_nan_col:
            continue
        
        target_col = col_names[target_idx]
        is_cat = target_idx in cat_col_indices

        missing_rows = np.isnan(data_np[:, target_idx])
        known_rows = ~missing_rows

        X_known = data_np[known_rows][:, [i for i in range(n_cols) if i != target_idx]]
        y_known = data_np[known_rows, target_idx]
        X_missing = data_np[missing_rows][:, [i for i in range(n_cols) if i != target_idx]]
        missing_indices = np.where(missing_rows)[0]

        # Feature types (categorical/numerical)
        feat_types = [
            'categorical' if i in cat_col_indices else 'numerical'
            for i in range(n_cols) if i != target_idx
        ]

        # Impute each missing row
        for i, row in enumerate(X_missing):
            mask = ~np.isnan(row)
            if not np.any(mask):
                continue

            # Compute distances to known rows
            diffs = X_known[:, mask] - row[mask]
            dists = np.zeros(len(X_known))
            for j, col_type in enumerate(np.array(feat_types)[mask]):
                if col_type == 'numerical':
                    dists += diffs[:, j] ** 2
                else:  # categorical
                    dists += diffs[:, j] != 0
            dists = np.sqrt(dists)

            # Get k nearest neighbors
            k_idx = np.argpartition(dists, k)[:k]
            neighbors = y_known[k_idx]

            # Impute
            if is_cat:
                imputed = Counter(neighbors[~np.isnan(neighbors)]).most_common(1)
                if imputed:
                    data_np[missing_indices[i], target_idx] = imputed[0][0]
            else:
                valid_vals = neighbors[~np.isnan(neighbors)]
                if len(valid_vals) > 0:
                    data_np[missing_indices[i], target_idx] = np.mean(valid_vals)

    # Convert back to DataFrame
    df_imputed = pd.DataFrame(data_np, columns=col_names)

    # Decode categorical columns
    for col in cat_cols:
        col_idx = col_index_map[col]
        decoder = decoders[col]
        df_imputed[col] = df_imputed[col].round().astype(int).map(decoder)

    return df_imputed