import pandas as pd
import plotly.express as px
import math
import numpy as np
from collections import Counter
import random

def normalize(column: str, dataframe):
    assert isinstance(dataframe, pd.DataFrame)

    col_min = dataframe[column].min()
    col_max = dataframe[column].max()
    if col_max - col_min != 0: # Avoid division by zero
        dataframe[column] = (dataframe[column] - col_min) / (col_max - col_min)

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

def handle_nan_values(df):
    """
    Replaces NaN values in a pandas DataFrame with the median of each column for numeric data,
    and replaces NaN and '???' with 'Unknown' in the CellType column.
    
    Parameters:
    df (pandas.DataFrame): Input DataFrame with possible NaN values
    
    Returns:
    pandas.DataFrame: DataFrame with NaN and '???' handled appropriately
    """
    # Create a copy of the DataFrame to avoid modifying the original
    df_filled = df.copy()
    
    # Iterate through each column
    for column in df_filled.columns:
        # Check if the column has numeric data
        if pd.api.types.is_numeric_dtype(df_filled[column]):
            # Calculate median for the column
            median_value = df_filled[column].median()
            # Replace NaN with median
            df_filled[column] = df_filled[column].fillna(median_value)
        
        # Handle CellType column specifically
        elif column == 'CellType':
            # Replace both NaN and '???' with 'Unknown'
            df_filled[column] = df_filled[column].replace('???', 'Unknown')
            df_filled[column] = df_filled[column].fillna('Unknown')

    return df_filled

def boxplot_outlier_removal(X, exclude=['']):
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
            Q1 = X[col].quantile(0.25)
            Q3 = X[col].quantile(0.75)
            IQR = Q3 - Q1
            # define outliers and remove them
            filter_ = (X[col] > Q1 - 1.5 * IQR) & (X[col] < Q3 + 1.5 *IQR)
            X = X[filter_]

    after = len(X)
    diff = before-after
    percent = diff/before*100
    print('{} ({:.2f}%) outliers removed'.format(diff, percent))
    return X

def replace_nan_with_knn_custom(df, k=5):
    """
    Replaces NaN values in a pandas DataFrame using a custom KNN imputation method.
    
    Parameters:
    df (pandas.DataFrame): Input DataFrame with possible NaN values
    k (int, optional): Number of neighbors to use for imputation (default is 5)
    
    Returns:
    pandas.DataFrame: DataFrame with NaN values replaced by KNN predictions
    """
    # Create a copy of the DataFrame to avoid modifying the original
    df_filled = df.copy()
    
    # Separate numeric and categorical columns
    numeric_cols = df_filled.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df_filled.select_dtypes(exclude=[np.number]).columns.tolist()
    
    # Function to compute Euclidean distance between two rows, ignoring NaN
    def nan_euclidean_distance(row1, row2, cols):
        diff = 0
        count = 0
        for col in cols:
            if not pd.isna(row1[col]) and not pd.isna(row2[col]):
                diff += (row1[col] - row2[col]) ** 2
                count += 1
        return np.sqrt(diff) if count > 0 else np.inf
    
    # Function to find k nearest neighbors for a row
    def find_k_nearest_neighbors(row_idx, df, cols, k):
        distances = []
        target_row = df.iloc[row_idx]
        for idx, row in df.iterrows():
            if idx != row_idx:  # Skip the row itself
                dist = nan_euclidean_distance(target_row, row, cols)
                distances.append((dist, idx))
        distances.sort()  # Sort by distance
        return [idx for _, idx in distances[:k]]
    
    # Encode categorical columns to numeric for distance computation
    df_encoded = df_filled.copy()
    label_encoders = {}
    for col in categorical_cols:
        # Map unique values to integers, NaN remains NaN
        unique_vals = df_encoded[col].dropna().unique()
        mapping = {val: i for i, val in enumerate(unique_vals)}
        df_encoded[col] = df_encoded[col].map(lambda x: mapping[x] if pd.notna(x) else np.nan)
        label_encoders[col] = mapping
    
    # Impute NaN values
    for col in df_encoded.columns:
        # Rows with NaN in this column
        nan_rows = df_encoded[col].isna()
        if nan_rows.any():
            for idx in df_encoded[nan_rows].index:
                # Find k nearest neighbors based on all columns except the one being imputed
                other_cols = [c for c in df_encoded.columns if c != col]
                neighbors_idx = find_k_nearest_neighbors(idx, df_encoded, other_cols, k)
                
                # Get neighbor values for the column
                neighbor_values = df_encoded.loc[neighbors_idx, col].dropna()
                
                if col in numeric_cols and len(neighbor_values) > 0:
                    # For numeric columns, use mean of neighbors
                    df_filled.loc[idx, col] = neighbor_values.mean()
                elif col in categorical_cols and len(neighbor_values) > 0:
                    # For categorical columns, use mode (most frequent value)
                    mode_val = neighbor_values.mode()
                    df_filled.loc[idx, col] = mode_val[0] if not mode_val.empty else np.nan
    
    # Decode categorical columns back to original values
    for col in categorical_cols:
        inverse_mapping = {v: k for k, v in label_encoders[col].items()}
        df_filled[col] = df_filled[col].map(lambda x: inverse_mapping[int(x)] if pd.notna(x) and int(x) in inverse_mapping else x)
    
    return df_filled

def encode_categorical(data, column_names):
    encoders = {}
    for col in column_names:
        unique = list({row[col] for row in data if row[col] is not None})
        encoders[col] = {val: i for i, val in enumerate(unique)}
    return encoders

def compute_distance(a, b, feature_types):
    dist = 0
    for key in a:
        if a[key] is None or b[key] is None:
            continue
        if feature_types[key] == 'categorical':
            dist += 0 if a[key] == b[key] else 1
        else:
            dist += (a[key] - b[key]) ** 2
    return math.sqrt(dist)

def knn_impute_all(df, k=3):
    df = df.copy()
    
    # Detect categorical columns
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    # Label encode categorical columns
    encoders = {}
    decoders = {}
    for col in cat_cols:
        unique = df[col].dropna().unique()
        encoder = {val: i for i, val in enumerate(unique)}
        decoder = {i: val for val, i in encoder.items()}
        encoders[col] = encoder
        decoders[col] = decoder
        df[col] = df[col].map(encoder)

    # Identify columns with missing values
    columns_with_nans = df.columns[df.isna().any()].tolist()

    # Impute each column
    for target_column in columns_with_nans:
        missing_mask = df[target_column].isna()
        df_missing = df[missing_mask]
        df_known = df[~missing_mask]

        feature_cols = [col for col in df.columns if col != target_column]

        # Feature types
        feature_types = {
            col: 'categorical' if col in cat_cols else 'numerical'
            for col in feature_cols
        }

        for idx, row in df_missing.iterrows():
            distances = []
            for _, ref_row in df_known.iterrows():
                dist = 0
                for col in feature_cols:
                    val1 = row[col]
                    val2 = ref_row[col]
                    if pd.isna(val1) or pd.isna(val2):
                        continue
                    if feature_types[col] == 'numerical':
                        dist += (val1 - val2) ** 2
                    else:
                        dist += 0 if val1 == val2 else 1
                distances.append((np.sqrt(dist), ref_row[target_column]))

            if not distances:
                continue

            distances.sort(key=lambda x: x[0])
            k_neighs = [val for _, val in distances[:k]]

            if target_column in cat_cols:
                imputed_value = Counter(k_neighs).most_common(1)[0][0]
            else:
                imputed_value = np.mean(k_neighs)

            df.at[idx, target_column] = imputed_value

    # Decode categorical columns back to original labels
    for col in cat_cols:
        decoder = decoders[col]
        df[col] = df[col].round().astype(int).map(decoder)

    return df

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

def undersample_data(df, target_column, random_seed=42):
    """
    Perform random undersampling on a dataset to balance classes without sklearn.
    
    Parameters:
    df (pandas.DataFrame): Input dataframe containing features and target
    target_column (str): Name of the target column with class labels
    random_seed (int): Seed for reproducibility (default: 42)
    
    Returns:
    pandas.DataFrame: Undersampled dataframe
    """
    # Set random seed for reproducibility
    random.seed(random_seed)
    
    # Separate majority and minority classes
    class_counts = df[target_column].value_counts()
    minority_class = class_counts.idxmin()
    majority_class = class_counts.idxmax()
    
    df_minority = df[df[target_column] == minority_class]
    df_majority = df[df[target_column] == majority_class]
    
    # Get the size of the minority class
    minority_size = len(df_minority)
    
    # Randomly select indices from the majority class
    majority_indices = df_majority.index.tolist()
    undersampled_indices = random.sample(majority_indices, minority_size)
    
    # Create undersampled majority dataframe
    df_majority_undersampled = df_majority.loc[undersampled_indices]
    
    # Combine minority class with undersampled majority class
    df_undersampled = pd.concat([df_majority_undersampled, df_minority])
    
    # Shuffle the resulting dataframe
    df_undersampled = df_undersampled.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    
    return df_undersampled

def oversample_data_dup(df, target_column, random_seed=42):
    """
    Perform oversampling by duplicating random rows from the minority class to balance the dataset.
    
    Parameters:
    df (pandas.DataFrame): Input DataFrame containing features and target
    target_column (str): Name of the target column with class labels
    random_seed (int): Seed for reproducibility (default: 42)
    
    Returns:
    pandas.DataFrame: Oversampled DataFrame with balanced classes
    """
    # Input validation
    assert isinstance(df, pd.DataFrame), "Input must be a pandas DataFrame"
    assert target_column in df.columns, f"Target column '{target_column}' not found in DataFrame"
    
    # Set random seed for reproducibility
    random.seed(random_seed)
    
    # Identify majority and minority classes
    class_counts = df[target_column].value_counts()
    if len(class_counts) < 2:
        raise ValueError("At least two classes are required for oversampling")
    
    majority_class = class_counts.idxmax()
    minority_class = class_counts.idxmin()
    majority_size = class_counts[majority_class]
    minority_size = class_counts[minority_class]
    
    if majority_size == minority_size:
        return df.copy()  # No oversampling needed if classes are already balanced
    
    # Separate majority and minority classes
    df_minority = df[df[target_column] == minority_class]
    df_majority = df[df[target_column] == majority_class]
    
    # Calculate how many additional minority samples are needed
    samples_needed = majority_size - minority_size
    
    # Randomly duplicate rows from the minority class
    oversampled_indices = random.choices(df_minority.index.tolist(), k=samples_needed)
    df_minority_oversampled = df_minority.loc[oversampled_indices]
    
    # Combine majority and oversampled minority classes
    df_oversampled = pd.concat([df_majority, df_minority, df_minority_oversampled])
    
    # Shuffle the resulting DataFrame
    df_oversampled = df_oversampled.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    
    return df_oversampled

def oversample_smote(df, target_column, random_seed=42, k_neighbors=5):
    """
    Perform oversampling using SMOTE (Synthetic Minority Oversampling Technique) 
    without imblearn or sklearn.
    
    Parameters:
    df (pandas.DataFrame): Input DataFrame containing features and target
    target_column (str): Name of the target column with class labels
    random_seed (int): Seed for reproducibility (default: 42)
    k_neighbors (int): Number of nearest neighbors to use for SMOTE (default: 5)
    
    Returns:
    pandas.DataFrame: Oversampled DataFrame with balanced classes
    """
    # Input validation
    assert isinstance(df, pd.DataFrame), "Input must be a pandas DataFrame"
    assert target_column in df.columns, f"Target column '{target_column}' not found in DataFrame"
    
    # Set random seed for reproducibility
    np.random.seed(random_seed)
    random.seed(random_seed)
    
    # Separate features and target
    X = df.drop(columns=[target_column]).values
    y = df[target_column].values
    
    # Ensure X contains only numerical data
    if not all(df.drop(columns=[target_column]).dtypes.apply(lambda x: np.issubdtype(x, np.number))):
        raise ValueError("All feature columns must be numerical for SMOTE")
    
    # Identify majority and minority classes
    class_counts = pd.Series(y).value_counts()
    if len(class_counts) < 2:
        raise ValueError("At least two classes are required for oversampling")
    
    majority_class = class_counts.idxmax()
    minority_class = class_counts.idxmin()
    majority_size = class_counts[majority_class]
    minority_size = class_counts[minority_class]
    
    if majority_size == minority_size:
        return df.copy()  # No oversampling needed if classes are balanced
    
    # Extract minority class samples
    minority_indices = np.where(y == minority_class)[0]
    X_minority = X[minority_indices]
    
    # Adjust k_neighbors if minority class is too small
    if minority_size <= k_neighbors:
        k_neighbors = max(1, minority_size - 1)
        print(f"Warning: Adjusted k_neighbors to {k_neighbors} due to small minority class size")
    
    # Function to compute k-nearest neighbors (Euclidean distance)
    def get_k_nearest_neighbors(X, point, k):
        distances = np.sqrt(np.sum((X - point) ** 2, axis=1))
        # Exclude the point itself by setting its distance to infinity
        distances[np.where((X == point).all(axis=1))[0]] = np.inf
        # Get indices of k nearest neighbors
        nearest_indices = np.argsort(distances)[:k]
        return nearest_indices
    
    # Generate synthetic samples
    samples_needed = majority_size - minority_size
    synthetic_samples = []
    
    for _ in range(samples_needed):
        # Randomly select a minority sample
        idx = random.choice(range(minority_size))
        sample = X_minority[idx]
        
        # Find k-nearest neighbors within minority class
        neighbor_indices = get_k_nearest_neighbors(X_minority, sample, k_neighbors)
        neighbor = X_minority[random.choice(neighbor_indices)]
        
        # Generate synthetic sample
        alpha = np.random.uniform(0, 1)
        synthetic_sample = sample + alpha * (neighbor - sample)
        synthetic_samples.append(synthetic_sample)
    
    # Combine original and synthetic samples
    X_synthetic = np.array(synthetic_samples)
    y_synthetic = np.array([minority_class] * samples_needed)
    
    # Reconstruct DataFrame
    X_combined = np.vstack([X, X_synthetic])
    y_combined = np.hstack([y, y_synthetic])
    
    df_resampled = pd.DataFrame(X_combined, columns=df.drop(columns=[target_column]).columns)
    df_resampled[target_column] = y_combined
    
    # Shuffle the resulting DataFrame
    df_resampled = df_resampled.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    
    return df_resampled