import pandas as pd
import plotly.express as px
import math

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


    