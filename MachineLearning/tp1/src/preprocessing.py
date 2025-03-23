import pandas as pd

def handle_nan_values(dataframe):
    assert isinstance(dataframe, pd.DataFrame)
    return dataframe.dropna()  

def normalize(column: str, dataframe):
    assert isinstance(dataframe, pd.DataFrame)

    col_min = dataframe[column].min()
    col_max = dataframe[column].max()
    if col_max - col_min != 0: # Avoid division by zero
        dataframe[column] = (dataframe[column] - col_min) / (col_max - col_min)

def convert_area_sqft_to_m2(dataframe):
    assert isinstance(dataframe, pd.DataFrame)
    for index, row in dataframe.iterrows():
        if row['area_units'] == 'sqft':
            dataframe.loc[index, 'area'] = row['area'] / 10.76
            dataframe.loc[index, 'area_units'] = 'm2'
    
# Feature Engeneering

def create_feature_area_per_room(dataframe):
    assert isinstance(dataframe, pd.DataFrame)
    areas_per_room = []
    for _ , row in dataframe.iterrows():
        areas_per_room.append(row['area'] / row['rooms'])
    dataframe['area_per_room'] = areas_per_room

def create_feature_house_and_pool(dataframe):
    assert isinstance(dataframe, pd.DataFrame)
    list = []
    for _ , row in dataframe.iterrows():
        if row['is_house'] and row['has_pool']:
            list.append(1)
        else:
            list.append(0) 
    dataframe['has_pool_and_house'] = list

def create_feature_age_range(dataframe):
    assert isinstance(dataframe, pd.DataFrame)
    age_new_list = []
    age_mid_list = []
    age_old_list = []
    for _ , row in dataframe.iterrows():
        if row['age'] <= 5:
            age_new_list.append(1)
            age_mid_list.append(0)
            age_old_list.append(0)
        elif 5 < row['age'] <= 14:
            age_new_list.append(0)
            age_mid_list.append(1)
            age_old_list.append(0)
        else:
            age_new_list.append(0)
            age_mid_list.append(0)
            age_old_list.append(1)
    dataframe['age_new'] = age_new_list
    dataframe['age_mid'] = age_mid_list
    dataframe['age_old'] = age_old_list

def expand_features(df):

    new_columns = []
    
    for exp in range(2, 44):
        for col in df.columns:
            if col != 'price':
                new_col_name = f'{col}^{exp}'
                new_col = df[col] ** exp
                new_columns.append(pd.DataFrame({new_col_name: new_col}, index=df.index))

    expanded_df = pd.concat([df] + new_columns, axis=1)
    
    return expanded_df