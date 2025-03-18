import numpy as np
import pandas as pd

#/home/ig/Desktop/prueba/MachineLearning/tp1/data/raw/casas_dev.csv

file_path = 'MachineLearning/tp1/data/raw/casas_dev.csv'
df = pd.read_csv(file_path)

def handle_nan_values(dataframe):
    assert isinstance(dataframe, pd.DataFrame)
    return dataframe.dropna()  

def normalize(column: str, dataframe):
    assert isinstance(dataframe, pd.DataFrame)

    col_min = dataframe[column].min()
    col_max = dataframe[column].max()
    if col_max - col_min != 0: # Avoid division by zero
        dataframe[column] = (dataframe[column] - col_min) / (col_max - col_min)

def convert_area_sqft(dataframe):
    assert isinstance(dataframe, pd.DataFrame)
    for index, row in dataframe.iterrows():
        if row['area_units'] == 'm2':
            dataframe.loc[index,'area'] = row['area'] * 10.76
            dataframe.loc[index,'area_units'] = 'sqft'

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
            



df_cleaned = handle_nan_values(df)
df_cleaned_m2 = handle_nan_values(df)
df_cleaned_features = handle_nan_values(df)
convert_area_sqft_to_m2(df_cleaned_m2)
convert_area_sqft(df_cleaned)
convert_area_sqft(df_cleaned_features)
df_cleaned = df_cleaned.drop(columns=['area_units'])
df_cleaned_features = df_cleaned_features.drop(columns=['area_units'])

create_feature_area_per_room(df_cleaned_features)
create_feature_house_and_pool(df_cleaned_features)

normalize('area',df_cleaned)
normalize('price',df_cleaned)
normalize('age',df_cleaned)
normalize('lat',df_cleaned)
normalize('lon',df_cleaned)
normalize('rooms',df_cleaned)

normalize('area_per_room',df_cleaned)

print(df_cleaned)

# df_cleaned.to_csv('MachineLearning/tp1/data/processed/cleaned_casas_dev.csv', index=False)
# df_cleaned_m2.to_csv('MachineLearning/tp1/data/processed/cleaned_m2_casas_dev.csv', index=False)

# row_index = df.loc[]

