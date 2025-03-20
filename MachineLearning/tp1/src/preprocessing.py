import numpy as np
import pandas as pd

#/home/ig/Desktop/prueba/MachineLearning/tp1/data/raw/casas_dev.csv

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
            


# before split

file_path = 'MachineLearning/tp1/data/raw/casas_dev.csv'
df = pd.read_csv(file_path)
df = handle_nan_values(df)
convert_area_sqft_to_m2(df)
df.to_csv('MachineLearning/tp1/data/processed/cleaned_casas_dev.csv', index=False)

# after split

file_path = 'MachineLearning/tp1/data/processed/train_casas_dev.csv'
df_train = pd.read_csv(file_path)
df_train = df_train.drop(columns=['area_units'])
# normalize('area',df_train)
# normalize('price',df_train)
# normalize('age',df_train)
# normalize('lat',df_train)
# normalize('lon',df_train)
# normalize('rooms',df_train)

df_train.to_csv('MachineLearning/tp1/data/processed/train_cleaned_casas_dev.csv', index=False)

# Own featured train

# df_cleaned_own_featured = handle_nan_values(df)
# convert_area_sqft_to_m2(df_cleaned_own_featured)
# df_cleaned_own_featured = df_train.drop(columns=['area_units'])
create_feature_area_per_room(df_train)
create_feature_house_and_pool(df_train)
create_feature_age_range(df_train)
# normalize('area',df_cleaned_own_featured)
# normalize('price',df_cleaned_own_featured)
# normalize('age',df_cleaned_own_featured)
# normalize('lat',df_cleaned_own_featured)
# normalize('lon',df_cleaned_own_featured)
# normalize('rooms',df_cleaned_own_featured)
# normalize('area_per_room',df_cleaned_own_featured)

df_train.to_csv('MachineLearning/tp1/data/processed/own_featured_train_casas_dev.csv', index=False)

# Amanda clean

file_path = 'MachineLearning/tp1/data/raw/vivienda_Amanda.csv'
amanda_df = pd.read_csv(file_path)
convert_area_sqft_to_m2(amanda_df)
amanda_df = amanda_df.drop(columns=['area_units'])
amanda_df.to_csv('MachineLearning/tp1/data/processed/cleaned_Amanda.csv', index=False)

# Amanda own featured

# amanda_cleaned_df = handle_nan_values(amanda_df)
# convert_area_sqft_to_m2(amanda_cleaned_df)
# amanda_cleaned_df = amanda_cleaned_df.drop(columns=['area_units'])
create_feature_area_per_room(amanda_df)
create_feature_house_and_pool(amanda_df)
create_feature_age_range(amanda_df)

amanda_df.to_csv('MachineLearning/tp1/data/processed/own_featured_Amanda.csv', index=False)

# Exponential features

# def expand_features(df):
#     expanded_df = df.copy()
#     for exp in range(2, 44):
#         for col in df.columns:
#             if col != 'price':
#                 expanded_df[f'{col}^{exp}'] = df[col] ** exp
#     return expanded_df

# file_path = 'MachineLearning/tp1/data/processed/train_cleaned_casas_dev.csv'
# df = pd.read_csv(file_path)
# df = df.drop(columns=['area_units'])

# df = expand_features(df)
# df = df.drop(columns=['rooms^43'])
# # print(df)

# df.to_csv('MachineLearning/tp1/data/processed/exponential_featured_train_casas_dev.csv', index=False)


# file_path = 'MachineLearning/tp1/data/processed/cleaned_Amanda.csv'
# df = pd.read_csv(file_path)

# df = expand_features(df)
# df = df.drop(columns=['rooms^43'])
# # print(df)

# df.to_csv('MachineLearning/tp1/data/processed/exponential_featured_Amanda.csv', index=False)

# Validation

file_path = 'MachineLearning/tp1/data/processed/val_casas_dev.csv'
val_df = pd.read_csv(file_path)
# convert_area_sqft_to_m2(val_df)
val_df = val_df.drop(columns=['area_units'])

val_df.to_csv('MachineLearning/tp1/data/processed/val_cleaned_casas_dev.csv', index=False)

# Own features validation

create_feature_area_per_room(val_df)
create_feature_house_and_pool(val_df)
create_feature_age_range(val_df)

val_df.to_csv('MachineLearning/tp1/data/processed/own_featured_val_casas_dev.csv', index=False)

