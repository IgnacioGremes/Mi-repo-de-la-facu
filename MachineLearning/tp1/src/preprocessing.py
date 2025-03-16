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
    
            



df_cleaned = handle_nan_values(df) 
convert_area_sqft(df_cleaned)
normalize('area',df_cleaned)
normalize('price',df_cleaned)
normalize('age',df_cleaned)
normalize('lat',df_cleaned)
normalize('lon',df_cleaned)
normalize('rooms',df_cleaned)
print(df_cleaned)

df_cleaned.to_csv('MachineLearning/tp1/data/processed/cleaned_casas_dev.csv', index=False)

# row_index = df.loc[]
