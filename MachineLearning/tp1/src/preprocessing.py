import numpy as np
import pandas as pd

file_path = '/home/ig/Desktop/prueba/MachineLearning/tp1/data/raw/casas_dev.csv'
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
    
            



df = handle_nan_values(df) 
convert_area_sqft(df)
print(df)

# row_index = df.loc[]
