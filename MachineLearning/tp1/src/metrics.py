import numpy as np
import pandas as pd

def mean_squared_error(y_true, y_pred):
    n = len(y_true)
    mse = np.sum((y_true - y_pred) ** 2) / n
    return mse

def mean_value_per_m2(dataframe):
    assert isinstance(dataframe, pd.DataFrame)
    mean_list = []
    for _ , row in dataframe.iterrows():
        mean_list.append(row['price'] / row['area'])
    mean = sum(mean_list) / dataframe.shape[0]
    return mean

def mean_absolute_error(y_true, y_pred):
    mae = np.mean(np.abs(y_true - y_pred))
    return mae

def root_mean_squared_error(y_true, y_pred):
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    return rmse