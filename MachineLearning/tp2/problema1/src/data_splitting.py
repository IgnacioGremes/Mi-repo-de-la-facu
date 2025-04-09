import numpy as np
import pandas as pd
from src.metrics import *

def train_val_split(dataframe):

    n_samples = len(dataframe)
    np.random.seed(40)
    indices = np.random.permutation(n_samples)

    split_point = int(n_samples * 0.8)

    train_indices = indices[:split_point]
    val_indices = indices[split_point:]

    train_df = dataframe.iloc[train_indices]
    val_df = dataframe.iloc[val_indices]

    return train_df, val_df
