import numpy as np
import pandas as pd

file_path = 'MachineLearning/tp1/data/processed/cleaned_casas_dev.csv'
df = pd.read_csv(file_path)

n_samples = len(df)
indices = np.random.permutation(n_samples)

split_point = int(n_samples * 0.8)

train_indices = indices[:split_point]
val_indices = indices[split_point:]

train_df = df.iloc[train_indices]
val_df = df.iloc[val_indices]

train_df.to_csv('MachineLearning/tp1/data/processed/train_cleaned_casas_dev.csv', index=False)
val_df.to_csv('MachineLearning/tp1/data/processed/val_cleaned_casas_dev.csv', index=False)