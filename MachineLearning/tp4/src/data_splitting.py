import numpy as np

def train_val_split(array, seed=42):
    np.random.seed(seed)
    n_samples = len(array)
    indices = np.random.permutation(n_samples)
    split_point = int(n_samples * 0.8)
    train_idx, val_idx = indices[:split_point], indices[split_point:]
    return array[train_idx], array[val_idx]