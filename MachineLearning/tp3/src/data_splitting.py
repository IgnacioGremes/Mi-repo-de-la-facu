import numpy as np

def train_val_test_split(X, y, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    # Verify ratios sum to approximately 1
    if not (0.99 <= train_ratio + val_ratio + test_ratio <= 1.01):
        raise ValueError("train_ratio + val_ratio + test_ratio must sum to approximately 1.0")
    
    n_samples = X.shape[0]
    np.random.seed(40)  # Uncomment for reproducibility
    indices = np.random.permutation(n_samples)
    
    # Calculate split points
    train_split = int(n_samples * train_ratio)
    val_split = int(n_samples * (train_ratio + val_ratio))
    
    # Split indices
    train_indices = indices[:train_split]
    val_indices = indices[train_split:val_split]
    test_indices = indices[val_split:]
    
    # Split arrays
    X_train = X[train_indices]
    y_train = y[train_indices]
    X_val = X[val_indices]
    y_val = y[val_indices]
    X_test = X[test_indices]
    y_test = y[test_indices]
    
    return X_train, y_train, X_val, y_val, X_test, y_test