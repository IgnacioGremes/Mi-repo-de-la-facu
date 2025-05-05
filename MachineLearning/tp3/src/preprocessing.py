import numpy as np

def one_hot_encode(Y, num_classes):
    """
    One-hot encode a vector of integer labels.
    Y: numpy array of shape (n_samples,) with integer labels
    num_classes: number of classes (e.g., 49)
    Returns: numpy array of shape (n_samples, num_classes)
    """
    return np.eye(num_classes)[Y]