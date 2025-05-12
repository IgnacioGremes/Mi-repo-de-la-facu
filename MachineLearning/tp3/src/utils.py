import numpy as np

def confusion_matrix(Y_true, Y_pred_probs):
    Y_pred = np.argmax(Y_pred_probs, axis=1)
    conf_matrix = np.zeros((49, 49), dtype=int)
    for t, p in zip(Y_true, Y_pred):
        conf_matrix[t, p] += 1
    return conf_matrix