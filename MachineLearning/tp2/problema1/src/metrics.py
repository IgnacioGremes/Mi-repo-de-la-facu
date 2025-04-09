import pandas as pd
import numpy as np

def calculate_confusion_matrix(y_true, y_pred):
    """
    Calculate the confusion matrix for binary classification.
    
    Parameters:
    - y_true (array-like): True binary labels (0 or 1).
    - y_pred (array-like): Predicted binary labels (0 or 1).
    
    Returns:
    - np.ndarray: 2x2 confusion matrix [[TN, FP], [FN, TP]].
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    tp = np.sum((y_true == 1) & (y_pred == 1))
    
    return np.array([[tn, fp], [fn, tp]])

def calculate_precision(y_true, y_pred):
    """
    Calculate precision for binary classification.
    
    Parameters:
    - y_true (array-like): True binary labels (0 or 1).
    - y_pred (array-like): Predicted binary labels (0 or 1).
    
    Returns:
    - float: Precision score.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    
    return tp / (tp + fp) if (tp + fp) > 0 else 0

def calculate_accuracy(y_true, y_pred):
    """
    Calculate accuracy for binary classification.
    
    Parameters:
    - y_true (array-like): True binary labels (0 or 1).
    - y_pred (array-like): Predicted binary labels (0 or 1).
    
    Returns:
    - float: Accuracy score.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    correct = np.sum(y_true == y_pred)
    total = len(y_true)
    
    return correct / total

def calculate_recall(y_true, y_pred):
    """
    Calculate recall for binary classification.
    
    Parameters:
    - y_true (array-like): True binary labels (0 or 1).
    - y_pred (array-like): Predicted binary labels (0 or 1).
    
    Returns:
    - float: Recall score.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    
    return tp / (tp + fn) if (tp + fn) > 0 else 0

def calculate_f1_score(y_true, y_pred):
    """
    Calculate F1-score for binary classification.
    
    Parameters:
    - y_true (array-like): True binary labels (0 or 1).
    - y_pred (array-like): Predicted binary labels (0 or 1).
    
    Returns:
    - float: F1-score.
    """
    precision = calculate_precision(y_true, y_pred)
    recall = calculate_recall(y_true, y_pred)
    
    return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

def calculate_precision_recall_curve(y_true, y_prob):
    """
    Calculate precision-recall curve for binary classification.
    
    Parameters:
    - y_true (array-like): True binary labels (0 or 1).
    - y_prob (array-like): Predicted probabilities for the positive class (1).
    
    Returns:
    - tuple: (precision_values, recall_values) as lists.
    """
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    
    sorted_indices = np.argsort(y_prob)[::-1]
    y_prob_sorted = y_prob[sorted_indices]
    y_true_sorted = y_true[sorted_indices]
    
    thresholds = np.concatenate([[1.1], y_prob_sorted])
    thresholds = np.unique(thresholds)[::-1]
    
    precision_vals = []
    recall_vals = []
    for thresh in thresholds:
        y_pred_thresh = (y_prob >= thresh).astype(int)
        tp = np.sum((y_true == 1) & (y_pred_thresh == 1))
        fp = np.sum((y_true == 0) & (y_pred_thresh == 1))
        fn = np.sum((y_true == 1) & (y_pred_thresh == 0))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        precision_vals.append(prec)
        recall_vals.append(rec)
    
    return precision_vals, recall_vals

def calculate_roc_curve(y_true, y_prob):
    """
    Calculate ROC curve for binary classification.
    
    Parameters:
    - y_true (array-like): True binary labels (0 or 1).
    - y_prob (array-like): Predicted probabilities for the positive class (1).
    
    Returns:
    - tuple: (fpr_values, tpr_values) as lists.
    """
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    
    sorted_indices = np.argsort(y_prob)[::-1]
    y_prob_sorted = y_prob[sorted_indices]
    y_true_sorted = y_true[sorted_indices]
    
    thresholds = np.concatenate([[1.1], y_prob_sorted])
    thresholds = np.unique(thresholds)[::-1]
    
    fpr_vals = []
    tpr_vals = []
    for thresh in thresholds:
        y_pred_thresh = (y_prob >= thresh).astype(int)
        tp = np.sum((y_true == 1) & (y_pred_thresh == 1))
        fp = np.sum((y_true == 0) & (y_pred_thresh == 1))
        fn = np.sum((y_true == 1) & (y_pred_thresh == 0))
        tn = np.sum((y_true == 0) & (y_pred_thresh == 0))
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr_vals.append(fpr)
        tpr_vals.append(tpr)
    
    return fpr_vals, tpr_vals

def calculate_auc_roc(y_true, y_prob):
    """
    Calculate AUC-ROC for binary classification using the trapezoidal rule.
    
    Parameters:
    - y_true (array-like): True binary labels (0 or 1).
    - y_prob (array-like): Predicted probabilities for the positive class (1).
    
    Returns:
    - float: AUC-ROC score.
    """
    fpr_vals, tpr_vals = calculate_roc_curve(y_true, y_prob)
    auc_roc = np.sum(np.diff(fpr_vals) * (np.array(tpr_vals[:-1]) + np.array(tpr_vals[1:])) / 2)
    return abs(auc_roc)  # Ensure positive value

def calculate_auc_pr(y_true, y_prob):
    """
    Calculate AUC-PR for binary classification using the trapezoidal rule.
    
    Parameters:
    - y_true (array-like): True binary labels (0 or 1).
    - y_prob (array-like): Predicted probabilities for the positive class (1).
    
    Returns:
    - float: AUC-PR score.
    """
    precision_vals, recall_vals = calculate_precision_recall_curve(y_true, y_prob)
    auc_pr = np.sum(np.diff(recall_vals) * (np.array(precision_vals[:-1]) + np.array(precision_vals[1:])) / 2)
    return abs(auc_pr)  # Ensure positive value