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

def print_results_table_from_lists(all_metrics):
    """
    Print a formatted table of model performance metrics from a list of lists.
    
    Parameters:
    all_metrics (list of lists): List where each inner list contains the metrics for all models.
                                 Expected structure: [accuracies, precisions, recalls, f_scores, auc_rocs, auc_prs]
                                 Each inner list should have 5 values in order: 
                                 [Sin rebalanceo, Undersampling, Oversampling duplicate, Oversampling SMOTE, Cost re-weighting]
    """
    # Define the expected models and metrics
    expected_models = ['Sin rebalanceo', 'Undersampling', 'Oversampling duplicate', 
                       'Oversampling SMOTE', 'Cost re-weighting']
    expected_metrics = ['Accuracy', 'Precision', 'Recall', 'F-Score', 'AUC-ROC', 'AUC-PR']
    
    # Validate the input
    if len(all_metrics) != len(expected_metrics):
        raise ValueError(f"Expected {len(expected_metrics)} metric lists, got {len(all_metrics)}")
    for metric_list in all_metrics:
        if len(metric_list) != len(expected_models):
            raise ValueError(f"Each metric list must contain {len(expected_models)} values, got {len(metric_list)}")
    
    # Organize the data into a dictionary for easier access
    results_dict = {}
    for model_idx, model in enumerate(expected_models):
        results_dict[model] = {}
        for metric_idx, metric in enumerate(expected_metrics):
            results_dict[model][metric] = all_metrics[metric_idx][model_idx]
    
    # Define column widths
    col_widths = {
        'Modelo': max(len('Modelo'), max(len(model) for model in expected_models)),
        'Accuracy': max(len('Accuracy'), max(len(f"{results_dict[model]['Accuracy']:.3f}") for model in expected_models)),
        'Precision': max(len('Precision'), max(len(f"{results_dict[model]['Precision']:.3f}") for model in expected_models)),
        'Recall': max(len('Recall'), max(len(f"{results_dict[model]['Recall']:.3f}") for model in expected_models)),
        'F-Score': max(len('F-Score'), max(len(f"{results_dict[model]['F-Score']:.3f}") for model in expected_models)),
        'AUC-ROC': max(len('AUC-ROC'), max(len(f"{results_dict[model]['AUC-ROC']:.3f}") for model in expected_models)),
        'AUC-PR': max(len('AUC-PR'), max(len(f"{results_dict[model]['AUC-PR']:.3f}") for model in expected_models))
    }
    
    # Create the header
    header = (f"| {{:<{col_widths['Modelo']}}} "
              f"| {{:<{col_widths['Accuracy']}}} "
              f"| {{:<{col_widths['Precision']}}} "
              f"| {{:<{col_widths['Recall']}}} "
              f"| {{:<{col_widths['F-Score']}}} "
              f"| {{:<{col_widths['AUC-ROC']}}} "
              f"| {{:<{col_widths['AUC-PR']}}} |")
    
    # Print header
    print(header.format('Modelo', 'Accuracy', 'Precision', 'Recall', 'F-Score', 'AUC-ROC', 'AUC-PR'))
    
    # Print separator
    separator = (f"|{'-' * (col_widths['Modelo'] + 2)}"
                 f"|{'-' * (col_widths['Accuracy'] + 2)}"
                 f"|{'-' * (col_widths['Precision'] + 2)}"
                 f"|{'-' * (col_widths['Recall'] + 2)}"
                 f"|{'-' * (col_widths['F-Score'] + 2)}"
                 f"|{'-' * (col_widths['AUC-ROC'] + 2)}"
                 f"|{'-' * (col_widths['AUC-PR'] + 2)}|")
    print(separator)
    
    # Print rows
    row_template = (f"| {{:<{col_widths['Modelo']}}} "
                    f"| {{:<{col_widths['Accuracy']}}} "
                    f"| {{:<{col_widths['Precision']}}} "
                    f"| {{:<{col_widths['Recall']}}} "
                    f"| {{:<{col_widths['F-Score']}}} "
                    f"| {{:<{col_widths['AUC-ROC']}}} "
                    f"| {{:<{col_widths['AUC-PR']}}} |")
    
    for model in expected_models:
        metrics = results_dict[model]
        print(row_template.format(
            model,
            f"{metrics['Accuracy']:.3f}",
            f"{metrics['Precision']:.3f}",
            f"{metrics['Recall']:.3f}",
            f"{metrics['F-Score']:.3f}",
            f"{metrics['AUC-ROC']:.3f}",
            f"{metrics['AUC-PR']:.3f}"
        ))