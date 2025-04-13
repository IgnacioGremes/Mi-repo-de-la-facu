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

# def calculate_precision_recall_curve(y_true, y_prob):
#     """
#     Calculate precision-recall curve for binary classification.
    
#     Parameters:
#     - y_true (array-like): True binary labels (0 or 1).
#     - y_prob (array-like): Predicted probabilities for the positive class (1).
    
#     Returns:
#     - tuple: (precision_values, recall_values) as lists.
#     """
#     y_true = np.array(y_true)
#     y_prob = np.array(y_prob)
    
#     sorted_indices = np.argsort(y_prob)[::-1]
#     y_prob_sorted = y_prob[sorted_indices]
#     y_true_sorted = y_true[sorted_indices]
    
#     thresholds = np.concatenate([[1.1], y_prob_sorted])
#     thresholds = np.unique(thresholds)[::-1]
    
#     precision_vals = []
#     recall_vals = []
#     for thresh in thresholds:
#         y_pred_thresh = (y_prob >= thresh).astype(int)
#         tp = np.sum((y_true == 1) & (y_pred_thresh == 1))
#         fp = np.sum((y_true == 0) & (y_pred_thresh == 1))
#         fn = np.sum((y_true == 1) & (y_pred_thresh == 0))
#         prec = tp / (tp + fp) if (tp + fp) > 0 else 0
#         rec = tp / (tp + fn) if (tp + fn) > 0 else 0
#         precision_vals.append(prec)
#         recall_vals.append(rec)
    
#     return precision_vals, recall_vals

def calculate_precision_recall_curve(y_true, y_prob):
    """
    Calculate precision-recall curve for binary classification.
    
    Parameters:
    - y_true (array-like): True binary labels (0 or 1).
    - y_prob (array-like): Predicted probabilities for the positive class (1).
    
    Returns:
    - tuple: (precision_values, recall_values, thresholds) as lists.
    """
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    
    # Validate inputs
    if y_true.shape != y_prob.shape:
        raise ValueError(f"y_true and y_prob must have the same shape. "
                         f"Got {y_true.shape} for y_true and {y_prob.shape} for y_prob.")
    if not np.all(np.isin(y_true, [0, 1])):
        raise ValueError(f"y_true must contain only 0 or 1. Got {np.unique(y_true)}.")
    
    # Sort by probabilities in descending order
    sorted_indices = np.argsort(y_prob)[::-1]
    y_prob_sorted = y_prob[sorted_indices]
    y_true_sorted = y_true[sorted_indices]
    
    # Initialize variables
    precision_vals = []
    recall_vals = []
    thresholds = []
    
    tp = 0
    fp = 0
    fn = np.sum(y_true == 1)  # Total positives
    pos = fn  # Total positives for recall
    
    # Start with threshold above max probability
    precision_vals.append(1.0)  # Precision is 1 when no positives are predicted
    recall_vals.append(0.0)     # Recall is 0 when no positives are predicted
    thresholds.append(1.1)
    
    # Iterate through sorted probabilities
    for i in range(len(y_prob_sorted)):
        if y_true_sorted[i] == 1:
            tp += 1
            fn -= 1
        else:
            fp += 1
        
        # Compute precision and recall
        prec = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        rec = tp / pos if pos > 0 else 0.0
        
        precision_vals.append(prec)
        recall_vals.append(rec)
        thresholds.append(y_prob_sorted[i])
    
    # Ensure the curve ends at recall=1
    if recall_vals[-1] != 1.0:
        precision_vals.append(tp / (tp + fp) if (tp + fp) > 0 else 1.0)
        recall_vals.append(1.0)
        thresholds.append(0.0)
    
    return precision_vals, recall_vals, thresholds



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
    precision_vals, recall_vals, thresh = calculate_precision_recall_curve(y_true, y_prob)
    auc_pr = np.sum(np.diff(recall_vals) * (np.array(precision_vals[:-1]) + np.array(precision_vals[1:])) / 2)
    return abs(auc_pr)  # Ensure positive value

def print_results_table_from_lists(all_metrics,exp_models,exp_metrics):
    """
    Print a formatted table of model performance metrics from a list of lists.
    
    Parameters:
    all_metrics (list of lists): List where each inner list contains the metrics for all models.
                                 Expected structure: [accuracies, precisions, recalls, f_scores, auc_rocs, auc_prs]
                                 Each inner list should have 5 values in order: 
                                 [Sin rebalanceo, Undersampling, Oversampling duplicate, Oversampling SMOTE, Cost re-weighting]
    """
    # Define the expected models and metrics
    # expected_models = ['Sin rebalanceo', 'Undersampling', 'Oversampling duplicate', 
    #                    'Oversampling SMOTE', 'Cost re-weighting']
    # expected_metrics = ['Accuracy', 'Precision', 'Recall', 'F-Score', 'AUC-ROC', 'AUC-PR']
    expected_models = exp_models
    expected_metrics = exp_metrics
    
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

def count_categories(df, column_name):
    """
    Count the number of samples for each category (1, 2, 3) in a specified column of a DataFrame.
    
    Parameters:
    df (pandas.DataFrame): Input DataFrame containing the categorical column
    column_name (str): Name of the column to analyze
    
    Returns:
    dict: Dictionary with categories (1, 2, 3) as keys and their counts as values
    """
    # Input validation
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame")
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in DataFrame")
    
    # Check that all values in the column are 1, 2, or 3
    valid_categories = {0, 1}
    unique_values = set(df[column_name].unique())
    if not unique_values.issubset(valid_categories):
        invalid_values = unique_values - valid_categories
        raise ValueError(f"Column '{column_name}' contains invalid categories: {invalid_values}. Expected only 1, 2, 3.")
    
    # Count occurrences of each category
    counts = df[column_name].value_counts().reindex([0, 1], fill_value=0)
    
    # Convert to dictionary
    counts_dict = counts.to_dict()
    
    return counts_dict