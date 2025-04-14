import pandas as pd
import numpy as np
from collections import Counter

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
    valid_categories = {1, 2, 3}
    unique_values = set(df[column_name].unique())
    if not unique_values.issubset(valid_categories):
        invalid_values = unique_values - valid_categories
        raise ValueError(f"Column '{column_name}' contains invalid categories: {invalid_values}. Expected only 1, 2, 3.")
    
    # Count occurrences of each category
    counts = df[column_name].value_counts().reindex([1, 2, 3], fill_value=0)
    
    # Convert to dictionary
    counts_dict = counts.to_dict()
    
    return counts_dict

def print_results_table_from_lists_2(all_metrics):
    """
    Print a formatted table of model performance metrics from a list of lists.
    
    Parameters:
    all_metrics (list of lists): List where each inner list contains the metrics for all models.
                                 Expected structure: [accuracies, precisions, recalls, f_scores, auc_rocs, auc_prs]
                                 Each inner list should have 5 values in order: 
                                 [Sin rebalanceo, Undersampling, Oversampling duplicate, Oversampling SMOTE, Cost re-weighting]
    """
    # Define the expected models and metrics
    expected_models = ['Logistic', 'LDA', 'Random Forest']
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

def calculate_confusion_matrix_multi(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    if y_true.shape != y_pred.shape:
        raise ValueError(f"y_true and y_pred must have the same shape. "
                         f"Got {y_true.shape} for y_true and {y_pred.shape} for y_pred.")
    
    classes = np.unique(np.concatenate([y_true, y_pred]))
    expected_classes = np.array([1, 2, 3])
    if not np.all(np.isin(classes, expected_classes)):
        raise ValueError(f"Classes must be 1, 2, or 3. Got {classes}.")
    
    if not np.all(np.isin(expected_classes, classes)):
        classes = expected_classes
    
    n_classes = len(expected_classes)
    conf_matrix = np.zeros((n_classes, n_classes), dtype=int)
    
    for true_label, pred_label in zip(y_true, y_pred):
        # Map class labels (1, 2, 3) to indices (0, 1, 2)
        true_idx = int(true_label) - 1
        pred_idx = int(pred_label) - 1
        conf_matrix[true_idx, pred_idx] += 1
    
    conf_matrix_list = conf_matrix.tolist()
    return conf_matrix_list

def calculate_accuracy_multi(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    if y_true.shape != y_pred.shape:
        raise ValueError(f"y_true and y_pred must have the same shape. "
                         f"Got {y_true.shape} for y_true and {y_pred.shape} for y_pred.")
    
    classes = np.unique(y_true)
    if not np.all(np.isin(classes, [1, 2, 3])):
        raise ValueError(f"Classes must be 1, 2, or 3. Got {classes}.")
    
    accuracies = []
    for c in [1, 2, 3]:
        correct = np.sum((y_true == c) & (y_pred == c))
        total = np.sum(y_true == c)
        class_accuracy = correct / total if total > 0 else 0.0
        accuracies.append(class_accuracy)
    
    # Macro-average
    macro_accuracy = np.mean(accuracies)
    return macro_accuracy

def calculate_precision_multi(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    if y_true.shape != y_pred.shape:
        raise ValueError(f"y_true and y_pred must have the same shape. "
                         f"Got {y_true.shape} for y_true and {y_pred.shape} for y_pred.")
    
    classes = np.unique(y_true)
    if not np.all(np.isin(classes, [1, 2, 3])):
        raise ValueError(f"Classes must be 1, 2, or 3. Got {classes}.")
    
    precisions = []
    for c in [1, 2, 3]:
        tp = np.sum((y_pred == c) & (y_true == c))
        fp = np.sum((y_pred == c) & (y_true != c))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        precisions.append(precision)
    
    # Macro-average
    macro_precision = np.mean(precisions)
    return macro_precision

def calculate_recall_multi(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    if y_true.shape != y_pred.shape:
        raise ValueError(f"y_true and y_pred must have the same shape. "
                         f"Got {y_true.shape} for y_true and {y_pred.shape} for y_pred.")
    
    classes = np.unique(y_true)
    if not np.all(np.isin(classes, [1, 2, 3])):
        raise ValueError(f"Classes must be 1, 2, or 3. Got {classes}.")
    
    recalls = []
    for c in [1, 2, 3]:
        tp = np.sum((y_pred == c) & (y_true == c))
        fn = np.sum((y_pred != c) & (y_true == c))
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        recalls.append(recall)
    
    # Macro-average
    macro_recall = np.mean(recalls)
    return macro_recall

def calculate_fscore_multi(y_true, y_pred, beta=1.0):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    if y_true.shape != y_pred.shape:
        raise ValueError(f"y_true and y_pred must have the same shape. "
                         f"Got {y_true.shape} for y_true and {y_pred.shape} for y_pred.")
    
    classes = np.unique(y_true)
    if not np.all(np.isin(classes, [1, 2, 3])):
        raise ValueError(f"Classes must be 1, 2, or 3. Got {classes}.")
    
    fscores = []
    for c in [1, 2, 3]:
        tp = np.sum((y_pred == c) & (y_true == c))
        fp = np.sum((y_pred == c) & (y_true != c))
        fn = np.sum((y_pred != c) & (y_true == c))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        denom = beta**2 * precision + recall
        fscore = (1 + beta**2) * (precision * recall) / denom if denom > 0 else 0.0
        fscores.append(fscore)
    
    # Macro-average
    macro_fscore = np.mean(fscores)
    return macro_fscore

def calculate_auc_roc_multi(y_true, y_prob):
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)  
    
    if y_true.shape[0] != y_prob.shape[0]:
        raise ValueError(f"y_true and y_prob must have the same number of samples. "
                         f"Got {y_true.shape[0]} for y_true and {y_prob.shape[0]} for y_prob.")
    
    classes = np.unique(y_true)
    if not np.all(np.isin(classes, [1, 2, 3])):
        raise ValueError(f"Classes must be 1, 2, or 3. Got {classes}.")
    
    if y_prob.shape[1] != len(classes):
        raise ValueError(f"y_prob must have probabilities for each class. "
                         f"Got {y_prob.shape[1]} columns, but expected {len(classes)} classes.")
    
    auc_scores = []
    for idx, c in enumerate([1, 2, 3]):
        y_true_binary = (y_true == c).astype(int)
        y_prob_c = y_prob[:, idx]
        
        sorted_indices = np.argsort(y_prob_c)[::-1]
        y_true_sorted = y_true_binary[sorted_indices]
        
        tpr = []
        fpr = []
        tp = 0
        fp = 0
        pos = np.sum(y_true_binary)
        neg = len(y_true_binary) - pos
        
        for i in range(len(y_true_binary)):
            if y_true_sorted[i] == 1:
                tp += 1
            else:
                fp += 1
            tpr.append(tp / pos if pos > 0 else 0)
            fpr.append(fp / neg if neg > 0 else 0)
        
        auc = 0
        for i in range(1, len(fpr)):
            auc += (fpr[i] - fpr[i-1]) * (tpr[i] + tpr[i-1]) / 2
        auc_scores.append(auc)
    
    # Macro-average
    macro_auc = np.mean(auc_scores)
    return macro_auc

def calculate_auc_pr_multi(y_true, y_prob):
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    
    if y_true.shape[0] != y_prob.shape[0]:
        raise ValueError(f"y_true and y_prob must have the same number of samples. "
                         f"Got {y_true.shape[0]} for y_true and {y_prob.shape[0]} for y_prob.")
    
    classes = np.unique(y_true)
    if not np.all(np.isin(classes, [1, 2, 3])):
        raise ValueError(f"Classes must be 1, 2, or 3. Got {classes}.")
    
    if y_prob.shape[1] != len(classes):
        raise ValueError(f"y_prob must have probabilities for each class. "
                         f"Got {y_prob.shape[1]} columns, but expected {len(classes)} classes.")
    
    auc_pr_scores = []
    for idx, c in enumerate([1, 2, 3]):
        y_true_binary = (y_true == c).astype(int)
        y_prob_c = y_prob[:, idx]
        
        sorted_indices = np.argsort(y_prob_c)[::-1]
        y_true_sorted = y_true_binary[sorted_indices]
        
        precision_vals = []
        recall_vals = []
        tp = 0
        fp = 0
        fn = np.sum(y_true_binary)
        
        for i in range(len(y_true_binary)):
            if y_true_sorted[i] == 1:
                tp += 1
                fn -= 1
            else:
                fp += 1
            precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            precision_vals.append(precision)
            recall_vals.append(recall)
        
        auc_pr = 0
        for i in range(1, len(recall_vals)):
            auc_pr += (recall_vals[i] - recall_vals[i-1]) * (precision_vals[i] + precision_vals[i-1]) / 2
        auc_pr_scores.append(auc_pr)
    
    # Macro-average
    macro_auc_pr = np.mean(auc_pr_scores)
    return macro_auc_pr

def calculate_precision_recall_curve_multi(y_true, y_prob):
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    
    if y_true.shape[0] != y_prob.shape[0]:
        raise ValueError(f"y_true and y_prob must have the same number of samples. "
                         f"Got {y_true.shape[0]} for y_true and {y_prob.shape[0]} for y_prob.")
    
    classes = np.unique(y_true)
    if not np.all(np.isin(classes, [1, 2, 3])):
        raise ValueError(f"Classes must be 1, 2, or 3. Got {classes}.")
    
    if y_prob.shape[1] != len(classes):
        raise ValueError(f"y_prob must have probabilities for each class. "
                         f"Got {y_prob.shape[1]} columns, but expected {len(classes)} classes.")
    
    all_precisions = []
    all_recalls = []
    all_thresholds = []
    
    for idx, c in enumerate([1, 2, 3]):
        y_true_binary = (y_true == c).astype(int)
        y_prob_c = y_prob[:, idx]
        
        sorted_indices = np.argsort(y_prob_c)[::-1]
        y_prob_sorted = y_prob_c[sorted_indices]
        
        thresholds = np.concatenate([[1.1], y_prob_sorted])
        thresholds = np.unique(thresholds)[::-1]
        
        precision_vals = []
        recall_vals = []
        tp = 0
        fp = 0
        fn = np.sum(y_true_binary)
        
        for thresh in thresholds:
            y_pred_binary = (y_prob_c >= thresh).astype(int)
            tp = np.sum((y_pred_binary == 1) & (y_true_binary == 1))
            fp = np.sum((y_pred_binary == 1) & (y_true_binary == 0))
            fn = np.sum((y_pred_binary == 0) & (y_true_binary == 1))
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            
            precision_vals.append(precision)
            recall_vals.append(recall)
        
        all_precisions.append(precision_vals)
        all_recalls.append(recall_vals)
        all_thresholds.append(thresholds)
    
    # Compute macro-averaged curve
    recall_points = np.linspace(0, 1, 1000)
    interpolated_precisions = []
    
    for prec, rec in zip(all_precisions, all_recalls):
        sorted_idx = np.argsort(rec)
        rec_sorted = np.array(rec)[sorted_idx]
        prec_sorted = np.array(prec)[sorted_idx]
        interp_prec = np.interp(recall_points, rec_sorted, prec_sorted, left=1.0, right=0.0)
        interpolated_precisions.append(interp_prec)
    
    avg_precision = np.mean(interpolated_precisions, axis=0)
    avg_recall = recall_points
    avg_thresholds = np.linspace(0, 1, len(recall_points))
    
    return avg_precision, avg_recall, avg_thresholds

def calculate_roc_curve_multi(y_true, y_prob):
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    
    if y_true.shape[0] != y_prob.shape[0]:
        raise ValueError(f"y_true and y_prob must have the same number of samples. "
                         f"Got {y_true.shape[0]} for y_true and {y_prob.shape[0]} for y_prob.")
    
    classes = np.unique(y_true)
    if not np.all(np.isin(classes, [1, 2, 3])):
        raise ValueError(f"Classes must be 1, 2, or 3. Got {classes}.")
    
    if y_prob.shape[1] != len(classes):
        raise ValueError(f"y_prob must have probabilities for each class. "
                         f"Got {y_prob.shape[1]} columns, but expected {len(classes)} classes.")
    
    all_fprs = []
    all_tprs = []
    all_thresholds = []
    
    for idx, c in enumerate([1, 2, 3]):
        y_true_binary = (y_true == c).astype(int)
        y_prob_c = y_prob[:, idx]
        
        sorted_indices = np.argsort(y_prob_c)[::-1]
        y_prob_sorted = y_prob_c[sorted_indices]
        
        thresholds = np.concatenate([[1.1], y_prob_sorted])
        thresholds = np.unique(thresholds)[::-1]
        
        fpr_vals = []
        tpr_vals = []
        tp = 0
        fp = 0
        fn = np.sum(y_true_binary)
        tn = len(y_true_binary) - fn
        
        for thresh in thresholds:
            y_pred_binary = (y_prob_c >= thresh).astype(int)
            tp = np.sum((y_pred_binary == 1) & (y_true_binary == 1))
            fp = np.sum((y_pred_binary == 1) & (y_true_binary == 0))
            fn = np.sum((y_pred_binary == 0) & (y_true_binary == 1))
            tn = np.sum((y_pred_binary == 0) & (y_true_binary == 0))
            
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
            
            tpr_vals.append(tpr)
            fpr_vals.append(fpr)
        
        all_fprs.append(fpr_vals)
        all_tprs.append(tpr_vals)
        all_thresholds.append(thresholds)
    
    # Compute macro-averaged curve
    fpr_points = np.linspace(0, 1, 1000)
    interpolated_tprs = []
    
    for fpr, tpr in zip(all_fprs, all_tprs):
        sorted_idx = np.argsort(fpr)
        fpr_sorted = np.array(fpr)[sorted_idx]
        tpr_sorted = np.array(tpr)[sorted_idx]
        interp_tpr = np.interp(fpr_points, fpr_sorted, tpr_sorted, left=0.0, right=1.0)
        interpolated_tprs.append(interp_tpr)
    
    avg_tpr = np.mean(interpolated_tprs, axis=0)
    avg_fpr = fpr_points
    avg_thresholds = np.linspace(0, 1, len(fpr_points)) 
    return avg_fpr, avg_tpr, avg_thresholds