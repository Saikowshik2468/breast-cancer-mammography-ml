"""
Evaluation metrics for imbalanced classification
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report, balanced_accuracy_score
)


def evaluate_model(y_true, y_pred, y_pred_proba=None):
    """
    Evaluate model performance with multiple metrics
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_pred_proba: Predicted probabilities (optional)
    
    Returns:
        dict: Dictionary of evaluation metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1_score': f1_score(y_true, y_pred, zero_division=0),
    }
    
    if y_pred_proba is not None:
        metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
        metrics['pr_auc'] = average_precision_score(y_true, y_pred_proba)
    
    return metrics


def print_classification_report(y_true, y_pred):
    """
    Print detailed classification report
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
    """
    print("\n" + "="*50)
    print("Classification Report")
    print("="*50)
    print(classification_report(y_true, y_pred, target_names=['Benign', 'Malignant']))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    print("="*50)


def print_metrics_summary(metrics):
    """
    Print a formatted summary of evaluation metrics
    
    Args:
        metrics: Dictionary of metrics
    """
    print("\n" + "="*50)
    print("Model Performance Metrics")
    print("="*50)
    for metric_name, value in metrics.items():
        if isinstance(value, float):
            print(f"{metric_name.capitalize()}: {value:.4f}")
    print("="*50)

