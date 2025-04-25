import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                            confusion_matrix, classification_report, roc_curve, auc,
                            precision_recall_curve, average_precision_score, balanced_accuracy_score)

def evaluate_model(y_true, y_pred, y_proba=None):
    """
    Evaluate the model and return various performance metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Predicted probabilities for positive class (optional)
    
    Returns:
        dict: Dictionary containing evaluation metrics
    """
    # Basic classification metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    
    # Generate classification report
    report = classification_report(y_true, y_pred)
    
    # Initialize AUC metrics
    roc_auc = None
    pr_auc = None
    
    # Calculate ROC and PR AUC if probabilities are provided
    if y_proba is not None:
        # ROC AUC
        roc_auc = auc(*roc_curve(y_true, y_proba)[:2])
        
        # Precision-Recall AUC
        pr_auc = average_precision_score(y_true, y_proba)
    
    # Return metrics as a dictionary
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'balanced_accuracy': balanced_acc,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'classification_report': report
    }

def plot_confusion_matrix(y_true, y_pred):
    """
    Plot a confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
    
    Returns:
        matplotlib.figure.Figure: The confusion matrix figure
    """
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Create figure and axes
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot confusion matrix
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=['No Churn', 'Churn'],
        yticklabels=['No Churn', 'Churn'],
        ax=ax
    )
    
    # Set labels
    ax.set_xlabel('Predicted label')
    ax.set_ylabel('True label')
    ax.set_title('Confusion Matrix')
    
    # Improve layout
    plt.tight_layout()
    
    return fig

def plot_roc_curve(y_true, y_proba):
    """
    Plot a ROC curve.
    
    Args:
        y_true: True labels
        y_proba: Predicted probabilities for positive class
    
    Returns:
        matplotlib.figure.Figure: The ROC curve figure
    """
    # Compute ROC curve and ROC area
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)
    
    # Create figure and axes
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot ROC curve
    ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    
    # Set labels and title
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic (ROC) Curve')
    ax.legend(loc="lower right")
    
    # Add annotation for best threshold (optional)
    # Find the point on the curve closest to (0, 1)
    optimal_idx = np.argmin(np.sqrt((1-tpr)**2 + fpr**2))
    optimal_threshold = thresholds[optimal_idx]
    
    # Add annotation
    ax.annotate(f'Threshold: {optimal_threshold:.3f}',
                xy=(fpr[optimal_idx], tpr[optimal_idx]),
                xytext=(fpr[optimal_idx]+0.1, tpr[optimal_idx]-0.1),
                arrowprops=dict(facecolor='black', shrink=0.05))
    
    # Mark the point
    ax.plot(fpr[optimal_idx], tpr[optimal_idx], 'ro')
    
    # Improve layout
    plt.tight_layout()
    
    return fig
