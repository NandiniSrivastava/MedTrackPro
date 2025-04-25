import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
from sklearn.inspection import permutation_importance
from eli5.sklearn import explain_weights
from eli5.formatters import format_as_dict

def plot_feature_importance(model, model_type, feature_names):
    """
    Plot feature importance for the model.
    
    Args:
        model: Trained model
        model_type (str): Type of model
        feature_names (list): Names of features
    
    Returns:
        matplotlib.figure.Figure: Feature importance figure
    """
    # Create figure and axes
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Different models provide feature importance in different ways
    if model_type == "Logistic Regression":
        # For logistic regression, use coefficients as importance
        importances = np.abs(model.coef_[0])
        indices = np.argsort(importances)[::-1]
        
        # Plot feature importance
        ax.barh(range(len(indices)), importances[indices], align='center')
        ax.set_yticks(range(len(indices)))
        ax.set_yticklabels([feature_names[i] for i in indices])
        ax.set_title('Feature Importance (Logistic Regression Coefficients)')
        ax.set_xlabel('Absolute Coefficient Magnitude')
        
    elif model_type in ["Random Forest", "Gradient Boosting"]:
        # For tree-based models, use built-in feature importance
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        # Plot feature importance
        ax.barh(range(len(indices)), importances[indices], align='center')
        ax.set_yticks(range(len(indices)))
        ax.set_yticklabels([feature_names[i] for i in indices])
        ax.set_title(f'Feature Importance ({model_type})')
        ax.set_xlabel('Importance')
    
    else:
        ax.text(0.5, 0.5, f"Feature importance is not directly available for {model_type}", 
                ha='center', va='center', fontsize=12)
    
    # Improve layout
    plt.tight_layout()
    
    return fig

def explain_prediction_permutation(model, X_test, y_test, feature_names):
    """
    Generate permutation feature importance for model explanation.
    
    Args:
        model: Trained model
        X_test: Test data features
        y_test: Test data labels
        feature_names (list): Names of features
    
    Returns:
        matplotlib.figure.Figure: Permutation importance plot
    """
    # Calculate permutation importance
    perm_importance = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42)
    
    # Sort features by importance
    sorted_idx = perm_importance.importances_mean.argsort()[::-1]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot sorted importance values
    ax.barh(range(len(sorted_idx)), perm_importance.importances_mean[sorted_idx])
    ax.set_yticks(range(len(sorted_idx)))
    ax.set_yticklabels([feature_names[i] for i in sorted_idx])
    ax.set_title("Permutation Feature Importance")
    ax.set_xlabel("Importance (decrease in model performance when feature is permuted)")
    
    plt.tight_layout()
    return fig

def plot_prediction_explanation(model, model_type, sample, feature_names):
    """
    Generate a simple explanation for an individual prediction based on feature contributions.
    
    Args:
        model: Trained model
        model_type (str): Type of model
        sample: Single data sample to explain
        feature_names (list): Names of features
    
    Returns:
        matplotlib.figure.Figure: Feature contribution plot
    """
    # Create a figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Different visualization based on model type
    if model_type == "Logistic Regression":
        # For logistic regression, use coefficients to show contribution
        # Get feature values and multiply by coefficients
        feature_values = sample.values[0]
        coefficients = model.coef_[0]
        
        # Calculate contribution
        contributions = feature_values * coefficients
        
        # Sort by absolute contribution
        sorted_idx = np.abs(contributions).argsort()[::-1]
        
        # Plot top contributions
        top_n = min(10, len(sorted_idx))
        colors = ['green' if c > 0 else 'red' for c in contributions[sorted_idx[:top_n]]]
        
        ax.barh(range(top_n), contributions[sorted_idx[:top_n]], color=colors)
        ax.set_yticks(range(top_n))
        ax.set_yticklabels([feature_names[i] for i in sorted_idx[:top_n]])
        ax.set_title('Feature Contributions to Prediction')
        ax.set_xlabel('Contribution (positive = higher churn probability)')
        
    elif model_type in ["Random Forest", "Gradient Boosting"]:
        # For tree models, we'll plot feature contributions based on their importance
        # This is a simplification since individual predictions are complex in these models
        importances = model.feature_importances_
        feature_values = sample.values[0]
        
        # Scale feature values to [0,1] for visualization
        scaled_values = (feature_values - np.min(feature_values)) / (np.max(feature_values) - np.min(feature_values) + 1e-10)
        
        # Calculate simplified contribution (importance * scaled value)
        contributions = importances * scaled_values
        
        # Sort by contribution
        sorted_idx = contributions.argsort()[::-1]
        
        # Plot top contributions
        top_n = min(10, len(sorted_idx))
        ax.barh(range(top_n), contributions[sorted_idx[:top_n]])
        ax.set_yticks(range(top_n))
        ax.set_yticklabels([feature_names[i] for i in sorted_idx[:top_n]])
        ax.set_title('Estimated Feature Contributions')
        ax.set_xlabel('Estimated Contribution to Prediction')
        
    else:
        ax.text(0.5, 0.5, f"Individual prediction explanation is not available for {model_type}", 
                ha='center', va='center', fontsize=12)
    
    plt.tight_layout()
    return fig
