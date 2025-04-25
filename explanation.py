import numpy as np
import matplotlib.pyplot as plt
import shap
import streamlit as st

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

def explain_prediction_shap(model, model_type, X_sample, feature_names):
    """
    Generate SHAP values for model explanation.
    
    Args:
        model: Trained model
        model_type (str): Type of model
        X_sample: Sample data to explain (can be subset of test data)
        feature_names (list): Names of features
    
    Returns:
        matplotlib.figure.Figure: SHAP summary plot
    """
    # Convert to numpy for compatibility with SHAP
    X_numpy = X_sample.values
    
    # Different explainers for different model types
    if model_type == "Logistic Regression":
        # Linear explainer for linear models
        explainer = shap.LinearExplainer(model, X_numpy, feature_perturbation="interventional")
    
    elif model_type in ["Random Forest", "Gradient Boosting"]:
        # Tree explainer for tree-based models
        explainer = shap.TreeExplainer(model)
    
    elif model_type == "Support Vector Machine":
        # Kernel explainer for SVM (computationally intensive)
        X_background = shap.sample(X_numpy, 100)  # Sample for background
        explainer = shap.KernelExplainer(model.predict_proba, X_background)
    
    else:
        # Default to Kernel explainer for other models
        X_background = shap.sample(X_numpy, 100)  # Sample for background
        explainer = shap.KernelExplainer(model.predict_proba, X_background)
    
    # Calculate SHAP values
    shap_values = explainer.shap_values(X_numpy)
    
    # For models that return a list of arrays (one per class)
    if isinstance(shap_values, list):
        # Take values for the positive class (assuming binary classification)
        shap_values = shap_values[1]
    
    # Create SHAP summary plot
    plt.figure()
    fig = plt.gcf()
    shap.summary_plot(shap_values, X_numpy, feature_names=feature_names, show=False)
    plt.tight_layout()
    
    return fig

def plot_individual_shap(model, model_type, sample, feature_names):
    """
    Generate SHAP waterfall plot for an individual prediction.
    
    Args:
        model: Trained model
        model_type (str): Type of model
        sample: Single data sample to explain
        feature_names (list): Names of features
    
    Returns:
        matplotlib.figure.Figure: SHAP waterfall plot
    """
    # Convert to numpy for compatibility with SHAP
    sample_numpy = sample.values
    
    # Different explainers for different model types
    if model_type == "Logistic Regression":
        # Linear explainer for linear models
        explainer = shap.LinearExplainer(model, sample_numpy, feature_perturbation="interventional")
    
    elif model_type in ["Random Forest", "Gradient Boosting"]:
        # Tree explainer for tree-based models
        explainer = shap.TreeExplainer(model)
    
    else:
        # Default to Kernel explainer for other models
        explainer = shap.KernelExplainer(model.predict_proba, sample_numpy)
    
    # Calculate SHAP values
    shap_values = explainer.shap_values(sample_numpy)
    
    # For models that return a list of arrays (one per class)
    if isinstance(shap_values, list):
        # Take values for the positive class (assuming binary classification)
        shap_values = shap_values[1]
    
    # Create SHAP waterfall plot
    plt.figure(figsize=(10, 8))
    fig = plt.gcf()
    shap.waterfall_plot(explainer.expected_value[1] if isinstance(explainer.expected_value, np.ndarray) 
                      else explainer.expected_value, 
                      shap_values[0], 
                      feature_names=feature_names,
                      show=False)
    plt.tight_layout()
    
    return fig
