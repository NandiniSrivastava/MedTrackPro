import streamlit as st

def display_step_info(step):
    """
    Display informational text for each step of the ML workflow.
    
    Args:
        step (str): The current workflow step
    """
    if step == "data_exploration":
        st.markdown("""
        Data exploration is the first step in any machine learning project. It helps you understand:
        
        - The structure and size of your dataset
        - The distribution of your features and target variable
        - Relationships between variables
        - Potential issues like missing values or outliers
        
        Use the tools below to explore the Telco Customer Churn dataset.
        """)
    
    elif step == "data_preprocessing":
        st.markdown("""
        Data preprocessing is crucial for preparing your data for modeling:
        
        - **Handling Missing Values**: Incomplete data can lead to biased models
        - **Categorical Encoding**: Converting text categories to numbers
        - **Target Preparation**: Ensuring the target variable is in the right format
        
        These steps help create clean, machine-learning-ready data.
        """)
    
    elif step == "feature_engineering":
        st.markdown("""
        Feature engineering improves your model by creating better inputs:
        
        - **Scaling**: Normalizing numerical features to similar ranges
        - **Feature Selection**: Choosing the most informative variables
        - **Data Splitting**: Creating training and testing datasets
        
        Good feature engineering can significantly improve model performance.
        """)
    
    elif step == "model_selection":
        st.markdown("""
        Model selection involves choosing and training the right algorithm:
        
        - Different models have different strengths and weaknesses
        - Parameters control the model's behavior and performance
        - Training fits the model to patterns in your data
        
        Select a model and adjust its parameters before training.
        """)
    
    elif step == "model_evaluation":
        st.markdown("""
        Model evaluation assesses how well your model performs:
        
        - **Classification Metrics**: Accuracy, precision, recall, F1-score
        - **Confusion Matrix**: Shows correct and incorrect predictions
        - **ROC Curve**: Visualizes the trade-off between true and false positives
        
        These metrics help you understand if your model is making reliable predictions.
        """)
    
    elif step == "model_interpretation":
        st.markdown("""
        Model interpretation explains why your model makes certain predictions:
        
        - **Feature Importance**: Which variables influence predictions the most
        - **SHAP Values**: How each feature contributes to individual predictions
        - **Individual Explanations**: Understanding specific customer predictions
        
        Interpretability is crucial for gaining actionable insights and building trust in your model.
        """)

def display_metrics_explanation():
    """
    Display explanations for common evaluation metrics.
    """
    st.markdown("""
    ### Evaluation Metrics Explained
    
    **Accuracy**
    - The proportion of correct predictions (both true positives and true negatives)
    - Formula: (TP + TN) / (TP + TN + FP + FN)
    - When to use: Best for balanced datasets
    
    **Precision**
    - The proportion of positive identifications that were actually correct
    - Formula: TP / (TP + FP)
    - When to use: When false positives are costly (e.g., unnecessary retention offers)
    
    **Recall (Sensitivity)**
    - The proportion of actual positives that were correctly identified
    - Formula: TP / (TP + FN)
    - When to use: When false negatives are costly (e.g., missing customers who will churn)
    
    **F1 Score**
    - The harmonic mean of precision and recall
    - Formula: 2 * (Precision * Recall) / (Precision + Recall)
    - When to use: When you need to balance precision and recall
    
    **Balanced Accuracy**
    - The average of recall for each class
    - Useful for imbalanced datasets
    
    **ROC AUC**
    - Area Under the Receiver Operating Characteristic curve
    - Measures the model's ability to distinguish between classes
    - Range: 0.5 (random guess) to 1.0 (perfect classification)
    
    **PR AUC**
    - Area Under the Precision-Recall curve
    - Useful for imbalanced datasets
    
    ### Classification Report Terminology
    
    **Support**
    - The number of samples in each class
    
    **Macro Avg**
    - The unweighted mean of metrics for each class
    
    **Weighted Avg**
    - The weighted mean of metrics for each class, where weights are based on class frequency
    """)
