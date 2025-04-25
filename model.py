import streamlit as st
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

def get_model_params(model_type):
    """
    Display and collect parameters for the selected model.
    
    Args:
        model_type (str): Type of model to get parameters for
    
    Returns:
        dict: Model parameters
    """
    params = {}
    
    # Display parameters based on model type
    if model_type == "Logistic Regression":
        st.write("Logistic Regression Parameters:")
        
        params['C'] = st.slider(
            "Regularization strength (C)",
            0.01, 10.0, 1.0,
            help="Lower values indicate stronger regularization"
        )
        
        params['solver'] = st.selectbox(
            "Solver algorithm",
            ['liblinear', 'lbfgs', 'newton-cg', 'sag', 'saga'],
            help="Algorithm to use in the optimization problem"
        )
        
        params['max_iter'] = st.slider(
            "Maximum iterations",
            100, 1000, 100,
            help="Maximum number of iterations for solver convergence"
        )
        
        with st.expander("What is Logistic Regression?"):
            st.markdown("""
            **Logistic Regression** is a classification algorithm used to predict binary outcomes. Despite its name, it's used for classification, not regression problems.
            
            **How it works:**
            - Models the probability that an input belongs to a certain class
            - Uses the logistic function to transform its output to a probability between 0 and 1
            - If the probability is greater than 0.5, the input is classified as the positive class, otherwise as the negative class
            
            **Key parameters:**
            - **C**: Controls the regularization strength (inverse of regularization strength)
            - **Solver**: Algorithm used to find the model coefficients
            - **Max iterations**: Maximum number of iterations for the solver to converge
            
            **Advantages:**
            - Simple and interpretable
            - Outputs well-calibrated probabilities
            - Works well for linearly separable classes
            - Doesn't require much computational power
            
            **Disadvantages:**
            - Can't learn complex non-linear relationships
            - May underperform with imbalanced datasets
            """)
    
    elif model_type == "Random Forest":
        st.write("Random Forest Parameters:")
        
        params['n_estimators'] = st.slider(
            "Number of trees",
            50, 500, 100,
            help="The number of trees in the forest"
        )
        
        params['max_depth'] = st.slider(
            "Maximum tree depth",
            None, 50, 10,
            help="Maximum depth of the trees. None means unlimited."
        )
        
        params['min_samples_split'] = st.slider(
            "Minimum samples required to split",
            2, 20, 2,
            help="Minimum number of samples required to split an internal node"
        )
        
        params['min_samples_leaf'] = st.slider(
            "Minimum samples in leaf node",
            1, 20, 1,
            help="Minimum number of samples required to be at a leaf node"
        )
        
        params['random_state'] = 42  # For reproducibility
        
        with st.expander("What is Random Forest?"):
            st.markdown("""
            **Random Forest** is an ensemble learning method that builds multiple decision trees and combines their predictions.
            
            **How it works:**
            - Creates multiple decision trees on random subsets of data and features
            - For classification, takes the majority vote of all trees as the final prediction
            - Reduces overfitting compared to single decision trees
            
            **Key parameters:**
            - **n_estimators**: Number of trees in the forest
            - **max_depth**: Maximum depth of each tree
            - **min_samples_split**: Minimum samples needed to split a node
            - **min_samples_leaf**: Minimum samples required at a leaf node
            
            **Advantages:**
            - Handles non-linear relationships well
            - Robust to outliers and noisy data
            - Provides feature importance measures
            - Reduces overfitting by averaging multiple trees
            
            **Disadvantages:**
            - Less interpretable than a single decision tree
            - Can be computationally expensive with many trees
            - May overfit on noisy datasets if parameters not tuned properly
            """)
    
    elif model_type == "Gradient Boosting":
        st.write("Gradient Boosting Parameters:")
        
        params['n_estimators'] = st.slider(
            "Number of boosting stages",
            50, 500, 100,
            help="The number of boosting stages to perform"
        )
        
        params['learning_rate'] = st.slider(
            "Learning rate",
            0.01, 1.0, 0.1,
            help="Controls the contribution of each tree to the final outcome"
        )
        
        params['max_depth'] = st.slider(
            "Maximum tree depth",
            1, 20, 3,
            help="Maximum depth of the individual regression estimators"
        )
        
        params['min_samples_split'] = st.slider(
            "Minimum samples required to split",
            2, 20, 2,
            help="Minimum number of samples required to split an internal node"
        )
        
        params['random_state'] = 42  # For reproducibility
        
        with st.expander("What is Gradient Boosting?"):
            st.markdown("""
            **Gradient Boosting** is an ensemble technique that builds trees sequentially, with each tree correcting the errors of its predecessors.
            
            **How it works:**
            - Builds trees one at a time, where each new tree helps to correct errors made by previously trained trees
            - Trees are added until no further improvements can be made
            - Optimization is done using gradient descent
            
            **Key parameters:**
            - **n_estimators**: Number of boosting stages (trees)
            - **learning_rate**: How much each tree contributes to the final prediction
            - **max_depth**: Maximum depth of each tree
            - **min_samples_split**: Minimum samples needed to split a node
            
            **Advantages:**
            - Often provides higher accuracy than Random Forest
            - Handles complex, non-linear relationships well
            - Can capture feature interactions
            - Less prone to overfitting with proper tuning
            
            **Disadvantages:**
            - Can overfit on noisy data if not tuned properly
            - More sensitive to hyperparameter choices
            - More computationally intensive than Random Forest
            - Training is sequential (cannot be easily parallelized)
            """)
            
    elif model_type == "Support Vector Machine":
        st.write("Support Vector Machine Parameters:")
        
        params['C'] = st.slider(
            "Regularization parameter (C)",
            0.1, 10.0, 1.0,
            help="Penalty parameter of the error term"
        )
        
        params['kernel'] = st.selectbox(
            "Kernel type",
            ['rbf', 'linear', 'poly', 'sigmoid'],
            help="Specifies the kernel type to be used in the algorithm"
        )
        
        if params['kernel'] == 'rbf' or params['kernel'] == 'poly' or params['kernel'] == 'sigmoid':
            params['gamma'] = st.selectbox(
                "Kernel coefficient (gamma)",
                ['scale', 'auto', 0.001, 0.01, 0.1, 1, 10, 100],
                help="Kernel coefficient. 'scale' uses 1/(n_features*X.var()) and 'auto' uses 1/n_features"
            )
            
            # Convert gamma to float if it's a number
            if params['gamma'] not in ['scale', 'auto']:
                params['gamma'] = float(params['gamma'])
        
        params['probability'] = True  # Needed for ROC curve
        params['random_state'] = 42  # For reproducibility
        
        with st.expander("What is Support Vector Machine?"):
            st.markdown("""
            **Support Vector Machine (SVM)** is a powerful classifier that finds the optimal hyperplane that best separates classes.
            
            **How it works:**
            - Identifies the hyperplane that best separates different classes
            - Maximizes the margin between the closest points (support vectors) of different classes
            - Can map data to higher dimensions using kernel functions to find better separation
            
            **Key parameters:**
            - **C**: Regularization parameter (trade-off between smooth decision boundary and classifying training points correctly)
            - **Kernel**: Function used to transform the data (common ones: rbf, linear, poly)
            - **Gamma**: Defines how far the influence of a single training example reaches
            
            **Advantages:**
            - Effective in high-dimensional spaces
            - Memory efficient as it uses a subset of training points (support vectors)
            - Versatile through different kernel functions
            - Works well when classes are clearly separable
            
            **Disadvantages:**
            - Doesn't scale well to large datasets
            - Requires careful tuning of parameters
            - Doesn't directly provide probability estimates (requires additional computation)
            - Can be sensitive to the choice of kernel and regularization
            """)
            
    elif model_type == "K-Nearest Neighbors":
        st.write("K-Nearest Neighbors Parameters:")
        
        params['n_neighbors'] = st.slider(
            "Number of neighbors (K)",
            1, 30, 5,
            help="Number of neighbors to use for classification"
        )
        
        params['weights'] = st.selectbox(
            "Weight function",
            ['uniform', 'distance'],
            help="Weight function used in prediction"
        )
        
        params['algorithm'] = st.selectbox(
            "Algorithm used to compute nearest neighbors",
            ['auto', 'ball_tree', 'kd_tree', 'brute'],
            help="Algorithm used to compute the nearest neighbors"
        )
        
        params['p'] = st.selectbox(
            "Power parameter for Minkowski metric",
            [1, 2],
            format_func=lambda x: 'Manhattan (p=1)' if x == 1 else 'Euclidean (p=2)',
            help="Power parameter for the Minkowski metric. p=1 for Manhattan, p=2 for Euclidean"
        )
        
        with st.expander("What is K-Nearest Neighbors?"):
            st.markdown("""
            **K-Nearest Neighbors (KNN)** is a simple, instance-based learning algorithm that classifies data points based on how their neighbors are classified.
            
            **How it works:**
            - Stores all available cases
            - Classifies new cases based on a similarity measure (e.g., distance)
            - New data point is assigned the class most common among its K nearest neighbors
            
            **Key parameters:**
            - **n_neighbors (K)**: Number of neighbors to consider
            - **weights**: Whether all neighbors are weighted equally or closer neighbors have more influence
            - **algorithm**: Method used to compute nearest neighbors
            - **p**: Distance metric (1 for Manhattan, 2 for Euclidean)
            
            **Advantages:**
            - Simple to understand and implement
            - No training phase (lazy learning)
            - Naturally handles multi-class problems
            - Few parameters to tune
            
            **Disadvantages:**
            - Computationally expensive for large datasets
            - Sensitive to irrelevant features
            - Requires feature scaling
            - Memory-intensive as it stores the entire training dataset
            """)
    
    return params

def select_model(model_type, params):
    """
    Select and configure a model based on the specified type and parameters.
    
    Args:
        model_type (str): Type of model to create
        params (dict): Parameters for the model
    
    Returns:
        object: Configured model instance
    """
    if model_type == "Logistic Regression":
        return LogisticRegression(**params)
    
    elif model_type == "Random Forest":
        return RandomForestClassifier(**params)
    
    elif model_type == "Gradient Boosting":
        return GradientBoostingClassifier(**params)
    
    elif model_type == "Support Vector Machine":
        return SVC(**params)
    
    elif model_type == "K-Nearest Neighbors":
        return KNeighborsClassifier(**params)
    
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

def train_model(model, X_train, y_train):
    """
    Train the model on the training data.
    
    Args:
        model: The model to train
        X_train: Training features
        y_train: Training labels
    
    Returns:
        object: Trained model
    """
    model.fit(X_train, y_train)
    return model
