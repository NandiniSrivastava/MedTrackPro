import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve
import pickle
import os
import io
from io import BytesIO

# Set page configuration
st.set_page_config(
    page_title="Healthcare Model SHAP Explainer",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better appearance
st.markdown("""
<style>
    .main .block-container {
        padding-top: 2rem;
    }
    .stTabs [data-baseweb="tab-panel"] {
        padding-top: 1rem;
    }
    h1, h2, h3 {
        color: #0072b2;
    }
    .stButton>button {
        background-color: #0072b2;
        color: white;
    }
    .stProgress .st-bo {
        background-color: #0072b2;
    }
</style>
""", unsafe_allow_html=True)

# Create sidebar
st.sidebar.title("🏥 Healthcare ML Explainer")
st.sidebar.markdown("---")

# Main navigation
app_mode = st.sidebar.selectbox(
    "Choose the app mode",
    ["Introduction", "Dataset Exploration", "Model Training & Evaluation", "SHAP Explanations"]
)

# Function to load data
@st.cache_data
def load_data():
    # Load diabetes dataset from sklearn
    from sklearn.datasets import load_diabetes, load_breast_cancer
    
    if selected_dataset == "Diabetes":
        # Using regression dataset but converting to classification
        data = load_diabetes()
        X = pd.DataFrame(data.data, columns=data.feature_names)
        y = pd.Series(data.target)
        # Convert to binary classification (above/below median)
        y_binary = (y > y.median()).astype(int)
        return X, y_binary, data.feature_names, ["Non-diabetic", "Diabetic"]
    else:  # Breast Cancer
        data = load_breast_cancer()
        X = pd.DataFrame(data.data, columns=data.feature_names)
        y = pd.Series(data.target)
        return X, y, data.feature_names, ["Malignant", "Benign"]

# Dataset selection in sidebar
selected_dataset = st.sidebar.radio(
    "Select Healthcare Dataset",
    ["Diabetes", "Breast Cancer"]
)

# Load the selected dataset
X, y, feature_names, class_names = load_data()

# Introduction page
if app_mode == "Introduction":
    st.title("Explaining Healthcare Models with SHAP")
    
    st.markdown("""
    ## Welcome to the Healthcare ML Explainer
    
    This application demonstrates how to use SHAP (SHapley Additive exPlanations) to interpret
    machine learning models for healthcare predictions.
    
    ### What you can do with this app:
    
    1. **Explore healthcare datasets**: Examine the structure and characteristics of healthcare data
    2. **Train predictive models**: Build and evaluate machine learning models on healthcare data
    3. **Interpret models with SHAP**: Understand how models make predictions and which features are most important
    
    ### Why Model Interpretability Matters in Healthcare
    
    Machine learning models are increasingly being used in healthcare for diagnosis, prognosis, and treatment planning.
    However, black-box models that make predictions without explanations are difficult to trust and implement in clinical settings.
    
    SHAP provides a unified approach to explain the output of any machine learning model, helping clinicians understand
    why certain predictions are made, which can:
    
    - Increase trust in model predictions
    - Reveal potential biases in models
    - Provide clinical insights about important health factors
    - Aid in regulatory compliance and model validation
    
    ### Selected Dataset: {selected_dataset}
    
    You've selected the **{selected_dataset}** dataset. Use the sidebar to navigate through the app's sections.
    """)
    
    # Display dataset overview
    st.markdown(f"### {selected_dataset} Dataset Overview")
    
    # Create two columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Sample Data**")
        st.dataframe(X.head())
    
    with col2:
        st.write("**Dataset Statistics**")
        st.write(f"- **Number of samples**: {X.shape[0]}")
        st.write(f"- **Number of features**: {X.shape[1]}")
        st.write(f"- **Target class distribution**:")
        fig, ax = plt.subplots(figsize=(6, 4))
        y_counts = pd.Series(y).value_counts().sort_index()
        ax.bar(
            [class_names[i] for i in y_counts.index],
            y_counts.values,
            color=['#ff9999', '#66b3ff']
        )
        for i, v in enumerate(y_counts.values):
            ax.text(i, v + 5, str(v), ha='center')
        
        ax.set_xlabel('Class')
        ax.set_ylabel('Count')
        ax.set_title('Target Distribution')
        st.pyplot(fig)

# Dataset Exploration
elif app_mode == "Dataset Exploration":
    st.title(f"{selected_dataset} Dataset Exploration")
    
    st.markdown("""
    Exploring the dataset is a crucial first step in building interpretable models. Let's examine the feature 
    distributions and relationships to better understand the data.
    """)
    
    # Display data overview
    st.subheader("Data Overview")
    st.dataframe(X.head())
    
    # Feature statistics
    st.subheader("Feature Statistics")
    st.dataframe(X.describe())
    
    # Feature distributions
    st.subheader("Feature Distributions")
    
    # Select features to visualize
    selected_features = st.multiselect(
        "Select features to visualize",
        feature_names,
        default=feature_names[:3]
    )
    
    if selected_features:
        # Plot histograms
        fig, axes = plt.subplots(nrows=1, ncols=len(selected_features), figsize=(4*len(selected_features), 4))
        if len(selected_features) == 1:
            axes = [axes]  # Make iterable if only one feature selected
            
        for i, feature in enumerate(selected_features):
            sns.histplot(data=X, x=feature, hue=pd.Series(y).map({0: class_names[0], 1: class_names[1]}), 
                        ax=axes[i], kde=True, bins=20)
            axes[i].set_title(f'Distribution of {feature}')
            axes[i].set_xlabel(feature)
            
        plt.tight_layout()
        st.pyplot(fig)
    
    # Correlation heatmap
    st.subheader("Feature Correlation Heatmap")
    # Create correlation matrix and plot
    corr_matrix = X.corr()
    plt.figure(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr_matrix, mask=mask, annot=True, cmap="coolwarm", vmin=-1, vmax=1, 
                fmt=".2f", linewidths=.5, ax=ax)
    ax.set_title('Feature Correlation Heatmap')
    st.pyplot(fig)
    
    # Feature importance based on correlation with target
    st.subheader("Feature Correlation with Target")
    corr_with_target = pd.DataFrame({
        'Feature': feature_names,
        'Correlation': [np.corrcoef(X[feature], y)[0, 1] for feature in feature_names]
    }).sort_values('Correlation', ascending=False)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(
        corr_with_target['Feature'], 
        corr_with_target['Correlation'].abs(),
        color=[('#ff9999' if c < 0 else '#66b3ff') for c in corr_with_target['Correlation']]
    )
    ax.set_xlabel('Absolute Correlation with Target')
    ax.set_title('Feature Correlation with Target')
    
    # Add correlation values as text
    for i, bar in enumerate(bars):
        ax.text(
            bar.get_width() + 0.01, 
            bar.get_y() + bar.get_height()/2, 
            f'{corr_with_target["Correlation"].iloc[i]:.3f}', 
            va='center'
        )
    
    st.pyplot(fig)
    
    # Pairplot for selected features
    st.subheader("Feature Relationships")
    st.markdown("Select a few features (2-4 recommended) to visualize relationships:")
    pair_features = st.multiselect(
        "Select features for pairplot",
        feature_names,
        default=feature_names[:3] if len(feature_names) >= 3 else feature_names
    )
    
    if pair_features and len(pair_features) >= 2:
        # Create a dataframe with selected features and target
        pair_df = X[pair_features].copy()
        pair_df['Target'] = pd.Series(y).map({0: class_names[0], 1: class_names[1]})
        
        # Create pairplot
        plt.figure(figsize=(12, 10))
        pair_plot = sns.pairplot(pair_df, hue='Target', corner=True)
        pair_plot.fig.suptitle(f'Relationships between selected features', y=1.02)
        st.pyplot(pair_plot.fig)
    else:
        st.info("Please select at least 2 features for the pairplot.")

# Model Training & Evaluation
elif app_mode == "Model Training & Evaluation":
    st.title("Model Training & Evaluation")
    
    st.markdown("""
    In this section, we'll train machine learning models to predict health outcomes and evaluate their performance.
    These models will be used in the next section for SHAP explanations.
    """)
    
    # Model selection
    model_type = st.selectbox(
        "Select model type",
        ["Logistic Regression", "Random Forest", "Gradient Boosting"]
    )
    
    # Split data into train and test sets
    test_size = st.slider("Test set size", 0.1, 0.5, 0.2, 0.05)
    
    # Show splitting process
    with st.expander("Data splitting details"):
        st.markdown(f"""
        - Training set: {int((1-test_size) * 100)}% of data ({int((1-test_size) * X.shape[0])} samples)
        - Test set: {int(test_size * 100)}% of data ({int(test_size * X.shape[0])} samples)
        - Random state: 42 (for reproducibility)
        """)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    # Model hyperparameters
    st.subheader("Model Hyperparameters")
    
    if model_type == "Logistic Regression":
        C = st.slider("Regularization strength (C)", 0.01, 10.0, 1.0, 0.1)
        model = LogisticRegression(C=C, max_iter=1000, random_state=42)
    
    elif model_type == "Random Forest":
        n_estimators = st.slider("Number of trees", 10, 500, 100, 10)
        max_depth = st.slider("Maximum tree depth", 1, 50, 10, 1)
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42
        )
    
    elif model_type == "Gradient Boosting":
        n_estimators = st.slider("Number of boosting stages", 10, 500, 100, 10)
        learning_rate = st.slider("Learning rate", 0.01, 1.0, 0.1, 0.01)
        max_depth = st.slider("Maximum tree depth", 1, 10, 3, 1)
        model = GradientBoostingClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=42
        )
    
    # Create pipeline with preprocessing
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', model)
    ])
    
    # Train model button
    if st.button("Train Model"):
        with st.spinner(f"Training {model_type} model..."):
            # Fit the model
            pipeline.fit(X_train, y_train)
            
            # Make predictions
            y_pred = pipeline.predict(X_test)
            y_prob = pipeline.predict_proba(X_test)[:, 1]
            
            # Save the model for SHAP explanation
            if not os.path.exists('models'):
                os.makedirs('models')
            
            with open(f'models/{selected_dataset}_{model_type.replace(" ", "_")}.pkl', 'wb') as f:
                pickle.dump({'pipeline': pipeline, 'feature_names': feature_names, 'class_names': class_names}, f)
            
            # Save to session state for SHAP explanations
            st.session_state['model'] = pipeline
            st.session_state['X_train'] = X_train
            st.session_state['X_test'] = X_test
            st.session_state['y_train'] = y_train
            st.session_state['y_test'] = y_test
            st.session_state['feature_names'] = feature_names
            st.session_state['class_names'] = class_names
            st.session_state['selected_dataset'] = selected_dataset
            st.session_state['model_type'] = model_type
            
            # Display success message
            st.success("Model trained successfully! You can now view the evaluation results below.")
        
        # Model evaluation
        st.subheader("Model Evaluation")
        
        # Create tabs for different evaluation metrics
        eval_tab1, eval_tab2, eval_tab3 = st.tabs(["Performance Metrics", "Confusion Matrix", "ROC & PR Curves"])
        
        # Tab 1: Performance metrics
        with eval_tab1:
            col1, col2, col3, col4, col5 = st.columns(5)
            
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_prob)
            
            col1.metric("Accuracy", f"{accuracy:.3f}")
            col2.metric("Precision", f"{precision:.3f}")
            col3.metric("Recall", f"{recall:.3f}")
            col4.metric("F1 Score", f"{f1:.3f}")
            col5.metric("AUC-ROC", f"{auc:.3f}")
            
            st.markdown("""
            #### Understanding these metrics:
            - **Accuracy**: Overall proportion of correct predictions
            - **Precision**: Proportion of positive predictions that were actually positive
            - **Recall**: Proportion of actual positives that were correctly predicted
            - **F1 Score**: Harmonic mean of precision and recall
            - **AUC-ROC**: Area under the Receiver Operating Characteristic curve
            """)
        
        # Tab 2: Confusion Matrix
        with eval_tab2:
            st.markdown("""
            The confusion matrix shows:
            - True Positives (TP): Correctly predicted positives
            - False Positives (FP): Incorrectly predicted positives
            - True Negatives (TN): Correctly predicted negatives
            - False Negatives (FN): Incorrectly predicted negatives
            """)
            
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(
                cm, 
                annot=True, 
                fmt="d", 
                cmap="Blues",
                xticklabels=class_names,
                yticklabels=class_names,
                ax=ax
            )
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            ax.set_title("Confusion Matrix")
            st.pyplot(fig)
            
        # Tab 3: ROC and PR curves
        with eval_tab3:
            # Create ROC curve
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            
            # Create PR curve
            precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_prob)
            
            # Plot both curves
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # ROC curve
            ax1.plot(fpr, tpr, label=f'AUC = {auc:.3f}')
            ax1.plot([0, 1], [0, 1], 'k--')
            ax1.set_xlabel('False Positive Rate')
            ax1.set_ylabel('True Positive Rate')
            ax1.set_title('ROC Curve')
            ax1.legend(loc='lower right')
            
            # PR curve
            ax2.plot(recall_curve, precision_curve, label=f'Precision-Recall')
            ax2.axhline(y=sum(y_test)/len(y_test), color='r', linestyle='--', 
                       label=f'No Skill = {sum(y_test)/len(y_test):.3f}')
            ax2.set_xlabel('Recall')
            ax2.set_ylabel('Precision')
            ax2.set_title('Precision-Recall Curve')
            ax2.legend(loc='lower left')
            
            plt.tight_layout()
            st.pyplot(fig)
            
            st.markdown("""
            #### Interpreting these curves:
            
            **ROC Curve**:
            - Plots the True Positive Rate vs. False Positive Rate
            - The diagonal line represents a random classifier (AUC = 0.5)
            - A perfect classifier would have an AUC of 1.0
            
            **Precision-Recall Curve**:
            - Shows the trade-off between precision and recall at different thresholds
            - Especially useful for imbalanced datasets
            - The "No Skill" line shows the performance of a random classifier
            """)

# SHAP Explanations
elif app_mode == "SHAP Explanations":
    st.title("SHAP Explanations")
    
    if 'model' not in st.session_state:
        st.warning("Please train a model first in the 'Model Training & Evaluation' section.")
    else:
        st.markdown(f"""
        ## Explaining {st.session_state['model_type']} Model for {st.session_state['selected_dataset']} Dataset
        
        SHAP (SHapley Additive exPlanations) is a game theoretic approach to explain the output of any machine learning model. 
        It connects optimal credit allocation with local explanations using the classic Shapley values from game theory and their extensions.
        """)
        
        # Function to calculate SHAP values with simulation
        def calculate_shapley_values(model, X_background, X_explain, feature_names):
            """
            Calculate approximate SHAP values through simulation
            """
            # Get the mean prediction as base value
            X_background_scaled = model.named_steps['scaler'].transform(X_background)
            base_value = model.named_steps['model'].predict_proba(X_background_scaled)[:, 1].mean()
            
            # Scale the data for explanation
            X_explain_scaled = model.named_steps['scaler'].transform(X_explain)
            
            # Get number of features
            n_features = X_explain.shape[1]
            
            # Initialize SHAP values
            shap_values = np.zeros((X_explain.shape[0], n_features))
            
            # Number of Monte Carlo samples for approximation
            n_samples = 100
            
            # For each instance to explain
            for i in range(X_explain.shape[0]):
                single_instance = X_explain.iloc[i:i+1]
                single_instance_scaled = X_explain_scaled[i:i+1]
                
                # Get the prediction for this instance
                pred = model.named_steps['model'].predict_proba(single_instance_scaled)[0, 1]
                
                # For each feature
                for j in range(n_features):
                    # Create random orderings of features
                    feature_contributions = []
                    
                    for _ in range(n_samples):
                        # Generate random permutation of features
                        perm = np.random.permutation(n_features)
                        
                        # Find position of current feature in permutation
                        feature_idx = np.where(perm == j)[0][0]
                        
                        # Create two scenarios - with and without the feature
                        coalition_with = perm[:feature_idx+1]  # Including current feature
                        coalition_without = perm[:feature_idx]  # Excluding current feature
                        
                        # Create synthetic instances for both scenarios
                        instance_with = X_background.iloc[0].copy()
                        instance_without = X_background.iloc[0].copy()
                        
                        # Set values for coalitions
                        instance_with[coalition_with] = single_instance.iloc[0][coalition_with]
                        instance_without[coalition_without] = single_instance.iloc[0][coalition_without]
                        
                        # Transform instances
                        instance_with = np.array(instance_with).reshape(1, -1)
                        instance_without = np.array(instance_without).reshape(1, -1)
                        
                        with_scaled = model.named_steps['scaler'].transform(instance_with)
                        without_scaled = model.named_steps['scaler'].transform(instance_without)
                        
                        # Get predictions
                        pred_with = model.named_steps['model'].predict_proba(with_scaled)[0, 1]
                        pred_without = model.named_steps['model'].predict_proba(without_scaled)[0, 1]
                        
                        # The contribution is the difference
                        feature_contributions.append(pred_with - pred_without)
                    
                    # Average the contributions across all permutations
                    shap_values[i, j] = sum(feature_contributions) / n_samples
            
            # Create a dataframe with the SHAP values
            shap_df = pd.DataFrame(shap_values, columns=feature_names)
            
            return shap_df, base_value
        
        # Function to create SHAP summary plot
        def create_shap_summary_plot(shap_values, features, feature_names):
            """Create a SHAP summary plot showing feature importance"""
            # Sort features by absolute SHAP value
            mean_abs_shap = np.abs(shap_values).mean(axis=0)
            feature_importance = pd.DataFrame({
                'Feature': feature_names,
                'Importance': mean_abs_shap
            }).sort_values('Importance', ascending=False)
            
            # Create figure
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # For each feature (starting with most important)
            for i, feature in enumerate(feature_importance['Feature']):
                feature_idx = feature_names.index(feature)
                feature_shap = shap_values[:, feature_idx]
                feature_value = features[:, feature_idx]
                
                # Normalize feature values
                feature_value_norm = (feature_value - feature_value.min()) / (feature_value.max() - feature_value.min())
                
                # Plot points in a scatter pattern, colored by feature value
                sc = ax.scatter(
                    feature_shap,  # x position (SHAP value)
                    len(feature_importance) - 1 - i,  # y position (feature rank)
                    c=feature_value_norm,  # color by normalized feature value
                    cmap='coolwarm',  # colormap
                    alpha=0.8,  # transparency
                    s=30  # point size
                )
            
            # Add feature names to y-axis
            ax.set_yticks(range(len(feature_importance)))
            ax.set_yticklabels(feature_importance['Feature'])
            
            # Add labels and title
            ax.set_xlabel('SHAP Value (Impact on Prediction)')
            ax.set_title('Feature Importance (Mean |SHAP Value|)')
            
            # Add colorbar
            cbar = plt.colorbar(sc, ax=ax)
            cbar.set_label('Normalized Feature Value')
            
            # Add vertical line at x=0
            ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
            
            plt.tight_layout()
            return fig
        
        # Function to create SHAP waterfall plot
        def create_shap_waterfall_plot(shap_values, feature_names, base_value, instance_prediction, title=None):
            """Create a SHAP waterfall plot for a single prediction"""
            # Sort SHAP values by absolute magnitude
            idx = np.argsort(np.abs(shap_values))[::-1]
            sorted_features = [feature_names[i] for i in idx]
            sorted_shap = shap_values[idx]
            
            # Take top 10 features
            top_n = min(10, len(sorted_features))
            top_features = sorted_features[:top_n]
            top_shap = sorted_shap[:top_n]
            
            # Create figure
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Start with base value
            cumulative = [base_value]
            
            # Add each feature contribution
            for i in range(top_n):
                cumulative.append(cumulative[-1] + top_shap[i])
            
            # Set positions for bars
            pos = np.arange(len(cumulative))
            
            # Plot base value
            ax.bar(pos[0], cumulative[0], bottom=0, color='grey', width=0.6, alpha=0.7)
            
            # Plot each feature contribution
            colors = ['red' if x < 0 else 'green' for x in top_shap]
            for i in range(top_n):
                # Plot from previous cumulative to new value
                start = cumulative[i]
                end = cumulative[i+1]
                height = end - start
                bottom = min(start, end)
                
                ax.bar(pos[i+1], height, bottom=bottom, color=colors[i], width=0.6)
            
            # Plot final prediction
            ax.bar(pos[-1], instance_prediction, bottom=0, color='blue', width=0.6, alpha=0.7)
            
            # Add connecting lines
            for i in range(len(cumulative)-1):
                ax.plot([pos[i], pos[i+1]], [cumulative[i], cumulative[i]], color='black', alpha=0.3)
            
            # Add labels
            labels = ['Base value'] + top_features + ['Final prediction']
            ax.set_xticks(pos)
            ax.set_xticklabels(labels, rotation=45, ha='right')
            
            # Add value labels
            ax.text(0, cumulative[0]/2, f'{cumulative[0]:.3f}', ha='center', va='center')
            for i in range(top_n):
                if top_shap[i] > 0:
                    txt = f'+{top_shap[i]:.3f}'
                else:
                    txt = f'{top_shap[i]:.3f}'
                midpoint = cumulative[i] + top_shap[i]/2
                ax.text(i+1, midpoint, txt, ha='center', va='center')
            
            ax.text(pos[-1], instance_prediction/2, f'{instance_prediction:.3f}', ha='center', va='center')
            
            # Set title and labels
            if title:
                ax.set_title(title)
            else:
                ax.set_title('SHAP Waterfall Plot: Feature Contributions to Prediction')
            ax.set_ylabel('Prediction value')
            
            plt.tight_layout()
            return fig
        
        # Function to create SHAP force plot
        def create_shap_force_plot(shap_values, feature_names, feature_values, base_value, title=None):
            """Create a SHAP force plot for a single prediction"""
            # Sort by absolute contribution
            idx = np.argsort(np.abs(shap_values))[::-1]
            sorted_shap = shap_values[idx]
            sorted_features = [feature_names[i] for i in idx]
            sorted_values = feature_values[idx]
            
            # Take top 10 features
            top_n = min(10, len(sorted_features))
            
            # Create figure
            fig, ax = plt.subplots(figsize=(10, 4))
            
            # Starting position
            left_pos = 0
            right_pos = 0
            
            # Plot positive and negative contributions
            for i in range(top_n):
                width = abs(sorted_shap[i])
                
                # Color based on contribution direction
                if sorted_shap[i] > 0:  # Positive contribution
                    ax.barh(0, width, height=0.8, left=right_pos, color='red', alpha=0.7)
                    # Add feature name
                    if width > 0.02:  # Only add text if bar is wide enough
                        ax.text(
                            right_pos + width/2, 
                            0, 
                            f'{sorted_features[i]}', 
                            ha='center', 
                            va='center',
                            color='white',
                            fontsize=9
                        )
                    right_pos += width
                else:  # Negative contribution
                    ax.barh(0, width, height=0.8, left=left_pos-width, color='blue', alpha=0.7)
                    # Add feature name
                    if width > 0.02:  # Only add text if bar is wide enough
                        ax.text(
                            left_pos - width/2, 
                            0, 
                            f'{sorted_features[i]}', 
                            ha='center', 
                            va='center',
                            color='white', 
                            fontsize=9
                        )
                    left_pos -= width
            
            # Add axis information
            ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
            
            # Add values
            final_pred = base_value + np.sum(shap_values)
            ax.text(left_pos-0.05, 0, f'{base_value:.3f}', ha='right', va='center', fontsize=10)
            ax.text(right_pos+0.05, 0, f'{final_pred:.3f}', ha='left', va='center', fontsize=10)
            
            # Remove y-axis ticks and labels
            ax.set_yticks([])
            ax.set_yticklabels([])
            
            # Increase horizontal space
            x_min = min(left_pos-0.2, -0.5)
            x_max = max(right_pos+0.2, 0.5)
            ax.set_xlim(x_min, x_max)
            
            # Set title
            if title:
                ax.set_title(title)
            else:
                ax.set_title('SHAP Force Plot: Feature Contributions')
            
            # Add foot note
            plt.figtext(
                0.5, 0.01, 
                'Red bars push prediction higher, blue bars push prediction lower',
                ha='center', fontsize=9
            )
            
            return fig
        
        # Get model, data, and feature names from session state
        model = st.session_state['model']
        X_train = st.session_state['X_train']
        X_test = st.session_state['X_test']
        y_test = st.session_state['y_test']
        feature_names = st.session_state['feature_names']
        model_type = st.session_state['model_type']
        
        # Explain SHAP values interface
        st.subheader("Choose what to explain")
        
        explanation_type = st.radio(
            "Select explanation type",
            ["Global Model Interpretation", "Individual Prediction Explanation"]
        )
        
        # If global interpretation
        if explanation_type == "Global Model Interpretation":
            st.markdown("""
            Global interpretation shows how each feature impacts the model's predictions across the entire dataset. 
            """)
            
            # Sample size for explanation (fewer samples for faster computation)
            sample_size = st.slider(
                "Number of samples to use for SHAP calculations", 
                min_value=10, 
                max_value=min(100, X_test.shape[0]), 
                value=min(50, X_test.shape[0])
            )
            
            with st.spinner("Calculating SHAP values (this may take a moment)..."):
                # Randomly sample data
                idx = np.random.choice(X_test.shape[0], sample_size, replace=False)
                X_sampled = X_test.iloc[idx]
                
                # Calculate SHAP values
                shap_df, base_value = calculate_shapley_values(
                    model,
                    X_train,
                    X_sampled,
                    feature_names
                )
                
                # Display results in tabs
                shap_tab1, shap_tab2, shap_tab3 = st.tabs(["SHAP Summary Plot", "Feature Importance", "SHAP Values Table"])
                
                with shap_tab1:
                    st.markdown("### SHAP Summary Plot")
                    st.markdown("""
                    This plot shows how each feature impacts predictions across multiple instances:
                    - Features are ordered by importance (top = most important)
                    - Each point represents one sample in the dataset
                    - Position on x-axis shows impact on prediction
                    - Color represents the feature's value (red = high, blue = low)
                    """)
                    
                    fig = create_shap_summary_plot(shap_df.values, X_sampled.values, feature_names)
                    st.pyplot(fig)
                
                with shap_tab2:
                    st.markdown("### Feature Importance Based on SHAP Values")
                    
                    # Calculate mean absolute SHAP value for each feature
                    mean_abs_shap = np.abs(shap_df).mean().sort_values(ascending=False)
                    
                    # Create bar chart
                    fig, ax = plt.subplots(figsize=(10, 8))
                    bars = ax.barh(mean_abs_shap.index, mean_abs_shap.values)
                    
                    # Add data labels
                    for i, bar in enumerate(bars):
                        ax.text(
                            bar.get_width() + 0.002,
                            bar.get_y() + bar.get_height()/2,
                            f'{mean_abs_shap.values[i]:.3f}',
                            va='center'
                        )
                    
                    ax.set_xlabel('Mean |SHAP Value|')
                    ax.set_title('Feature Importance')
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    st.markdown("""
                    This plot shows the average magnitude of each feature's impact on model output:
                    - Larger values indicate more important features
                    - This considers the absolute impact, regardless of direction
                    """)
                
                with shap_tab3:
                    st.markdown("### SHAP Values")
                    
                    # Display SHAP values as a table
                    st.dataframe(shap_df)
                    
                    st.markdown("""
                    This table shows the raw SHAP values for each feature and instance:
                    - Positive values increase the prediction
                    - Negative values decrease the prediction
                    - The magnitude represents the strength of the impact
                    """)
        
        # If individual prediction explanation
        else:
            st.markdown("""
            Individual explanation shows how features contribute to a specific prediction for a single patient.
            """)
            
            # Select a sample to explain
            sample_index = st.number_input(
                "Select sample index from test set to explain",
                min_value=0,
                max_value=X_test.shape[0] - 1,
                value=0
            )
            
            # Get the sample
            sample = X_test.iloc[[sample_index]]
            actual_label = y_test.iloc[sample_index]
            
            # Make prediction
            y_pred_prob = model.predict_proba(sample)[0, 1]
            y_pred = 1 if y_pred_prob >= 0.5 else 0
            
            # Display sample information
            st.subheader("Sample Information")
            
            # Create two columns
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"**Actual diagnosis:** {st.session_state['class_names'][actual_label]}")
                st.markdown(f"**Predicted diagnosis:** {st.session_state['class_names'][y_pred]}")
                
                # Check if prediction is correct
                if y_pred == actual_label:
                    st.markdown("**Prediction Status:** ✅ Correct")
                else:
                    st.markdown("**Prediction Status:** ❌ Incorrect")
            
            with col2:
                st.markdown(f"**Probability of positive class:** {y_pred_prob:.3f}")
                
                # Create gauge chart for probability
                fig, ax = plt.subplots(figsize=(4, 0.8))
                
                # Draw progress bar
                ax.barh(0, 1, height=0.6, color='lightgray', alpha=0.3)
                ax.barh(0, y_pred_prob, height=0.6, color='blue' if y_pred_prob < 0.5 else 'red')
                
                # Add probability text
                ax.text(y_pred_prob, 0, f' {y_pred_prob:.2f}', va='center')
                
                # Add labels for low/high risk
                ax.text(0.01, 0, 'Low Risk', va='center', ha='left', fontsize=8, alpha=0.7)
                ax.text(0.99, 0, 'High Risk', va='center', ha='right', fontsize=8, alpha=0.7)
                
                # Remove axes
                ax.set_xlim(0, 1)
                ax.set_ylim(-0.5, 0.5)
                ax.axis('off')
                
                st.pyplot(fig)
            
            # Display sample feature values
            st.markdown("#### Patient Data")
            st.dataframe(sample.T.rename(columns={sample_index: "Value"}))
            
            # Calculate SHAP values for this instance
            with st.spinner("Calculating SHAP values..."):
                shap_df, base_value = calculate_shapley_values(
                    model,
                    X_train,
                    sample,
                    feature_names
                )
                
                shap_values = shap_df.values[0]
                
                # Display explanation in tabs
                exp_tab1, exp_tab2 = st.tabs(["Waterfall Plot", "Force Plot"])
                
                with exp_tab1:
                    st.markdown("### SHAP Waterfall Plot")
                    st.markdown("""
                    This waterfall plot shows how each feature contributes to the final prediction:
                    - Starting from the base value (average model output)
                    - Each bar shows a feature's contribution
                    - Red bars push prediction higher, green bars push it lower
                    - The final blue bar shows the model's prediction for this patient
                    """)
                    
                    waterfall_fig = create_shap_waterfall_plot(
                        shap_values, 
                        feature_names, 
                        base_value, 
                        y_pred_prob,
                        f"Feature Contributions for Patient #{sample_index}"
                    )
                    st.pyplot(waterfall_fig)
                
                with exp_tab2:
                    st.markdown("### SHAP Force Plot")
                    st.markdown("""
                    This force plot provides a simplified view of feature contributions:
                    - Base value is on the left
                    - Final prediction is on the right
                    - Red bars push prediction higher (toward positive class)
                    - Blue bars push prediction lower (toward negative class)
                    """)
                    
                    force_fig = create_shap_force_plot(
                        shap_values, 
                        feature_names, 
                        sample.values[0],
                        base_value,
                        f"SHAP Force Plot for Patient #{sample_index}"
                    )
                    st.pyplot(force_fig)
            
            # Clinical implications section
            st.subheader("Clinical Implications")
            
            # Get top positive and negative features
            feature_impacts = list(zip(feature_names, shap_values))
            # Sort by absolute value
            feature_impacts.sort(key=lambda x: abs(x[1]), reverse=True)
            
            # Get top factors increasing and decreasing risk
            increasing_risk = [(f, v) for f, v in feature_impacts if v > 0]
            decreasing_risk = [(f, v) for f, v in feature_impacts if v < 0]
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Factors Increasing Risk:")
                if increasing_risk:
                    for feature, value in increasing_risk[:5]:  # Show top 5
                        st.markdown(f"- **{feature}**: +{value:.3f}")
                else:
                    st.markdown("No significant factors found.")
            
            with col2:
                st.markdown("#### Factors Decreasing Risk:")
                if decreasing_risk:
                    for feature, value in decreasing_risk[:5]:  # Show top 5
                        st.markdown(f"- **{feature}**: {value:.3f}")
                else:
                    st.markdown("No significant factors found.")
            
            # Add clinical interpretation
            st.markdown("#### Potential Clinical Actions:")
            
            if st.session_state['selected_dataset'] == "Diabetes":
                st.markdown("""
                Based on the SHAP analysis, consider:
                - Monitoring and addressing the highest risk factors
                - Discussing lifestyle modifications that could impact the most significant features
                - Scheduling appropriate follow-up based on risk level
                """)
            else:  # Breast Cancer
                st.markdown("""
                Based on the SHAP analysis, consider:
                - Further diagnostic testing focusing on the key risk indicators
                - Discussing the specific risk factors with the patient
                - Creating a personalized monitoring plan based on the most significant features
                """)