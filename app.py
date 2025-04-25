import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split

# Import custom modules
from data_loader import load_telco_data
from preprocessing import preprocess_data, encode_categorical_features, handle_missing_values
from model import select_model, train_model, get_model_params
from evaluation import evaluate_model, plot_confusion_matrix, plot_roc_curve
from explanation import plot_feature_importance, explain_prediction_permutation, plot_prediction_explanation
from utils import display_step_info, display_metrics_explanation

# Set page configuration
st.set_page_config(
    page_title="Telco Customer Churn Prediction",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state for navigation
if 'current_step' not in st.session_state:
    st.session_state.current_step = 'intro'
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'data_processed' not in st.session_state:
    st.session_state.data_processed = False
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'data' not in st.session_state:
    st.session_state.data = None
if 'X_train' not in st.session_state:
    st.session_state.X_train = None
if 'X_test' not in st.session_state:
    st.session_state.X_test = None
if 'y_train' not in st.session_state:
    st.session_state.y_train = None
if 'y_test' not in st.session_state:
    st.session_state.y_test = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'preprocessed_data' not in st.session_state:
    st.session_state.preprocessed_data = None
if 'predictions' not in st.session_state:
    st.session_state.predictions = None
if 'probabilities' not in st.session_state:
    st.session_state.probabilities = None
if 'feature_names' not in st.session_state:
    st.session_state.feature_names = None

# Main title
st.title("Telco Customer Churn Prediction")
st.markdown("### A Complete Machine Learning Workflow for Beginners")

# Sidebar for navigation
st.sidebar.title("Navigation")
steps = {
    'intro': "Introduction",
    'data_exploration': "1. Data Exploration",
    'data_preprocessing': "2. Data Preprocessing",
    'feature_engineering': "3. Feature Engineering",
    'model_selection': "4. Model Selection & Training",
    'model_evaluation': "5. Model Evaluation",
    'model_interpretation': "6. Model Interpretation",
    'conclusion': "Conclusion"
}

selected_step = st.sidebar.radio("Go to Step:", list(steps.values()))
st.session_state.current_step = list(steps.keys())[list(steps.values()).index(selected_step)]

# Display current step content
if st.session_state.current_step == 'intro':
    st.markdown("## Welcome to the Telco Customer Churn Prediction App!")
    st.markdown("""
    This application will guide you through the complete machine learning workflow to predict customer churn for a telecom company.
    
    ### What is Customer Churn?
    Customer churn refers to when customers stop using a company's products or services. For telecom companies, this is particularly important because acquiring new customers can be 5-25 times more expensive than retaining existing ones.
    
    ### Machine Learning Workflow
    This app will guide you through these steps:
    1. **Data Exploration** - Understanding the dataset through statistics and visualizations
    2. **Data Preprocessing** - Cleaning and preparing the data for modeling
    3. **Feature Engineering** - Creating and selecting relevant features
    4. **Model Selection & Training** - Choosing and training the right predictive model
    5. **Model Evaluation** - Assessing model performance
    6. **Model Interpretation** - Understanding why the model makes certain predictions
    
    ### Telco Customer Churn Dataset
    The dataset contains information about customers and whether they left the company (churned) or not.
    
    Let's get started by clicking the '1. Data Exploration' step in the sidebar!
    """)
    
    # Show a sample image
    st.markdown("### Machine Learning Workflow")
    st.image("https://miro.medium.com/max/1400/1*NLs-vsUbWUgNGnL98m9Zbg.png", width=700)

elif st.session_state.current_step == 'data_exploration':
    st.markdown("## 1. Data Exploration")
    display_step_info("data_exploration")
    
    # Load data button
    if not st.session_state.data_loaded:
        if st.button("Load Telco Churn Dataset"):
            with st.spinner("Loading data..."):
                try:
                    st.session_state.data = load_telco_data()
                    st.session_state.data_loaded = True
                    st.success("Data loaded successfully!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error loading data: {e}")
    
    # If data is loaded, show exploration tools
    if st.session_state.data_loaded and st.session_state.data is not None:
        data = st.session_state.data
        
        # Basic dataset information
        st.markdown("### Dataset Overview")
        st.write(f"Dataset shape: {data.shape[0]} rows and {data.shape[1]} columns")
        
        # Display a sample of the data
        st.markdown("### Data Sample")
        st.dataframe(data.head())
        
        # Data statistics
        st.markdown("### Data Statistics")
        st.dataframe(data.describe().T)
        
        # Missing values
        st.markdown("### Missing Values")
        missing_values = data.isnull().sum()
        missing_percent = (missing_values / len(data)) * 100
        missing_data = pd.DataFrame({
            'Missing Values': missing_values,
            'Percentage': missing_percent
        })
        st.dataframe(missing_data[missing_data['Missing Values'] > 0] if not missing_data.empty else 
                    "No missing values found in the dataset.")
        
        # Data types
        st.markdown("### Data Types")
        st.dataframe(pd.DataFrame(data.dtypes, columns=['Data Type']))
        
        # Target variable distribution
        st.markdown("### Target Variable: Churn")
        if 'Churn' in data.columns:
            churn_counts = data['Churn'].value_counts()
            st.write(f"No Churn: {churn_counts.get('No', 0)} ({churn_counts.get('No', 0)/len(data)*100:.2f}%)")
            st.write(f"Churn: {churn_counts.get('Yes', 0)} ({churn_counts.get('Yes', 0)/len(data)*100:.2f}%)")
            
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.pie(churn_counts, labels=churn_counts.index, autopct='%1.1f%%', startangle=90)
            ax.axis('equal')
            st.pyplot(fig)

            # Convert to binary for further analysis
            data['Churn_Binary'] = data['Churn'].map({'Yes': 1, 'No': 0})
        else:
            st.warning("Churn column not found in the dataset.")
        
        # EDA visualizations
        st.markdown("### Exploratory Data Analysis")
        
        # Categorical variables
        categorical_cols = data.select_dtypes(include=['object']).columns.tolist()
        if 'Churn' in categorical_cols:
            categorical_cols.remove('Churn')
        
        st.markdown("#### Categorical Variables")
        if categorical_cols:
            selected_cat_col = st.selectbox(
                "Select a categorical variable to visualize:", 
                categorical_cols
            )
            
            fig, ax = plt.subplots(figsize=(10, 6))
            cat_counts = data[selected_cat_col].value_counts()
            sns.countplot(x=selected_cat_col, data=data, ax=ax, hue='Churn')
            plt.xticks(rotation=45)
            plt.title(f'Distribution of {selected_cat_col}')
            st.pyplot(fig)
            
            # Churn rate by category
            if 'Churn' in data.columns:
                st.markdown(f"#### Churn Rate by {selected_cat_col}")
                churn_by_cat = pd.crosstab(data[selected_cat_col], data['Churn'])
                churn_by_cat['Churn Rate'] = churn_by_cat['Yes'] / (churn_by_cat['Yes'] + churn_by_cat['No'])
                
                fig = px.bar(
                    churn_by_cat.reset_index(), 
                    x=selected_cat_col, 
                    y='Churn Rate',
                    title=f'Churn Rate by {selected_cat_col}'
                )
                st.plotly_chart(fig)
        
        # Numerical variables
        numerical_cols = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
        if 'Churn_Binary' in numerical_cols:
            numerical_cols.remove('Churn_Binary')
            
        st.markdown("#### Numerical Variables")
        if numerical_cols:
            selected_num_col = st.selectbox(
                "Select a numerical variable to visualize:", 
                numerical_cols
            )
            
            # Histogram
            fig = px.histogram(
                data, 
                x=selected_num_col, 
                color='Churn' if 'Churn' in data.columns else None,
                marginal='box',
                title=f'Distribution of {selected_num_col}'
            )
            st.plotly_chart(fig)
            
            # Correlation with churn
            if 'Churn_Binary' in data.columns:
                corr = data[selected_num_col].corr(data['Churn_Binary'])
                st.write(f"Correlation with Churn: {corr:.4f}")
        
        # Correlation matrix
        st.markdown("#### Correlation Matrix")
        if len(numerical_cols) > 1:
            # Add Churn_Binary to numerical columns for correlation
            if 'Churn_Binary' in data.columns and 'Churn_Binary' not in numerical_cols:
                numerical_cols.append('Churn_Binary')
                
            # Compute correlation matrix
            corr_matrix = data[numerical_cols].corr()
            
            # Plot using heatmap
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax, fmt='.2f')
            plt.title('Correlation Matrix')
            st.pyplot(fig)
    else:
        st.info("Please load the dataset using the button above.")

elif st.session_state.current_step == 'data_preprocessing':
    st.markdown("## 2. Data Preprocessing")
    display_step_info("data_preprocessing")
    
    if not st.session_state.data_loaded:
        st.warning("Please load the dataset first by going to the Data Exploration step.")
    else:
        data = st.session_state.data
        
        st.markdown("### Handling Missing Values")
        # Display missing values info
        missing_values = data.isnull().sum()
        missing_percent = (missing_values / len(data)) * 100
        missing_data = pd.DataFrame({
            'Missing Values': missing_values,
            'Percentage': missing_percent
        })
        
        missing_cols = missing_data[missing_data['Missing Values'] > 0]
        if not missing_cols.empty:
            st.write("Columns with missing values:")
            st.dataframe(missing_cols)
            
            # Options for handling missing values
            missing_strategy = st.radio(
                "Select strategy for handling missing values:",
                ["Drop rows with missing values", 
                 "Fill numerical with mean, categorical with mode", 
                 "Fill numerical with median, categorical with mode"]
            )
        else:
            st.success("No missing values found in the dataset.")
            missing_strategy = None
        
        st.markdown("### Handling Categorical Variables")
        categorical_cols = data.select_dtypes(include=['object']).columns.tolist()
        
        # Remove CustomerID and Churn from encoding
        encoding_cols = categorical_cols.copy()
        if 'customerID' in encoding_cols:
            encoding_cols.remove('customerID')
        if 'Churn' in encoding_cols:
            encoding_cols.remove('Churn')
        
        if encoding_cols:
            st.write("Categorical columns to encode:")
            st.write(encoding_cols)
            
            encoding_strategy = st.radio(
                "Select encoding strategy for categorical variables:",
                ["One-Hot Encoding", "Label Encoding"]
            )
        else:
            st.info("No categorical variables to encode.")
            encoding_strategy = None
        
        # Handle target variable
        st.markdown("### Target Variable Preparation")
        if 'Churn' in data.columns:
            st.write("Converting 'Churn' to binary (Yes=1, No=0)")
        
        # Process button
        if st.button("Preprocess Data"):
            with st.spinner("Preprocessing data..."):
                try:
                    # Handle missing values if needed
                    if missing_strategy:
                        if missing_strategy == "Drop rows with missing values":
                            processed_data = handle_missing_values(data, strategy='drop')
                        elif missing_strategy == "Fill numerical with mean, categorical with mode":
                            processed_data = handle_missing_values(data, strategy='mean')
                        else:
                            processed_data = handle_missing_values(data, strategy='median')
                    else:
                        processed_data = data.copy()
                    
                    # Handle categorical encoding
                    if encoding_strategy and encoding_cols:
                        if encoding_strategy == "One-Hot Encoding":
                            processed_data = encode_categorical_features(processed_data, encoding_cols, method='onehot')
                        else:
                            processed_data = encode_categorical_features(processed_data, encoding_cols, method='label')
                    
                    # Convert target variable to binary
                    if 'Churn' in processed_data.columns:
                        processed_data['Churn'] = processed_data['Churn'].map({'Yes': 1, 'No': 0})
                    
                    # Remove customerID if exists
                    if 'customerID' in processed_data.columns:
                        processed_data = processed_data.drop('customerID', axis=1)
                    
                    # Store in session state
                    st.session_state.preprocessed_data = processed_data
                    st.session_state.data_processed = True
                    
                    st.success("Data preprocessing completed!")
                    
                    # Show sample of processed data
                    st.markdown("### Processed Data Sample")
                    st.dataframe(processed_data.head())
                    
                    # Show dataset info after preprocessing
                    st.markdown("### Processed Dataset Information")
                    st.write(f"Processed dataset shape: {processed_data.shape[0]} rows and {processed_data.shape[1]} columns")
                    st.write("Data types:")
                    st.write(processed_data.dtypes)
                    
                except Exception as e:
                    st.error(f"Error during preprocessing: {e}")
        
        # Show processed data if already done
        elif st.session_state.data_processed and st.session_state.preprocessed_data is not None:
            st.success("Data has been preprocessed!")
            
            # Show sample of processed data
            st.markdown("### Processed Data Sample")
            st.dataframe(st.session_state.preprocessed_data.head())
            
            # Show dataset info after preprocessing
            st.markdown("### Processed Dataset Information")
            st.write(f"Processed dataset shape: {st.session_state.preprocessed_data.shape[0]} rows and {st.session_state.preprocessed_data.shape[1]} columns")
            st.write("Data types:")
            st.write(st.session_state.preprocessed_data.dtypes)
    
elif st.session_state.current_step == 'feature_engineering':
    st.markdown("## 3. Feature Engineering")
    display_step_info("feature_engineering")
    
    if not st.session_state.data_processed:
        st.warning("Please preprocess the data first by going to the Data Preprocessing step.")
    else:
        data = st.session_state.preprocessed_data
        
        st.markdown("### Feature Selection and Engineering")
        
        # Determine numerical columns for scaling
        numerical_cols = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
        if 'Churn' in numerical_cols:
            numerical_cols.remove('Churn')
        
        # Feature scaling options
        st.markdown("#### Feature Scaling")
        if numerical_cols:
            scaling_method = st.radio(
                "Select scaling method for numerical features:",
                ["None", "StandardScaler", "MinMaxScaler", "RobustScaler"]
            )
        else:
            st.info("No numerical features to scale.")
            scaling_method = "None"
        
        # Feature selection options
        st.markdown("#### Feature Selection")
        feature_selection_method = st.radio(
            "Select feature selection method:",
            ["None", "Remove Low Variance", "Select K Best (ANOVA)", "Recursive Feature Elimination (RFE)"]
        )
        
        if feature_selection_method != "None":
            if feature_selection_method == "Remove Low Variance":
                variance_threshold = st.slider("Variance threshold", 0.0, 1.0, 0.1, 0.05)
            elif feature_selection_method == "Select K Best (ANOVA)":
                k_best = st.slider("Number of features to select", 1, len(data.columns)-1, min(10, len(data.columns)-1))
            elif feature_selection_method == "Recursive Feature Elimination (RFE)":
                n_features_to_select = st.slider("Number of features to select", 1, len(data.columns)-1, min(10, len(data.columns)-1))
        
        # Train-test split options
        st.markdown("#### Train-Test Split")
        test_size = st.slider("Test set size (%)", 10, 40, 20)
        random_state = st.number_input("Random state (for reproducibility)", 0, 1000, 42)
        
        # Process features button
        if st.button("Engineer Features and Split Data"):
            with st.spinner("Engineering features and splitting data..."):
                try:
                    # Separate features and target
                    X = data.drop('Churn', axis=1)
                    y = data['Churn']
                    
                    # Scale features if selected
                    if scaling_method != "None":
                        from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
                        
                        if scaling_method == "StandardScaler":
                            from sklearn.preprocessing import StandardScaler
                            scaler = StandardScaler()
                        elif scaling_method == "MinMaxScaler":
                            from sklearn.preprocessing import MinMaxScaler
                            scaler = MinMaxScaler()
                        else:  # RobustScaler
                            from sklearn.preprocessing import RobustScaler
                            scaler = RobustScaler()
                        
                        # Scale only numerical columns
                        X_scaled = X.copy()
                        X_scaled[numerical_cols] = scaler.fit_transform(X[numerical_cols])
                        X = X_scaled
                        
                        st.write(f"Applied {scaling_method} to {len(numerical_cols)} numerical features.")
                    
                    # Apply feature selection if chosen
                    if feature_selection_method != "None":
                        if feature_selection_method == "Remove Low Variance":
                            from sklearn.feature_selection import VarianceThreshold
                            selector = VarianceThreshold(threshold=variance_threshold)
                            X_selected = selector.fit_transform(X)
                            selected_cols = X.columns[selector.get_support()]
                            
                        elif feature_selection_method == "Select K Best (ANOVA)":
                            from sklearn.feature_selection import SelectKBest, f_classif
                            selector = SelectKBest(f_classif, k=k_best)
                            X_selected = selector.fit_transform(X, y)
                            selected_cols = X.columns[selector.get_support()]
                            
                        elif feature_selection_method == "Recursive Feature Elimination (RFE)":
                            from sklearn.feature_selection import RFE
                            from sklearn.ensemble import RandomForestClassifier
                            estimator = RandomForestClassifier(n_estimators=10, random_state=random_state)
                            selector = RFE(estimator, n_features_to_select=n_features_to_select)
                            X_selected = selector.fit_transform(X, y)
                            selected_cols = X.columns[selector.get_support()]
                        
                        # Create a new DataFrame with selected features
                        X = X[selected_cols]
                        st.success(f"Selected {len(selected_cols)} features: {', '.join(selected_cols)}")
                    
                    # Split the data
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=test_size/100, random_state=random_state, stratify=y
                    )
                    
                    # Store in session state
                    st.session_state.X_train = X_train
                    st.session_state.X_test = X_test
                    st.session_state.y_train = y_train
                    st.session_state.y_test = y_test
                    st.session_state.feature_names = X.columns.tolist()
                    
                    st.success(f"Data split into training ({X_train.shape[0]} samples) and test ({X_test.shape[0]} samples) sets!")
                    
                    # Display class distribution in the splits
                    train_class_dist = y_train.value_counts(normalize=True) * 100
                    test_class_dist = y_test.value_counts(normalize=True) * 100
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("#### Training Set Class Distribution")
                        st.write(f"Class 0 (No Churn): {train_class_dist.get(0, 0):.2f}%")
                        st.write(f"Class 1 (Churn): {train_class_dist.get(1, 0):.2f}%")
                    
                    with col2:
                        st.markdown("#### Test Set Class Distribution")
                        st.write(f"Class 0 (No Churn): {test_class_dist.get(0, 0):.2f}%")
                        st.write(f"Class 1 (Churn): {test_class_dist.get(1, 0):.2f}%")
                    
                    # Display feature information
                    st.markdown("#### Feature Information")
                    st.write(f"Number of features: {X.shape[1]}")
                    st.write("Feature names:")
                    st.write(X.columns.tolist())
                    
                except Exception as e:
                    st.error(f"Error during feature engineering: {e}")
        
        # Display if already processed
        elif st.session_state.X_train is not None:
            st.success("Features have been engineered and data has been split!")
            
            # Display dataset information
            st.markdown("#### Dataset Splits")
            st.write(f"Training set: {st.session_state.X_train.shape[0]} samples, {st.session_state.X_train.shape[1]} features")
            st.write(f"Test set: {st.session_state.X_test.shape[0]} samples, {st.session_state.X_test.shape[1]} features")
            
            # Display class distribution in the splits
            train_class_dist = st.session_state.y_train.value_counts(normalize=True) * 100
            test_class_dist = st.session_state.y_test.value_counts(normalize=True) * 100
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### Training Set Class Distribution")
                st.write(f"Class 0 (No Churn): {train_class_dist.get(0, 0):.2f}%")
                st.write(f"Class 1 (Churn): {train_class_dist.get(1, 0):.2f}%")
            
            with col2:
                st.markdown("#### Test Set Class Distribution")
                st.write(f"Class 0 (No Churn): {test_class_dist.get(0, 0):.2f}%")
                st.write(f"Class 1 (Churn): {test_class_dist.get(1, 0):.2f}%")
            
            # Display feature information
            st.markdown("#### Feature Information")
            st.write(f"Number of features: {st.session_state.X_train.shape[1]}")
            if st.session_state.feature_names:
                st.write("Feature names:")
                st.write(st.session_state.feature_names)

elif st.session_state.current_step == 'model_selection':
    st.markdown("## 4. Model Selection & Training")
    display_step_info("model_selection")
    
    if (st.session_state.X_train is None or 
        st.session_state.X_test is None or 
        st.session_state.y_train is None or 
        st.session_state.y_test is None):
        st.warning("Please complete the Feature Engineering step first.")
    else:
        st.markdown("### Select a Model")
        
        model_type = st.selectbox(
            "Choose a model type:",
            ["Logistic Regression", "Random Forest", "Gradient Boosting", "Support Vector Machine", "K-Nearest Neighbors"]
        )
        
        # Display model parameters based on selection
        model_params = get_model_params(model_type)
        
        # Train model button
        if st.button("Train Model"):
            with st.spinner(f"Training {model_type} model..."):
                try:
                    # Select and train the model with chosen parameters
                    model = select_model(model_type, model_params)
                    model = train_model(
                        model, 
                        st.session_state.X_train, 
                        st.session_state.y_train
                    )
                    
                    # Make predictions on test set
                    predictions = model.predict(st.session_state.X_test)
                    
                    # Get prediction probabilities for ROC curve if available
                    try:
                        probabilities = model.predict_proba(st.session_state.X_test)[:, 1]
                    except:
                        probabilities = None
                    
                    # Store model and predictions in session state
                    st.session_state.model = model
                    st.session_state.model_type = model_type
                    st.session_state.predictions = predictions
                    st.session_state.probabilities = probabilities
                    st.session_state.model_trained = True
                    
                    st.success(f"{model_type} model trained successfully!")
                    
                    # Show basic model information
                    st.markdown("### Model Information")
                    st.write(f"Model type: {model_type}")
                    st.write("Model parameters:")
                    st.write(model.get_params())
                    
                except Exception as e:
                    st.error(f"Error during model training: {e}")
        
        # Display if model already trained
        elif st.session_state.model_trained and st.session_state.model is not None:
            st.success(f"Model '{st.session_state.model_type}' has been trained!")
            
            # Show model information
            st.markdown("### Model Information")
            st.write(f"Model type: {st.session_state.model_type}")
            st.write("Model parameters:")
            st.write(st.session_state.model.get_params())

elif st.session_state.current_step == 'model_evaluation':
    st.markdown("## 5. Model Evaluation")
    display_step_info("model_evaluation")
    
    if not st.session_state.model_trained:
        st.warning("Please train a model first by going to the Model Selection & Training step.")
    else:
        st.markdown("### Model Performance Metrics")
        
        # Evaluate model and display metrics
        model = st.session_state.model
        X_test = st.session_state.X_test
        y_test = st.session_state.y_test
        predictions = st.session_state.predictions
        probabilities = st.session_state.probabilities
        
        # Get and display evaluation metrics
        metrics = evaluate_model(y_test, predictions)
        
        # Display metrics in a nice format
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Classification Metrics")
            st.write(f"Accuracy: {metrics['accuracy']:.4f}")
            st.write(f"Precision: {metrics['precision']:.4f}")
            st.write(f"Recall: {metrics['recall']:.4f}")
            st.write(f"F1 Score: {metrics['f1']:.4f}")
        
        with col2:
            st.markdown("#### Additional Metrics")
            st.write(f"Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
            if probabilities is not None and metrics['roc_auc'] is not None:
                st.write(f"ROC AUC: {metrics['roc_auc']:.4f}")
                st.write(f"PR AUC: {metrics['pr_auc']:.4f}")
            else:
                st.write("ROC AUC: Not available")
                st.write("PR AUC: Not available")
            
        # Display classification report
        st.markdown("#### Classification Report")
        st.text(metrics['classification_report'])
        
        # Explanation of metrics
        with st.expander("What do these metrics mean?"):
            display_metrics_explanation()
        
        # Plot confusion matrix
        st.markdown("### Confusion Matrix")
        cm_fig = plot_confusion_matrix(y_test, predictions)
        st.pyplot(cm_fig)
        
        with st.expander("How to interpret the confusion matrix?"):
            st.markdown("""
            The confusion matrix shows the counts of true positives, false positives, true negatives, and false negatives:
            
            - **True Positives (TP)**: Customers who actually churned and were correctly predicted to churn
            - **False Positives (FP)**: Customers who didn't churn but were incorrectly predicted to churn
            - **True Negatives (TN)**: Customers who didn't churn and were correctly predicted not to churn
            - **False Negatives (FN)**: Customers who actually churned but were incorrectly predicted not to churn
            
            For a churn prediction model:
            - A high number of false negatives means you're missing customers who will churn
            - A high number of false positives means you might waste resources on retention for customers who wouldn't churn anyway
            """)
        
        # Plot ROC curve if probabilities available
        if probabilities is not None:
            st.markdown("### ROC Curve")
            roc_fig = plot_roc_curve(y_test, probabilities)
            st.pyplot(roc_fig)
            
            with st.expander("What is the ROC curve?"):
                st.markdown("""
                The Receiver Operating Characteristic (ROC) curve plots the True Positive Rate against the False Positive Rate at various threshold settings:
                
                - The area under the ROC curve (AUC) measures the model's ability to distinguish between classes
                - AUC values range from 0 to 1:
                  - 0.5 means the model has no discrimination capacity (equivalent to random guessing)
                  - 1.0 means perfect discrimination
                  - Values above 0.7 generally indicate a good model
                
                The curve shows the tradeoff between catching true positives and avoiding false positives as you change the classification threshold.
                """)

elif st.session_state.current_step == 'model_interpretation':
    st.markdown("## 6. Model Interpretation")
    display_step_info("model_interpretation")
    
    if not st.session_state.model_trained:
        st.warning("Please train a model first by going to the Model Selection & Training step.")
    else:
        st.markdown("### Understanding the Model's Decisions")
        
        # Feature importance
        st.markdown("#### Feature Importance")
        
        if (st.session_state.model_type in ["Logistic Regression", "Random Forest", "Gradient Boosting"] and 
            st.session_state.feature_names is not None):
            
            feature_importance_fig = plot_feature_importance(
                st.session_state.model, 
                st.session_state.model_type,
                st.session_state.feature_names
            )
            
            st.pyplot(feature_importance_fig)
            
            with st.expander("How to interpret feature importance?"):
                st.markdown("""
                Feature importance shows which variables have the strongest influence on the model's predictions:
                
                - For **Logistic Regression**, coefficients show the change in log-odds for a one-unit increase in the feature
                - For **Tree-based models** (Random Forest, Gradient Boosting), importance typically represents how much a feature reduces impurity on average
                
                Higher importance means the feature has a stronger influence on the churn prediction. Features can have positive impact (increasing churn probability) or negative impact (decreasing churn probability).
                """)
        
        # Permutation importance for model interpretation
        st.markdown("#### Permutation Feature Importance")
        
        if st.button("Generate Feature Importance (may take a moment)"):
            with st.spinner("Generating permutation feature importance..."):
                try:
                    # Get a sample of the test data to explain
                    if len(st.session_state.X_test) > 100:
                        sample_indices = np.random.choice(len(st.session_state.X_test), 100, replace=False)
                        X_sample = st.session_state.X_test.iloc[sample_indices]
                    else:
                        X_sample = st.session_state.X_test
                    
                    # Generate SHAP-like summary plot
                    from explanation import create_shap_summary_plot
                    summary_plot = create_shap_summary_plot(
                        st.session_state.model,
                        st.session_state.model_type, 
                        st.session_state.X_test,
                        st.session_state.feature_names
                    )
                    
                    st.pyplot(summary_plot)
                    
                    with st.expander("How to interpret SHAP Summary Plots?"):
                        st.markdown("""
                        This SHAP-like summary plot shows the impact of each feature on the model's predictions:
                        
                        - Each point represents one sample/instance in the data
                        - Features are ordered by importance (top to bottom)
                        - Position on x-axis shows the impact on prediction:
                          - Points to the right (positive values) push prediction higher (toward churn)
                          - Points to the left (negative values) push prediction lower (away from churn)
                        - Color represents the feature value:
                          - Red = high feature value
                          - Blue = low feature value
                          
                        When you see a clustering of red points on the right side for a feature, it means high values of that feature tend to increase churn probability.
                        
                        Similarly, blue points on the left mean low values of that feature tend to decrease churn probability.
                        """)
                    
                except Exception as e:
                    st.error(f"Error generating feature importance plot: {e}")
        
        # Individual prediction explanation
        st.markdown("### Explain Individual Predictions")
        
        # Allow user to select data point to explain 
        if len(st.session_state.X_test) > 0:
            sample_index = st.number_input(
                "Select sample index to explain (from test set):", 
                0, 
                len(st.session_state.X_test) - 1, 
                0
            )
            
            if st.button("Explain this prediction"):
                with st.spinner("Generating explanation..."):
                    try:
                        # Get the selected data point
                        x_to_explain = st.session_state.X_test.iloc[[sample_index]]
                        actual_label = st.session_state.y_test.iloc[sample_index]
                        predicted_prob = st.session_state.model.predict_proba(x_to_explain)[0, 1]
                        predicted_label = 1 if predicted_prob >= 0.5 else 0
                        
                        # Display prediction info
                        st.markdown("#### Prediction Analysis")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown(f"**Actual Label:** {'Churn' if actual_label == 1 else 'No Churn'}")
                            st.markdown(f"**Predicted Label:** {'Churn' if predicted_label == 1 else 'No Churn'}")
                        
                        with col2:
                            st.markdown(f"**Churn Probability:** {predicted_prob:.4f}")
                            # Determine if prediction matches actual
                            prediction_correct = actual_label == predicted_label
                            st.markdown(f"**Prediction Status:** {'✅ Correct' if prediction_correct else '❌ Incorrect'}")
                        
                        # Display customer data
                        st.markdown("#### Customer Information")
                        st.dataframe(x_to_explain.T.rename(columns={sample_index: "Value"}))
                        
                        # Interpret with feature contributions
                        st.markdown("#### Feature Contributions")
                        from explanation import plot_prediction_explanation
                        
                        explanation_fig = plot_prediction_explanation(
                            st.session_state.model, 
                            st.session_state.model_type, 
                            x_to_explain,
                            st.session_state.feature_names
                        )
                        
                        st.pyplot(explanation_fig)
                        
                        with st.expander("How to interpret this plot?"):
                            st.markdown("""
                            This SHAP-like waterfall plot shows how each feature contributes to the prediction:
                            
                            - The base value (gray) represents the average model output over the training dataset
                            - Red bars push the prediction higher (toward churn)
                            - Green bars push the prediction lower (away from churn)
                            - The blue bar shows the final prediction value
                            - Each feature's contribution is shown with the exact value impact
                            
                            Features are ordered by their impact on this specific prediction.
                            The connecting lines show how the prediction changes as each feature is added.
                            """)
                    
                    except Exception as e:
                        st.error(f"Error explaining prediction: {e}")

elif st.session_state.current_step == 'conclusion':
    st.markdown("## Conclusion and Next Steps")
    
    st.markdown("""
    ### Congratulations!
    
    You've completed a full machine learning workflow for Telco Customer Churn prediction. Here's a summary of what you've accomplished:
    
    1. **Explored Data** - Examined the dataset structure, distributions, and relationships
    2. **Preprocessed Data** - Handled missing values and encoded categorical features
    3. **Engineered Features** - Selected relevant features and prepared data for modeling
    4. **Selected and Trained a Model** - Chose a model and optimized its parameters
    5. **Evaluated Model Performance** - Assessed how well the model works
    6. **Interpreted the Model** - Understood which factors contribute most to churn
    
    ### Potential Business Actions
    
    Based on the model insights, here are some actions a telecom company might take:
    
    - Target retention offers to customers with high churn probability
    - Address the most influential factors in customer churn
    - Implement early warning systems to identify at-risk customers
    - Develop personalized retention strategies based on specific churn factors
    
    ### Potential Improvements
    
    To enhance this ML solution, consider:
    
    - **Dataset Enrichment** - Add more features like customer service interactions
    - **Advanced Feature Engineering** - Create more sophisticated features
    - **Model Tuning** - Perform hyperparameter optimization
    - **Ensemble Methods** - Combine multiple models for better performance
    - **Addressing Class Imbalance** - Try techniques like SMOTE or class weighting
    - **Deployment Strategy** - Move from analysis to production
    
    ### Learning Resources
    
    To deepen your understanding of ML for churn prediction:
    
    - [Scikit-learn Documentation](https://scikit-learn.org/stable/documentation.html)
    - [Machine Learning Mastery](https://machinelearningmastery.com)
    - [Towards Data Science](https://towardsdatascience.com)
    - [Feature Importance in scikit-learn](https://scikit-learn.org/stable/modules/permutation_importance.html)
    
    Thank you for using this Telco Customer Churn Prediction application!
    """)
    
    # Return to start
    if st.button("Start Over"):
        # Reset session state
        for key in ['current_step', 'data_loaded', 'data_processed', 'model_trained',
                   'data', 'X_train', 'X_test', 'y_train', 'y_test', 'model',
                   'preprocessed_data', 'predictions', 'probabilities']:
            if key in st.session_state:
                del st.session_state[key]
        
        # Go back to intro
        st.session_state.current_step = 'intro'
        st.rerun()
