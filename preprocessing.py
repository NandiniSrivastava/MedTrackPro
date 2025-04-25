import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

def handle_missing_values(data, strategy='mean'):
    """
    Handle missing values in the dataset.
    
    Args:
        data (DataFrame): The input dataset
        strategy (str): Strategy to handle missing values
                       'drop' - Drop rows with missing values
                       'mean' - Fill numerical with mean, categorical with mode
                       'median' - Fill numerical with median, categorical with mode
    
    Returns:
        DataFrame: Dataset with handled missing values
    """
    df = data.copy()
    
    if strategy == 'drop':
        # Drop rows with missing values
        df = df.dropna()
        
    else:
        # Fill missing values based on data type and strategy
        # Get numerical and categorical columns
        numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        
        # Fill numerical missing values
        for col in numerical_cols:
            if df[col].isnull().sum() > 0:
                if strategy == 'mean':
                    df[col] = df[col].fillna(df[col].mean())
                else:  # median
                    df[col] = df[col].fillna(df[col].median())
        
        # Fill categorical missing values with mode
        for col in categorical_cols:
            if df[col].isnull().sum() > 0:
                # Get the mode (most frequent value)
                mode_value = df[col].mode()[0]
                df[col] = df[col].fillna(mode_value)
    
    return df

def encode_categorical_features(data, columns, method='onehot'):
    """
    Encode categorical features.
    
    Args:
        data (DataFrame): The input dataset
        columns (list): List of categorical columns to encode
        method (str): Encoding method - 'onehot' or 'label'
    
    Returns:
        DataFrame: Dataset with encoded categorical features
    """
    df = data.copy()
    
    if method == 'onehot':
        # One-hot encode categorical features
        df = pd.get_dummies(df, columns=columns, drop_first=False)
        
    else:  # label encoding
        # Create a label encoder
        encoder = LabelEncoder()
        
        # Apply label encoding to each categorical column
        for col in columns:
            df[col] = encoder.fit_transform(df[col])
    
    return df

def preprocess_data(data, missing_strategy='mean', encoding_method='onehot'):
    """
    Preprocess the dataset by handling missing values and encoding categorical features.
    
    Args:
        data (DataFrame): The input dataset
        missing_strategy (str): Strategy to handle missing values
        encoding_method (str): Method to encode categorical features
    
    Returns:
        DataFrame: Preprocessed dataset
    """
    # Make a copy of the data
    df = data.copy()
    
    # Handle missing values
    df = handle_missing_values(df, strategy=missing_strategy)
    
    # Get categorical columns (excluding 'customerID' and 'Churn')
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    if 'customerID' in categorical_cols:
        categorical_cols.remove('customerID')
    if 'Churn' in categorical_cols:
        categorical_cols.remove('Churn')
    
    # Encode categorical features
    df = encode_categorical_features(df, categorical_cols, method=encoding_method)
    
    # Convert target variable to binary if it exists
    if 'Churn' in df.columns:
        df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    
    # Remove customerID if it exists
    if 'customerID' in df.columns:
        df = df.drop('customerID', axis=1)
    
    return df
