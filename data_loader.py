import pandas as pd
import streamlit as st

def load_telco_data():
    """
    Load the Telco Customer Churn dataset from IBM.
    
    Returns:
        pandas.DataFrame: The loaded Telco Customer Churn dataset
    """
    # URL to the dataset
    url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
    
    try:
        # Load the dataset
        df = pd.read_csv(url)
        
        # Basic data cleaning
        # Convert TotalCharges to numeric (it contains spaces for customers with 0 tenure)
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        
        # Add a note about conversion
        st.info("Note: 'TotalCharges' column has been converted to numeric format. " 
                "Some values with spaces (representing customers with 0 tenure) may have been converted to NaN.")
        
        return df
    
    except Exception as e:
        raise Exception(f"Error loading Telco Customer Churn dataset: {e}")
