# Healthcare ML Explainer with SHAP Visualizations

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.15+-red.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

A comprehensive Streamlit application for machine learning model interpretability, focusing on healthcare predictive modeling and advanced model explanation techniques with Shapley value (SHAP) visualizations.

![image](https://github.com/user-attachments/assets/fbd65456-9387-427a-b6c4-b70188d0e72d)


## ✨ Features

- **Complete ML Pipeline**: Data loading, exploration, preprocessing, model training, and explanation
- **Healthcare Datasets**: Built-in breast cancer and diabetes datasets from scikit-learn
- **Interactive Data Exploration**: Visualize feature distributions and correlations with target variables
- **Advanced Preprocessing**: Handle missing values, encode categorical features, scale numerical features
- **Multiple ML Models**: Train and evaluate Logistic Regression, Random Forest, and Gradient Boosting
- **SHAP Visualizations**: 
  - Summary plots showing feature importance across all predictions
  - Waterfall plots explaining individual prediction contributions
  - Force plots visualizing the push/pull effects on model decisions
- **Clinical Insights**: Translate model outputs into actionable medical insights
- **Interactive UI**: Intuitive, menu-based navigation for seamless workflow

## 🚀 Demo

You can access a live demo of the application [here](https://healthcare-shap-explainer.streamlit.app).

## 💻 Installation

To run this project locally, follow these steps:

1. Clone the repository:
```bash
git clone https://github.com/yourusername/healthcare-shap-app.git
cd healthcare-shap-app
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the required dependencies:
```bash
pip install streamlit pandas numpy matplotlib scikit-learn seaborn plotly eli5
```

4. Run the Streamlit app:
```bash
streamlit run healthcare_shap_app.py
```

The app will open in your default web browser at [http://localhost:5000](http://localhost:5000).

## 📁 Project Structure

```
healthcare-shap-app/
├── healthcare_shap_app.py   # Main Streamlit application
├── data_loader.py           # Dataset loading utilities
├── preprocessing.py         # Data preprocessing functions
├── model.py                 # Model selection and training
├── evaluation.py            # Model evaluation metrics
├── explanation.py           # SHAP visualization functions
├── utils.py                 # Helper utilities
├── .streamlit/              # Streamlit configuration
│   └── config.toml          # Streamlit server settings
├── models/                  # Saved model files (gitignored)
└── README.md                # Project documentation
```

## 📊 Usage

The application follows a step-by-step workflow that guides you through the machine learning pipeline:

1. **Introduction**: Overview of the project and machine learning interpretability
2. **Data Selection**: Choose between breast cancer and diabetes datasets
3. **Data Exploration**: Explore the distribution of features and their relationship with the target
4. **Data Preprocessing**: Handle missing values, encode categorical features, scale numerical features
5. **Model Training**: Select and train various machine learning models
6. **Model Evaluation**: Review performance metrics and evaluation charts
7. **SHAP Explanations**: Visualize model predictions with global and individual explanations

Navigate through these steps using the sidebar menu on the left.

## 🔄 Machine Learning Pipeline

### Data Selection
- Choose between Breast Cancer and Diabetes datasets from scikit-learn
- View dataset characteristics and descriptions
- Option to upload custom healthcare datasets (future feature)

### Exploratory Data Analysis
- Distribution of target variable (Malignant/Benign or Diabetes Positive/Negative)
- Feature visualization with histograms and box plots
- Correlation analysis with heatmaps
- Descriptive statistics for all features
![image](https://github.com/user-attachments/assets/36a59e5f-7e1e-45d1-9e86-5e5db12968a1)
![image](https://github.com/user-attachments/assets/8c7bb7b3-e84f-47b7-a78e-bda68cb7c231)

### Data Preprocessing
- Handle missing values with imputation
- Scale numerical features using StandardScaler
- Split data into training and testing sets
- Preview processed dataset before model training


### Model Training & Evaluation
- Train various models: 
  - Logistic Regression
  - Random Forest
  - Gradient Boosting
- Tune hyperparameters for optimal performance
- Evaluate models using:
  - Accuracy, Precision, Recall, F1-Score
  - Confusion Matrix
  - ROC Curve and AUC Score
  - Cross-validation scores
  ![image](https://github.com/user-attachments/assets/986cb93b-b46e-46d2-a191-17ce7be89c24)
![image](https://github.com/user-attachments/assets/13f9c420-7b79-4c5a-af3f-3728dcf11209)
![image](https://github.com/user-attachments/assets/0039c809-2959-4d93-be71-f090fdf8cadc)


### SHAP Explanations
- Global interpretability with SHAP summary plots
- Feature importance rankings
- Individual prediction explanations with waterfall plots
- Force plots showing feature contributions
- Clinical implications of model decisions
![image](https://github.com/user-attachments/assets/4274119e-f874-4c79-98c3-054fdacc583f)

![image](https://github.com/user-attachments/assets/4a9d8dc9-4a7e-486b-97ce-90f999460970)

## 📊 Datasets

The application uses two primary healthcare datasets:

### Breast Cancer Wisconsin Dataset
- Features computed from digitized images of fine needle aspirates (FNA) of breast mass
- Includes cell characteristics like radius, texture, perimeter, area, smoothness
- Binary classification: Malignant (M) or Benign (B)

### Diabetes Dataset
- Diagnostic measurements for diabetes prediction
- Includes glucose concentration, blood pressure, skin thickness, insulin, BMI
- Binary classification: Diabetes positive (1) or negative (0)

## 🤖 Models

The application supports the following machine learning models:

- **Logistic Regression**: A linear model suitable for binary classification in medical contexts
- **Random Forest**: An ensemble of decision trees offering robust performance for complex relationships
- **Gradient Boosting**: Advanced ensemble method with often superior performance for medical predictions

Each model can be customized with different hyperparameters through the UI.

## 📊 SHAP Visualizations

The application provides sophisticated SHAP-based visualizations:

- **Summary Plots**: Show how each feature impacts predictions across multiple patients
- **Bar Plots**: Rank features by their global importance to the model
- **Waterfall Plots**: Explain individual patient predictions by showing each feature's contribution
- **Force Plots**: Visualize the push/pull effect of features on a specific medical prediction

These visualizations help medical professionals understand not just what the model predicts, but why it makes specific predictions.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

Built with ❤️ using Python, Streamlit, scikit-learn, and matplotlib.
