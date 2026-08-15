# Customer Churn Prediction using Explainable AI

An end-to-end Machine Learning project that predicts customer churn and explains the key factors influencing model predictions using Explainable AI.

---

## Project Overview

Customer churn is a major challenge for subscription-based businesses. Identifying customers who are likely to leave can help organizations take proactive actions to improve customer retention.

This project covers the complete Machine Learning workflow, including data preprocessing, exploratory data analysis, feature engineering, model training, evaluation, explainability using SHAP, new customer prediction, and deployment through a Streamlit web application.

---

## Objectives

- Analyze customer churn patterns.
- Clean and preprocess customer data.
- Perform Exploratory Data Analysis (EDA).
- Engineer meaningful features.
- Train and compare multiple Machine Learning models.
- Evaluate model performance using classification metrics.
- Select and save a final model.
- Explain model predictions using SHAP.
- Predict churn for new customers.
- Develop an interactive Streamlit application.

---

## Project Workflow

```text
Raw Customer Data
        ↓
Data Cleaning & Preprocessing
        ↓
Exploratory Data Analysis
        ↓
Feature Engineering
        ↓
Model Training & Comparison
        ↓
Model Evaluation
        ↓
Explainable AI with SHAP
        ↓
Final Model Selection
        ↓
New Customer Prediction
        ↓
Streamlit Web Application
```

---

## Machine Learning Models

The following models were trained and compared:

- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier
- XGBoost Classifier

---

## Model Evaluation

Models were evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix
- ROC Curve
- ROC-AUC Score

Evaluation data and results are available in the `Results/` directory.

---

## Feature Engineering

The following features were engineered to capture customer behavior and service usage patterns:

- `Tenure Group`
- `Total Aditional Services`
- `Total Streaming Services`
- `Total Security Services`
- `Has Multiple Services`

---

## Explainable AI

This project uses **SHAP (SHapley Additive exPlanations)** to improve the transparency and interpretability of model predictions.

SHAP analysis helps to:

- Identify the most influential features.
- Understand the overall impact of features on churn predictions.
- Analyze positive and negative feature contributions.
- Interpret individual customer predictions.

---

## New Customer Prediction

The final model predicts whether a customer is:

- **Likely to Churn**
- **Likely to Stay**

It also provides the customer's **churn probability**.

```text
Customer Input
      ↓
Feature Alignment
      ↓
Data Preprocessing
      ↓
Trained Model
      ↓
Churn Prediction + Probability
```

---

## Streamlit Web Application

An interactive Streamlit application was developed to make the model accessible through a user-friendly interface.

Users can enter customer information and receive:

- Churn prediction
- Churn probability
- Customer risk interpretation

---

## Project Structure

```text
Customer-Churn-Prediction-using-Explainable-AI/
│
├── Dataset/
├── Models/
│   ├── churn_models
│   ├── final_churn_model
│   ├── Model.pkl
│   └── Preprocessor.pkl
│
├── Notebooks/
├── Results/
│   ├── Evaluation data
│   └── evaluation_results
│
├── app.py
├── requirements.txt
└── README.md
```

---

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- XGBoost
- SHAP
- Joblib
- Streamlit
- Google Colab
- GitHub

---

## Run Locally

Clone the repository:

```bash
git clone https://github.com/YOUR-USERNAME/YOUR-REPOSITORY-NAME.git
```

Navigate to the project directory:

```bash
cd Customer-Churn-Prediction-using-Explainable-AI
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
streamlit run app.py
```

---

## Key Skills Demonstrated

- Data Cleaning and Preprocessing
- Exploratory Data Analysis
- Feature Engineering
- Machine Learning Classification
- Model Training and Comparison
- Model Evaluation
- Explainable AI with SHAP
- Predictive Modeling
- New Customer Prediction
- Streamlit Application Development
- GitHub Project Organization

---

## Author

**Asima Shafiq**  
BS Data Science Student  
Government College University Faisalabad
