# Customer Churn Prediction using Explainable AI

An end-to-end Machine Learning project that predicts customer churn and explains the key factors influencing model predictions using Explainable AI.

---

## Project Overview

Customer churn is a major challenge for subscription-based businesses. Identifying customers who are likely to leave can help organizations take proactive actions to improve customer retention.

This project covers the complete Machine Learning workflow, including data preprocessing, exploratory data analysis, feature engineering, model training, evaluation, explainability using SHAP, new customer prediction, and deployment through a Streamlit web application.

The application not only predicts whether a customer is likely to churn but also provides the **churn probability, risk level, and key factors that increased or reduced the churn risk**.

---

## Objectives

* Analyze customer churn patterns.
* Clean and preprocess customer data.
* Perform Exploratory Data Analysis (EDA).
* Engineer meaningful features.
* Train and compare multiple Machine Learning models.
* Evaluate model performance using classification metrics.
* Select and save a final model.
* Explain model predictions using SHAP.
* Predict churn for new customers.
* Identify factors increasing and reducing churn risk.
* Develop an interactive Streamlit application.

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
Churn Prediction + Probability
        ↓
Prediction Explanation
        ↓
Streamlit Web Application
```

---

## Machine Learning Models

The following models were trained and compared:

* Logistic Regression
* Decision Tree Classifier
* Random Forest Classifier
* XGBoost Classifier

The models were evaluated and compared to select the final model for customer churn prediction.

---

## Model Evaluation

Models were evaluated using:

* Accuracy
* Precision
* Recall
* F1-Score
* Confusion Matrix
* ROC Curve
* ROC-AUC Score

Evaluation data and results are available in the `Results/` directory.

---

## Feature Engineering

The following features were engineered to capture customer behavior and service usage patterns:

* `Tenure Group`
* `Total Aditional Services`
* `Total Streaming Services`
* `Total Security Services`
* `Has Multiple Services`

These features provide additional information about customer service usage and behavior.

---

## Explainable AI

This project uses **SHAP (SHapley Additive exPlanations)** to improve the transparency and interpretability of model predictions.

SHAP analysis helps to:

* Identify the most influential features.
* Understand the overall impact of features on churn predictions.
* Analyze positive and negative feature contributions.
* Interpret individual customer predictions.
* Identify factors increasing churn risk.
* Identify factors reducing churn risk.

The Streamlit application integrates SHAP-based explanations for individual customer predictions, allowing users to understand not only **what the model predicted**, but also **which factors influenced that prediction**.

---

## New Customer Prediction

The final model predicts whether a customer is:

* **Likely to Churn**
* **Likely to Stay**

It also provides the customer's **churn probability** and churn risk interpretation.

```text
Customer Input
      ↓
Feature Engineering
      ↓
Feature Alignment
      ↓
Data Preprocessing
      ↓
Trained Model
      ↓
Churn Prediction + Probability
      ↓
Risk Interpretation
      ↓
SHAP Explanation
      ↓
Factors Increasing / Reducing Churn Risk
```

---

## Streamlit Web Application

An interactive Streamlit application was developed to make the model accessible through a user-friendly interface.

Users can enter customer information including:

* Personal information
* Service information
* Internet and additional services
* Contract details
* Payment information
* Tenure information
* Monthly charges
* Total charges
* Customer Lifetime Value (CLTV)

After clicking **Predict Churn**, the application provides:

* Customer churn prediction
* Churn probability
* Churn risk interpretation
* Factors increasing churn risk
* Factors reducing churn risk
* Customer-specific feature contributions
* Prediction summary

This makes the prediction more transparent and helps users understand the reasons behind the model's decision.

🚀 Live Application

🔗 Click here to try the live application

---

## Project Structure

```text
Customer-Churn-Prediction-using-Explainable-AI/
│
├── Dataset/
├── Models/
│   ├── churn_models/
│   ├── final_churn_model/
│   ├── Model.pkl
│   └── Preprocessor.pkl
│
├── Notebooks/
│
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

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* XGBoost
* SHAP
* Joblib
* Streamlit
* Google Colab
* GitHub

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

The application will then open in your browser.

---

## Key Skills Demonstrated

* Data Cleaning and Preprocessing
* Exploratory Data Analysis
* Feature Engineering
* Machine Learning Classification
* Model Training and Comparison
* Model Evaluation
* Explainable AI with SHAP
* Predictive Modeling
* New Customer Prediction
* Customer-Level Prediction Explanation
* Streamlit Application Development
* Model Deployment Preparation
* GitHub Project Organization

---

## Author

**Asima Shafiq**

BS Data Science Student

Government College University Faisalabad
