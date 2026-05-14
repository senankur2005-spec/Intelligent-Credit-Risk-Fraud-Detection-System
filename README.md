# Intelligent Credit Risk & Fraud Detection System

## Overview

An end-to-end Financial AI platform that combines **Credit Risk Prediction** and **Fraud Detection** using Machine Learning and interactive Streamlit dashboards.

This project helps financial institutions:

- Predict loan approval risk
- Detect suspicious financial transactions
- Analyze fraud patterns
- Generate business insights and recommendations
- Perform predictive analytics through interactive dashboards

The platform integrates data preprocessing, exploratory data analysis (EDA), machine learning model development, dashboard deployment, and business intelligence workflows.

---

# Features

## Credit Risk Prediction Module

- Loan approval prediction
- Credit risk assessment
- Applicant profile analysis
- Logistic Regression based prediction system
- Financial risk recommendation engine

## Fraud Detection Module

- Credit card fraud detection
- Fraud probability scoring
- Risk categorization
- Batch transaction prediction using CSV upload
- Fraud analytics and transaction monitoring

## Dashboard & Analytics

- Interactive Streamlit dashboard
- Exploratory Data Analysis (EDA)
- Feature importance visualization
- Confusion matrix and ROC curve analysis
- Business recommendations dashboard
- Dark/Light theme support

---

# Tech Stack

## Programming Languages

- Python

## Libraries & Frameworks

- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Imbalanced-learn (SMOTE)
- Matplotlib
- Seaborn
- Plotly
- Streamlit
- Joblib

---

# Machine Learning Models

## Fraud Detection Models

- Logistic Regression
- Random Forest
- XGBoost
- Isolation Forest

### Final Selected Model

- Random Forest Classifier

## Credit Risk Prediction Models

- Logistic Regression
- Random Forest
- XGBoost

### Final Selected Model

- Logistic Regression

---

# Project Workflow

Data Collection → Data Cleaning → Preprocessing → EDA → Feature Engineering → Model Training → Model Evaluation → Risk Prediction → Dashboard Deployment → Business Recommendations

---

# Project Structure

```bash
Intelligent-Credit-Risk-Fraud-Detection-System/
│
├── dashboard/
│   └── app.py
│
├── models/
│
├── notebooks/
│
├── reports/
│
├── screenshots/  
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

# Exploratory Data Analysis

The project includes:

- Fraud transaction analysis
- Credit risk analysis
- Transaction amount distribution
- Loan approval trends
- Applicant income analysis
- Fraud rate analysis
- Class imbalance visualization
- Financial KPI monitoring

---

# Dashboard Modules

## Home Page

- Project overview
- System architecture
- KPI metrics
- Workflow summary
- Tech stack overview

## Fraud Prediction

- Transaction-based fraud detection
- Fraud probability gauge
- Risk categorization
- Batch CSV prediction

## Credit Risk Prediction

- Loan approval prediction
- Credit risk analysis
- Applicant financial profiling

## EDA Dashboard

- Fraud analytics
- Credit risk analytics
- Interactive visualizations

## Model Insights

- Model comparison
- ROC curve
- Confusion matrix
- Feature importance analysis

## Business Recommendations

- Fraud prevention insights
- Credit risk management insights
- Financial AI recommendations

---

# Installation

## Clone Repository

```bash
git clone https://github.com/senankur2005-spec/Intelligent-Credit-Risk-Fraud-Detection-System.git
```

## Navigate to Project Folder

```bash
cd Intelligent-Credit-Risk-Fraud-Detection-System
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Run Streamlit App

```bash
streamlit run dashboard/app.py
```

---

# Dataset Information

## Fraud Detection Dataset

- Credit card transaction dataset
- Highly imbalanced fraud detection problem
- SMOTE used for imbalance handling

## Credit Risk Dataset

- Loan approval prediction dataset
- Applicant financial information
- Credit history-based risk analysis

---

# Results

## Fraud Detection

- High fraud detection capability
- Strong F1-score using Random Forest
- Risk probability-based monitoring system

## Credit Risk Prediction

- High loan approval prediction accuracy
- Logistic Regression performed best
- Credit history identified as a major influencing factor

---

# Future Improvements

- SHAP Explainable AI integration
- Real-time API deployment
- Cloud deployment using AWS/GCP
- Docker containerization
- Real-time fraud monitoring pipeline
- Database integration
- Authentication system
- Advanced business intelligence dashboards

---

# Author

Ankur Sen\
AIML Student | Data Science & Financial AI Enthusiast

---

# License

This project is intended for educational and portfolio purposes.
