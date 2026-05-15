import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

import plotly.graph_objects as go

st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="💳",
    layout="wide"
)

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "fraud_detection_model.pkl")
DATA_PATH = os.path.join(BASE_DIR, "dataset", "credit_risk_cleaned.csv")
FEATURE_PATH = os.path.join(BASE_DIR, "reports", "feature_importance.csv")

CREDIT_MODEL_PATH = os.path.join(BASE_DIR, "models", "credit_risk_model.pkl")
CREDIT_SCALER_PATH = os.path.join(BASE_DIR, "models", "credit_scaler.pkl")
CREDIT_ENCODER_PATH = os.path.join(BASE_DIR, "models", "credit_label_encoders.pkl")
CREDIT_COLUMNS_PATH = os.path.join(BASE_DIR, "models", "credit_feature_columns.pkl")

# Load model/data
model = joblib.load(MODEL_PATH)
credit_model = joblib.load(CREDIT_MODEL_PATH)
credit_scaler = joblib.load(CREDIT_SCALER_PATH)
credit_label_encoders = joblib.load(CREDIT_ENCODER_PATH)
credit_feature_columns = joblib.load(CREDIT_COLUMNS_PATH)
df = pd.read_csv(DATA_PATH)
feature_importance = pd.read_csv(FEATURE_PATH)

st.title("💳 Intelligent Credit Risk & Fraud Detection System")
st.sidebar.title("💳 Financial AI Platform")
st.write("A machine learning based financial AI platform for credit risk prediction and fraud detection.")

# Sidebar
page = st.sidebar.radio(
    "Navigation",
    [
        "Home",
        "Fraud Prediction",
        "Credit Risk Prediction",
        "EDA Dashboard",
        "Model Insights",
        "Business Recommendations"
    ]
)

def risk_category(probability):
    if probability < 0.30:
        return "Low Risk"
    elif probability < 0.70:
        return "Medium Risk"
    else:
        return "High Risk"

if page == "Home":
    st.header("Project Overview")

    st.write("""
    This project is an end-to-end Financial AI system that combines
    Credit Risk Prediction and Fraud Detection using machine learning.
    It helps financial institutions assess loan approval risk and identify
    suspicious transactions.
    """)

    st.subheader("System Modules")

    col1, col2 = st.columns(2)

    with col1:
        st.info("🏦 Credit Risk Prediction")
        st.write("""
        Predicts whether a loan applicant is likely to be approved or rejected
        based on income, credit history, employment status, loan amount, and
        property area.
        """)

    with col2:
        st.warning("💳 Fraud Detection")
        st.write("""
        Detects suspicious credit card transactions using machine learning
        and assigns fraud probability with risk category.
        """)

    st.subheader("Project KPIs")

    col1, col2, col3 = st.columns(3)

    col1.metric("Fraud Transactions", int(df["Class"].sum()))
    col2.metric("Fraud Rate", f"{df['Class'].mean() * 100:.4f}%")
    col3.metric("ML Modules", "2")

    st.subheader("Tech Stack")

    st.write("""
    Python, Pandas, NumPy, Scikit-learn, XGBoost, SMOTE, Random Forest,
    Logistic Regression, Streamlit, Plotly, Matplotlib, Seaborn.
    """)

    st.subheader("Project Workflow")

    st.write("""
    Data Collection → Data Cleaning → EDA → Preprocessing → Model Training →
    Model Evaluation → Risk Prediction → Dashboard Deployment → Business Recommendations
    """)

elif page == "Fraud Prediction":
    st.header("Fraud Prediction")

    st.write("Select a transaction sample from the dataset and predict fraud risk.")

    sample_index = st.number_input(
        "Select Transaction Index",
        min_value=0,
        max_value=len(df) - 1,
        value=0
    )

    selected_transaction = df.drop("Class", axis=1).iloc[[sample_index]]
    actual_class = df["Class"].iloc[sample_index]

    st.subheader("Selected Transaction")
    st.dataframe(selected_transaction, width="stretch")

    if actual_class == 1:
        st.warning("⚠️ This is an actual FRAUD transaction.")
    else:
        st.info("✅ This is a NORMAL transaction.")

    if st.button("Predict Fraud Risk"):
        probability = model.predict_proba(selected_transaction)[0][1]
        prediction = model.predict(selected_transaction)[0]
        risk = risk_category(probability)

        st.subheader("Prediction Result")

        col1, col2, col3 = st.columns(3)

        with col1:
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=probability * 100,
                title={'text': "Fraud Probability"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "red"},
                    'steps': [
                        {'range': [0, 30], 'color': "green"},
                        {'range': [30, 70], 'color': "yellow"},
                        {'range': [70, 100], 'color': "red"}
                    ]
                }
            ))

            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.metric("Risk Category", risk)

        with col3:
            st.metric(
                "Actual Class",
                "Fraud" if actual_class == 1 else "Normal"
            )

        if prediction == 1:
            st.error("⚠️ Fraudulent Transaction Detected")
        else:
            st.success("✅ Normal Transaction")

    st.divider()

    st.subheader("Batch Fraud Prediction")

    uploaded_file = st.file_uploader("Upload a CSV file for batch prediction", type=["csv"])

    if uploaded_file is not None:
        batch_df = pd.read_csv(uploaded_file)

        st.write("Uploaded Data Preview")
        st.dataframe(batch_df.head(), width="stretch")

        if "Class" in batch_df.columns:
            batch_features = batch_df.drop("Class", axis=1)
        else:
            batch_features = batch_df.copy()

        if st.button("Run Batch Prediction"):
            batch_probabilities = model.predict_proba(batch_features)[:, 1]
            batch_predictions = model.predict(batch_features)

            batch_results = batch_df.copy()
            batch_results["Fraud Probability"] = batch_probabilities
            batch_results["Risk Category"] = [
                risk_category(p) for p in batch_probabilities
            ]
            batch_results["Prediction"] = [
                "Fraud" if pred == 1 else "Normal"
                for pred in batch_predictions
            ]

            st.success("Batch prediction completed successfully")
            st.dataframe(batch_results.head(20), width="stretch")

            csv = batch_results.to_csv(index=False).encode("utf-8")

            st.download_button(
                label="Download Prediction Results",
                data=csv,
                file_name="fraud_prediction_results.csv",
                mime="text/csv"
            )   

elif page == "Credit Risk Prediction":

    st.header("Credit Risk Prediction")
    st.write("Enter applicant details to predict loan approval risk.")

    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("Gender", credit_label_encoders["Gender"].classes_)
        married = st.selectbox("Married", credit_label_encoders["Married"].classes_)
        dependents = st.selectbox("Dependents", credit_label_encoders["Dependents"].classes_)
        education = st.selectbox("Education", credit_label_encoders["Education"].classes_)
        self_employed = st.selectbox("Self Employed", credit_label_encoders["Self_Employed"].classes_)

    with col2:
        applicant_income = st.number_input("Applicant Income", min_value=0, value=5000)
        coapplicant_income = st.number_input("Coapplicant Income", min_value=0.0, value=0.0)
        loan_amount = st.number_input("Loan Amount", min_value=0.0, value=120.0)
        loan_amount_term = st.number_input("Loan Amount Term", min_value=0.0, value=360.0)
        credit_history = st.selectbox("Credit History", [0.0, 1.0])
        property_area = st.selectbox("Property Area", credit_label_encoders["Property_Area"].classes_)

    input_credit = pd.DataFrame([{
        "Gender": gender,
        "Married": married,
        "Dependents": dependents,
        "Education": education,
        "Self_Employed": self_employed,
        "ApplicantIncome": applicant_income,
        "CoapplicantIncome": coapplicant_income,
        "LoanAmount": loan_amount,
        "Loan_Amount_Term": loan_amount_term,
        "Credit_History": credit_history,
        "Property_Area": property_area
    }])

    # Encode categorical columns
    categorical_cols = ["Gender", "Married", "Dependents", "Education", "Self_Employed", "Property_Area"]

    for col in categorical_cols:
        input_credit[col] = credit_label_encoders[col].transform(input_credit[col])

    # Feature engineering
    input_credit["Total_Income"] = input_credit["ApplicantIncome"] + input_credit["CoapplicantIncome"]
    input_credit["Total_Income_Log"] = np.log1p(input_credit["Total_Income"])
    input_credit["LoanAmount_Log"] = np.log1p(input_credit["LoanAmount"])
    input_credit["Debt_Income_Ratio"] = input_credit["LoanAmount"] / (input_credit["Total_Income"] + 1)

    if input_credit["Total_Income"].iloc[0] > 20000:
        st.warning(
            "This income value is much higher than most training data. "
            "Prediction may be less reliable for extreme income values."
        )

    # Final feature columns used during model training
    credit_feature_columns = [
        "Credit_History",
        "Total_Income_Log",
        "LoanAmount_Log",
        "Debt_Income_Ratio",
        "Married",
        "Education",
        "Property_Area"
    ]

    input_ready = input_credit[credit_feature_columns]

    if st.button("Predict Credit Risk"):

        prediction = credit_model.predict(input_ready)[0]
        probability = credit_model.predict_proba(input_ready)[0]

        rejection_prob = probability[0]
        approval_prob = probability[1]

        st.subheader("Prediction Result")

        col1, col2 = st.columns(2)

        col1.metric("Loan Approval Probability", f"{approval_prob * 100:.2f}%")
        col2.metric("Credit Risk Probability", f"{rejection_prob * 100:.2f}%")

        if prediction == 1:
            st.success("✅ Loan Approved / Low Credit Risk")
        else:
            st.error("❌ Loan Rejected / High Credit Risk")

        st.subheader("Risk Interpretation")

        if approval_prob >= 0.75:
            st.success("The applicant has a strong credit profile.")
        elif approval_prob >= 0.50:
            st.warning("The applicant has moderate credit risk.")
        else:
            st.error("The applicant has high credit risk.")

        st.subheader("Input Summary")
        st.dataframe(input_ready)

elif page == "EDA Dashboard":
    st.header("Exploratory Data Analysis")

    tab1, tab2 = st.tabs(["Fraud Detection EDA", "Credit Risk EDA"])

    with tab1:
        st.subheader("Fraud Dataset Overview")

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Transactions", len(df))
        col2.metric("Fraud Transactions", int(df["Class"].sum()))
        col3.metric("Fraud Rate", f"{df['Class'].mean() * 100:.4f}%")

        st.subheader("Class Distribution")
        st.bar_chart(df["Class"].value_counts())

        st.subheader("Transaction Amount Distribution")
        st.line_chart(df["Amount"].head(1000))

        st.subheader("Average Transaction Amount by Class")
        st.bar_chart(df.groupby("Class")["Amount"].mean())

        st.subheader("Dataset Summary")
        st.dataframe(df.describe(), width="stretch")

    with tab2:
        st.subheader("Credit Risk Dataset Overview")

        credit_df = pd.read_csv(os.path.join(BASE_DIR, "dataset", "credit_risk_processed.csv"))

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Loan Records", len(credit_df))
        col2.metric("Approved Loans", int(credit_df["Loan_Status"].sum()))
        col3.metric("Approval Rate", f"{credit_df['Loan_Status'].mean() * 100:.2f}%")

        st.subheader("Loan Status Distribution")
        st.bar_chart(credit_df["Loan_Status"].value_counts())

        st.subheader("Average Loan Amount by Loan Status")
        st.bar_chart(credit_df.groupby("Loan_Status")["LoanAmount"].mean())

        st.subheader("Average Applicant Income by Loan Status")
        st.bar_chart(credit_df.groupby("Loan_Status")["ApplicantIncome"].mean())

        st.subheader("Credit Risk Dataset Summary")
        st.dataframe(credit_df.describe(), width="stretch")

elif page == "Model Insights":
    st.header("Model Insights")

    tab1, tab2 = st.tabs(["Fraud Detection Model", "Credit Risk Model"])

    with tab1:
        st.subheader("Fraud Detection Model")

        st.write("Final Model: Random Forest")

        st.subheader("Top Important Features")
        st.bar_chart(
            feature_importance.set_index("Feature")["Importance"].head(10)
        )

        st.subheader("Model Performance")

        col1, col2 = st.columns(2)

        with col1:
            st.image(os.path.join(BASE_DIR, "reports", "confusion_matrix.png"))

        with col2:
            st.image(os.path.join(BASE_DIR, "reports", "roc_curve.png"))

        st.success("""
        Random Forest was selected because it achieved the best F1-score and provided
        a strong balance between precision and recall for fraud detection.
        """)

    with tab2:
        st.subheader("Credit Risk Model")

        st.write("Final Model: Random Forest Classifier")

        credit_results = pd.read_csv(os.path.join(BASE_DIR, "reports", "credit_model_results.csv"))

        st.subheader("Credit Model Comparison")
        st.dataframe(credit_results, width="stretch")

        st.success("""
        Random Forest was selected as the final model because it provided a good balance
        between prediction performance and interpretability. Although XGBoost achieved
        slightly higher accuracy, Random Forest showed healthier feature importance
        distribution by considering Credit History, Debt-Income Ratio, Income, and Loan Amount
        together instead of depending too heavily on one feature.
        """)

        st.subheader("SHAP Explainability Analysis")

        st.write("""
        SHAP analysis explains how each feature influences the model's loan approval prediction.
        Positive SHAP values push the prediction toward loan approval, while negative SHAP values
        push the prediction toward rejection or higher credit risk.
        """)

        st.image(
        os.path.join(BASE_DIR, "reports", "credit_shap_summary.png"),
        caption="SHAP Summary Plot for Credit Risk Model"
        )

        st.markdown("""
        ### Key Insights from SHAP

        - **Credit_History** is the strongest factor in loan approval prediction.
        - **Debt_Income_Ratio** is an important affordability indicator.
        - **LoanAmount_Log** shows the effect of loan burden on approval chances.
        - **Total_Income_Log** contributes moderately, but the model does not depend only on income.
        - **Married, Education, and Property_Area** have smaller but useful effects.

        This shows that the Random Forest model is using multiple financial indicators
        instead of relying only on Credit History.
        """)

elif page == "Business Recommendations":
    st.header("Business Recommendations")

    tab1, tab2 = st.tabs(["Fraud Prevention", "Credit Risk Management"])

    with tab1:
        st.subheader("Fraud Prevention Insights")

        st.write("""
        - Fraud cases are extremely rare, so class imbalance handling is essential.
        - Random Forest provides the best balance between fraud detection and false alert reduction.
        - High-risk transactions should be flagged for manual verification.
        - Batch prediction can help monitor multiple transactions at once.
        - The fraud model should be retrained regularly with new transaction data.
        """)

        st.subheader("Recommended Actions")

        st.success("""
        Use fraud probability scoring for real-time transaction monitoring and
        trigger alerts for high-risk transactions.
        """)

    with tab2:
        st.subheader("Credit Risk Insights")

        st.write("""
        - Credit history is one of the most important factors in loan approval.
        - Applicant income, loan amount, and property area help assess repayment risk.
        - Random Forest is suitable for credit scoring because it provides good performance and interpretability.
        - Medium-risk applicants should be reviewed manually before rejection.
        - The credit risk model can support faster and more consistent loan decisions.
        """)

        st.subheader("Recommended Actions")

        st.success("""
        Use the credit risk model as a decision-support system, not as the only
        approval authority. Combine ML predictions with financial policy rules.
        """)

    st.subheader("Final Project Conclusion")

    st.info("""
    This project provides a complete Financial AI workflow by combining fraud detection
    and credit risk prediction into one dashboard-based decision support system.
    """)

st.markdown("---")
st.markdown(
    "Developed by Ankur Sen | AIML | Financial AI & Fraud Analytics Project"
)