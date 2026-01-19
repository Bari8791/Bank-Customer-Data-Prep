import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import roc_auc_score, precision_recall_curve

# --- 1. CONFIG & ASSET LOADING ---
st.set_page_config(page_title="Bank Churn Dashboard", layout="wide", page_icon="🏦")

@st.cache_resource
def load_assets():
    model_path = "churn_model.pkl"
    scaler_path = "scaler.pkl"
    threshold_path = "threshold.pkl"
    
    if not os.path.exists(model_path) or not os.path.exists(scaler_path) or not os.path.exists(threshold_path):
        st.error("❌ Assets not found! Please run your notebook to generate .pkl files")
        st.stop()
    
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    threshold = joblib.load(threshold_path)
    return model, scaler, threshold

rf_model, scaler_model, best_threshold = load_assets()

REQUIRED_FEATURES = [
    'CreditScore','Gender_num','Age','Tenure','Balance','NumOfProducts',
    'HasCrCard','IsActiveMember','EstimatedSalary','ProductPerYear','balance_to_income','income_v_product'
]

# --- 2. CORE LOGIC ---
def get_predictions(df_input):
    df = df_input.copy()
    if 'Gender_num' not in df.columns and 'Gender' in df.columns:
        df['Gender_num'] = np.where(df['Gender'].astype(str).str.title() == 'Female', 1, 0)
    df['ProductPerYear'] = df['NumOfProducts'] / (df['Tenure'] + 0.1)
    df['balance_to_income'] = df['Balance'] / (df['EstimatedSalary'] + 1)
    df['income_v_product'] = df['EstimatedSalary'] / (df['NumOfProducts'] + 1)
    for col in REQUIRED_FEATURES:
        if col not in df.columns:
            df[col] = 0
    X = df[REQUIRED_FEATURES]
    X_scaled = scaler_model.transform(X)
    probs = rf_model.predict_proba(X_scaled)[:, 1]
    df['Churn Probability'] = probs
    df['Verdict'] = np.where(probs >= best_threshold, "🔴 HIGH RISK", "🟢 LOW RISK")
    return df

# --- 3. SIDEBAR NAVIGATION ---
st.sidebar.title("Navigation")
section = st.sidebar.radio("Go to:", ["Single Prediction","Batch Analysis","Feature Importance","Model Health","What-If Strategy"])

# --- 4. SINGLE CUSTOMER PREDICTION ---
if section == "Single Prediction":
    st.header("🎯 Individual Customer Risk Assessment")
    with st.form("input_form"):
        c1,c2,c3 = st.columns(3)
        with c1:
            credit_score = st.number_input("Credit Score",300,850,650)
            gender = st.selectbox("Gender", ["Male","Female"])
            age = st.slider("Age",18,95,40)
        with c2:
            tenure = st.number_input("Tenure (Years)",0,10,5)
            balance = st.number_input("Account Balance ($)",0.0,250000.0,50000.0)
            num_products = st.slider("Number of Products",1,4,2)
        with c3:
            has_card = st.selectbox("Has Credit Card?", [1,0])
            active = st.selectbox("Is Active Member?", [1,0])
            salary = st.number_input("Estimated Salary ($)",0.0,200000.0,75000.0)
        submit = st.form_submit_button("Run Analysis")

    if submit:
        input_data = pd.DataFrame([{ 'CreditScore':credit_score,'Gender':gender,'Age':age,'Tenure':tenure,'Balance':balance,'NumOfProducts':num_products,'HasCrCard':has_card,'IsActiveMember':active,'EstimatedSalary':salary }])
        results = get_predictions(input_data)
        prob = results['Churn Probability'].values[0]
        verdict = results['Verdict'].values[0]
        st.divider()
        k1,k2 = st.columns(2)
        k1.metric("Churn Probability", f"{prob:.2%}")
        k2.subheader(f"Status: {verdict}")
        if prob >= best_threshold:
            st.error("⚠️ ACTION REQUIRED: This customer shows high churn signals.")
        else:
            st.success("✅ STABLE: Customer is likely to stay.")

# --- 5. BATCH ANALYSIS ---
elif section == "Batch Analysis":
    st.header("📊 Batch Risk Portfolio")
    uploaded_file = st.file_uploader("Upload Customer CSV", type="csv")
    if uploaded_file:
        df_batch = pd.read_csv(uploaded_file)
        processed_df = get_predictions(df_batch)
        high_risk_n = len(processed_df[processed_df['Churn Probability'] >= best_threshold])
        col1,col2,col3 = st.columns(3)
        col1.metric("Total Customers", len(processed_df))
        col2.metric("High Risk Identified", high_risk_n, delta=f"{high_risk_n/len(processed_df):.1%}")
        col3.metric("System Threshold", f"{best_threshold:.3f}")
        fig,ax = plt.subplots(figsize=(10,4))
        sns.histplot(processed_df['Churn Probability'], kde=True, bins=30, color='royalblue', ax=ax)
        ax.axvline(best_threshold, color='red', linestyle='--', label='Risk Threshold')
        ax.set_title("Distribution of Risk Scores")
        st.pyplot(fig)
        st.subheader("Filtered High-Risk List")
        st.dataframe(processed_df[processed_df['Churn Probability'] >= best_threshold].sort_values('Churn Probability',ascending=False))

# --- 6. FEATURE IMPORTANCE ---
elif section == "Feature Importance":
    st.header("🌟 Global Feature Importance")
    importances = pd.Series(rf_model.feature_importances_, index=REQUIRED_FEATURES).sort_values()
    fig,ax = plt.subplots()
    importances.plot(kind='barh', color='teal', ax=ax)
    ax.set_title("Random Forest Feature Importance")
    st.pyplot(fig)

# --- 7. MODEL HEALTH ---
elif section == "Model Health":
    st.header("📈 Model Health & Performance")
    st.info("View Train/Test AUC, threshold analysis, and prediction distribution.")
    # Dummy train/test (replace with real if available)
    y_train = joblib.load('y_train.pkl') if os.path.exists('y_train.pkl') else None
    x_train_scaled = joblib.load('x_train_scaled.pkl') if os.path.exists('x_train_scaled.pkl') else None
    y_test = joblib.load('y_test.pkl') if os.path.exists('y_test.pkl') else None
    x_test_scaled = joblib.load('x_test_scaled.pkl') if os.path.exists('x_test_scaled.pkl') else None

    if y_train is not None and x_train_scaled is not None:
        train_auc = roc_auc_score(y_train, rf_model.predict_proba(x_train_scaled)[:,1])
        test_auc = roc_auc_score(y_test, rf_model.predict_proba(x_test_scaled)[:,1])
        st.metric("Train AUC", f"{train_auc:.4f}")
        st.metric("Test AUC", f"{test_auc:.4f}")
        st.write(f"AUC Gap: {train_auc - test_auc:.4f}")
        fig,ax = plt.subplots(figsize=(8,4))
        sns.histplot(rf_model.predict_proba(x_test_scaled)[:,1], bins=30, kde=True, ax=ax, color='orange')
        ax.axvline(best_threshold, color='red', linestyle='--')
        ax.set_title("Test Set Churn Probability Distribution")
        st.pyplot(fig)
    else:
        st.warning("Training/test artifacts not found; cannot compute AUC")

# --- 8. WHAT-IF STRATEGY ---
elif section == "What-If Strategy":
    st.header("💰 Cost Optimization & Scenario Simulation")
    st.write("Simulate marketing interventions on high-risk customers.")
    reduction = st.slider("Retention Offer Effectiveness (%)",0,100,20)
    st.info("Upload batch data in 'Batch Analysis' first to simulate impact.")
