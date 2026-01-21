import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- 1. PAGE CONFIG ---
st.set_page_config(page_title="BANK CUSTOMER CHURN APP", layout="wide", page_icon="🏦")

# --- 2. ASSET LOADING ---
@st.cache_resource
def load_assets():
    paths = {
        "m": "models/churn_model.pkl",
        "s": "models/scaler.pkl",
        "t": "models/threshold.pkl"
    }
    if not all(os.path.exists(v) for v in paths.values()):
        st.error("❌ Model assets missing in /models folder!")
        st.stop()
    return joblib.load(paths["m"]), joblib.load(paths["s"]), joblib.load(paths["t"])

rf_model, scaler_model, best_threshold = load_assets()

# --- 3. DATA PROCESSING AND FEATURE ENGINEERING ---
def process_data(df):
    df = df.copy()
    alias_map = {
        'CreditScore': ['creditscore', 'score', 'credit_rating', 'cr_score'],
        'Gender': ['gender', 'sex', 'gen', 'gender_type'],
        'Age': ['age', 'years', 'customer_age'],
        'Tenure': ['tenure', 'years_with_bank', 'membership'],
        'Balance': ['balance', 'account_balance', 'money', 'wealth'],
        'NumOfProducts': ['numofproducts', 'products', 'services_used'],
        'HasCrCard': ['hascrcard', 'creditcard', 'card_holder'],
        'IsActiveMember': ['isactivemember', 'active', 'is_active', 'status'],
        'EstimatedSalary': ['estimatedsalary', 'salary', 'income', 'annual_revenue']
    }
    found_cols = {}
    for official_name, aliases in alias_map.items():
        for col in df.columns:
            clean_col = col.lower().replace(" ", "").replace("_", "")
            if clean_col in aliases:
                found_cols[col] = official_name
                break
    df = df.rename(columns=found_cols)

    # Feature engineering
    df['Gender_num'] = np.where(df['Gender'].astype(str).str.strip().str.lower().str.startswith('f'), 1, 0)
    df['ProductPerYear'] = df['NumOfProducts'] / (df['Tenure'] + 0.1)
    df['balance_to_income'] = df['Balance'] / (df['EstimatedSalary'] + 1)
    df['income_v_product'] = df['EstimatedSalary'] / (df['NumOfProducts'] + 1)
    
    model_features = ['CreditScore','Gender_num','Age','Tenure','Balance','NumOfProducts',
                     'HasCrCard','IsActiveMember','EstimatedSalary',
                     'ProductPerYear','balance_to_income','income_v_product']
    
    X_scaled = scaler_model.transform(df[model_features])
    df['Prob'] = rf_model.predict_proba(X_scaled)[:, 1]
    
    cond = [(df['Prob'] < 0.3), (df['Prob'] < 0.5), (df['Prob'] < 0.8), (df['Prob'] >= 0.8)]
    choices = ["🟢 Stay (Safe)", "🟡 Likely Stay", "🟠 Likely Leave", "🔴 Highly Leave (Churn)"]
    df['AI_Verdict'] = np.select(cond, choices, default="Unknown")
    
    return df, model_features

# --- 4. SIDEBAR ---
with st.sidebar:
    st.header("📂 Dashboard Mode")

    # Info section for users
    st.markdown(
        "ℹ️ **Mode Info:**\n\n"
        "- **Internal Demo**: Example dataset to explore app functionality.\n"
        "- **Client Batch Analysis**: Upload your CSV for AI predictions.\n\n"
        "**Expected Data Columns for Client Upload:**\n"
        "- CustomerId, Surname, CreditScore, Geography, Gender, Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary\n"
        "- Download the template CSV below to match these column names."
    )

    # Mode selection with Internal Demo default
    mode = st.radio(
        "Select Mode:",
        options=["Internal Demo", "Client Batch Analysis"],
        index=0  # Internal Demo is default
    )

    # Template CSV download
    template = pd.DataFrame({
        'CustomerId': [0],
        'Surname': ['Test'],
        'CreditScore': [650],
        'Geography': ['France'],
        'Gender': ['Female'],
        'Age': [40],
        'Tenure': [3],
        'Balance': [60000],
        'NumOfProducts': [2],
        'HasCrCard': [1],
        'IsActiveMember': [1],
        'EstimatedSalary': [75000]
    })
    st.download_button("📥 Download Template CSV", template.to_csv(index=False), "template.csv")

# --- 5. DATA SOURCE ---
if mode == "Client Batch Analysis":
    st.title("📤 Client Batch Analysis")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    if not uploaded_file: st.stop()
    raw_df = pd.read_csv(uploaded_file)
else:
    st.title("🏦 Internal Demo - BANK CUSTOMER CHURN APP")
    raw_df = pd.read_csv("data/processed/Bank_Churn_Final_With_NumericClusters.csv")

df_results, model_feats = process_data(raw_df)

# --- 6. GLOBAL FILTERS ---
with st.container(border=True):
    st.subheader("🕵️ Global Portfolio Search & Filters")
    f1, f2, f3, f4 = st.columns(4)
    geo_col = next((c for c in ['Geography', 'Country'] if c in df_results.columns), None)
    with f1:
        countries = df_results[geo_col].unique() if geo_col else ["Global"]
        country_sel = st.multiselect("Geography", options=countries, default=countries)
    with f2:
        age_sel = st.slider("Global Age Range", 18, 95, (18, 95))
    with f3:
        bal_sel = st.slider("Global Balance Range ($)", 0, 250000, (0, 250000))
    with f4:
        verdict_sel = st.multiselect("AI Risk Verdict", options=df_results.AI_Verdict.unique(), default=df_results.AI_Verdict.unique())

mask = (df_results.Age.between(age_sel[0], age_sel[1])) & \
       (df_results.Balance.between(bal_sel[0], bal_sel[1])) & \
       (df_results.AI_Verdict.isin(verdict_sel))
if geo_col: mask &= (df_results[geo_col].isin(country_sel))
filtered_df = df_results[mask]

# --- 7. TOP KPIs ---
k1, k2, k3, k4 = st.columns(4)
at_risk_money = filtered_df[filtered_df['Prob'] >= best_threshold]['Balance'].sum()
k1.metric("💰 Exposure", f"${at_risk_money:,.0f}")
k2.metric("📉 Risk Avg", f"{filtered_df['Prob'].mean():.1%}")
k3.metric("🚨 Critical Alerts", len(filtered_df[filtered_df.Prob >= 0.8]))
with k4:
    st.caption("🤖 AI Health")
    st.progress(0.88)

# --- 8. SINGLE CUSTOMER AI ASSESSMENT ---
st.divider()
st.subheader("👤 Single Customer AI Assessment")
with st.expander("Analyze & Export Profile", expanded=False):
    i1, i2, i3, i4, i5 = st.columns(5)
    with i1:
        in_age = st.number_input("Age", 18, 100, 40)
        in_tenure = st.number_input("Tenure (Years)", 0, 10, 5)
        in_credit = st.number_input("Credit Score", 300, 850, 650)
    with i2:
        in_bal = st.number_input("Balance ($)", 0.0, 500000.0, 50000.0)
        in_products = st.slider("Number of Products", 1, 4, 1)
    with i3:
        in_active = st.selectbox("Status", ["Active", "Not Active"])
        in_card = st.selectbox("Has Credit Card?", [1, 0])
    with i4:
        in_salary = st.number_input("Estimated Salary ($)", 0.0, 200000.0, 75000.0)
    with i5:
        in_gender = st.selectbox("Gender", ["Male", "Female"])

    if st.button("🚀 Run AI Analysis"):
        test_data = pd.DataFrame([{
            'CreditScore': in_credit,
            'Gender': in_gender,
            'Age': in_age,
            'Tenure': in_tenure,
            'Balance': in_bal,
            'NumOfProducts': in_products,
            'HasCrCard': in_card,
            'IsActiveMember': 1 if in_active=="Active" else 0,
            'EstimatedSalary': in_salary
        }])
        res, _ = process_data(test_data)
        verdict = res['AI_Verdict'].values[0]
        prob = res['Prob'].values[0]

        st.write(f"#### Result: {verdict}")
        st.metric("Churn Probability", f"{prob:.2%}")

        st.subheader("📝 Customer Feature Table")
        st.dataframe(res, use_container_width=True)

        ind_csv = res.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Export Individual Profile (CSV)", ind_csv, "individual_assessment.csv")

# --- 9. WHAT-IF SIMULATION & ROI ---
st.divider()
st.subheader("💰 What-If Simulation & Retention ROI")
# ... keep your previous What-If Simulation code unchanged ...
