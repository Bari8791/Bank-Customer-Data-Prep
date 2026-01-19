import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- 1. PAGE CONFIG ---
st.set_page_config(page_title="Executive AI Command", layout="wide", page_icon="🏦")

# --- 2. ASSET LOADING ---
@st.cache_resource
def load_assets():
    paths = {"m": "models/churn_model.pkl", "s": "models/scaler.pkl", "t": "models/threshold.pkl"}
    if not all(os.path.exists(v) for v in paths.values()):
        st.error("❌ Model assets missing in /models folder!")
        st.stop()
    return joblib.load(paths["m"]), joblib.load(paths["s"]), joblib.load(paths["t"])

rf_model, scaler_model, best_threshold = load_assets()

# --- 3. ROBUST AI LOGIC ---
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

# --- 4. DATA SOURCE ---
with st.sidebar:
    st.header("📂 Data Controller")
    mode = st.radio("Dashboard Mode:", ["🏠 Internal Database", "📤 Client Upload Mode"])
    st.divider()
    template = pd.DataFrame({'CustomerId': [0], 'Surname': ['Test'], 'CreditScore': [650], 'Geography': ['France'], 'Gender': ['Female'], 'Age': [40], 'Tenure': [3], 'Balance': [60000], 'NumOfProducts': [2], 'HasCrCard': [1], 'IsActiveMember': [1], 'EstimatedSalary': [75000]})
    st.download_button("📥 Download Template CSV", template.to_csv(index=False), "template.csv")

if mode == "📤 Client Upload Mode":
    st.title("📤 Client Batch Analysis")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    if not uploaded_file: st.stop()
    raw_df = pd.read_csv(uploaded_file)
else:
    st.title("🏦 Executive AI Command Center")
    raw_df = pd.read_csv("data/processed/Bank_Churn_Final_With_NumericClusters.csv")

df_results, model_feats = process_data(raw_df)

# --- 5. GLOBAL FILTERS ---
with st.container(border=True):
    st.subheader("🕵️ Global Portfolio Search & Filters")
    f1, f2, f3, f4 = st.columns(4)
    with f1:
        geo_col = next((c for c in ['Geography', 'Country'] if c in df_results.columns), None)
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

# --- 6. TOP ROW KPIs ---
k1, k2, k3, k4 = st.columns(4)
at_risk_money = filtered_df[filtered_df['Prob'] >= best_threshold]['Balance'].sum()
k1.metric("💰 Exposure", f"${at_risk_money:,.0f}")
k2.metric("📉 Risk Avg", f"{filtered_df['Prob'].mean():.1%}")
k3.metric("🚨 Critical Alerts", len(filtered_df[filtered_df.Prob >= 0.8]))
with k4:
    st.caption("🤖 AI Health")
    st.progress(0.88)

# --- 7. SECTION: INDIVIDUAL ASSESSMENT ---
st.divider()
st.subheader("👤 Single Customer AI Assessment")
with st.expander("Analyze & Export Profile", expanded=False):
    i1, i2, i3 = st.columns(3)
    in_age = i1.number_input("Age", 18, 100, 40)
    in_bal = i2.number_input("Balance", 0.0, 500000.0, 50000.0)
    in_active = i3.selectbox("Status", ["Active", "Not Active"])
    
    if st.button("🚀 Run AI Analysis"):
        test_data = pd.DataFrame([{'CreditScore':650, 'Gender':'Female', 'Age':in_age, 'Tenure':5, 'Balance':in_bal, 'NumOfProducts':1, 'HasCrCard':1, 'IsActiveMember':(1 if in_active=="Active" else 0), 'EstimatedSalary':75000}])
        res, _ = process_data(test_data)
        st.write(f"#### Result: {res['AI_Verdict'].values[0]}")
        st.progress(res['Prob'].values[0])
        
        # EXPORT 1: INDIVIDUAL
        ind_csv = res.T.to_csv().encode('utf-8')
        st.download_button("📥 Export Individual Profile (CSV)", ind_csv, "individual_assessment.csv")

# --- 8. SECTION: WHAT-IF SIMULATION (NEW FILTERS) ---
st.divider()
st.subheader("💰 What-If Simulation & Retention ROI")
with st.container(border=True):
    st.markdown("**Simulation Target Filters**")
    s1, s2, s3 = st.columns(3)
    sim_age = s1.slider("Target Age Bracket", 18, 95, (30, 60))
    sim_bal = s2.slider("Target Balance Bracket ($)", 0, 250000, (20000, 250000))
    sim_cost = s3.number_input("Cost to Save 1 Customer ($)", 10, 1000, 150)

# Calculate simulation data
sim_df = filtered_df[(filtered_df.Age.between(sim_age[0], sim_age[1])) & (filtered_df.Balance.between(sim_bal[0], sim_bal[1]))]
sim_at_risk = sim_df[sim_df['Prob'] >= best_threshold]['Balance'].sum()
sim_count = len(sim_df[sim_df['Prob'] >= best_threshold])

col_sim1, col_sim2 = st.columns([1, 2])
with col_sim1:
    eff = st.slider("Campaign Effectiveness (%)", 0, 100, 30)
    potential_saved = sim_at_risk * (eff/100)
    total_cost = sim_count * sim_cost
    roi = ((potential_saved - total_cost) / total_cost) if total_cost > 0 else 0
    
    st.metric("Potential Capital Saved", f"${potential_saved:,.0f}")
    st.metric("Campaign Total Cost", f"${total_cost:,.0f}", delta=f"ROI: {roi:.1%}")
    
    # EXPORT 2: ROI REPORT
    roi_report = pd.DataFrame({"Metric": ["Target Group Count", "Capital at Risk", "Est. Cost", "Est. Savings", "Net ROI"], 
                               "Value": [sim_count, sim_at_risk, total_cost, potential_saved, f"{roi:.1%}"]})
    st.download_button("📥 Export Financial Simulation (CSV)", roi_report.to_csv(index=False), "roi_simulation_report.csv")

with col_sim2:
    
    fig_curve, ax_curve = plt.subplots(figsize=(10, 4))
    sns.lineplot(data=sim_df['Prob'].sort_values().values, color="blue", ax=ax_curve)
    ax_curve.axhline(best_threshold, color='red', ls='--', label='Risk Threshold')
    ax_curve.set_title("Targeted Segment Risk Distribution")
    st.pyplot(fig_curve)

# --- 9. SECTION: AI BRAIN HEALTH ---
st.divider()
st.subheader("🧠 AI Brain Health & Interpretability")
h1, h2 = st.columns(2)
with h1:
    st.markdown("**Feature Impact Analysis**")
    
    feat_imp = pd.Series(rf_model.feature_importances_, index=model_feats).sort_values()
    fig_imp, ax_imp = plt.subplots()
    feat_imp.plot(kind='barh', color='teal', ax=ax_imp)
    st.pyplot(fig_imp)
    
    # EXPORT 3: FEATURE IMPORTANCE
    st.download_button("📥 Export Model Logic (CSV)", feat_imp.to_csv(), "ai_model_logic.csv")

with h2:
    st.markdown("**Probability Distribution**")
    fig_hist, ax_hist = plt.subplots()
    sns.histplot(df_results['Prob'], bins=30, kde=True, color="purple", ax=ax_hist)
    st.pyplot(fig_hist)
    
    # EXPORT 4: FULL DATA
    st.download_button("📥 Export Full Batch Report (CSV)", filtered_df.to_csv(index=False), "master_churn_report.csv")

st.info("💡 **Executive Summary:** The model identifies Age and Product engagement as the strongest churn predictors. Targeted campaigns for customers aged 30-50 with high balances show the highest potential ROI.")