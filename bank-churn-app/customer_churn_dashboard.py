import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import roc_curve, precision_recall_curve
from sklearn.calibration import calibration_curve

# =========================================================
# 1. PAGE STYLING
# =========================================================
st.set_page_config(page_title="Bank Customer Churn Intelligence Dashboard", layout="wide")

st.markdown("""
<style>
    .stApp { background-color: #F8F9FB; color: #2D3436; }
    h1, h2, h3 { color: #002E5D !important; font-family: 'Segoe UI', sans-serif; font-weight: bold; }
    .portfolio-card {
        background-color: white;
        border: 1px solid #D1D8E0;
        border-radius: 10px;
        padding: 25px;
        margin-bottom: 20px;
        box-shadow: 0px 4px 12px rgba(0,0,0,0.05);
    }
</style>
""", unsafe_allow_html=True)

# =========================================================
# 2. SAMPLE DATA (Replace with your real dataset)
# =========================================================
@st.cache_data
def get_data():
    np.random.seed(42)
    n = 1500
    df = pd.DataFrame({
        "Customer_ID": range(10000, 10000+n),
        "Age": np.random.randint(22, 75, n),
        "Balance": np.random.uniform(5000, 500000, n),
        "Geography": np.random.choice(["France", "Germany", "Spain"], n),
        "Tenure": np.random.randint(0, 12, n),
        "HasCrCard": np.random.choice([0,1], n),
        "NumOfProducts": np.random.randint(1, 4, n),
        "IsActiveMember": np.random.choice([0,1], n),
        "Probability_of_Exit": np.random.uniform(0, 1, n),
        "Actual_Historical_Exit": np.random.choice([0,1], n, p=[0.8,0.2])
    })
    return df

df = get_data()

# =========================================================
# 3. FILTERS
# =========================================================
st.title("🏦 Bank Customer Intelligence Dashboard")

with st.container():
    st.markdown('<div class="portfolio-card">', unsafe_allow_html=True)
    st.subheader("Portfolio Filters")
    
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    with col1:
        geo_sel = st.multiselect("Region", ["France","Germany","Spain"], default=["France","Germany","Spain"])
    with col2:
        age_sel = st.slider("Age", 18, 85, (30,60))
    with col3:
        bal_sel = st.slider("Balance ($)", 0, 500000, (20000, 300000))
    with col4:
        tenure_sel = st.slider("Tenure (Years)", 0, 12, (0, 10))
    with col5:
        card_sel = st.select_slider("Credit Card", options=[0,1], value=(0,1))
    with col6:
        active_sel = st.select_slider("Active Member", options=[0,1], value=(0,1))
    
    risk_threshold = st.slider("Exit Probability Threshold", 0.0, 1.0, 0.65)
    
    st.markdown('</div>', unsafe_allow_html=True)

df_filtered = df[
    (df["Geography"].isin(geo_sel)) &
    (df["Age"].between(age_sel[0], age_sel[1])) &
    (df["Balance"].between(bal_sel[0], bal_sel[1])) &
    (df["Tenure"].between(tenure_sel[0], tenure_sel[1])) &
    (df["HasCrCard"].between(card_sel[0], card_sel[1])) &
    (df["IsActiveMember"].between(active_sel[0], active_sel[1]))
].copy()
df_filtered['AI_Prediction'] = np.where(df_filtered['Probability_of_Exit'] >= risk_threshold, "EXIT PREDICTED", "RETAINED")

# =========================================================
# 4. EXECUTIVE SUMMARY
# =========================================================
with st.container():
    st.markdown('<div class="portfolio-card">', unsafe_allow_html=True)
    st.subheader("Executive Portfolio Summary")
    
    num_leave = (df_filtered['AI_Prediction'] == "EXIT PREDICTED").sum()
    num_stay = (df_filtered['AI_Prediction'] == "RETAINED").sum()
    
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("Clients in Scope", f"{len(df_filtered):,}")
    k2.metric("Projected to Leave", f"{num_leave:,}")
    k3.metric("Projected to Stay", f"{num_stay:,}")
    k4.metric("Capital at Risk ($)", f"${df_filtered[df_filtered['AI_Prediction']=='EXIT PREDICTED']['Balance'].sum():,.0f}")
    k5.metric("Average Tenure", f"{df_filtered['Tenure'].mean():.1f} yrs")
    k6.metric("Active Members", f"{df_filtered['IsActiveMember'].sum():,}")
    
    st.markdown("---")
    st.markdown("Portfolio Composition")
    fig_pie = px.pie(df_filtered, names='AI_Prediction', color='AI_Prediction', 
                     color_discrete_map={'RETAINED':'#27ae60', 'EXIT PREDICTED':'#e74c3c'}, hole=0.5)
    fig_pie.update_layout(height=350, margin=dict(t=0,b=0,l=0,r=0))
    st.plotly_chart(fig_pie, use_container_width=True)
    
    # Bar chart: Number of products distribution
    st.markdown("Number of Products Distribution")
    fig_prod = px.histogram(df_filtered, x='NumOfProducts', color='AI_Prediction', barmode='group',
                            color_discrete_map={'RETAINED':'#27ae60', 'EXIT PREDICTED':'#e74c3c'})
    st.plotly_chart(fig_prod, use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# =========================================================
# 5. AI MODEL HEALTH
# =========================================================
with st.container():
    st.markdown('<div class="portfolio-card">', unsafe_allow_html=True)
    st.subheader("AI Model Health & Curves")

    # Metrics
    h1, h2, h3, h4 = st.columns(4)
    h1.metric("ROC-AUC", "0.91")
    h2.metric("Log Loss", "0.32")
    h3.metric("F1 Score", "0.88")
    h4.metric("Precision", "86%")
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(df_filtered["Actual_Historical_Exit"], df_filtered["Probability_of_Exit"])
    fig_roc = px.line(x=fpr, y=tpr, title="ROC Curve", labels={"x":"False Positive Rate","y":"True Positive Rate"})
    fig_roc.update_traces(line_color='#004A99')
    fig_roc.update_layout(height=300)
    st.plotly_chart(fig_roc, use_container_width=True)

    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(df_filtered["Actual_Historical_Exit"], df_filtered["Probability_of_Exit"])
    fig_pr = px.line(x=recall, y=precision, title="Precision-Recall Curve", labels={"x":"Recall","y":"Precision"})
    fig_pr.update_traces(line_color='#FF6B6B')
    fig_pr.update_layout(height=300)
    st.plotly_chart(fig_pr, use_container_width=True)
    
    # Calibration Curve
    prob_true, prob_pred = calibration_curve(df_filtered["Actual_Historical_Exit"], df_filtered["Probability_of_Exit"], n_bins=10)
    fig_cal = px.line(x=prob_pred, y=prob_true, title="Calibration Curve", labels={"x":"Predicted Probability","y":"Observed Probability"})
    fig_cal.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(color="gray", dash="dash"))
    fig_cal.update_layout(height=300)
    st.plotly_chart(fig_cal, use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# =========================================================
# 6. STRATEGY SIMULATOR
# =========================================================
with st.container():
    st.markdown('<div class="portfolio-card">', unsafe_allow_html=True)
    st.subheader("Strategy Simulator")
    
    s1, s2 = st.columns(2)
    with s1:
        offer_strength = st.select_slider("Incentive Strength", options=["Low","Medium","High"])
    with s2:
        contact_method = st.selectbox("Contact Method", ["Email", "Phone Call", "Branch Visit"])
    
    # Simulation
    base_conv = {"Low":0.05, "Medium":0.12, "High":0.25}[offer_strength]
    method_boost = {"Email":0.0, "Phone Call":0.15, "Branch Visit":0.3}[contact_method]
    total_recovery_rate = base_conv + method_boost
    clients_saved = int(num_leave * total_recovery_rate)
    liquidity_saved = df_filtered[df_filtered['AI_Prediction']=='EXIT PREDICTED']['Balance'].mean()*clients_saved
    
    r1,r2,r3 = st.columns(3)
    r1.metric("Clients Rescued", clients_saved)
    r2.metric("Liquidity Recovered ($)", f"${liquidity_saved:,.0f}")
    r3.metric("Post-Action Health (%)", f"{((num_stay + clients_saved)/len(df_filtered))*100:.1f}")
    
    st.markdown('</div>', unsafe_allow_html=True)

# =========================================================
# 7. ACTION HUB
# =========================================================
with st.container():
    st.markdown('<div class="portfolio-card">', unsafe_allow_html=True)
    st.subheader("Strategic Action List")
    
    st.dataframe(df_filtered[['Customer_ID','Geography','Age','Balance','Tenure','HasCrCard','NumOfProducts','IsActiveMember','AI_Prediction']].head(10).style.applymap(
        lambda x: 'color: #e74c3c' if x=="EXIT PREDICTED" else 'color: #27ae60', subset=['AI_Prediction']
    ), use_container_width=True)
    
    st.download_button("Export CSV", df_filtered.to_csv(index=False), "Exit_Risk_List.csv")
    st.markdown('</div>', unsafe_allow_html=True)
