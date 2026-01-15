🏦 **Bank Customer Churn – Strategic Command Center**

Operationalizing AI for Proactive Asset Retention

This Streamlit application transforms predictive churn research into an interactive Strategic Command Center for banking decision-makers.

It enables Portfolio Managers and Executives to move from reactive churn reporting to proactive liquidity protection, using high-fidelity machine learning and transparent model governance.

The dashboard operationalizes analytical outputs into an executive-ready environment for monitoring risk, simulating interventions, and activating retention strategies.

🚀 Key Capabilities
📊 Portfolio Risk Surveillance

Capital at Risk ($): Real-time visibility into the dollar value of deposits exposed to churn, driven by AI-predicted exit probabilities

Dynamic Recalculation: Risk metrics update instantly as filters and probability thresholds are adjusted

🧠 Behavioral Micro-Segmentation

Granular controls allow users to identify high-value risk pockets, not generic churn averages:

Demographics: Geography, Age tiers

Financial Profile: Balance / Liquidity tiers, Credit Score

Engagement Signals: Product count, Tenure, Active Member status, Credit Card ownership

🧪 Model Governance Vault

Designed to maintain executive trust in AI decisions through transparency and auditability:

Core Metrics:

ROC-AUC
=
0.8351
ROC-AUC=0.8351

Precision and Recall

Visual Audit: ROC Curve displaying discrimination power across all risk thresholds

🎯 Strategy Simulator (ROI Engine)

An interactive What-If environment for testing retention strategies before execution:

Predicts:

Clients recovered

Liquidity preserved

Post-action portfolio health (%)

Control Variables:

Incentive strength

Contact method (Automated vs. High-Touch)

📥 Actionable CRM Integration

Prioritized Hit Lists: Dynamic generation of high-risk customer segments

One-Click Export: CSV files formatted for CRM uploads or Relationship Manager outreach

🛠️ Technical Architecture

The dashboard is engineered for modularity, speed, and production realism:

Inference Engine:
Random Forest Classifier

ROC-AUC
=
0.8351
ROC-AUC=0.8351

Artifact Integration:
Serialized model and scaler dynamically loaded from the /models directory

Frontend:
Streamlit (Python-based) for a responsive, data-driven executive UI

Data Layer:
Uses cleaned and feature-engineered datasets produced by the notebook pipeline

💻 Setup & Local Deployment

Run the Strategic Command Center locally from the project root.

1️⃣ Install Dependencies
pip install -r requirements.txt

2️⃣ Launch the Dashboard
streamlit run bank-churn-app/customer_churn_dashboard.py


The application will open automatically in your browser.

🧭 Executive User Guide
🔧 Risk Thresholding

Use the Churn Sensitivity slider to control intervention strictness.
For high-certainty, high-ROI actions, focus on customers where:

𝑃
(
Exit
)
>
0.80
P(Exit)>0.80
🧠 Priority Strategy: Cluster 0

Profile: High Balance / High Salary

Strategic Value:

Largest liquidity exposure

Highest return on retention investment

Recommended Action:
Deepen engagement or cross-sell before exit occurs

🎯 Simulate Recovery

Input your proposed retention effort intensity to instantly view:

Liquidity Recovered ($)

Portfolio Health Recovery (%)

✅ Purpose & Business Value

This application bridges the gap between data science research and real-world banking decisions.

It enables leaders to:

Quantify churn risk in financial terms

Trust AI outputs through built-in transparency

Allocate retention budgets intelligently

Act before value exits the portfolio

📌 Deployment Status

Current: Local / Portfolio-ready

Designed for: Streamlit Cloud or internal enterprise deployment

Artifacts: Models, scalers, and datasets fully versioned and reproducible
