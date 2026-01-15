
# 🏦 Bank Customer Churn Intelligence Dashboard

**End-to-End Machine Learning Pipeline for Financial Risk Management**

This project delivers a high-fidelity data science pipeline to solve the problem of customer attrition in retail banking. It moves from raw, "messy" data to a production-ready model that identifies high-value risk segments.

> **Source:** Original analysis from Maven Analytics (Kaggle Dataset)
> **Engineering Focus:** Feature creation, stratified validation, and numeric clustering.
> **Scope:** 10,000 Total Samples (, ).

---

## 🏗️ 1. Technical Data Architecture

The foundation of this project is a robust cleaning and feature engineering suite designed for banking data.

### 🧹 Data Sanitization

* **Currency Normalization:** Transformed `Balance` and `EstimatedSalary` strings into numeric types for mathematical processing.
* **Integrity Checks:** Automated duplicate removal and median-based imputation for missing values to prevent data leakage.
* **Encoding:** Converted categorical fields (`Geography`, `Gender`) into machine-readable formats while retaining geographic business distinctions.

### 🛠️ Feature Engineering (The "Alpha" Features)

Three advanced financial ratios were engineered to capture customer behavior more effectively:

1. **Product Density ():**  — Measures engagement depth over time.
2. **Liquidity Leverage:**  — Indicates the proportion of personal wealth held at the bank.
3. **Revenue Efficiency:**  — Estimates cross-sell potential relative to product usage.

---

## 🎯 2. Modeling Performance & Audit

A **Random Forest Classifier** was utilized to capture non-linear relationships in customer behavior, significantly outperforming the Logistic Regression baseline.

### 📊 Model Performance Benchmark

| Metric   | Logistic Regression (Baseline) | Random Forest (Final) | Improvement |
| -------- | ------------------------------ | --------------------- | ----------- |
| Accuracy | 81.25%                         | 85.25%                | +4.00%      |
| ROC-AUC  | 0.7814                         | 0.8351                | +6.87%      |

### 🔍 Top Churn Drivers

The model identifies these features as the primary predictors of customer exit:

1. **Age ( Importance):** Risk peaks significantly in the **40–60** demographic.
2. **Product Count ( Importance):** Churn rate drops drastically as product holdings move from 1 to 2.
3. **Income vs Product ( Importance):** Identifies under-utilization of products as a key exit indicator.

---

## 🧭 3. Strategic Segmentation (K-Means)

Using **K-Means Clustering ()**, I categorized the portfolio into actionable segments to move from "General Predictions" to "Targeted Interventions."

### 📊 Risk Hierarchy & Recommendations

| Cluster       | Profile                        | Strategy                                                             | Risk Level   |
| ------------- | ------------------------------ | -------------------------------------------------------------------- | ------------ |
| **Cluster 0** | **High Balance / High Salary** | **VIP Retention:** Secure assets by cross-selling a second product.  | **CRITICAL** |
| **Cluster 2** | High Balance / Low Salary      | **Debt Stabilization:** Offer financial counseling or consolidation. | **HIGH**     |
| **Cluster 1** | Low Tenure / High Product      | **Onboarding Support:** High-touch service to survive Year 1.        | **MODERATE** |
| **Cluster 3** | High Tenure / Low Balance      | **Loyalty Maintenance:** Low-cost rewards for a stable base.         | **STABLE**   |

---

## 💾 4. Project Outputs

The notebook generates the following assets for business integration:

* `Bank_Churn_Final_With_NumericClusters.csv` — Master dataset for analysis and dashboards.
* `cluster_summary_unscaled.csv` — Non-technical segment summary for stakeholders.
* `recommendations_final.md` — Strategic playbook for retention campaigns.

---

## 📦 Bank-Customer-Churn-Project

bank-churn-app
customer_churn_dashboard.py
requirements.txt
README.md

data/raw
Bank_Churn.csv
Bank_Churn_Messy.xlsx
Bank_Churn_Data_Dictionary.csv

data/processed
Bank_Churn_Preprocessed.csv
Bank_Churn_Final_With_NumericClusters.csv
Bank_Churn_Final_Predictions.csv

models
churn_model.pkl
scaler.pkl

notebooks
Bank Customer Churn Analysis.ipynb
Bank Customer Churn Analysis.py

outputs
cluster_summary.csv
recommendations_final.md

requirements.txt
README.md

---

## ✅ Conclusion

This project demonstrates a **full end-to-end banking churn solution**:

* From raw customer data to **engineered features** and predictive models.
* **Random Forest** provides actionable, high-confidence predictions with measurable ROI improvements.
* **Strategic segmentation and dashboards** allow portfolio managers to prioritize high-value customers and reduce attrition risk.
* All outputs are **production-ready** and structured for **real-world deployment**, ensuring clarity, auditability, and immediate business impact.

---

