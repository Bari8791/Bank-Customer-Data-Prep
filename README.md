# 🏦 Bank Customer Churn: Predictive Intelligence & Strategic Research

**End-to-End Machine Learning Pipeline for Financial Risk Management**

This project delivers a high-fidelity data science pipeline to solve the problem of customer attrition in retail banking. It moves from raw, "messy" data to a production-ready model that identifies high-value risk segments.

> **Source:** Original analysis from Maven Analytics (Kaggle Dataset)
> **Engineering Focus:** Feature creation, stratified validation, and numeric clustering.
> **Scope:** 10,000 Total Samples (, ).

---

## 🏗️ 1. Technical Data Architecture

The foundation of this project is a robust cleaning and feature engineering suite designed for banking data.

### 🧹 Data Sanitization

* **Currency Normalization:** Cleaned `Balance` and `Salary` strings into float64 types for mathematical processing.
* **Integrity Checks:** Automated duplicate removal and median imputation for missing financial fields to prevent data leakage.
* **Encoding:** Transformed categorical geography and gender into a machine-readable format while preserving market distinctions.

### 🛠️ Feature Engineering (The "Alpha" Features)

I engineered three specific financial ratios to capture customer behavior more effectively than raw data alone:

1. **Product Density ():** `NumOfProducts / Tenure` — Measures the depth of the relationship over time.
2. **Liquidity Leverage:** `Balance / (EstimatedSalary + 1)` — Indicates how much of the customer's total wealth is "parked" at the bank.
3. **Revenue Efficiency:** `EstimatedSalary / (NumOfProducts + 1)` — Estimates potential cross-sell value per product held.

---

## 🎯 2. Modeling Performance & Audit

I utilized a **Random Forest Classifier** to capture non-linear relationships in customer behavior, specifically outperforming the Logistic Regression baseline.

| Metric | Logistic Regression | **Random Forest (Final)** |
| --- | --- | --- |
| **Accuracy** |  | **** |
| **ROC-AUC** |  | **** |

### 🔍 Top Churn Drivers

The model identifies the following as the most critical predictors of exit:

1. **Age ( Importance):** Risk peaks significantly in the **40–60** demographic.
2. **Product Count ( Importance):** Churn rate drops from  (1 product) to  (2 products).
3. **Income vs Product ( Importance):** Validates that product under-utilization leads to higher exit rates.

---

## 🧭 3. Strategic Segmentation (K-Means)

Using **K-Means Clustering ()**, I categorized the portfolio into actionable segments to move from "General Predictions" to "Targeted Interventions."

### 📊 Risk Hierarchy & Recommendations

| Cluster | Profile | Strategy | Risk Level |
| --- | --- | --- | --- |
| **Cluster 0** | **High Balance / High Salary** | **VIP Retention:** Focus on locking in assets with a 2nd product. | **CRITICAL** |
| **Cluster 2** | High Balance / Low Salary | **Debt Stabilization:** Offer counseling or consolidation. | **HIGH** |
| **Cluster 1** | Low Tenure / High Product | **Onboarding Support:** Prevent early-exit in Year 1. | **MODERATE** |
| **Cluster 3** | High Tenure / Low Balance | **Loyalty Maintenance:** Low-cost rewards; stable base. | **STABLE** |

---

## 💾 4. Project Outputs

The pipeline generates the following assets for the business:

* `Bank_Churn_Final_With_NumericClusters.csv`: The master dataset for the Streamlit dashboard.
* `cluster_summary_unscaled.csv`: A non-technical summary for Portfolio Managers.
* `recommendations_final.md`: The strategic playbook for the marketing team.

---

### ✅ Conclusion

By combining **predictive power ()** with **strategic segmentation**, this project allows a bank to prioritize its retention budget toward the customers who represent the highest financial value and the highest probability of exit.
