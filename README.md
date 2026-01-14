# 🏦 Bank Customer Churn: Predictive Intelligence & Strategic Research

**End-to-End Machine Learning Pipeline for Financial Risk Management**

This project delivers a high-fidelity data science pipeline to solve the problem of customer attrition in retail banking. It moves from raw, "messy" data to a production-ready model that identifies high-value risk segments.

> **Source:** Original analysis from Maven Analytics (Kaggle Dataset)
> **Engineering Focus:** Feature creation, stratified validation, and numeric clustering.
> **Scope:** 10,000 Total Samples

---

## 🏗️ 1. Technical Data Architecture

The foundation of this project is a robust cleaning and feature engineering suite designed for banking data.

### 🧹 Data Sanitization

* **Currency Normalization:** Cleaned `Balance` and `EstimatedSalary` into numeric types for proper calculations.
* **Integrity Checks:** Removed duplicates and imputed missing values with medians to prevent data leakage.
* **Encoding:** Converted categorical fields (`Geography`, `Gender`) into machine-readable formats while retaining business meaning.

### 🛠️ Feature Engineering (The "Alpha" Features)

Three advanced financial ratios were created to capture customer behavior more effectively:

1. **Product Density ():** `NumOfProducts / Tenure` — Measures the depth of customer engagement over time.
2. **Liquidity Leverage:** `Balance / (EstimatedSalary + 1)` — Indicates the proportion of wealth held at the bank.
3. **Revenue Efficiency:** `EstimatedSalary / (NumOfProducts + 1)` — Estimates cross-sell potential per product.

---

## 🎯 2. Modeling Performance & Audit

A **Random Forest Classifier** was used to capture non-linear relationships, outperforming the Logistic Regression baseline.

| Metric       | Logistic Regression | **Random Forest (Final)** |
| ------------ | ------------------- | ------------------------- |
| **Accuracy** |                     | ****                      |
| **ROC-AUC**  |                     | ****                      |

### 🔍 Top Churn Drivers

The most influential predictors of customer exit are:

1. **Age:** Peak churn occurs in the **40–60** age range.
2. **Product Count:** Customers with fewer products are more likely to leave.
3. **Income vs Product:** Highlights under-utilized products as a churn driver.

---

## 🧭 3. Strategic Segmentation (K-Means)

Using **K-Means Clustering**, customers were grouped into actionable segments for targeted interventions.

### 📊 Risk Hierarchy & Recommendations

| Cluster       | Profile                        | Strategy                                                             | Risk Level   |
| ------------- | ------------------------------ | -------------------------------------------------------------------- | ------------ |
| **Cluster 0** | **High Balance / High Salary** | **VIP Retention:** Encourage a second product to secure assets.      | **CRITICAL** |
| **Cluster 2** | High Balance / Low Salary      | **Debt Stabilization:** Offer counseling or financial consolidation. | **HIGH**     |
| **Cluster 1** | Low Tenure / High Product      | **Onboarding Support:** Retain early customers in Year 1.            | **MODERATE** |
| **Cluster 3** | High Tenure / Low Balance      | **Loyalty Maintenance:** Low-cost rewards for stable base.           | **STABLE**   |

---

## 💾 4. Project Outputs

The notebook produces:

* `Bank_Churn_Final_With_NumericClusters.csv` — Master dataset for analysis and dashboards.
* `cluster_summary_unscaled.csv` — Non-technical segment summary for managers.
* `recommendations_final.md` — Strategic recommendations for retention campaigns.

---

### ✅ Conclusion

By combining **predictive modeling** with **strategic segmentation**, this project enables the bank to **focus retention efforts on high-value, high-risk customers**, maximizing ROI and reducing customer attrition effectively.


