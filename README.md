# 🏦 Bank Customer Churn Analysis & Predictive Modeling

This project delivers a **full end-to-end data science pipeline** for bank customer churn analysis, combining data preparation, exploratory analysis, predictive modeling, and customer segmentation. The outcome: actionable insights and strategies to retain high-value clients.

> **Source:** Original analysis from Maven Analytics
> **Data:** Kaggle Bank Customer Churn Dataset
> **Organization & Editing:** Adapted, organized, and restructured by me for clarity, predictive focus, and actionable insights.
> **Data Split:** Train (8000 samples), Test (2000 samples) [**Cell 8**].

---

## 🎯 Project Goals

The project was executed to achieve three key business objectives:

1.  **Predict Churn:** Build a robust machine learning model (Random Forest) to accurately forecast which customers are likely to exit the bank.
2.  **Diagnose Risk Factors:** Identify and quantify the key drivers of churn (e.g., **Age**, **Balance**, Product Count) to understand the *why*.
3.  **Segment Customers:** Define distinct, actionable customer clusters (**k=4**) to enable targeted and cost-effective retention strategies.

---

## 1. 🧱 Data Foundation & Feature Engineering

This section details the initial setup, cleaning, feature creation, and scaling—the building blocks for all subsequent analysis and modeling.

### 🏗️ Project Setup & Deep Cleaning (Cell 1, 3)

| Notebook Step | Description | Cell(s) |
| :--- | :--- | :--- |
| **Data Ingestion** | Robustly loads data from multiple paths. | **Cell 1** |
| **Duplicate Handling** | Exact duplicate rows are checked and removed. | **Cell 3** |
| **Currency Conversion** | Handles currency symbols in `Balance` and `EstimatedSalary`. | **Cell 3** |
| **Binary Encoding** | Converts `HasCrCard` and `IsActiveMember` to (1/0). | **Cell 3** |
| **Label Unification** | Cleans and standardizes **Geography** and **Gender** labels. | **Cell 3** |
| **Missing Values** | Numeric NaNs filled with the **median**. | **Cell 3** |

### 🛠️ Engineering Predictive Ratios (Cell 4)

Three high-impact, business-specific features were created:

1.  **`ProductPerYear`**: Measures customer engagement (Products / Tenure).
2.  **`balance_to_income`**: Assesses financial stability/leverage.
3.  **`income_v_product`**: Assesses spending power relative to product usage.

### 🧱 Feature Matrix, Scaling, & Split (Cell 8, 9)

* **Scaling**: All numeric features (including engineered ratios) were processed using **`StandardScaler`** to ensure equal weighting.
* **Train/Test Split**: Data was split (80/20) using **stratification** to preserve the target distribution.

---

## 2. 🔍 Exploratory Data Analysis (EDA) & Key Drivers

Visual exploration confirmed the strongest risk factors driving customer attrition.

* **Top Positive Driver (Risk):** **Age** (risk peaks in the **40–60** bracket) .
* **Top Retention Factor:** **Number of Products** (Churn rate drops from **$\approx 24\%$ at 1 product** to **$\approx 7\%$ at 2 products**).
* **Geographic Risk:** **Germany** exhibits the highest churn ($\approx 20\%$) combined with the highest average balance, identifying a crucial high-asset risk area.

---

## 3. 🎯 Predictive Modeling & Performance

The **Random Forest Classifier** was selected for final deployment due to its superior performance over the Logistic Regression baseline.

| Model | Test Accuracy | Test ROC AUC | Decision | Cell(s) |
| :--- | :--- | :--- | :--- | :--- |
| Logistic Regression (LR) | 0.8125 | 0.7814 | Baseline Benchmark | **Cell 10** |
| Random Forest (RF) | **0.8525** | **0.8351** | **Selected for Deployment** | **Cell 11, 20** |

### Key Feature Importances (RF Model)

The feature importance analysis validates the EDA, confirming the top drivers:

| Rank | Feature | Importance |
| :--- | :--- | :--- |
| **1** | **Age** | **0.217** |
| 2 | NumOfProducts | 0.112 |
| 3 | income\_v\_product | 0.110 |

---

## 4. 🧭 Customer Segmentation & Actionable Insights

K-Means Clustering (`k=4`, numeric features only) created four distinct segments, establishing a clear risk hierarchy for targeted action.

### Cluster Churn Rate Hierarchy (Cell 17a)

| Rank | Cluster\_Num | Churn Rate | Risk Level |
| :--- | :--- | :--- | :--- |
| **1** | **0** | **26.17%** | **Highest Priority** |
| 2 | **2** | 23.41% | High Priority |
| 3 | **1** | 17.73% | Moderate Risk |
| 4 | **3** | 12.35% | Lowest Risk |

### Detailed Cluster Recommendations (Cell 17d)

| Cluster | Key Profile Traits (Unscaled Means) | Recommendation (Priority) |
| :--- | :--- | :--- |
| **0** | High Balance (\$107K) & High Salary (\$153K), but low product holding (1.20). | **Primary Target:** Implement **VIP Retention Program** immediately. Focus on cross-selling a second product to lock in assets. |
| **2** | Highest Balance (\$114K) but **Lowest Salary** (\$49K), indicating severe financial pressure. | **Secondary Target:** Offer **Specialized Financial Counseling** or debt consolidation products to stabilize highly unbalanced, high-value clients. |
| **1** | Lowest average Tenure (1.12 years) and high `ProductPerYear`. | **Moderate Risk:** Deploy a **High-Touch Welcome Program** to ensure they pass the critical first-year churn window. |
| **3** | Highest Tenure (6.36 years) and highest products (1.99), but lowest Credit Score/Balance. | **Lowest Risk:** Maintain satisfaction with **Low-Cost Loyalty Rewards**. Stable base, minimal intervention needed. |

---

## 5. 📈 Actionable Strategy

1.  **🚨 VIP Retention Priority:** Launch the program targeting **Cluster 0** (26.17% churn). Pilot the initiative in the **German market** to protect the highest concentration of volatile high-asset clients.
2.  **🤝 Cross-Selling Lock-in:** Encourage customers in high-risk groups (0 & 2) to increase product holdings, exploiting the steep drop in churn observed when moving from one to two products.
3.  **👵 Age-targeted Retention:** Prioritize the **40–60 age group** with retirement planning and specialized wealth advisory programs.

---

## 6. 💾 Outputs & Conclusion

| File | Description | Cell(s) |
| :--- | :--- | :--- |
| `Bank_Churn_Final_With_NumericClusters.csv` | Final dataset with engineered features and the **Cluster\_Num** label. | **Cell 18** |
| `cluster_summary_unscaled.csv` | Cluster averages for stakeholder review. | **Cell 21** |
| `recommendations\_final.md` | Structured business recommendations document. | **Cell 21** |

### ✅ Conclusion

The project successfully delivered a robust churn analysis pipeline:

1.  **Predictive Power:** The Random Forest model achieved a high performance ($\text{AUC}$ of $0.8351$), providing a reliable tool to identify clients at risk.
2.  **Key Risk Factors:** **Age** and **low product count** were confirmed as the primary drivers of attrition.
3.  **Targeted Strategy:** Customers were segmented into four distinct clusters, enabling the bank to implement **cost-effective and targeted retention strategies** for high-value clients.

The analysis empowers banking teams to convert predictive insights into actionable business outcomes.
