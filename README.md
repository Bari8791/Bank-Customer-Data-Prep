# 🏦 Bank Customer Churn Analysis & Predictive Modeling

This project delivers a **full end-to-end data science pipeline** for **bank customer churn analysis**, combining data preparation, exploratory analysis, predictive modeling, and customer segmentation. The outcome: actionable insights and strategies to retain high-value clients.
this project delivers a full end-to-end data science pipeline for bank customer churn analysis, combining data preparation, exploratory analysis, predictive modeling, and customer segmentation. The outcome: actionable insights and strategies to retain high-value clients.

Source: Original analysis from Maven Analytics

Data: Kaggle Bank Customer Churn Dataset

Organization & Editing: Adapted, organized, and restructured by me for clarity, predictive focus, and actionable insights.

> **Data Split:** Train (8000 samples), Test (2000 samples) \[**Cell 8**].

---

## 🎯 Project Goals

1.  **Predict churn**: Build a robust machine learning model to forecast customer exit.
2.  **Diagnose risk factors**: Identify key drivers of churn (**Age**, **Balance**, Product Count, etc.).
3.  **Segment customers**: Define actionable clusters (**k=4**) for targeted retention strategies.

---

## 1. 🧱 Data Foundation & Feature Engineering

| Notebook Section | Purpose | Cell(s) |
| :--- | :--- | :--- |
| **Project Setup & Deep Cleaning** | Load raw data, remove duplicates, convert currency, unify categorical labels, and handle missing values. | **Cell 1, 3** |
| **Engineering Predictive Ratios** | Generated features like `ProductPerYear`, `balance_to_income`, `income_v_product` to strengthen predictive power. | **Cell 4** |
| **Feature Matrix & Scaling** | Scaled numeric predictors (`StandardScaler`) and prepared **stratified train/test splits**. | **Cell 8, 9** |

✅ **Outcome:** `df_scaled` is validated and ready for modeling and clustering.

---

## 2. 🔍 Exploratory Data Analysis (EDA)

Visual exploration confirmed key churn drivers and inter-feature relationships. 

* **Top Positive Driver (Risk):** **Age** (risk peaks **40–60** bracket) \[**Cell 7**].
* **Top Retention Factor:** **Number of Products** (Churn rate drops from $\approx 24\%$ at 1 product to $\approx 7\%$ at 2 products) \[**Cell 7**].
* **Geographic insights:** **Germany** exhibits the highest churn ($\approx 20\%$) with the highest average balance ($\approx \$120K$), identifying a major high-asset risk area \[**Cell 7**].

---

## 3. 🎯 Predictive Modeling

The **Random Forest Classifier** was selected for final deployment due to superior performance and clear feature importance insights.

| Model | Test Accuracy | Test ROC AUC | Decision | Cell |
| :--- | :--- | :--- | :--- | :--- |
| Logistic Regression (LR) | 0.8125 | 0.7814 | Baseline benchmark | **Cell 10** |
| Random Forest (RF) | **0.8525** | **0.8351** | **Selected for deployment** | **Cell 11, 20** |

#### Key Feature Importances (RF Model)

The top churn drivers are confirmed by the model:

| Rank | Feature | Importance |
| :--- | :--- | :--- |
| **1** | **Age** | **0.217** (The single strongest predictor) |
| 2 | NumOfProducts | 0.112 |
| 3 | income\_v\_product | 0.110 |

---

## 4. 🧭 Customer Segmentation & Actionable Insights

**K-Means Clustering** (`k=4`, numeric features) uncovered distinct customer personas. The cluster numbering shifted during execution, making **Cluster 0** the highest-risk group.

| Cluster | Profile (Based on Unscaled Means) | Churn Rate | Recommendation | Cell(s) |
| :--- | :--- | :--- | :--- | :--- |
| **0** | **Primary Target/Highest Risk** (High Bal: \$107K, High Salary: \$153K, Low Products: 1.20) | **26.17%** | **Launch VIP Retention Program** targeting this high-value/high-risk group. | **Cell 16, 17a, 17d** |
| **2** | **Secondary Target/High Risk** (Highest Bal: \$114K, Lowest Salary: \$49K, Low Products: 1.30) | 23.41% | Offer financial planning/consolidation due to highly unbalanced financials. | **Cell 16, 17a, 17d** |
| **1** | **Moderate Risk/New/Low Tenure** (Low Tenure: 1.12, High ProductPerYear: 1.44) | 17.73% | High-touch welcome campaigns and early loyalty encouragement. | **Cell 16, 17a, 17d** |
| **3** | **Lowest Risk/Stable Base** (Lowest Bal: \$12K, Low Products: 1.99, High Tenure: 6.36) | 12.35% | Stable base. Cost-effective retention, minimal intervention needed. | **Cell 16, 17a, 17d** |

---

## 5. 📈 Actionable Strategy

1.  **🚨 VIP Retention Priority:** The program must target **Cluster 0** (26.17% churn). Pilot the initiative in the **German market** to protect the highest concentration of volatile high-asset clients.
2.  **🤝 Cross-Selling Lock-in:** Encourage customers in high-risk groups (0 & 2) to increase product holdings: **1 product** ($\approx 24\%$ churn) $\rightarrow$ **2 products** ($\approx 7\%$ churn).
3.  **👵 Age-targeted retention:** Prioritize the **40–60 age group** with retirement planning and specialized wealth advisory programs.

---

## 6. 💾 Outputs & Visual Highlights

| File | Description | Cell(s) |
| :--- | :--- | :--- |
| `Bank_Churn_Final_With_NumericClusters.csv` | Final dataset with engineered features and the **Cluster_Num** for direct deployment. | **Cell 18** |
| `cluster_summary_unscaled.csv` | Cluster averages for unscaled features for stakeholder review. | **Cell 21** |
| `recommendations_final.md` | Structured business recommendations document. | **Cell 21** |

### 🖼️ Key Metrics at a Glance (Cell 7, 17a)

* **Visual Interpretation Check:** The **Churn % by Cluster** bar chart \[**Cell 17a**] should visually confirm that **Cluster 0** is the tallest bar, followed closely by **Cluster 2**, validating the updated hierarchy used in this report.
* **Cluster Centers Heatmap** \[**Cell 17c**] provides the visual foundation for profiling, showing how each segment is defined by high/low scaled feature scores.

---

### ✅ Conclusion

This project empowers banking teams to **predict churn, identify risk factors, segment customers effectively, and implement actionable, data-driven retention strategies** for high-value clients, validated by an $\text{AUC}$ of $0.8351$.
