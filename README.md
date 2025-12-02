# 🏦 Bank Customer Churn Analysis & Predictive Modeling

This project delivers a **full end-to-end data science pipeline** for **bank customer churn analysis**, combining data preparation, exploratory analysis, predictive modeling, and customer segmentation. The outcome: actionable insights and strategies to retain high-value clients.

> **Source:** Original analysis from [Maven Analytics](https://www.mavenanalytics.io)
> **Data:** [Kaggle Bank Customer Churn Dataset](https://www.kaggle.com/datasets)
> **Organization & Editing:** Adapted, organized, and restructured by me for clarity, predictive focus, and actionable insights.

---

## 🎯 Project Goals

1. **Predict churn**: Build a robust machine learning model to forecast customer exit.
2. **Diagnose risk factors**: Identify key drivers of churn (Age, Balance, Product Count, etc.).
3. **Segment customers**: Define actionable clusters (**k=4**) for targeted retention strategies.

---

## 1. 🧱 Data Foundation & Feature Engineering

We start by building a clean, robust dataset with engineered features to maximize predictive signal.

| Notebook Section                  | Purpose                                                                                                                     |
| :-------------------------------- | :-------------------------------------------------------------------------------------------------------------------------- |
| **Project Setup & Deep Cleaning** | Load raw data, remove duplicates, convert currency, unify categorical labels (Gender/Geography), and handle missing values. |
| **Engineering Predictive Ratios** | Generate features like `ProductPerYear`, `balance_to_income`, `income_v_product` to strengthen predictive power.            |
| **Feature Matrix & Scaling**      | Scale numeric predictors (`StandardScaler`) and prepare **stratified train/test splits**.                                   |

✅ **Outcome:** `df_scaled` is validated and ready for modeling and clustering.

---

## 2. 🔍 Exploratory Data Analysis (EDA)

Visual exploration confirmed key churn drivers and inter-feature relationships.

* **Positive churn drivers:** Age (risk peaks 40–60) and Balance.
* **Retention factors:** Number of Products and Active Membership.
* **Geographic insights:** Germany exhibits the highest churn (~20%) with the highest average balance (~$120K).

Visuals include **density plots, correlation matrices, and pairplots** for multivariate analysis.

---

## 3. 🎯 Predictive Modeling

The **Random Forest Classifier** was selected for final deployment due to superior performance and clear feature importance insights.

| Model                    | Test Accuracy | Test ROC AUC | Decision                    |
| :----------------------- | :------------ | :----------- | :-------------------------- |
| Logistic Regression (LR) | 0.79          | 0.83         | Baseline benchmark          |
| Random Forest (RF)       | **0.84**      | **0.88**     | **Selected for deployment** |

**ROC curve visualization** confirms Random Forest’s strong ability to discriminate churners vs. non-churners.

---

## 4. 🧭 Customer Segmentation & Actionable Insights

**K-Means Clustering** (`k=4`, numeric-only features) uncovered distinct customer personas:

| Cluster | Profile                                    | Churn Rate         | Recommendation                                        |
| :------ | :----------------------------------------- | :----------------- | :---------------------------------------------------- |
| **1**   | High Balance & Credit Score (**VIP Risk**) | **Highest (~33%)** | **VIP retention program**, personalized engagement    |
| **3**   | Low Balance & Credit Score                 | High (~21%)        | Low-touch loyalty & financial education               |
| **0**   | Young, Affluent (Low Age, High Salary)     | Moderate           | Cross-sell investments & long-term wealth products    |
| **2**   | Older, Stable, Low-Income                  | Low                | Cost-effective retention, retirement/insurance offers |

---

## 5. 📈 Actionable Strategy

1. **🚨 VIP Retention:** Launch **Cluster 1-focused program**, initially in Germany.
2. **🤝 Cross-Selling:** Encourage customers to increase product holdings: 1 product (~24% churn) → 2 products (~7% churn).
3. **👵 Age-targeted retention:** Focus on 40–60 age group with retirement planning and wealth advisory programs.

---

## 6. 💾 Outputs & Visual Highlights

| File                                        | Description                                                    |
| :------------------------------------------ | :------------------------------------------------------------- |
| `Bank_Churn_Final_With_NumericClusters.csv` | Final dataset with engineered features and `Cluster_Num`.      |
| `cluster_summary_unscaled.csv`              | Cluster averages for unscaled features for stakeholder review. |
| `recommendations_final.md`                  | Structured business recommendations document.                  |

### 🖼️ Key Metrics at a Glance

* **Top-left:** Churn % by country 🇩🇪
* **Top-right:** Average balance by country 💰
* **Bottom-left:** Churn % by cluster 📊
* **Bottom-right:** Avg products per cluster 🛍️

This synthesis provides a **concise visual summary** for quick stakeholder understanding.

---

✅ **Conclusion:** This project empowers banking teams to **predict churn, identify risk factors, segment customers effectively, and implement actionable retention strategies** for high-value clients.



