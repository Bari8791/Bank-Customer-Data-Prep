
# 🏦 Bank Customer Churn: Predictive Intelligence & Strategic Research

### *Operationalizing AI for Proactive Asset Retention*

---

## 🔍 Overview

This Streamlit application transforms predictive churn research into a **Strategic Command Center** for banking decision-makers.

It enables **Portfolio Managers and Executives** to shift from **reactive churn reporting** to **proactive liquidity protection**, using explainable machine learning and real-time decision tools.

> **Purpose:** Turn AI predictions into **financial action**, not just dashboards.

---

## 🚀 What This App Delivers

### 📊 **Portfolio Risk Surveillance**

* **Capital at Risk ($):** Live view of deposits exposed to churn
* **Dynamic Recalculation:** KPIs update instantly as filters and thresholds change
* **Executive Focus:** Risk expressed in **money**, not probabilities

---

### 🧠 **Behavioral Micro-Segmentation**

Identify *where* churn risk actually matters:

**Filters include**

* 🌍 Geography
* 👥 Age & Tenure
* 💳 Credit Card Ownership
* 📦 Number of Products
* ⚡ Active vs Inactive Customers
* 💰 Balance / Liquidity Tiers

> Enables precision targeting instead of blanket campaigns.

---

### 🧪 **Model Governance & Trust**

Built to earn **executive confidence in AI decisions**:

* **ROC-AUC:** `0.8351`
* **Precision & Recall:** Displayed live
* **ROC Curve:** Visual audit of model discrimination power

[
\text{Model: Random Forest} \quad (\text{AUC} = 0.8351)
]

---

### 🎯 **Strategy Simulator (ROI Engine)**

A controlled *What-If* environment to test retention actions **before spending money**.

**Simulates**

* Clients recovered
* Liquidity preserved ($)
* Portfolio health after intervention

**Variables**

* Incentive strength
* Contact method (Automated vs High-Touch)

---

### 📥 **Actionable CRM Outputs**

* Prioritized **High-Risk Client Lists**
* One-click **CSV export**
* Ready for CRM or Relationship Manager execution

---

## 🛠️ Technical Architecture

Designed to reflect **real banking analytics systems**, not demos.

| Layer           | Description                                         |
| --------------- | --------------------------------------------------- |
| **Model**       | Random Forest Classifier                            |
| **Artifacts**   | Serialized model & scaler (`/models`)               |
| **Frontend**    | Streamlit (Python)                                  |
| **Data Layer**  | Cleaned & engineered outputs from notebook pipeline |
| **Design Goal** | Fast, explainable, decision-ready                   |

---

## 💻 Local Setup & Run

Run from the **project root**.

### 1️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 2️⃣ Launch the dashboard

```bash
streamlit run bank-churn-app/customer_churn_dashboard.py
```

The app will open automatically in your browser.

---

## 🧭 Executive Usage Guide

### 🔧 Risk Threshold Control

Adjust **Churn Sensitivity** to control intervention strictness.

> Recommended focus:
> [
> P(\text{Exit}) > 0.80
> ]

High certainty → High ROI → Fewer wasted incentives.

---

### 🧠 Priority Segment: *Cluster 0*

* **Profile:** High Balance / High Salary
* **Risk:** Largest liquidity exposure
* **Strategy:** Early engagement + product deepening

This segment delivers the **highest return on retention spend**.

---

### 🎯 Simulate Recovery

Test retention intensity and instantly view:

* Liquidity recovered ($)
* Portfolio health improvement (%)

---

## ✅ Business Value

This application closes the gap between **data science** and **banking decisions**.

It enables leaders to:

* Quantify churn in **financial terms**
* Trust AI through built-in transparency
* Allocate budgets **intelligently**
* Act **before value exits the bank**

---

## 📌 Deployment Status

* **Current:** Local / Portfolio-ready
* **Designed for:** Streamlit Cloud or internal bank deployment
* **Reproducibility:** Models, scalers, and datasets fully versioned

---

