# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Bank Customer Churn Analysis and Modeling Notebook 

# %% [markdown]
#  ### 1. 🧹 Data Foundation & Feature Engineering

# %% [markdown]
# 🏗️ Project Setup & Data Ingestion

# %%
# Import necessary libraries (Ensuring all are available)
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from IPython.display import display, Markdown


# %%
# 1. Load Data (Assuming your file paths are correct)
xlsx_path = "/workspaces/Bank-Customer-Data-Prep/Bank_Churn_Messy.xlsx"
csv_path  = "/workspaces/Bank-Customer-Data-Prep/Bank_Churn.csv"
csv_alt   = "/workspaces/Bank-Customer-Data-Prep/Bank_Churn_Modelling.csv"

if os.path.exists(csv_path):
    df = pd.read_csv(csv_path)
elif os.path.exists(csv_alt):
    df = pd.read_csv(csv_alt)
elif os.path.exists(xlsx_path):
    try:
        cust = pd.read_excel(xlsx_path, sheet_name="Customer_Info")
        acct = pd.read_excel(xlsx_path, sheet_name="Account_Info")
        df = pd.merge(cust, acct, on="CustomerId", how="left")
    except:
        df = pd.read_excel(xlsx_path)
else:
    raise FileNotFoundError("Data file not found.")

# %%
# 2. Feature Engineering (Creates the necessary columns)
if 'NumOfProducts' in df.columns and 'Tenure' in df.columns:
    df['ProductPerYear'] = np.where(df['Tenure'] == 0, df['NumOfProducts'], df['NumOfProducts'] / df['Tenure'])
else:
    df['ProductPerYear'] = 0.0

if 'Balance' in df.columns and 'EstimatedSalary' in df.columns:
    df['balance_to_income'] = df['Balance'] / (df['EstimatedSalary'] + 1)
    df['income_v_product'] = df['EstimatedSalary'] / (df['NumOfProducts'] + 1)
else:
    df['balance_to_income'] = 0.0
    df['income_v_product'] = 0.0

    # Basic cleaning (Necessary for Gender_num)
df.columns = df.columns.str.strip().str.replace(" ", "_")
if 'Gender' in df.columns:
    df['Gender'] = df['Gender'].astype(str).str.strip().str.title()
    df['Gender_num'] = np.where(df['Gender'] == 'Female', 1, 0)
else:
    df['Gender_num'] = 0

# %%
# 3. Define Features and One-Hot Encode Geography
features = [
    'CreditScore','Gender_num','Age','Tenure','Balance','NumOfProducts',
    'HasCrCard','IsActiveMember','EstimatedSalary','ProductPerYear',
    'balance_to_income','income_v_product'
]
features = [c for c in features if c in df.columns]

df_prep = df.copy()
for c in ['France','Germany','Spain']:
    # Standardize Geography labels before one-hot encoding if needed
    if 'Geography' in df_prep.columns:
        df_prep[c] = (df_prep['Geography'].astype(str).str.strip().str.title() == c).astype(int)
    else:
        df_prep[c] = 0

cols_to_scale = features + [c for c in ['France','Germany','Spain'] if c in df_prep.columns]

# %%
# 4. Scale the Data and Create the Final df_scaled DataFrame
scaler_cluster_geo = StandardScaler()

df_scaled_features = pd.DataFrame(
    scaler_cluster_geo.fit_transform(df_prep[cols_to_scale]),
    columns=cols_to_scale,
    index=df_prep.index
)


# %%
# 5. Combine Scaled Features with Target and Categorical Columns
df_scaled = df_scaled_features.copy()
df_scaled['Exited'] = df_prep['Exited']
df_scaled['Geography'] = df_prep['Geography']

display(Markdown("### ✅ All Data Prep Complete. `df_scaled` is Ready for Clustering."))
display(df_scaled.head())

# %% [markdown]
# 🧼 Deep Data Cleaning & Standardization

# %%
# --- Cell 3: 🧹 Duplicates, Missing Values, and Type Cleaning ---

# 1. Drop duplicates
dups = df.duplicated().sum()
display(Markdown(f"**Exact duplicate rows found:** {dups}"))
if dups > 0:
    df.drop_duplicates(inplace=True)
    df.reset_index(drop=True, inplace=True)
    display(Markdown("Duplicates removed."))


# %%
# 2. Function to convert currency-like columns robustly
def to_numeric_currency(df_local, col):
    if col in df_local.columns:
        # remove currency symbols, commas, and coerce to numeric
        df_local[col] = (
            df_local[col]
            .astype(str)
            .str.replace(r'[^\d\.\-]', '', regex=True)
            .replace('', np.nan)
        )
        df_local[col] = pd.to_numeric(df_local[col], errors='coerce')
    return df_local

df = to_numeric_currency(df, 'Balance')
df = to_numeric_currency(df, 'EstimatedSalary')

# %%
# 3. Convert Binary features (Yes/No to 1/0)
if 'HasCrCard' in df.columns:
    df['HasCrCard'] = df['HasCrCard'].replace({'Yes':1,'No':0, 'yes':1, 'no':0}).astype('Int64')
if 'IsActiveMember' in df.columns:
    df['IsActiveMember'] = df['IsActiveMember'].replace({'Yes':1,'No':0, 'yes':1, 'no':0}).astype('Int64')


# %%
# 4. Clean Geography and Gender labels, then encode Gender
if 'Geography' in df.columns:
    df['Geography'] = df['Geography'].astype(str).str.strip().str.title()
    # Unify country labels
    df['Geography'] = df['Geography'].replace({
        'Fra':'France','Fr':'France','French':'France','FraNce':'France',
        'Spain':'Spain','Sp':'Spain','Germany':'Germany','Ger':'Germany','GerM':'Germany'
    })

if 'Gender' in df.columns:
    df['Gender'] = df['Gender'].astype(str).str.strip().str.title()
    df['Gender_num'] = np.where(df['Gender'] == 'Female', 1, 0)
else:
    df['Gender_num'] = 0


# %%
# 5. Handle missing values
for col in df.columns:
    if df[col].isnull().any():
        if df[col].dtype == object or df[col].dtype.name == 'category':
            df[col] = df[col].fillna("MISSING")
        else:
            # numeric: fill NaNs with median
            med = df[col].median()
            df[col] = df[col].fillna(med)

display(Markdown("**Cleaned Data Snapshot & Types**"))
display(df[['Balance','EstimatedSalary','Geography','Gender_num']].head())
display(df.dtypes)
display(Markdown(f"**Final Data Shape:** {df.shape}"))

# %% [markdown]
#  🛠️ Engineering Predictive Ratios

# %%
# --- Cell 4: ⚙️ Feature Engineering ---
# ProductPerYear: avoid division by zero
if 'NumOfProducts' in df.columns and 'Tenure' in df.columns:
    df['ProductPerYear'] = np.where(df['Tenure'] == 0, df['NumOfProducts'], df['NumOfProducts'] / df['Tenure'])
else:
    df['ProductPerYear'] = 0.0



# %%
# Balance to Income Ratio and Income vs Product
if 'Balance' in df.columns and 'EstimatedSalary' in df.columns:
    df['balance_to_income'] = df['Balance'] / (df['EstimatedSalary'] + 1)
    df['income_v_product'] = df['EstimatedSalary'] / (df['NumOfProducts'] + 1)
else:
    df['balance_to_income'] = 0.0
    df['income_v_product'] = 0.0

display(df[['ProductPerYear','balance_to_income','income_v_product']].head())

# %% [markdown]
# ### 2. 🔍 Exploratory Data Analysis (EDA) & Insights

# %%
import matplotlib.pyplot as plt   # <-- Missing import
import numpy as np

# --- Cell 5: 📊 Basic Statistics & Distributions ---
display(df.describe())

# Histograms for numeric columns
numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
df[numeric_cols].hist(figsize=(15,12), bins=20)
plt.suptitle("Feature Distributions (Histograms)", y=1.02)
plt.tight_layout()
plt.show()

display(Markdown("### Target distribution and Categorical Counts"))
display(df['Exited'].value_counts())
display(df['Gender'].value_counts())


# %% [markdown]
# 🔎 Churn Drivers: Univariate Analysis

# %%
import matplotlib.pyplot as plt
import seaborn as sns 
# Assuming 'df' is already defined in a previous cell

# CreditScore by churn
plt.figure(figsize=(8, 5))
sns.histplot(df, x='CreditScore', hue='Exited', bins=30, kde=True, palette={0:'blue',1:'red'}, alpha=0.4)
plt.title("CreditScore Distribution by Churn")
plt.show()

# %%
# Age vs Churn
plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x='Exited', y='Age')
plt.title("Age Distribution by Churn")
plt.show()


# %%
# Balance vs Churn
plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x='Exited', y='Balance')
plt.title("Balance Distribution by Churn")
plt.show()

# %%
# Churn Rate by Geography
plt.figure(figsize=(8, 5))
geo_churn_rate = df.groupby('Geography')['Exited'].mean().sort_values(ascending=False) * 100
sns.barplot(x=geo_churn_rate.index, y=geo_churn_rate.values)
plt.ylabel('Churn Rate (%)')
plt.title('Churn Rate by Geography')
plt.show()

# %% [markdown]
#  🌐 Inter-Feature Relationships (Pairplot)

# %%
sns.pairplot(df)

# %% [markdown]
# ### 3. 🎯 Predictive Modeling & Performance

# %% [markdown]
# 🧱 Feature Matrix, Scaling, & Train/Test Split

# %%
# --- Cell 8: 🧩 Prepare Data & Split ---
from sklearn.model_selection import train_test_split # Ensure import is present

# Drop identifier/non-numeric columns before preparing X, y
drop_cols = [c for c in ['CustomerId','Surname','Gender','Geography'] if c in df.columns]
df_model = df.drop(columns=drop_cols).copy()

# Define numeric features (consolidated list)
features_all = [
    'CreditScore','Gender_num','Age','Tenure','Balance','NumOfProducts',
    'HasCrCard','IsActiveMember','EstimatedSalary','ProductPerYear',
    'balance_to_income','income_v_product'
]
features = [c for c in features_all if c in df_model.columns and c != 'Exited']

X = df_model[features]
y = df_model['Exited']

# Train/Test Split (stratified)
x_train, x_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

display(Markdown(f"**Data Split:** Train ({x_train.shape[0]} samples), Test ({x_test.shape[0]} samples)"))

# %%
# --- Cell 9: ⚖️ Standardize Features ---
from sklearn.preprocessing import StandardScaler # Ensure import is present
import pandas as pd # Ensure pandas is imported

scaler = StandardScaler()
x_train_scaled = pd.DataFrame(scaler.fit_transform(x_train), columns=x_train.columns, index=x_train.index)
x_test_scaled  = pd.DataFrame(scaler.transform(x_test), columns=x_test.columns, index=x_test.index)

display(Markdown("**Features Scaled (x_train_scaled):**"))
display(x_train_scaled.head())

# %% [markdown]
# 🤖 Baseline Model: Logistic Regression

# %%
# --- Cell 10: 🤖 Logistic Regression (LR) ---
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from IPython.display import Markdown, display

logreg = LogisticRegression(max_iter=2000, random_state=42)
logreg.fit(x_train_scaled, y_train)

y_train_pred_lr = logreg.predict(x_train_scaled)
y_test_pred_lr  = logreg.predict(x_test_scaled)
y_test_proba_lr = logreg.predict_proba(x_test_scaled)[:,1]

display(Markdown("**Logistic Regression Results**"))
display(Markdown(f"Train Accuracy: {accuracy_score(y_train, y_train_pred_lr):.4f}"))
display(Markdown(f"Test Accuracy: {accuracy_score(y_test, y_test_pred_lr):.4f}"))
display(Markdown(f"Test ROC AUC: {roc_auc_score(y_test, y_test_proba_lr):.4f}"))

# Confusion Matrices
fig, axes = plt.subplots(1,2,figsize=(12,4))
ConfusionMatrixDisplay(confusion_matrix(y_train, y_train_pred_lr)).plot(ax=axes[0], cmap='viridis')
axes[0].set_title("LR Train Confusion Matrix")
ConfusionMatrixDisplay(confusion_matrix(y_test, y_test_pred_lr)).plot(ax=axes[1], cmap='viridis')
axes[1].set_title("LR Test Confusion Matrix")
plt.tight_layout()
plt.show()


# %% [markdown]
# 🌲 Advanced Model: Random Forest & Feature Importance

# %%
# --- Cell 11: 🌲 Random Forest (RF) Baseline ---
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from IPython.display import Markdown, display
import pandas as pd

rf = RandomForestClassifier(random_state=42, n_jobs=-1)
# NOTE: This assumes x_train_scaled, y_train, x_test_scaled, and y_test are defined.
rf.fit(x_train_scaled, y_train)

y_test_proba_rf = rf.predict_proba(x_test_scaled)[:,1]

display(Markdown("**Random Forest Results (Baseline)**"))
display(Markdown(f"RF Train Accuracy: {rf.score(x_train_scaled, y_train):.4f}"))
display(Markdown(f"RF Test Accuracy: {rf.score(x_test_scaled, y_test):.4f}"))
display(Markdown(f"RF Test ROC AUC: {roc_auc_score(y_test, y_test_proba_rf):.4f}")) # <-- FIX: Removed the extra ')'

# Feature importances
feat_imp = pd.Series(rf.feature_importances_, index=x_train_scaled.columns).sort_values(ascending=False)
display(Markdown("### RF Feature Importances"))
display(feat_imp.head(10).to_frame(name='Importance').style.bar(subset=['Importance'], color='#5fba7d'))

# %% [markdown]
# ⚖️ Model Comparison: ROC AUC

# %%
# --- Cell 12: 📈 ROC Curve Comparison ---
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score # <-- ADDED THIS IMPORT

# NOTE: This assumes y_test, y_test_proba_lr, and y_test_proba_rf are defined from previous cells.

fpr_lr, tpr_lr, _ = roc_curve(y_test, y_test_proba_lr)
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_test_proba_rf)

plt.figure(figsize=(8, 6))
plt.plot(fpr_lr, tpr_lr, label=f"LR AUC {roc_auc_score(y_test, y_test_proba_lr):.3f}")
plt.plot(fpr_rf, tpr_rf, label=f"RF AUC {roc_auc_score(y_test, y_test_proba_rf):.3f}")
plt.plot([0,1],[0,1],'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves: LR vs. RF')
plt.legend()
plt.show()

# %% [markdown]
# ### 4. 🧭 Customer Segmentation & Actionable Intelligence

# %% [markdown]
# 🧭 K-Means Setup & Optimal Cluster Selection

# %%
# --- Cell 13: 🧭 Geo-Clustering Setup and Elbow Method (Now works) ---
from sklearn.cluster import KMeans # Ensure import is present
import matplotlib.pyplot as plt
import seaborn as sns

# Features for Geo clustering: all scaled columns except 'Exited' and 'Geography' string
features_geo = [col for col in df_scaled.columns if col not in ['Exited', 'Geography']]
X_geo_scaled = df_scaled[features_geo].copy()

# Elbow Method
inertia_with_geo = []
K_range = range(2, 11)
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_geo_scaled)
    inertia_with_geo.append(km.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(list(K_range), inertia_with_geo, marker='o')
plt.xlabel("Number of clusters (k)")
plt.ylabel("Inertia")
plt.title("Elbow Method (Scaled Features + Geography)")
plt.show() #

# %%
# --- Cell 14: Fit K-Means and Add Cluster Labels (Geo) ---
from sklearn.cluster import KMeans # Ensure import is present

# Re-define features_geo and X_geo_scaled just in case the kernel was reset
features_geo = [col for col in df_scaled.columns if col not in ['Exited', 'Geography', 'Cluster_Num']] 
X_geo_scaled = df_scaled[features_geo].copy()

# Use the number of clusters (k) determined from the elbow plot (e.g., k=4)
k_geo = 4 
kmeans_geo = KMeans(n_clusters=k_geo, random_state=42, n_init=10)
kmeans_geo.fit(X_geo_scaled)

# Add cluster labels to the main scaled DataFrame
df_scaled['Cluster_Geo'] = kmeans_geo.labels_

display(Markdown(f"**Cluster Counts (k={k_geo} with Geography):**"))
display(df_scaled['Cluster_Geo'].value_counts().sort_index())

display(df_scaled[['Geography', 'France', 'Germany', 'Spain', 'Exited', 'Cluster_Geo']].head())

# %%
# --- Cell 15: 🔢 Numeric-Only Clustering Setup and Elbow Method ---

# Features for Numeric-Only clustering: exclude Geo one-hots, 'Exited', and 'Geography' string
geo_cols = ['France', 'Germany', 'Spain']
features_numeric = [col for col in df_scaled.columns if col not in ['Exited', 'Geography'] + geo_cols]
X_num_scaled = df_scaled[features_numeric].copy()

# Elbow Method
inertia_numeric = []
K_range = range(2, 11)
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_num_scaled)
    inertia_numeric.append(km.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(list(K_range), inertia_numeric, marker='o')
plt.xlabel("Number of clusters (k)")
plt.ylabel("Inertia")
plt.title("Elbow Method (Numeric Features Only)")
plt.show()

# %%
# --- Cell 16: Fit K-Means and Add Cluster Labels (Numeric-Only) ---

# Use the number of clusters (k) determined from the elbow plot (e.g., k=4)
k_numeric = 4
kmeans_numeric = KMeans(n_clusters=k_numeric, random_state=42, n_init=10)
kmeans_numeric.fit(X_num_scaled)

# Add cluster labels to the main scaled DataFrame
df_scaled['Cluster_Num'] = kmeans_numeric.labels_

display(Markdown(f"**Cluster Counts (k={k_numeric} without Geography):**"))
display(df_scaled['Cluster_Num'].value_counts().sort_index())

# %% [markdown]
# 🔍 Cluster Profiling & Interpretation Visuals

# %% [markdown]
# 1. Churn Rate (%) per Cluster (Bar Plot)

# %%
# --- Cell 17a: 📊 Churn Rate (%) per Cluster Plot (Final Fix) ---
import matplotlib.pyplot as plt
from IPython.display import display, Markdown
import seaborn as sns 
# Assuming df_scaled and Cluster_Num exist from prior cells

# 1. Calculate the churn rate per cluster
cluster_churn = df_scaled.groupby('Cluster_Num')['Exited'].mean() * 100

plt.figure(figsize=(8, 5))

# 2. Barplot command (Note: Ends on this line)
sns.barplot(
    x=cluster_churn.index, 
    y=cluster_churn.values, 
    hue=cluster_churn.index,  # Fixes the FutureWarning
    palette='viridis', 
    legend=False
) 

# 3. Title command starts on a NEW line
plt.title('Churn Rate (%) per Cluster') 

# 4. Other plot commands start on NEW lines
plt.xlabel('Cluster') 
plt.ylabel('Churn Rate (%)')
plt.show()

display(Markdown("### Cluster Churn Rate Table"))
display(cluster_churn.sort_index().to_frame(name='Churn Rate (%)').style.format('{:.2f}%'))

# %% [markdown]
#  Cell 17b: Country Composition (Counts) per Cluster (Stacked Bar Plot)

# %%
# --- Cell 17b: 🌍 Country Composition (Counts) per Cluster Plot ---
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display, Markdown

# NOTE: Requires Cell 16 to be run first

# Data for plot
country_counts_by_cluster = df_scaled.groupby('Cluster_Num')['Geography'].value_counts().unstack(fill_value=0)

plt.figure(figsize=(9, 6))
country_counts_by_cluster.plot(kind='bar', stacked=True, cmap='tab10', ax=plt.gca())
plt.title('Country Composition (Counts) per Cluster')
plt.xlabel('Cluster')
plt.ylabel('Customer Count')
plt.xticks(rotation=0)
plt.legend(title='Geography')
plt.show()

# %% [markdown]
# 3. Cell 17c: Cluster Centers (Scaled Features) (Heatmap)

# %%
# --- Cell 17c: 🔥 Cluster Centers (Scaled Features) Heatmap ---
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from IPython.display import display, Markdown

# NOTE: Requires Cell 16 to be run first to define kmeans_numeric and X_num_scaled

# Data for plot
centers_num = pd.DataFrame(kmeans_numeric.cluster_centers_, columns=X_num_scaled.columns)

plt.figure(figsize=(10, 8))
# Transpose centers so features are on the Y-axis for easier reading
sns.heatmap(centers_num.transpose(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title('Cluster Centers (Scaled Features)')
plt.xlabel('Cluster')
plt.ylabel('Scaled Feature')
plt.show()

# %%
# --- FIX: Ensure features_numeric is Clean (Must run before Cell 17d) ---
import pandas as pd # Ensure pandas is imported

# Redefine the full set of non-clustering columns we want to exclude from the feature list
cols_to_exclude = ['Exited', 'Geography', 'Cluster_Geo', 'Cluster_Num', 'France', 'Germany', 'Spain']

# Filter the columns in df_scaled to get the clean list of numeric features
# Note: df_scaled must exist from previous steps!
features_numeric = [col for col in df_scaled.columns if col not in cols_to_exclude]

display(Markdown(f"**Cleaned Features Numeric List:** {features_numeric}"))

# %% [markdown]
# Cell 17d: Unscaled Cluster Summary Table (Final Interpretation)

# %%
# --- Cell 17d: 📝 Unscaled Cluster Summary Table ---
import pandas as pd
from IPython.display import display, Markdown

# NOTE: Requires Cell 16 to be run first to define df_scaled['Cluster_Num'] and features_numeric
# Ensure Cluster_Num is on the original df for the unscaled mean calculation
df['Cluster_Num'] = df_scaled['Cluster_Num']

cluster_summary_unscaled = df.groupby('Cluster_Num')[features_numeric].mean()

display(Markdown("### Cluster Summary (Unscaled Means for Interpretation)"))
display(cluster_summary_unscaled.transpose().style.format('{:.2f}'))

# %% [markdown]
# 💾 Final Outputs & Strategic Recommendations

# %%
# --- Cell 18: 💾 Save Final Dataset ---
import pandas as pd
from IPython.display import display, Markdown

out_path_final = "/workspaces/Bank-Customer-Data-Prep/Bank_Churn_Final_With_NumericClusters.csv"
# The Cluster_Num label is already on 'df' from Cell 16/17d, but we ensure it's there.
df.to_csv(out_path_final, index=False)

display(Markdown(f"Saved final dataset with **Cluster_Num** labels to `{out_path_final}`"))

# %%
# --- Cell 20: 📈 Model diagnostics (ROC AUC) ---
# NOTE: This requires logreg, rf, x_test_scaled, and y_test to be defined from earlier modeling steps.
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt

# Get probabilities for the positive class (1 = Exited)
y_test_proba_lr = logreg.predict_proba(x_test_scaled)[:, 1]
y_test_proba_rf = rf.predict_proba(x_test_scaled)[:, 1]

display(Markdown("### Model Performance Summary"))
display(Markdown(f"**Logistic Regression (LR) Test AUC:** {roc_auc_score(y_test, y_test_proba_lr):.4f}"))
display(Markdown(f"**Random Forest (RF) Test AUC:** {roc_auc_score(y_test, y_test_proba_rf):.4f}"))

# ROC plot
fpr_lr, tpr_lr, _ = roc_curve(y_test, y_test_proba_lr)
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_test_proba_rf)
plt.figure(figsize=(8, 6))
plt.plot(fpr_lr, tpr_lr, label=f"LR AUC {roc_auc_score(y_test, y_test_proba_lr):.3f}")
plt.plot(fpr_rf, tpr_rf, label=f"RF AUC {roc_auc_score(y_test, y_test_proba_rf):.3f}")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves')
plt.legend()
plt.show()

# %%
# --- Cell 21: 💼 Save Outputs & Recommendations to Files ---
import pandas as pd
from IPython.display import display, Markdown

# 1. Save Cluster Summary (using the unscaled summary from Cell 17d)
# NOTE: This requires 'cluster_summary_unscaled' to be defined from Cell 17d.
summary_path = "/workspaces/Bank-Customer-Data-Prep/cluster_summary_unscaled.csv"
cluster_summary_unscaled.to_csv(summary_path, index=True)
display(Markdown(f"**Cluster summary (unscaled means) saved to** `{summary_path}`"))

# 2. Save Recommendations (text file)
rec_path = "/workspaces/Bank-Customer-Data-Prep/recommendations_final.md"
with open(rec_path, "w") as f:
    f.write("# Final Cluster Recommendations\n\n")
    f.write("--- Based on Numeric-Only Clustering (k=4) ---\n\n")
    f.write("* **Cluster 0 (Young Affluent):** Target with long-term wealth management.\n")
    f.write("* **Cluster 1 (High-Value, High-Churn):** Implement VIP retention program.\n")
    f.write("* **Cluster 2 (Older, Stable, Low-Income):** Focus on retirement planning and specialized insurance.\n")
    f.write("* **Cluster 3 (Low-Value, High-Churn):** Offer basic loyalty rewards and financial education.\n")
display(Markdown(f"**Recommendations saved to** `{rec_path}`"))

# %%
import matplotlib.pyplot as plt
import numpy as np

# --- DATA PREP ---
geo_churn = df.groupby('Geography')['Exited'].mean() * 100
geo_balance = df.groupby('Geography')['Balance'].mean() / 1000  # in K
cluster_churn = df_scaled.groupby('Cluster_Num')['Exited'].mean() * 100
cluster_products = df_scaled.groupby('Cluster_Num')['NumOfProducts'].mean()

# --- FIGURE SETUP ---
fig, axes = plt.subplots(2, 2, figsize=(8,8))
plt.subplots_adjust(wspace=0.3, hspace=0.3)

# --- 1. Donut: Churn by Geography ---
ax = axes[0,0]
wedges, texts, autotexts = ax.pie(
    geo_churn,
    labels=geo_churn.index,
    autopct='%1.0f%%',
    startangle=90,
    wedgeprops={'width':0.4, 'edgecolor':'w'},
    textprops={'fontsize':9}
)
ax.set_title("Churn % by Country", fontsize=10)

# --- 2. Donut: Avg Balance by Geography ---
ax = axes[0,1]
wedges, texts, autotexts = ax.pie(
    geo_balance,
    labels=geo_balance.index,
    autopct='%1.0fK',
    startangle=90,
    wedgeprops={'width':0.4, 'edgecolor':'w'},
    textprops={'fontsize':9}
)
ax.set_title("Avg Balance (K) by Country", fontsize=10)

# --- 3. Radial: Churn by Cluster ---
ax = axes[1,0]
N = len(cluster_churn)
angles = np.linspace(0, 2*np.pi, N, endpoint=False)
bars = ax.bar(
    angles, cluster_churn.values, width=0.5,
    color=plt.cm.magma(cluster_churn.values/cluster_churn.max())
)
ax.set_xticks(angles)
ax.set_xticklabels(cluster_churn.index)
ax.set_yticks([])
ax.set_title("Churn % by Cluster", fontsize=10)
for i, val in enumerate(cluster_churn.values):
    ax.text(angles[i], val+1, f"{val:.0f}%", ha='center', fontsize=8)

# --- 4. Radial: Avg Products per Cluster ---
ax = axes[1,1]
bars = ax.bar(
    angles, cluster_products.values, width=0.5,
    color=plt.cm.viridis(cluster_products.values/cluster_products.max())
)
ax.set_xticks(angles)
ax.set_xticklabels(cluster_products.index)
ax.set_yticks([])
ax.set_title("Avg Products per Cluster", fontsize=10)
for i, val in enumerate(cluster_products.values):
    ax.text(angles[i], val+0.05, f"{val:.1f}", ha='center', fontsize=8)

plt.show()


# %%
# FIX 1: Uncomment and define product_churn_rate
product_churn_rate = df.groupby('NumOfProducts')['Exited'].mean().reset_index()
product_churn_rate['Churn Rate (%)'] = product_churn_rate['Exited'] * 100

# FIX 2: Uncomment and define corr_to_exited
# Select only numeric columns including the target variable
numeric_cols = df.select_dtypes(include=np.number).columns.tolist() 
# Exclude columns that are ID-like or derived (as seen in your prior context)
cols_to_exclude_from_corr = ['CustomerId', 'RowNumber', 'Gender_num', 'Cluster_Num'] 
final_corr_cols = [col for col in numeric_cols if col not in cols_to_exclude_from_corr]

# Calculate correlation with the 'Exited' target
corr_to_exited = df[final_corr_cols].corr()['Exited'].drop('Exited').sort_values(ascending=False)

# --- 2-Row Layout: Top 2 charts + Bottom bubble ---
fig, axes = plt.subplots(2, 2, figsize=(14,10))
plt.subplots_adjust(hspace=0.4, wspace=0.3)

# --- Top-left: Age Distribution vs Churn ---
sns.kdeplot(
    data=df, x='Age', hue='Exited', fill=True,
    palette={0: 'darkblue', 1: 'darkred'}, alpha=0.6,
    ax=axes[0,0]
)
axes[0,0].set_title('Age Distribution by Churn')
axes[0,0].set_xlabel('Age')
axes[0,0].set_ylabel('Density')
axes[0,0].legend(title='Exited', labels=['Stayed (0)', 'Churned (1)'])

# --- Top-right: Product Count vs Churn (Donut) ---
ax = axes[0,1]
wedges, texts, autotexts = ax.pie(
    product_churn_rate['Churn Rate (%)'],
    labels=product_churn_rate['NumOfProducts'],
    autopct='%1.0f%%',
    startangle=90,
    wedgeprops={'width':0.4, 'edgecolor':'w'},
    textprops={'fontsize':9}
)
ax.set_title('Churn Rate by # Products')

# --- Bottom row: Feature Correlation as Bubble Plot ---
ax = axes[1,0]
axes[1,1].axis('off') 

bubble_sizes = np.abs(corr_to_exited.values) * 500 
scatter = ax.scatter(
    x=corr_to_exited.values,
    y=corr_to_exited.index,
    s=bubble_sizes,
    c=corr_to_exited.values,
    cmap='bwr',
    alpha=0.6,
    edgecolors='w'
)
ax.set_title('Feature Correlation with Churn')
ax.set_xlabel('Pearson Correlation')
ax.set_ylabel('Feature')
plt.colorbar(scatter, ax=ax, label='Correlation')

plt.show()

# %%
# --- Prepare aggregated data for bar plots ---
df_gender = df.groupby('Gender', as_index=False)['Exited'].mean()
df_gender['Churn Rate (%)'] = df_gender['Exited'] * 100

df_card = df.groupby('HasCrCard', as_index=False)['Exited'].mean()
df_card['Churn Rate (%)'] = df_card['Exited'] * 100

# --- 2x2 Layout (only 3 charts) ---
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
plt.subplots_adjust(wspace=0.3, hspace=0.35)

# --- 1. Gender vs Churn (Bar Plot) ---
sns.barplot(
    x='Gender', y='Churn Rate (%)', hue='Gender', data=df_gender,
    palette=['darkred', 'darkblue'], dodge=False, legend=False, ax=axes[0,0]
)
axes[0,0].set_title('A: Churn Rate (%) by Gender', fontsize=11)
axes[0,0].set_xlabel('Gender', fontsize=9)
axes[0,0].set_ylabel('Churn Rate (%)', fontsize=9)
axes[0,0].tick_params(axis='x', rotation=0)

# --- 2. Age vs Balance (Scatter Plot) ---
sns.scatterplot(
    x='Age', y='Balance', hue='Exited', data=df, 
    palette=['gray', 'red'], alpha=0.6, ax=axes[0,1]
)
axes[0,1].set_title('B: Age vs. Balance, Segmented by Churn', fontsize=11)
axes[0,1].set_xlabel('Age', fontsize=9)
axes[0,1].set_ylabel('Balance ($)', fontsize=9)
axes[0,1].legend(title='Exited', labels=['Stayed (0)', 'Churned (1)'], loc='upper left', fontsize=8)

# --- 3. Credit Card vs Churn (Bar Plot) ---
sns.barplot(
    x='HasCrCard', y='Churn Rate (%)', hue='HasCrCard', data=df_card,
    palette=['orange', 'purple'], dodge=False, legend=False, ax=axes[1,0]
)
axes[1,0].set_title('C: Churn Rate (%) by Credit Card Status', fontsize=11)
axes[1,0].set_xlabel('Has Credit Card (0=No, 1=Yes)', fontsize=9)
axes[1,0].set_ylabel('Churn Rate (%)', fontsize=9)
axes[1,0].tick_params(axis='x', rotation=0)

# --- Remove the last empty subplot for symmetry ---
axes[1,1].axis('off')

plt.show()


# %% [markdown]
#  ✅ Final, Corrected Recommendations
#
#  📌 Cluster Recommendations (Based on Numeric-Only Clustering)
#
# These recommendations are derived by combining the Unscaled Means (profile) and the Churn Rate (priority).
#
# * Cluster 0 (Primary Target: High-Value Churners):
#     * Profile: High Balance (\$107K) and High Estimated Salary (\$153K), but low product holding (1.20).
#     * Churn Rate: 26.17% (Highest)
#     * Recommendation: Implement a VIP Retention Program with dedicated Relationship Managers to secure these valuable assets. Priority: cross-sell a second product to reduce risk from 26% to $\approx 7\%$.
#
# * Cluster 2 (Secondary Target: Financially Stressed Churners):
#     * Profile: Highest Balance (\$114K) but the lowest Estimated Salary (\$49K), leading to a high debt/asset-to-income ratio.
#     * Churn Rate: 23.41% (High)
#     * Recommendation: Offer Specialized Financial Counseling or Debt Consolidation Products. Stabilize their finances to prevent attrition driven by economic stress.
#
# * Cluster 1 (Moderate Risk: New & Volatile Customers):
#     * Profile: Lowest average Tenure (1.12 years) and high product adoption rate (`ProductPerYear` is high).
#     * Churn Rate: 17.73% (Moderate)
#     * Recommendation: Focus on early engagement and loyalty. Deploy a High-Touch Welcome/Onboarding Program to ensure they pass the critical first-year churn window.
#
# * Cluster 3 (Lowest Risk: Stable, Low-Value Customers):
#     * Profile: Highest Tenure (6.36 years) and highest number of products (1.99), but lowest Credit Score and lowest Balance/Salary.
#     * Churn Rate: 12.35% (Lowest)
#     * Recommendation: Maintain satisfaction with Low-Cost Loyalty Rewards (e.g., basic fee waivers or extended warranties). They are the stable, low-touch base of the bank.

# %% [markdown]
#



