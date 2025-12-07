#!/usr/bin/env python
# coding: utf-8

# # Logistic Regression Baseline
# 
# ---

# ## Setup

# In[ ]:


import os
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import sqlite3
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    precision_recall_curve,
    auc,
    brier_score_loss,
    classification_report,
    confusion_matrix,
)

import seaborn as sns

# Make sure we can open the db file
try:
    root = subprocess.check_output(
        ["git", "rev-parse", "--show-toplevel"],
        text=True,
    ).strip()
    os.chdir(root)
except Exception:
    while not (Path.cwd() / "data").exists() and Path.cwd() != Path.home():
        os.chdir("..")

print("Working directory:", Path.cwd())
os.makedirs("results", exist_ok=True)


# ## Load labeled route data

# In[74]:


# Connect to routes_scores.db and load labeled routes
conn = sqlite3.connect("data/routes_scores.db")

df = pd.read_sql("SELECT * FROM routes;", conn)
print("Columns:", df.columns.tolist())
print("Number of routes:", len(df))


# ## Preprocessing and Train/Validation/Test Split

# In[75]:


numeric_features = ["impact_score"]
categorical_features = ["counties"]
target = "impacting_delivery"
df[numeric_features] = df[numeric_features].fillna(df[numeric_features].median())
df[categorical_features] = df[categorical_features].fillna("Unknown")

print("Label distribution:")
print(df[target].value_counts(normalize=True))

# 70/15/15 split
train, test = train_test_split(
    df,
    test_size=0.15,
    random_state=42,
    stratify=df[target],
)
train, val = train_test_split(
    train,
    test_size=0.1765,
    random_state=42,
    stratify=train[target],
)

print("Train:", train.shape)
print("Val:", val.shape)
print("Test:", test.shape)

X_train = train[numeric_features + categorical_features]
y_train = train[target]
X_val = val[numeric_features + categorical_features]
y_val = val[target]
X_test = test[numeric_features + categorical_features]
y_test = test[target]


# ## Model Definition and Training

# In[76]:


# ---------------------------------------------------------------
# Preprocessing: Standardization + One-Hot Encoding
# ---------------------------------------------------------------
numeric_transformer = Pipeline(steps=[
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

# Logistic Regression Pipeline
clf = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("model", LogisticRegression(
        penalty="l2",
        C=1.0,
        solver="liblinear",
        max_iter=500,
        class_weight="balanced"
    ))
])

clf.fit(X_train, y_train)


# ## Evaluation and Plots

# In[77]:


probs = clf.predict_proba(X_test)[:, 1]
precision, recall, thresholds = precision_recall_curve(y_test, probs)

auc_pr = auc(recall, precision)
brier  = brier_score_loss(y_test, probs)

print("AUPRC:", auc_pr)
print("Brier Score:", brier)
print("\nClassification Report:\n")
print(classification_report(y_test, clf.predict(X_test)))

# Save PR Curve
plt.plot(recall, precision)
plt.title(f"Precision-Recall Curve (AUC={auc_pr:.3f})")
plt.xlabel("Recall")
plt.ylabel("Precision")
os.makedirs("results", exist_ok=True)
plt.savefig("results/pr_curve_logreg_sklearn.png", dpi=150)
plt.close()


# In[78]:


# Confusion matrix
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

y_pred_lr = (probs >= 0.5).astype(int)
cm_lr = confusion_matrix(y_test, y_pred_lr)

plt.figure(figsize=(6,5))
sns.heatmap(cm_lr, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix – Logistic Regression")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.savefig("results/cm_logistic_regression.png", dpi=150)
plt.show()


# In[79]:


results = {
    "AUPRC": auc_pr,
    "brier_score": brier
}

pd.DataFrame([results]).to_csv("results/logreg_metrics_sklearn.csv", index=False)
np.save("results/lr_probs.npy", probs)
np.save("results/lr_y_test.npy", y_test.values)
print("\n✅ Logistic Regression (sklearn) baseline complete!")
conn.close()

