#!/usr/bin/env python
# coding: utf-8

# # Random Forest Baseline
# 
# ---

# ## Setup

# In[79]:


import os
import sqlite3
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, FunctionTransformer
from sklearn.metrics import (
    precision_recall_curve,
    auc,
    brier_score_loss,
    classification_report,
    confusion_matrix,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
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

# In[80]:


conn = sqlite3.connect("data/routes_scores.db")
df = pd.read_sql("SELECT * FROM routes;", conn)

df["county_count"] = df["counties"].str.count(",").fillna(0).astype(int) + 1

numeric_features = ["county_count"]
text_feature = "counties"
target = "impacting_delivery"

df[text_feature] = df[text_feature].fillna("")

print("Label distribution:\n", df[target].value_counts(normalize=True))


# ## Train/val/test split

# In[81]:


train, test = train_test_split(df, test_size=0.15, random_state=42, stratify=df[target])
train, val  = train_test_split(train, test_size=0.1765, random_state=42, stratify=train[target])

X_train, y_train = train[[text_feature] + numeric_features], train[target]
X_val,   y_val   = val[[text_feature] + numeric_features],   val[target]
X_test,  y_test  = test[[text_feature] + numeric_features],  test[target]


# ## Preprocessing and model

# In[82]:


# Convert single text column to bag-of-counties
to_1d = FunctionTransformer(lambda x: x.squeeze(), validate=False)

county_bow = Pipeline(
    steps=[
        ("to_1d", to_1d),
        ("vec", CountVectorizer(
            tokenizer=lambda s: [t.strip() for t in s.split(",") if t.strip()],
            lowercase=False,
            min_df=5,
        )),
    ]
)

numeric_pipe = Pipeline(steps=[("scaler", StandardScaler())])

preprocessor = ColumnTransformer(
    transformers=[
        ("county_bow", county_bow, [text_feature]),
        ("num", numeric_pipe, numeric_features),
    ]
)

rf = Pipeline(
    steps=[
        ("preprocess", preprocessor),
        ("model", RandomForestClassifier(
            n_estimators=500,
            max_depth=None,
            min_samples_leaf=1,
            class_weight="balanced",
            n_jobs=-1,
            random_state=42,
        )),
    ]
)

rf.fit(X_train, y_train)


# Threshold selection (validation)

# In[83]:


val_probs = rf.predict_proba(X_val)[:, 1]
prec_v, rec_v, thr_v = precision_recall_curve(y_val, val_probs)

f1_v = (2 * prec_v * rec_v) / (prec_v + rec_v + 1e-12)
best_idx = np.nanargmax(f1_v)
best_thr = thr_v[best_idx] if best_idx < len(thr_v) else 0.5
print(f"Chosen threshold from val (F1-optimal): {best_thr:.3f}")


# ## Evaluation and plots

# In[84]:


probs = rf.predict_proba(X_test)[:, 1]
precision, recall, thresholds = precision_recall_curve(y_test, probs)

auc_pr = auc(recall, precision)
brier  = brier_score_loss(y_test, probs)

print("AUPRC:", auc_pr)
print("Brier Score:", brier)
print("\nClassification Report:\n")
print(classification_report(y_test, rf.predict(X_test)))

os.makedirs("results", exist_ok=True)
plt.plot(recall, precision)
plt.title(f"Precision-Recall Curve (RF, AUC={auc_pr:.3f})")
plt.xlabel("Recall"); plt.ylabel("Precision")
plt.savefig("results/pr_curve_random_forest.png", dpi=150)
plt.close()


# In[85]:


# Confusion matrix
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

rf_preds = (rf.predict_proba(X_test)[:, 1] >= 0.5).astype(int)
cm_rf = confusion_matrix(y_test, rf_preds)

plt.figure(figsize=(6,5))
sns.heatmap(cm_rf, annot=True, fmt="d", cmap="Greens")
plt.title("Confusion Matrix – Random Forest")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.savefig("results/cm_random_forest.png", dpi=150)
plt.show()


# In[86]:


# RF Feature Importance

rf_model = rf.named_steps["model"]
pre = rf.named_steps["preprocess"]

vec = pre.named_transformers_["county_bow"].named_steps["vec"]
bow_features = vec.get_feature_names_out()

numeric_features = pre.named_transformers_["num"].get_feature_names_out()

feature_names = np.concatenate([bow_features, numeric_features])

print("Total feature count:", len(feature_names))

importances = rf_model.feature_importances_
idx = np.argsort(importances)[::-1]

top_k = 20
plt.figure(figsize=(8, 6))
plt.barh(np.array(feature_names)[idx][:top_k][::-1],
         importances[idx][:top_k][::-1])
plt.title("Random Forest Feature Importance (Top 20)")
plt.tight_layout()
plt.savefig("results/rf_feature_importance.png")
plt.close()


# In[88]:


pd.DataFrame([{"AUPRC": auc_pr, "brier_score": brier}]).to_csv(
    "results/random_forest_metrics.csv", index=False
)

np.save("results/rf_probs.npy", probs)
np.save("results/rf_y_test.npy", y_test.values)
print("✅ Random Forest run complete")

