import pandas as pd
import numpy as np
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef
)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# -------------------------------------------------
# 1. Setup
# -------------------------------------------------
os.makedirs("model", exist_ok=True)

# -------------------------------------------------
# 2. Load dataset
# -------------------------------------------------
df = pd.read_csv("breast_cancer.csv")

# -------------------------------------------------
# 3. Drop non-feature columns safely
# -------------------------------------------------
if "id" in df.columns:
    df.drop("id", axis=1, inplace=True)

# -------------------------------------------------
# 4. Encode target SAFELY (no NaNs)
# -------------------------------------------------
df["diagnosis"] = df["diagnosis"].replace({"M": 1, "B": 0})

# -------------------------------------------------
# 5. Explicit NaN handling (CRITICAL)
# -------------------------------------------------
# Drop rows where target is missing
df = df.dropna(subset=["diagnosis"])

# Separate features and target
X = df.drop("diagnosis", axis=1)
y = df["diagnosis"]

# Fill NaNs in features with column mean
X = X.fillna(X.mean())

# -------------------------------------------------
# 6. Train-test split
# -------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

# -------------------------------------------------
# 7. Scaling
# -------------------------------------------------
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

joblib.dump(scaler, "model/scaler.pkl")

# -------------------------------------------------
# 8. Models
# -------------------------------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "KNN": KNeighborsClassifier(),
    "Naive Bayes": GaussianNB(),
    "Random Forest": RandomForestClassifier(
        n_estimators=200, random_state=42
    ),
    "XGBoost": XGBClassifier(
        eval_metric="logloss",
        use_label_encoder=False,
        random_state=42
    )
}

# -------------------------------------------------
# 9. Train, Evaluate, Save
# -------------------------------------------------
results = []

for name, model in models.items():
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    results.append([
        name,
        accuracy_score(y_test, preds),
        roc_auc_score(y_test, probs),
        precision_score(y_test, preds),
        recall_score(y_test, preds),
        f1_score(y_test, preds),
        matthews_corrcoef(y_test, preds)
    ])

    joblib.dump(model, f"model/{name}.pkl")

# -------------------------------------------------
# 10. Save metrics
# -------------------------------------------------
metrics_df = pd.DataFrame(
    results,
    columns=["Model", "Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]
)

metrics_df.to_csv("model/metrics.csv", index=False)

print("\nTraining completed successfully!\n")
print(metrics_df)
