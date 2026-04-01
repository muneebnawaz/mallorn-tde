import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report
)

from catboost import CatBoostClassifier


# =========================================================
# 1. Load feature table
# =========================================================
df = pd.read_csv("artifacts/train_features.csv")

print("Shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())

print("\nTarget distribution:")
print(df["target"].value_counts())
print("\nTarget proportions:")
print(df["target"].value_counts(normalize=True))


# =========================================================
# 2. Create extra missingness-based features
# =========================================================
# limited_lc_flag: whether fade_time is missing
df["limited_lc_flag"] = df["fade_time"].isna().astype(int)

# nan_fraction: row-wise fraction of missing values across feature columns
exclude_for_nan_fraction = ["object_id", "target"]
feature_cols_for_nan_fraction = [c for c in df.columns if c not in exclude_for_nan_fraction]
df["nan_fraction"] = df[feature_cols_for_nan_fraction].isna().mean(axis=1)

print("\nAdded columns: limited_lc_flag, nan_fraction")


# =========================================================
# 3. Drop columns we should not use
# =========================================================
drop_cols = ["object_id", "target"]

# redshift_err is known to be useless in train (100% NaN)
if "redshift_err" in df.columns:
    drop_cols.append("redshift_err")

X = df.drop(columns=drop_cols)
y = df["target"]

print("\nFinal X shape:", X.shape)
print("Final y shape:", y.shape)

print("\nMissingness per feature:")
print(X.isna().mean().sort_values(ascending=False))


# =========================================================
# 4. Train / validation split
# =========================================================
X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTrain shape:", X_train.shape)
print("Validation shape:", X_val.shape)

print("\nTrain target distribution:")
print(y_train.value_counts())

print("\nValidation target distribution:")
print(y_val.value_counts())


# =========================================================
# 5. Define CatBoost model
# =========================================================
model = CatBoostClassifier(
    iterations=1000,
    learning_rate=0.03,
    depth=6,
    loss_function="Logloss",
    eval_metric="PRAUC",              # useful for imbalanced binary problems
    auto_class_weights="Balanced",    # important because TDE class is rare
    random_seed=42,
    verbose=100
)


# =========================================================
# 6. Train model
# =========================================================
model.fit(
    X_train,
    y_train,
    eval_set=(X_val, y_val),
    use_best_model=True,
    early_stopping_rounds=100
)


# =========================================================
# 7. Predict
# =========================================================
y_pred = model.predict(X_val).astype(int).ravel()
y_proba = model.predict_proba(X_val)[:, 1]


# =========================================================
# 8. Evaluation
# =========================================================
acc = accuracy_score(y_val, y_pred)
bal_acc = balanced_accuracy_score(y_val, y_pred)
prec = precision_score(y_val, y_pred, zero_division=0)
rec = recall_score(y_val, y_pred, zero_division=0)
f1 = f1_score(y_val, y_pred, zero_division=0)
roc_auc = roc_auc_score(y_val, y_proba)
pr_auc = average_precision_score(y_val, y_proba)

print("\n" + "=" * 60)
print("VALIDATION METRICS")
print("=" * 60)
print(f"Accuracy:           {acc:.4f}")
print(f"Balanced Accuracy:  {bal_acc:.4f}")
print(f"Precision:          {prec:.4f}")
print(f"Recall:             {rec:.4f}")
print(f"F1 Score:           {f1:.4f}")
print(f"ROC AUC:            {roc_auc:.4f}")
print(f"PR AUC:             {pr_auc:.4f}")

print("\nConfusion Matrix:")
print(confusion_matrix(y_val, y_pred))

print("\nClassification Report:")
print(classification_report(y_val, y_pred, zero_division=0))


# =========================================================
# 9. Feature importance
# =========================================================
feat_imp = pd.DataFrame({
    "feature": X.columns,
    "importance": model.get_feature_importance()
}).sort_values("importance", ascending=False)

print("\nTop 20 Features:")
print(feat_imp.head(20))