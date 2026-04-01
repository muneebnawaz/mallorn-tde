from pathlib import Path
import pandas as pd
from catboost import CatBoostClassifier


def add_missingness_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["limited_lc_flag"] = df["fade_time"].isna().astype(int)

    exclude_for_nan_fraction = ["object_id", "target"]
    feature_cols_for_nan_fraction = [c for c in df.columns if c not in exclude_for_nan_fraction]
    df["nan_fraction"] = df[feature_cols_for_nan_fraction].isna().mean(axis=1)

    return df


def main():
    # -----------------------------
    # Load feature tables
    # -----------------------------
    train_path = Path("artifacts/train_features.csv")
    test_path = Path("artifacts/test_features.csv")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    print("Loaded train:", train_df.shape)
    print("Loaded test :", test_df.shape)

    # -----------------------------
    # Add extra features
    # -----------------------------
    train_df = add_missingness_features(train_df)
    test_df = add_missingness_features(test_df)

    # -----------------------------
    # Save object_id for submission
    # -----------------------------
    test_object_ids = test_df["object_id"].copy()

    # -----------------------------
    # Prepare X, y
    # -----------------------------
    drop_cols_train = ["object_id", "target"]
    drop_cols_test = ["object_id"]

    if "redshift_err" in train_df.columns:
        drop_cols_train.append("redshift_err")
    if "redshift_err" in test_df.columns:
        drop_cols_test.append("redshift_err")

    X_train = train_df.drop(columns=drop_cols_train)
    y_train = train_df["target"]

    X_test = test_df.drop(columns=drop_cols_test)

    print("X_train shape:", X_train.shape)
    print("X_test shape :", X_test.shape)

    # -----------------------------
    # Train model on FULL training data
    # -----------------------------
    model = CatBoostClassifier(
        iterations=360,                 # from your best baseline iteration
        learning_rate=0.03,
        depth=6,
        loss_function="Logloss",
        eval_metric="PRAUC",
        auto_class_weights="Balanced",
        random_seed=42,
        verbose=100
    )

    model.fit(X_train, y_train)

    # -----------------------------
    # Predict test labels
    # -----------------------------
    test_pred = model.predict(X_test).astype(int).ravel()

    # -----------------------------
    # Build submission file
    # -----------------------------
    submission = pd.DataFrame({
        "object_id": test_object_ids,
        "prediction": test_pred
    })

    out_path = Path("artifacts/submission_catboost.csv")
    submission.to_csv(out_path, index=False)

    print(f"\nSaved submission to: {out_path}")
    print(submission.head())


if __name__ == "__main__":
    main()