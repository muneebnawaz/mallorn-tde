from pathlib import Path
import pandas as pd

from src.data.load_dataset import load_dataset
from src.features.gp_feature_extraction import build_feature_table


def main():
    train_data, test_data, train_log, test_log = load_dataset(
        apply_extinction_correction=True,
        drop_ebv=True,
        debug_extinction=False
    )

    print("Loaded data.")
    print("Train lightcurves shape:", train_data.shape)
    print("Test lightcurves shape:", test_data.shape)
    print("Train metadata shape:", train_log.shape)
    print("Test metadata shape:", test_log.shape)

    # -------------------------------
    # Build train feature table
    # -------------------------------
    print("\nBuilding train feature table...")
    train_features = build_feature_table(
        df_lc=train_data,
        df_meta=train_log,
        object_ids=None,
        n_jobs=-1,
        gp_n_restarts=0
    )

    print("Train feature table built.")
    print("Train feature shape:", train_features.shape)

    # -------------------------------
    # Build test feature table
    # -------------------------------
    print("\nBuilding test feature table...")
    test_features = build_feature_table(
        df_lc=test_data,
        df_meta=test_log,
        object_ids=None,
        n_jobs=-1,
        gp_n_restarts=0
    )

    print("Test feature table built.")
    print("Test feature shape:", test_features.shape)

    # -------------------------------
    # Save both
    # -------------------------------
    out_dir = Path("artifacts")
    out_dir.mkdir(parents=True, exist_ok=True)

    train_path = out_dir / "train_features.csv"
    test_path = out_dir / "test_features.csv"

    train_features.to_csv(train_path, index=False)
    test_features.to_csv(test_path, index=False)

    print(f"\nSaved train features to: {train_path}")
    print(f"Saved test features to: {test_path}")

    # -------------------------------
    # Column checks
    # -------------------------------
    train_cols = set(train_features.columns)
    test_cols = set(test_features.columns)

    print("\nColumns only in train:", sorted(train_cols - test_cols))
    print("Columns only in test:", sorted(test_cols - train_cols))


if __name__ == "__main__":
    main()