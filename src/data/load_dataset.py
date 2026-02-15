from pathlib import Path
import pandas as pd
import glob

def load_dataset():
    '''Loading full dataset, combining all split folders'''

    # Base data directory
    data_root = Path(__file__).resolve().parents[2] / "mallorn-astronomical-classification-challenge"

    # Load train/test logs
    train_log = pd.read_csv(data_root / "train_log.csv")
    test_log = pd.read_csv(data_root / "test_log.csv")

    # Find all split folders
    split_folders = sorted([p for p in data_root.glob("split_*") if p.is_dir()])

    # Merge all train ligthcurves files
    train_lcs = []
    test_lcs = []

    for folder in split_folders:
        train_lcs.append(pd.read_csv(folder / "train_full_lightcurves.csv"))
        test_lcs.append(pd.read_csv(folder / "test_full_lightcurves.csv"))

    train_data = pd.concat(train_lcs,ignore_index=True)
    test_data = pd.concat(test_lcs, ignore_index=True)

    # Merge metadata
    train_data = train_data.merge(
        train_log[['object_id', 'target', 'SpecType', 'Z', 'EBV']],
        on = 'object_id',
        how = 'left'
    )

    test_data = test_data.merge(
        test_log[['object_id', 'Z', 'Z_err', 'EBV']],
        on = 'object_id',
        how = 'left'
    )

    return train_data, test_data, train_log, test_log