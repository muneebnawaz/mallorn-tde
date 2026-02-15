from pathlib import Path
import pandas as pd
import numpy as np
from extinction import fitzpatrick99

EFF_WAVELENGTHS = {
    'u': 3641,
    'g': 4704,
    'r': 6155,
    'i': 7504,
    'z': 8695,
    'y': 10056
}

R_V = 3.1

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

    # -------------------------
    # Apply Galactic De-Extinction
    # -------------------------

    def apply_de_extinction(df):
        df = df.copy()
        df["Flux_corr"] = df["Flux"]

        for band, wavelength in EFF_WAVELENGTHS.items():
            mask = df["Filter"] == band

            if mask.any():
                ebv = df.loc[mask, "EBV"].values
                Av = ebv * R_V

                A_lambda = fitzpatrick99(
                    np.full_like(Av, wavelength, dtype=float),
                    Av
                )

                correction_factor = 10 ** (A_lambda / 2.5)

                df.loc[mask, "Flux_corr"] = (
                    df.loc[mask, "Flux"].values * correction_factor
                )

        return df

    train_data = apply_de_extinction(train_data)
    test_data = apply_de_extinction(test_data)

    # Drop EBV since flux is now corrected
    train_data = train_data.drop(columns=["EBV"])
    test_data = test_data.drop(columns=["EBV"])

    return train_data, test_data, train_log, test_log