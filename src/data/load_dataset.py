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


_K_BAND = None

def _get_k_band():
    global _K_BAND
    if _K_BAND is None:
        _K_BAND = {}
        for band, wavelength in EFF_WAVELENGTHS.items():
            A1 = fitzpatrick99(np.array([float(wavelength)], dtype=float), 1.0, R_V)
            _K_BAND[band] = float(A1[0])
    return _K_BAND

def load_dataset(apply_extinction_correction=True, drop_ebv=True, debug_extinction=False):
    """Load full dataset (all split folders), merge metadata, and optionally apply Galactic de-extinction.

    Returns
    -------
    train_data, test_data, train_log, test_log
    """

    data_root = Path(__file__).resolve().parents[2] / "mallorn-astronomical-classification-challenge"

    train_log = pd.read_csv(data_root / "train_log.csv")
    test_log = pd.read_csv(data_root / "test_log.csv")

    split_folders = sorted([p for p in data_root.glob("split_*") if p.is_dir()])

    train_lcs, test_lcs = [], []
    for folder in split_folders:
        train_lcs.append(pd.read_csv(folder / "train_full_lightcurves.csv"))
        test_lcs.append(pd.read_csv(folder / "test_full_lightcurves.csv"))

    train_data = pd.concat(train_lcs, ignore_index=True)
    test_data = pd.concat(test_lcs, ignore_index=True)

    # Merge metadata (needs EBV for correction)
    train_data = train_data.merge(
        train_log[['object_id', 'target', 'SpecType', 'Z', 'EBV']],
        on='object_id', how='left'
    )
    
    test_data = test_data.merge(
        test_log[['object_id', 'Z', 'Z_err', 'EBV']],
        on='object_id', how='left'
    )

    def apply_de_extinction(df):
        df = df.copy()
        df["Flux_corr"] = df["Flux"].to_numpy(dtype=float)

        filt = df["Filter"].to_numpy(dtype=str)
        ebv_all = df["EBV"].to_numpy(dtype=float)
        flux_all = df["Flux"].to_numpy(dtype=float)

        # Precompute A_lambda per unit Av for each band: k_band = A_lambda(Av=1)
        k_band = _get_k_band()

        # Apply correction band-by-band (vectorized over rows)
        for band in ("u", "g", "r", "i", "z", "y"):
            m = (filt == band)
            if not np.any(m):
                continue

            ebv = ebv_all[m]
            good = np.isfinite(ebv)
            if not np.any(good):
                continue

            Av = ebv[good] * R_V
            A_lambda = k_band[band] * Av
            # A_lambda / 2.5 -> log10(corr). Clip to avoid inf.
            log10_corr = np.clip(A_lambda / 2.5, -10.0, 10.0)
            corr = 10.0 ** log10_corr

            flux = flux_all[m]
            flux_corr = flux.copy()
            flux_corr[good] = flux_corr[good] * corr

            df.loc[m, "Flux_corr"] = flux_corr

        return df

    if apply_extinction_correction:
        train_data = apply_de_extinction(train_data)
        test_data = apply_de_extinction(test_data)
    else:
        # Still create Flux_corr for a stable schema
        train_data["Flux_corr"] = train_data["Flux"].to_numpy(dtype=float)
        test_data["Flux_corr"] = test_data["Flux"].to_numpy(dtype=float)

    if debug_extinction:
        ratio = train_data["Flux_corr"].to_numpy(float) / train_data["Flux"].to_numpy(float)
        print("De-extinction median Flux_corr/Flux:", np.nanmedian(ratio))
        for b in ["u","g","r","i","z","y"]:
            m = train_data["Filter"].to_numpy(str) == b
            rb = train_data.loc[m, "Flux_corr"].to_numpy(float) / train_data.loc[m, "Flux"].to_numpy(float)
            print(f"  {b}: median={np.nanmedian(rb):.4f}  95p={np.nanpercentile(rb,95):.4f}")

    if drop_ebv:
        train_data = train_data.drop(columns=["EBV"], errors="ignore")
        test_data = test_data.drop(columns=["EBV"], errors="ignore")

    return train_data, test_data, train_log, test_log