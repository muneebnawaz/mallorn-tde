import numpy as np


def compute_normalization_stats(df, per_spectrum=False, eps=1e-8):
    """
    per_spectrum set False by default. Per-spectrum normalization to by consciously selected.
    """
    stats = {}

    if per_spectrum:
        # Each row contains flux array for one spectrum
        stats["flux_mean"] = df["flux"].apply(np.mean)
        stats["flux_std"] = df["flux"].apply(np.std) + eps
        # Scale flux_err by the same per-spectrum std to preserve SNR
        stats["flux_err_scale"] = stats["flux_std"]
    else:
        # Global statistics over all flux values in training data
        all_flux = np.concatenate(df["flux"].values)
        stats["flux_mean"] = np.mean(all_flux)
        stats["flux_std"] = np.std(all_flux) + eps

        all_flux_err = np.concatenate(df["flux_err"].values)
        stats["flux_err_mean"] = np.mean(all_flux_err)
        stats["flux_err_std"] = np.std(all_flux_err) + eps

    # Redshift normalization (always global)
    stats["redshift_mean"] = df["redshift"].mean()
    stats["redshift_std"] = df["redshift"].std() + eps

    return stats