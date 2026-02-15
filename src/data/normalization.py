"""
Normalization utilities for lightcurve dataset.

This module computes global normalization statistics
for flux, flux_err, redshift, and EBV.

IMPORTANT:
Statistics must be computed on TRAIN SET ONLY.
"""

import pandas as pd


def compute_normalization_stats(df):
    """
    Compute global mean and standard deviation for:
        - flux
        - flux_err
        - redshift
        - ebv
    """

    stats = {}

    # Flux statistics
    stats["flux_mean"] = df["flux"].mean()
    stats["flux_std"] = df["flux"].std()

    # Flux error statistics
    stats["flux_err_mean"] = df["flux_err"].mean()
    stats["flux_err_std"] = df["flux_err"].std()

    # Redshift statistics
    stats["redshift_mean"] = df["redshift"].mean()
    stats["redshift_std"] = df["redshift"].std()

    # EBV statistics
    stats["ebv_mean"] = df["ebv"].mean()
    stats["ebv_std"] = df["ebv"].std()

    return stats