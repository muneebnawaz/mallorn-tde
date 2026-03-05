import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings

from src.data.load_dataset import load_dataset
from src.features.gp_feature_extraction import build_feature_table, fit_2d_gp_for_object, gp_predict, _get_flux_columns
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

RANDOM_SEED = 42
N_OBJECTS = 500

def main():
    # 1) Load
    data = load_dataset()

    # --- Adjust these two lines to match what load_dataset() returns ---
    # Common patterns: (train_lc, train_log, test_lc, test_log) OR dict
    if isinstance(data, dict):
        lcs = data.get("train_data") or data.get("train_lcs") or data.get("lightcurves")
        log = data.get("train_log") or data.get("meta") or data.get("log")
    else:
        # fallback: assume first item is lightcurves, second is log/meta
        lcs, log = data[0], data[1]

    assert isinstance(lcs, pd.DataFrame), "Expected lightcurves dataframe"
    assert "object_id" in lcs.columns, "lightcurves must contain object_id"

    # 2) Sample objects
    object_ids = lcs["object_id"].dropna().unique()
    rng = np.random.default_rng(RANDOM_SEED)
    sample_ids = rng.choice(object_ids, size=min(N_OBJECTS, len(object_ids)), replace=False)

    lcs_small = lcs[lcs["object_id"].isin(sample_ids)].copy()

    print(f"Loaded LCs: {len(lcs):,} rows | Small sample: {len(lcs_small):,} rows | Objects: {len(sample_ids)}")

    # 3) Build features (this is the core integration test)
    feat = build_feature_table(lcs_small, n_jobs=-1)
    assert isinstance(feat, pd.DataFrame), "build_feature_table must return a DataFrame"
    assert "object_id" in feat.columns, "feature table must contain object_id"

    print("\nFeature table shape:", feat.shape)
    print("Feature columns:", len(feat.columns) - 1)

    print("\nGP fit ok rate:", feat["gp_fit_ok"].mean())

    mask_ok = feat["gp_fit_ok"] == 1
    fade_nan_rate_ok = feat.loc[mask_ok, "fade_time"].isna().mean()
    print("fade_time NaN rate (gp_fit_ok==1):", fade_nan_rate_ok)

    color_cols = ["mean_color_gr", "std_color_gr", "color_cooling_rate"]
    for c in color_cols:
        rate = feat.loc[mask_ok, c].isna().mean()
        print(f"{c} NaN rate (gp_fit_ok==1): {rate}")

    # 4) NaN rate summary
    nan_rate = feat.drop(columns=["object_id"]).isna().mean().sort_values(ascending=False)
    print("\nTop NaN-rate features (first 30):")
    print(nan_rate.head(30).to_string())

    # Pick objects for spot checks
    X = feat.drop(columns=["object_id"])
    row_nan = X.isna().mean(axis=1)

    best_i = row_nan.idxmin()
    worst_i = row_nan.idxmax()
    med_i = row_nan.sort_values().index[len(row_nan)//2]

    best_id = feat.loc[best_i, "object_id"]
    worst_id = feat.loc[worst_i, "object_id"]
    med_id = feat.loc[med_i, "object_id"]

    print("\nBest:", best_id, "NaN rate:", float(row_nan.loc[best_i]))
    print("Worst:", worst_id, "NaN rate:", float(row_nan.loc[worst_i]))
    print("Median:", med_id, "NaN rate:", float(row_nan.loc[med_i]))

    # Inspect feature completeness for the worst object
    worst_row = feat[feat["object_id"] == worst_id].iloc[0]
    missing_cols = worst_row.index[worst_row.isna()].tolist()

    print("\nWorst object missing columns:", missing_cols)
    print("\nWorst object row (all columns):")
    print(worst_row.to_string())

    print("\nWorst gp_fit_ok:", int(worst_row.get("gp_fit_ok", -1)))
    print("Worst gp_fit_error:", worst_row.get("gp_fit_error", None))

    def summarize_lc(df_lc, oid):
        df = df_lc[df_lc["object_id"] == oid]
        n = len(df)
        tcol = "Time (MJD)"
        fcol = "Filter"

        span = float(df[tcol].max() - df[tcol].min()) if n else np.nan
        counts = df[fcol].value_counts().to_dict() if n else {}

        # Use your helper to get flux columns (since names can vary)
        # If _get_flux_columns isn't importable here, just inspect the columns directly.
        flux_col = "Flux_corr" if "Flux_corr" in df.columns else ("Flux" if "Flux" in df.columns else None)
        ferr_col = "Flux_err" if "Flux_err" in df.columns else None

        print(f"\nOID={oid}")
        print("  n_points:", n)
        print("  time_span_days:", span)
        print("  counts_by_filter:", counts)

        if flux_col:
            print("  flux finite rate:", float(np.isfinite(df[flux_col].to_numpy(float)).mean()))
        else:
            print("  flux finite rate: n/a (no 'flux' column)")

        if ferr_col:
            print("  flux_err finite rate:", float(np.isfinite(df[ferr_col].to_numpy(float)).mean()))
        else:
            print("  flux_err finite rate: n/a (no 'flux_err' column)")

    summarize_lc(lcs_small, best_id)
    summarize_lc(lcs_small, worst_id)
    summarize_lc(lcs_small, med_id)

    # === GP sanity check on the best object ===
    def gp_sanity_check(df_lc, oid, band="g", grid_points=100):
        df_obj = df_lc[df_lc["object_id"] == oid].copy()
        gp_res = fit_2d_gp_for_object(df_obj, n_restarts_optimizer=0)

        print(f"\nGP sanity check: oid={oid}, band={band}")
        if gp_res is None:
            print("  GP fit returned None")
            return

        t_obs = df_obj["Time (MJD)"].to_numpy(float)
        t_grid = np.linspace(np.nanmin(t_obs), np.nanmax(t_obs), grid_points).astype(float)

        y_pred, y_std = gp_predict(gp_res, t_grid, band, return_std=True)

        y_pred = np.asarray(y_pred, float)
        y_std = np.asarray(y_std, float)

        print("  pred finite rate:", float(np.isfinite(y_pred).mean()))
        print("  std  finite rate:", float(np.isfinite(y_std).mean()))
        print("  pred min/median/max:", float(np.nanmin(y_pred)), float(np.nanmedian(y_pred)), float(np.nanmax(y_pred)))
        print("  std  min/median/max:", float(np.nanmin(y_std)), float(np.nanmedian(y_std)), float(np.nanmax(y_std)))

    gp_sanity_check(lcs_small, best_id, band="g")

    def plot_gp_fit_one_band(df_lc, oid, band="g", grid_points=200, out_png="gp_debug.png"):
        df_obj = df_lc[df_lc["object_id"] == oid].copy()
        gp_res = fit_2d_gp_for_object(df_obj, n_restarts_optimizer=0)
        if gp_res is None:
            print(f"Plot skipped: GP fit None for {oid}")
            return

        # Extract observed arrays
        t_obs = df_obj["Time (MJD)"].to_numpy(float)
        f_obs = df_obj["Filter"].to_numpy(str)
        y_obs, yerr_obs = _get_flux_columns(df_obj)  # if not importable here, use your gp_feature_extraction import

        # Filter to one band
        m = (f_obs == band)
        t_b = t_obs[m]
        y_b = np.asarray(y_obs[m], float)
        e_b = np.asarray(yerr_obs[m], float)

        # Predict on grid
        t_grid = np.linspace(np.nanmin(t_obs), np.nanmax(t_obs), grid_points).astype(float)
        y_pred, y_std = gp_predict(gp_res, t_grid, band, return_std=True)

        plt.figure()
        plt.errorbar(t_b, y_b, yerr=e_b, fmt=".", capsize=0)
        plt.plot(t_grid, y_pred)
        plt.fill_between(t_grid, y_pred - y_std, y_pred + y_std, alpha=0.2)
        plt.title(f"{oid} | band={band}")
        plt.xlabel("Time (MJD)")
        plt.ylabel("Flux (scaled space)")
        plt.tight_layout()
        plt.savefig(out_png, dpi=150)
        plt.close()
        print(f"Saved plot: {out_png}")

    # ✅ CALLS GO HERE (still indented inside main(), but NOT inside the function)
    plot_gp_fit_one_band(lcs_small, best_id, band="g", out_png="gp_500_best_g.png")
    plot_gp_fit_one_band(lcs_small, best_id, band="r", out_png="gp_500_best_r.png")

    plot_gp_fit_one_band(lcs_small, worst_id, band="g", out_png="gp_500_worst_g.png")
    plot_gp_fit_one_band(lcs_small, worst_id, band="r", out_png="gp_500_worst_r.png")


    # 5) Per-object NaN distribution
    row_nan = feat.drop(columns=["object_id"]).isna().mean(axis=1)
    print("\nPer-object NaN rate:")
    print(row_nan.describe().to_string())

    # 6) Dump a small CSV for inspection
    feat.to_csv("debug_features_500.csv", index=False)
    print("\nWrote debug_features_500.csv")

if __name__ == "__main__":
    main()