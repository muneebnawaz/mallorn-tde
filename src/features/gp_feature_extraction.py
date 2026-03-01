from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Any

import numpy as np
import pandas as pd

from joblib import Parallel, delayed

from scipy.stats import skew, kurtosis
from scipy.optimize import curve_fit

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern

# Defining Filter Wavelengths
FILTER_WAVELENGTHS = {
    "u": 3641.0,
    "g": 4704.0,
    "r": 6155.0,
    "i": 7504.0,
    "z": 8695.0,
    "y": 10056.0,
}

# Defining Container for GP results
@dataclass
class GPFitResult:
    gp: GaussianProcessRegressor
    y_scale: float
    x_time0: float
    kernel: Any

# Creating safe float function to handle missing values if any
def _safe_float(x):
    try:
        if x is None:
            return np.nan
        return float(x)
    except:
        return np.nan

# Choosing which flux to use
def _get_flux_columns(df_obj):
    if "Flux_corr" in df_obj.columns:
        y = df_obj["Flux_corr"].to_numpy(dtype=float)
    else:
        y = df_obj["Flux"].to_numpy(dtype=float)
    yerr = df_obj["Flux_err"].to_numpy(dtype=float)
    return y, yerr

# Creating function to fit GP curve to each object
def fit_2d_gp_for_object(
    df_obj,
    min_points=8,
    min_sig_detections=2,
    sig_thresh=3.0,
    n_restarts_optimizer=0,
    random_state=42,
    ):

    # Skipping objects with not enough data
    if df_obj.shape[0] < min_points:
        return None
    

    # Ensuring existence of signal
    y, yerr = _get_flux_columns(df_obj)
    sig = (y > sig_thresh * yerr)
    if int(sig.sum()) < min_sig_detections:
        return None
    
    # Building GP input X = [time, wavelength]
    t = df_obj["Time (MJD)"].to_numpy(float)
    w = df_obj["Filter"].map(FILTER_WAVELENGTHS).to_numpy(float)
    t0 = float(np.nanmin(t))
    t_scaled = (t - t0)  # days since first observation (time starts at 0)
    w_scaled = w / 10000.0 # GP optimizer becomes more stable
    X = np.column_stack([t_scaled, w_scaled])

    # Scaling object's flux by its max flux
    y_scale = float(np.nanmax(np.abs(y)))
    y_scaled = y / y_scale
    yerr_scaled = yerr / y_scale

    # Shape setting of the GP curve
    kernel = ConstantKernel(1.0, (1e-2, 1e3)) * Matern(
        length_scale=[100.0, 0.6],
        length_scale_bounds=(1e-2, 1e4),
        nu=1.5,
    )

    # GP fitting
    alpha = (yerr_scaled)**2

    gp = GaussianProcessRegressor(
        kernel=kernel,
        alpha=np.square(np.clip(yerr_scaled, 1e-12, np.inf)),
        normalize_y=False,
        n_restarts_optimizer=n_restarts_optimizer,
        random_state=random_state,
    )
    gp.fit(X, y_scaled)

    return GPFitResult(gp=gp, y_scale=y_scale, x_time0=t0, kernel=gp.kernel_)

# Creating function to extract flux at any time/filter
def gp_predict (gp_res, t_mjd, filt, return_std=False):
    w = FILTER_WAVELENGTHS.get(filt) # Convert filter letter to wavelength
    if w is None:
        raise ValueError(f"Unknown filter '{filter}'")
    
    t_scaled = (t_mjd.astype(float) - gp_res.x_time0) # time starts at 0
    w_scaled = np.full_like(t_scaled, w / 10000.0, dtype=float) # builind the query input
    Xq = np.column_stack([t_scaled, w_scaled])

    if return_std:
        y_pred, y_std = gp_res.gp.predict(Xq, return_std=True)
        return y_pred * gp_res.y_scale, y_std * gp_res.y_scale
    else:
        y_pred = gp_res.gp.predict(Xq, return_std=False)
        return y_pred * gp_res.y_scale, None

# Defining power-law decay
def _powerlaw_model(dt, a, b):
    return a * (dt + 1.0) ** b

# TDE power-law error - Measures post-peak performance for TDE power-law decay
def _tde_powerlaw_error(t, y):
    peak_idx = int(np.argmax(y)) # find peak index (where flux is highest)

    # keeping only post-peak part
    t_post = t[peak_idx:]
    y_post = y[peak_idx:]

    # Normalize flux so peak becomes 1.0
    y_peak = float(np.max(y_post))
    if not np.isfinite(y_peak) or y_peak <= 0:
        return np.nan
    y_norm = y_post / y_peak 

    # turn time into days since peak
    dt = t_post - t_post[0]

    # fit the power-law curve to (dt, y_norm)
    try:
        popt, _ = curve_fit(
            _powerlaw_model,
            dt,
            y_norm,
            p0 = (1.0, -1.67),
            maxfev = 20000
        )

        # compute fitted curve and measure error
        y_fit = _powerlaw_model(dt, *popt)

        # RMSE
        err = np.sqrt(np.mean((y_norm - y_fit) ** 2))
        return float(err)
    
    except Exception:
        return np.nan

# Creating function to compare post-peak lightcurve shape to an idealized TDE template
def _template_chisq_tde(t, y):

    peak_idx = np.argmax(y) # finding peak
    
    # keeping only post-peak part
    t_post = t[peak_idx:]
    y_post = y[peak_idx:]
    
    # Normalize flux so peak becomes 1.0
    y_peak = float(np.max(y_post))
    if not np.isfinite(y_peak) or y_peak <= 0:
        return np.nan
    y_norm = y_post / y_peak 

    # build normalized time u from 0 to 1
    u = (t_post - t_post[0])
    if np.max(u) <= 0:
        return np.nan
    u = u / np.max(u)

    # creating simple ideal TDE decay template
    template = (u + 0.05) ** (-0.5 / 3.0)
    template = template / np.max(template)

    # RMSE
    diff = y_norm - template
    return float(np.sqrt(np.mean(diff ** 2)))

# Creating function to return basic curve shape features
def _basic_morphology_features(t, y):
    out = {}

    if len(t) < 10 or not np.isfinite(y).any():
        return {
            "rise_time": np.nan,
            "fade_time": np.nan,
            "fwhm": np.nan,
            "compactness": np.nan,
            "percentile_ratio_80_max": np.nan,
            "percentile_ratio_20_50": np.nan,
            "baseline_ratio": np.nan
        }
    
    # Find peak
    peak_idx = int(np.argmax(y))
    y_peak = float(y[peak_idx])
    if not np.isfinite(y_peak) or y_peak <= 0:
        return {
            "rise_time": np.nan,
            "fade_time": np.nan,
            "fwhm": np.nan,
            "compactness": np.nan,
            "percentile_ratio_80_max": np.nan,
            "percentile_ratio_20_50": np.nan,
            "baseline_ratio": np.nan
        }
    
    # Define threshold for 1 magnitude below peak
    thr_1mag = y_peak / 2.512

    # Rise time from thr_1mag to peak
    pre = y[: peak_idx + 1]
    t_pre = t[: peak_idx + 1]
    try:
        cross_pre = np.where(pre >= thr_1mag)[0]
        rise_time = float(t_pre[peak_idx] - t_pre[int(cross_pre[0])]) if len(cross_pre) else np.nan
    except Exception:
        rise_time = np.nan

    # Fade time from peak to thr-1mag
    post = y[peak_idx:]
    t_post = t [peak_idx:]
    try:
        below = np.where(post <= thr_1mag)[0]
        fade_time = float(t_post[int(below[0])] - t_post[0]) if len(below) else float(t_post[0])
    except Exception:
        fade_time = np.nan

    # FWHM (full width half max) - How wide is the bright peak
    half = y_peak / 2.0
    try:
        above = np.where(y >= half)[0]
        fwhm = float(t[int(above[-1])] - t[int(above[0])]) if len(above) else np.nan
    except Exception:
        fwhm = np.nan

    # Compactness of the peak = area / peak
    try:
        area = float(np.trapz(np.maximum(y, 0,0), t))
        compactness = area / y_peak
    except Exception:
        compactness = np.nan

    # Percentile ratios - Sharp peak or Plateau
    y_pos = y[y > 0]
    if y_pos.size >= 5:
        p80 = np.percentile(y_pos, 80)
        p20 = np.percentile(y_pos, 20)
        p50 = np.percentile(y_pos, 50)
        percentile_ratio_80_max = float(p80 / y_peak) if y_peak > 0 else np.nan
        percentile_ratio_20_50 = float(p20 / p50) if p50 > 0 else np.nan
    else:
        percentile_ratio_80_max = np.nan
        percentile_ratio_20_50 = np.nan

    # Baseline ratio
    try:
        n0 = max(3, int(0.15 * len(y)))
        baseline = float(np.nanmedian(y[:n0]))
        baseline_ratio = baseline / y_peak if y_peak > 0 else np.nan
    except Exception:
        baseline_ratio = np.nan

    out.update(
        rise_time=rise_time,
        fade_time=fade_time,
        fwhm=fwhm,
        compactness=compactness,
        percentile_ratio_80_max=percentile_ratio_80_max,
        percentile_ratio_20_50=percentile_ratio_20_50,
        baseline_ratio=baseline_ratio,
    )
    return out
    
# Creating function to extract features of one object
def extract_features_for_object(
        object_id,
        df_lc,
        df_meta=None,
        morphology_band="g",
        grid_points=100,
        gp_n_restarts=0
):
    
    # Filtering rows for this object
    df_obj = df_lc[df_lc["object_id"] == object_id].copy()
    out = {"object_id": object_id}

    if df_obj.empty:
        return out
    
    # Adding metadata features to the out dictionary
    if df_meta is not None and "object_id" in df_meta.columns:
        row = df_meta[df_meta["object_id"] == object_id]
        if not row.empty:
            if "Z" in row.columns:
                out["redshift"] = _safe_float(row.iloc[0]["Z"])
            if "Z_err" in row.columns:
                out["redshift_err"] = _safe_float(row.iloc[0]["Z_err"])
            if "target" in row.columns:
                out["target"] = row.iloc[0]["target"]
    else:
        # Use per-row repeated columns if present
        if "Z" in df_obj.columns:
            out["redshift"] = _safe_float(df_obj["Z"].iloc[0])
        if "Z_err" in df_obj.columns:
            out["redshift_err"] = _safe_float(df_obj["Z_err"].iloc[0])
        if "target" in df_obj.columns:
            out["target"] = df_obj["target"].iloc[0]

    # Fitting GP for this object
    gp_res = fit_2d_gp_for_object(df_obj, n_restarts_optimizer=gp_n_restarts)
    if gp_res is None:
        # Populate with NaNs so downstream model can handle missing
        out.update(
            ls_time=np.nan,
            ls_wave=np.nan,
            amplitude=np.nan,
            reduced_chi_square=np.nan,
            negative_flux_fraction=np.nan,
            duty_cycle=np.nan,
            robust_duration=np.nan,
            template_chisq_tde=np.nan,
            tde_power_law_error=np.nan,
            log_tde_error=np.nan,
            flux_skew=np.nan,
            flux_kurtosis=np.nan,
            ug_peak=np.nan,
            gr_peak=np.nan,
            ur_peak=np.nan,
            mean_color_gr=np.nan,
            std_color_gr=np.nan,
            color_cooling_rate=np.nan,
            rise_time=np.nan,
            fade_time=np.nan,
            fwhm=np.nan,
            compactness=np.nan,
            percentile_ratio_80_max=np.nan,
            percentile_ratio_20_50=np.nan,
            baseline_ratio=np.nan,
            rise_time_rest=np.nan,
            fade_time_rest=np.nan,
            fwhm_rest=np.nan,
            robust_duration_rest=np.nan,
        )
        return out
    
    # learned kernel parameters
    try:
        k = gp_res.kernel
        const = getattr(k, "k1", None)
        matern = getattr(k, "k2", None)
        if const is None or matern is None:
            const = getattr(k, "k2", None)
            matern = getattr(k, "k1", None)

        ls_time = float(matern.length_scale[0]) if matern is not None else np.nan # time length scale
        ls_wave = float(matern.length_scale[1]) if matern is not None else np.nan # wavelength length scale 

        # scale of the signal
        amp = (
            float(np.sqrt(getattr(const, "constant_value", np.nan)) * gp_res.y_scale)
            if const is not None
            else np.nan
        )
    except Exception:
        ls_time, ls_wave, amp = np.nan, np.nan, np.nan

    out["ls_time"] = ls_time
    out["ls_wave"] = ls_wave
    out["amplitude"] = amp

    # reduced chi-square on observed points
    # real observed values
    y_obs, yerr_obs = _get_flux_columns(df_obj)
    t_obs = df_obj["Time (MJD)"].to_numpy(float)
    f_obs = df_obj["Filter"].to_numpy(str)
    # Predicting GP value at every observed point
    y_pred_obs = np.empty_like(y_obs, dtype=float)
    for i, (ti, fi) in enumerate(zip(t_obs, f_obs)):
        yp, _ = gp_predict(gp_res, np.array([ti], dtype=float), fi, return_std=False)
        y_pred_obs[i] = yp[0]
    # resuduals
    resid = y_obs - y_pred_obs
    # chi-square
    denom = np.square(np.clip(yerr_obs, 1e-12, np.inf))
    chi2 = float(np.nansum(np.square(resid) / denom))
    # reduced chi-square
    dof = max(1, int(np.isfinite(resid).sum()) - 3) # '-3' is a rough correction
    out["reduced_chi_square"] = chi2 / dof

    # checking if flux is strongly negative
    out["negative_flux_fraction"] = float(np.mean(y_obs < -3.0 * yerr_obs))

    # Significant detections
    det = (y_obs > 3.0 * yerr_obs)
    # robust durations / Duty cycle (object active period)
    if det.any():
        det_times = t_obs[det]
        t10, t90 = np.percentile(det_times, [10, 90])
        out["robust_duration"] = float(t90 - t10)
        total_span = float(np.nanmax(t_obs) - np.nanmin(t_obs)) if len(t_obs) else np.nan
        out["duty_cycle"] = float((t90 - t10) / total_span) if total_span and total_span > 0 else np.nan
    else:
        out["robust_duration"] = np.nan
        out["duty_cycle"] = np.nan

    # predict smooth curve in morphology band and extract morphology + TDE-shape features
    t_grid = np.linspace(np.nanmin(t_obs), np.nanmax(t_obs), grid_points).astype(float) # making the time grid
    y_grid, _ = gp_predict(gp_res, t_grid, morphology_band, return_std=False) # predicting the smooth curve

    # morphology
    out.update(_basic_morphology_features(t_grid, y_grid)) # updating out table with morphology features

    # TDE-shape features
    out["template_chisq_tde"] = _template_chisq_tde(t_grid, y_grid)
    out["tde_power_law_error"] = _tde_powerlaw_error(t_grid, y_grid)
    out["log_tde_error"] = (
        float(np.log10(out["tde_power_law_error"] + 1e-12))
        if np.isfinite(out["tde_power_law_error"])
        else np.nan
    )

    # computing skew/kurtosis on (corrected/raw) flux points
    y_for_stats = y_obs[np.isfinite(y_obs)] # removing NaNs
    if y_for_stats.size >= 8:
        out["flux_skew"] = float(skew(y_for_stats))
        out["flux_kurtosis"] = float(kurtosis(y_for_stats, fisher=True))
    else:
        out["flux_skew"] = np.nan
        out["flux_kurtosis"] = np.nan

    # finding peak colors using GP at peak time (in morphology band)
    # finding peak color time
    peak_idx = int(np.nanargmax(y_grid)) if np.isfinite(y_grid).any() else 0
    t_peak = float(t_grid[peak_idx])

    # helper function that predicts flux at t_peak
    def pred_at_peak(band):
        yp, _ = gp_predict(gp_res, np.array([t_peak], dtype=float), band, return_std=False)
        return float(yp[0])

    try:
        fu = pred_at_peak("u")
        fg = pred_at_peak("g")
        fr = pred_at_peak("r")

        # flux differences between filters
        out["ug_peak"] = float(fu - fg)
        out["gr_peak"] = float(fg - fr)
        out["ur_peak"] = float(fu - fr)
    except Exception:
        out["ug_peak"] = np.nan
        out["gr_peak"] = np.nan
        out["ur_peak"] = np.nan

    # describing color evolution summary: sample 5 points from peak to peak+fade_time
    fade_time = out.get("fade_time", np.nan)
    if np.isfinite(fade_time) and fade_time > 0:
        t_samples = np.linspace(t_peak, t_peak + fade_time, 5).astype(float) # creating sample times
        try:
            g_s, _ = gp_predict(gp_res, t_samples, "g", return_std=False)
            r_s, _ = gp_predict(gp_res, t_samples, "r", return_std=False)
            gr = (g_s - r_s)

            out["mean_color_gr"] = float(np.nanmean(gr))
            out["std_color_gr"] = float(np.nanstd(gr))
            out["color_cooling_rate"] = float(gr[len(gr) // 2] - gr[0])
        except Exception:
            out["mean_color_gr"] = np.nan
            out["std_color_gr"] = np.nan
            out["color_cooling_rate"] = np.nan
    else:
        out["mean_color_gr"] = np.nan
        out["std_color_gr"] = np.nan
        out["color_cooling_rate"] = np.nan

    # describing rest-frame conversion (divide time features by (1+z)) - incorprating redshift for time correction
    z = out.get("redshift", np.nan)
    if np.isfinite(z) and z > -0.5:
        out["rise_time_rest"] = float(out.get("rise_time", np.nan) / (1.0 + z)) if np.isfinite(out.get("rise_time", np.nan)) else np.nan
        out["fade_time_rest"] = float(out.get("fade_time", np.nan) / (1.0 + z)) if np.isfinite(out.get("fade_time", np.nan)) else np.nan
        out["fwhm_rest"] = float(out.get("fwhm", np.nan) / (1.0 + z)) if np.isfinite(out.get("fwhm", np.nan)) else np.nan
        out["robust_duration_rest"] = float(out.get("robust_duration", np.nan) / (1.0 + z)) if np.isfinite(out.get("robust_duration", np.nan)) else np.nan
    else:
        out["rise_time_rest"] = np.nan
        out["fade_time_rest"] = np.nan
        out["fwhm_rest"] = np.nan
        out["robust_duration_rest"] = np.nan

# Creating function to extacts features from all objects
def build_feature_table(
    df_lc,
    df_meta=None,
    object_ids=None,
    n_jobs=-1,
    gp_n_restarts=0,
):
    """
    Build a one-row-per-object feature table from a long-format lightcurve dataframe.
    """
    if object_ids is None:
        object_ids = df_lc["object_id"].dropna().unique()

    feats = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(extract_features_for_object)(
            int(oid),
            df_lc,
            df_meta,
            gp_n_restarts=gp_n_restarts,
        )
        for oid in object_ids
    )
    return pd.DataFrame(feats)