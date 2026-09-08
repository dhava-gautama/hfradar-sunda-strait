#!/usr/bin/env python3
"""
Paper 1.2 (ASCMO-2026-18) — Reviewer response analyses.

Addresses the two referee reports by computing the statistical / diagnostic
evidence the reviewers asked for. Everything here is CPU-only (statsmodels +
numpy) and uses the SAME data, sea mask, and chronological splits as
train_paper12.py so the numbers are directly comparable to the submitted
manuscript (Table 2 / Phase 1).

Tasks (select with --task, default = all):
  validate   R2.3a  Replicate the submitted ARIMA (30 cells, 720 test steps) to
                    confirm we reproduce the manuscript's 12.14 / 14.10 cm/s.
  arima_full R2.3a, R2.minor2
                    Re-run ARIMA(1,0,1) on ALL 291 sea cells over the FULL
                    5,761-step test period -> overall RMSE, skill score, and a
                    per-cell spatial RMSE map.
  diagnostics R1.1, R1.3
                    ADF stationarity test, AIC order-selection grid (justifies
                    the uniform (1,0,1) order), Ljung-Box residual
                    autocorrelation test, and residual normality (Jarque-Bera)
                    on a representative 30-cell sample. Also saves ACF/PACF
                    arrays for a supplementary figure.
  ets        R1.4   Per-cell simple exponential smoothing (SES) one-step
                    forecast as an additional classical baseline.
  gap        R2.4   Gap-fill contamination analysis: per-cell gap fraction map
                    + RMSE split into gap-filled vs genuinely-observed test
                    targets (persistence and ARIMA).
  vector     R2.minor1
                    Vector-aware error metrics (speed RMSE, complex/vector RMSE,
                    direction MAE) for persistence and ARIMA.

Outputs land in paper12/output/reviewer/.

Usage:
  python reviewer_response.py --task all   [--workers 12]
  python reviewer_response.py --task arima_full
  python reviewer_response.py --task validate
"""

import argparse
import json
import os
import time
import warnings
from datetime import datetime
from multiprocessing import Pool

import numpy as np
import xarray as xr

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG — matches train_paper12.py
# ─────────────────────────────────────────────────────────────────────────────

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
DATA_PATH = os.path.join(REPO, "data/processed/BADA_hourly_qc.nc")
OUT_DIR = os.path.join(REPO, "paper12/output/reviewer")
EXTRAS_JSON = os.path.join(REPO, "paper12/output/extras_results.json")
os.makedirs(OUT_DIR, exist_ok=True)

TRAIN_END = datetime(2025, 1, 1)   # Dec 2022 – Dec 2024 training
VAL_END = datetime(2025, 7, 1)     # Jan – Jun 2025 validation
# Manuscript test window: Jul 2025 – Feb 2026 = 5,761 steps (idx 22560:28321).
# The file extends to 24 Mar 2026 (28,967 steps); we cap at TEST_END_IDX so the
# numbers line up with the submitted tables.
TEST_END_IDX = 28321

ARIMA_ORDER = (1, 0, 1)
N_SAMPLE = 30          # representative cells for diagnostics / validation
SAMPLE_SEED = 42
CONSTITS = ["M2", "S2", "K1", "O1", "N2", "K2", "P1", "M4"]


# ─────────────────────────────────────────────────────────────────────────────
# DATA
# ─────────────────────────────────────────────────────────────────────────────

def load_data():
    print(f"Loading {DATA_PATH} ...")
    ds = xr.open_dataset(DATA_PATH)
    U = ds["U"].values.astype(np.float64)         # (time, 21, 21), gap-filled, cm/s
    V = ds["V"].values.astype(np.float64)
    qc = ds["qc_flag"].values                      # 0=obs, 1=QC-removed(filled), 2=gap(filled)
    times = ds["time"].values
    lat = ds["latitude"].values
    lon = ds["longitude"].values
    ds.close()

    valid_frac = np.mean(np.isfinite(U), axis=0)
    sea_mask = valid_frac > 0.5                    # 291 sea cells
    U = np.nan_to_num(U, nan=0.0)
    V = np.nan_to_num(V, nan=0.0)

    idx_train = int(np.searchsorted(times, np.datetime64(TRAIN_END)))
    idx_test = int(np.searchsorted(times, np.datetime64(VAL_END)))
    test_end = min(TEST_END_IDX, len(times))

    print(f"  grid {U.shape[1]}x{U.shape[2]}, sea cells {int(sea_mask.sum())}, "
          f"timesteps {len(times)}")
    print(f"  train [:{idx_train}]  test [{idx_test}:{test_end}] "
          f"({test_end - idx_test} steps)")

    return dict(U=U, V=V, qc=qc, times=times, lat=lat, lon=lon,
                sea_mask=sea_mask, idx_train=idx_train, idx_test=idx_test,
                test_end=test_end, ny=U.shape[1], nx=U.shape[2])


def sea_cells(data):
    """List of (yi, xi) for the 291 sea cells, in row-major order."""
    return list(zip(*np.where(data["sea_mask"])))


def sample_cells(data, n=N_SAMPLE, seed=SAMPLE_SEED):
    """Same RandomState selection the manuscript used for the 30-cell subsample."""
    cells = sea_cells(data)
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(cells), min(n, len(cells)), replace=False)
    return [cells[i] for i in idx]


def rmse(pred, true):
    return float(np.sqrt(np.mean((np.asarray(pred) - np.asarray(true)) ** 2)))


# ─────────────────────────────────────────────────────────────────────────────
# ARIMA workers (module-level for multiprocessing)
# ─────────────────────────────────────────────────────────────────────────────

def _arima_onestep(series, idx_train, idx_test, test_end, order=ARIMA_ORDER):
    """Fit ARIMA on train, produce one-step-ahead predictions over the test
    window using fixed (train-estimated) parameters. Mirrors train_paper12.py:
    fit on [:idx_train], .apply over [:test_end], .predict(idx_test:test_end).
    Returns (pred_test, resid_train) or (None, None) on failure."""
    from statsmodels.tsa.arima.model import ARIMA
    try:
        fit = ARIMA(series[:idx_train], order=order).fit()
        res = fit.apply(series[:test_end])
        pred = np.asarray(res.predict(start=idx_test, end=test_end - 1))
        return pred, np.asarray(fit.resid)
    except Exception:
        return None, None


def _worker_cell(task):
    """Fit ARIMA for U and V of one cell. Returns predictions + diagnostics."""
    (yi, xi), u_full, v_full, idx_train, idx_test, test_end = task
    order = ARIMA_ORDER  # read current module global (forked workers inherit it)
    pu, ru = _arima_onestep(u_full, idx_train, idx_test, test_end, order)
    pv, rv = _arima_onestep(v_full, idx_train, idx_test, test_end, order)
    out = {"yi": int(yi), "xi": int(xi)}
    out["pred_u"] = pu.tolist() if pu is not None else None
    out["pred_v"] = pv.tolist() if pv is not None else None
    return out


def _build_tasks(data, cells):
    U, V = data["U"], data["V"]
    it, iv, te = data["idx_train"], data["idx_test"], data["test_end"]
    for (yi, xi) in cells:
        yield ((yi, xi), U[:te, yi, xi].copy(), V[:te, yi, xi].copy(), it, iv, te)


# ─────────────────────────────────────────────────────────────────────────────
# TASK: validate (reproduce submitted ARIMA numbers)
# ─────────────────────────────────────────────────────────────────────────────

def task_validate(data, workers):
    print("\n" + "=" * 70)
    print("VALIDATE — reproduce submitted ARIMA (30 cells, 720 test steps)")
    print("=" * 70)
    cells = sample_cells(data, N_SAMPLE)
    it, iv = data["idx_train"], data["idx_test"]
    te720 = iv + 720
    U, V = data["U"], data["V"]

    tasks = [((yi, xi), U[:te720, yi, xi].copy(), V[:te720, yi, xi].copy(),
              it, iv, te720) for (yi, xi) in cells]
    t0 = time.time()
    with Pool(workers) as pool:
        results = pool.map(_worker_cell, tasks)

    ue, ve = [], []
    for r in results:
        if r["pred_u"] is None:
            continue
        yi, xi = r["yi"], r["xi"]
        tu = U[iv:te720, yi, xi]
        tv = V[iv:te720, yi, xi]
        ue.append(np.mean((np.array(r["pred_u"]) - tu) ** 2))
        ve.append(np.mean((np.array(r["pred_v"]) - tv) ** 2))
    out = {"rmse_u": float(np.sqrt(np.mean(ue))),
           "rmse_v": float(np.sqrt(np.mean(ve))),
           "n_cells": len(ue), "n_test_steps": 720,
           "manuscript_rmse_u": 12.14, "manuscript_rmse_v": 14.10}
    print(f"  reproduced ARIMA: U={out['rmse_u']:.2f}  V={out['rmse_v']:.2f} cm/s "
          f"(manuscript 12.14 / 14.10)  [{time.time()-t0:.0f}s]")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# TASK: arima_full (all 291 cells, full test period)
# ─────────────────────────────────────────────────────────────────────────────

def _persistence_pred(series, idx_test, test_end):
    """One-step persistence: pred(t+1)=obs(t) over [idx_test:test_end]."""
    return series[idx_test - 1:test_end - 1]


def task_arima_full(data, workers, _cache={}):
    print("\n" + "=" * 70)
    print("ARIMA-FULL — all 291 sea cells, full test period")
    print("=" * 70)
    cells = sea_cells(data)
    U, V = data["U"], data["V"]
    iv, te = data["idx_test"], data["test_end"]
    ny, nx = data["ny"], data["nx"]

    t0 = time.time()
    with Pool(workers) as pool:
        results = pool.map(_worker_cell, list(_build_tasks(data, cells)),
                           chunksize=4)
    print(f"  fitted {len(results)} cells in {time.time()-t0:.0f}s")

    # store predictions in a (n_test, 21, 21) array for downstream tasks
    n_test = te - iv
    pred_u = np.full((n_test, ny, nx), np.nan)
    pred_v = np.full((n_test, ny, nx), np.nan)
    cell_rmse_u = np.full((ny, nx), np.nan)
    cell_rmse_v = np.full((ny, nx), np.nan)
    n_fail = 0
    for r in results:
        yi, xi = r["yi"], r["xi"]
        if r["pred_u"] is None:
            n_fail += 1
            continue
        pu = np.array(r["pred_u"]); pv = np.array(r["pred_v"])
        pred_u[:, yi, xi] = pu
        pred_v[:, yi, xi] = pv
        cell_rmse_u[yi, xi] = rmse(pu, U[iv:te, yi, xi])
        cell_rmse_v[yi, xi] = rmse(pv, V[iv:te, yi, xi])

    sm = data["sea_mask"]
    ovr_u = rmse(pred_u[:, sm], U[iv:te][:, sm])
    ovr_v = rmse(pred_v[:, sm], V[iv:te][:, sm])

    # skill score vs one-step persistence over the SAME full window
    pers_u = U[iv - 1:te - 1]
    pers_v = V[iv - 1:te - 1]
    mse_pers_u = np.mean((pers_u[:, sm] - U[iv:te][:, sm]) ** 2)
    mse_pers_v = np.mean((pers_v[:, sm] - V[iv:te][:, sm]) ** 2)
    ss_u = 1 - ovr_u ** 2 / mse_pers_u
    ss_v = 1 - ovr_v ** 2 / mse_pers_v

    # cache predictions on disk for gap / vector tasks. The (1,0,1) run keeps
    # the canonical filename (used by gap/vector); other orders get a suffix.
    otag = "" if ARIMA_ORDER == (1, 0, 1) else "_%d%d%d" % ARIMA_ORDER
    np.savez(os.path.join(OUT_DIR, f"arima_full_pred{otag}.npz"),
             pred_u=pred_u.astype(np.float32), pred_v=pred_v.astype(np.float32),
             cell_rmse_u=cell_rmse_u, cell_rmse_v=cell_rmse_v)

    # spatial-pattern correlation against the DL maps (Fig. 15) if available
    spatial_corr = {}
    if os.path.exists(EXTRAS_JSON):
        extras = json.load(open(EXTRAS_JSON))
        am_u = cell_rmse_u[sm]; am_v = cell_rmse_v[sm]
        for m in ("CNN", "GRU", "CNN-GRU"):
            if m in extras and "cell_rmse_u" in extras[m]:
                du = np.array(extras[m]["cell_rmse_u"])[sm]
                dv = np.array(extras[m]["cell_rmse_v"])[sm]
                ok_u = np.isfinite(am_u) & np.isfinite(du)
                ok_v = np.isfinite(am_v) & np.isfinite(dv)
                spatial_corr[m] = {
                    "r_u": float(np.corrcoef(am_u[ok_u], du[ok_u])[0, 1]),
                    "r_v": float(np.corrcoef(am_v[ok_v], dv[ok_v])[0, 1])}

    out = {"order": list(ARIMA_ORDER),
           "rmse_u": ovr_u, "rmse_v": ovr_v, "ss_u": float(ss_u),
           "ss_v": float(ss_v), "n_cells": len(cells) - n_fail,
           "n_failed": n_fail, "n_test_steps": int(n_test),
           "submitted_rmse_u": 12.14, "submitted_rmse_v": 14.10,
           "submitted_ss_u": 0.426, "submitted_ss_v": 0.514,
           "spatial_pattern_corr_vs_dl": spatial_corr,
           "cell_rmse_u": cell_rmse_u.tolist(),
           "cell_rmse_v": cell_rmse_v.tolist()}
    print(f"  FULL ARIMA{ARIMA_ORDER}: U={ovr_u:.2f} (SS={ss_u:.3f})  "
          f"V={ovr_v:.2f} (SS={ss_v:.3f})  cm/s")
    print(f"  submitted (30 cells/720 steps): U=12.14 (0.426)  V=14.10 (0.514)")
    if spatial_corr:
        print("  spatial-pattern corr vs DL maps:")
        for m, c in spatial_corr.items():
            print(f"    {m}: r_u={c['r_u']:.2f}  r_v={c['r_v']:.2f}")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# TASK: diagnostics (ADF, AIC grid, Ljung-Box, normality, ACF/PACF)
# ─────────────────────────────────────────────────────────────────────────────

def _adf_one(series):
    from statsmodels.tsa.stattools import adfuller
    try:
        r = adfuller(series, autolag="AIC")
        return {"stat": float(r[0]), "pvalue": float(r[1]),
                "stationary_5pct": bool(r[1] < 0.05)}
    except Exception:
        return None


def _aic_grid_one(series, idx_train):
    from statsmodels.tsa.arima.model import ARIMA
    best = None
    grid = {}
    for p in (0, 1, 2):
        for d in (0, 1):
            for q in (0, 1, 2):
                try:
                    aic = ARIMA(series[:idx_train], order=(p, d, q)).fit().aic
                    grid[f"{p}{d}{q}"] = float(aic)
                    if best is None or aic < best[1]:
                        best = ((p, d, q), aic)
                except Exception:
                    continue
    return best[0] if best else None, grid


def _diag_worker(task):
    """Parallel diagnostics worker for one cell-component training series:
    ADF test, AIC-best order, and Ljung-Box / normality of ARIMA(1,0,1)."""
    comp, yi, xi, series, idx_train = task
    adf = _adf_one(series[:idx_train])
    best, _ = _aic_grid_one(series, idx_train)
    lb = _ljungbox_normality(series, idx_train)
    return {"comp": comp, "yi": yi, "xi": xi, "adf": adf,
            "aic_best": ("%d%d%d" % best) if best else None, "lb": lb}


def _ljungbox_normality(series, idx_train, order=ARIMA_ORDER):
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.stats.diagnostic import acorr_ljungbox
    from scipy import stats
    try:
        fit = ARIMA(series[:idx_train], order=order).fit()
        resid = np.asarray(fit.resid)[1:]  # drop first (init transient)
        lb = acorr_ljungbox(resid, lags=[10, 20, 50], return_df=True)
        jb_stat, jb_p = stats.jarque_bera(resid)[:2]
        return {
            "lb_p_lag10": float(lb["lb_pvalue"].iloc[0]),
            "lb_p_lag20": float(lb["lb_pvalue"].iloc[1]),
            "lb_p_lag50": float(lb["lb_pvalue"].iloc[2]),
            "lb_resid_white_lag20": bool(lb["lb_pvalue"].iloc[1] > 0.05),
            "jb_pvalue": float(jb_p),
            "resid_skew": float(stats.skew(resid)),
            "resid_kurtosis": float(stats.kurtosis(resid, fisher=True)),
            "resid_std": float(np.std(resid))}
    except Exception:
        return None


def task_diagnostics(data, workers):
    print("\n" + "=" * 70)
    print("DIAGNOSTICS — ADF / AIC grid / Ljung-Box / normality / ACF-PACF")
    print("=" * 70)
    from statsmodels.tsa.stattools import acf, pacf

    cells = sample_cells(data, N_SAMPLE)
    U, V = data["U"], data["V"]
    it = data["idx_train"]

    # Build 60 cell-component tasks (30 cells x U/V) and run ADF + AIC grid +
    # Ljung-Box/normality in parallel.
    tasks = []
    for (yi, xi) in cells:
        for comp, arr in (("U", U), ("V", V)):
            tasks.append((comp, int(yi), int(xi), arr[:it, yi, xi].copy(), it))
    with Pool(workers) as pool:
        recs = pool.map(_diag_worker, tasks)

    adf = {"U": [], "V": []}
    order_counts = {}
    lb = {"U": [], "V": []}
    for r in recs:
        if r["adf"]:
            adf[r["comp"]].append(r["adf"])
        if r["aic_best"]:
            order_counts[r["aic_best"]] = order_counts.get(r["aic_best"], 0) + 1
        if r["lb"]:
            lb[r["comp"]].append(r["lb"])

    adf_summary = {}
    for comp in ("U", "V"):
        ps = [d["pvalue"] for d in adf[comp]]
        stats_ = [d["stat"] for d in adf[comp]]
        adf_summary[comp] = {
            "n": len(ps),
            "n_stationary_5pct": int(sum(p < 0.05 for p in ps)),
            "median_pvalue": float(np.median(ps)),
            "max_pvalue": float(np.max(ps)),
            "median_stat": float(np.median(stats_))}
    print(f"  ADF: U {adf_summary['U']['n_stationary_5pct']}/{adf_summary['U']['n']} "
          f"stationary at 5%; V {adf_summary['V']['n_stationary_5pct']}/"
          f"{adf_summary['V']['n']}")

    total_orders = sum(order_counts.values())
    print(f"  AIC-best orders across {total_orders} cell-series: {order_counts}")

    lb_summary = {}
    for comp in ("U", "V"):
        recs = lb[comp]
        lb_summary[comp] = {
            "n": len(recs),
            "median_lb_p_lag20": float(np.median([r["lb_p_lag20"] for r in recs])),
            "frac_resid_white_lag20": float(np.mean([r["lb_resid_white_lag20"] for r in recs])),
            "median_jb_p": float(np.median([r["jb_pvalue"] for r in recs])),
            "median_resid_kurtosis": float(np.median([r["resid_kurtosis"] for r in recs])),
            "median_resid_skew": float(np.median([r["resid_skew"] for r in recs]))}
    print(f"  Ljung-Box(lag20): U median p={lb_summary['U']['median_lb_p_lag20']:.3f}, "
          f"frac white={lb_summary['U']['frac_resid_white_lag20']:.2f}")

    # ACF/PACF for 4 representative cells (for supplementary figure)
    rep = cells[:4]
    acf_pacf = {}
    for (yi, xi) in rep:
        for comp, arr in (("U", U), ("V", V)):
            s = arr[:it, yi, xi]
            acf_pacf[f"{comp}_{yi}_{xi}"] = {
                "acf": acf(s, nlags=48, fft=True).tolist(),
                "pacf": pacf(s, nlags=48).tolist()}
    np.savez(os.path.join(OUT_DIR, "acf_pacf.npz"),
             **{k: np.array([v["acf"], v["pacf"]]) for k, v in acf_pacf.items()})

    return {"adf": adf_summary, "aic_order_counts": order_counts,
            "aic_total": total_orders, "ljung_box_normality": lb_summary,
            "acf_pacf_cells": list(acf_pacf.keys())}


# ─────────────────────────────────────────────────────────────────────────────
# TASK: ets (simple exponential smoothing one-step baseline)
# ─────────────────────────────────────────────────────────────────────────────

def _ses_alpha(series_train):
    """Optimise SES smoothing level on training data; return alpha."""
    from statsmodels.tsa.holtwinters import SimpleExpSmoothing
    try:
        fit = SimpleExpSmoothing(series_train, initialization_method="estimated").fit()
        return float(fit.params["smoothing_level"])
    except Exception:
        return None


def _ses_onestep(series, alpha, idx_test, test_end):
    """Recursive one-step SES forecast: l_t = a*y_t + (1-a)*l_{t-1};
    pred(t+1)=l_t. Evaluated over [idx_test:test_end]."""
    level = series[0]
    pred = np.empty(test_end)
    for t in range(test_end):
        pred[t] = level                 # forecast for time t = level after t-1
        level = alpha * series[t] + (1 - alpha) * level
    # pred[t] is the one-step-ahead forecast of y[t] given y[:t]
    return pred[idx_test:test_end]


def task_ets(data, workers):
    print("\n" + "=" * 70)
    print("ETS — per-cell simple exponential smoothing one-step baseline")
    print("=" * 70)
    cells = sea_cells(data)
    U, V = data["U"], data["V"]
    it, iv, te = data["idx_train"], data["idx_test"], data["test_end"]

    ue, ve, alphas = [], [], []
    for (yi, xi) in cells:
        su, sv = U[:te, yi, xi], V[:te, yi, xi]
        au = _ses_alpha(su[:it]); av = _ses_alpha(sv[:it])
        if au is None or av is None:
            continue
        pu = _ses_onestep(su, au, iv, te)
        pv = _ses_onestep(sv, av, iv, te)
        ue.append(np.mean((pu - su[iv:te]) ** 2))
        ve.append(np.mean((pv - sv[iv:te]) ** 2))
        alphas.append((au, av))
    out = {"rmse_u": float(np.sqrt(np.mean(ue))),
           "rmse_v": float(np.sqrt(np.mean(ve))),
           "n_cells": len(ue),
           "mean_alpha_u": float(np.mean([a[0] for a in alphas])),
           "mean_alpha_v": float(np.mean([a[1] for a in alphas]))}
    print(f"  SES: U={out['rmse_u']:.2f}  V={out['rmse_v']:.2f} cm/s  "
          f"(mean alpha_u={out['mean_alpha_u']:.2f})")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# TASK: gap (gap-fill contamination analysis)
# ─────────────────────────────────────────────────────────────────────────────

def task_gap(data, workers):
    print("\n" + "=" * 70)
    print("GAP — gap-filled vs observed test targets (persistence + ARIMA)")
    print("=" * 70)
    U, V, qc = data["U"], data["V"], data["qc"]
    iv, te = data["idx_test"], data["test_end"]
    sm = data["sea_mask"]
    ny, nx = data["ny"], data["nx"]

    # target = t+1 over the test window
    qc_target = qc[iv:te]                      # (n_test, 21, 21) flag of the TARGET step
    is_gap = qc_target != 0                    # filled (1) or missing (2) -> synthetic
    true_u = U[iv:te]; true_v = V[iv:te]

    # per-cell gap fraction among test targets (sea cells)
    gap_frac = np.full((ny, nx), np.nan)
    for (yi, xi) in sea_cells(data):
        gap_frac[yi, xi] = float(np.mean(is_gap[:, yi, xi]))

    def split_rmse(pred_u, pred_v):
        m = sm[None, :, :] & np.ones_like(is_gap, dtype=bool)
        gap = is_gap & sm[None, :, :]
        obs = (~is_gap) & sm[None, :, :]
        return {
            "rmse_u_obs": rmse(pred_u[obs], true_u[obs]),
            "rmse_v_obs": rmse(pred_v[obs], true_v[obs]),
            "rmse_u_gap": rmse(pred_u[gap], true_u[gap]),
            "rmse_v_gap": rmse(pred_v[gap], true_v[gap]),
            "n_obs": int(obs.sum()), "n_gap": int(gap.sum())}

    # persistence predictions
    pers_u = U[iv - 1:te - 1]; pers_v = V[iv - 1:te - 1]
    pers_split = split_rmse(pers_u, pers_v)

    result = {"overall_gap_fraction_sea": float(np.mean(is_gap[:, sm])),
              "persistence": pers_split}

    # boundary vs interior gap fraction (edge = cells with < 8 sea neighbours)
    from scipy.ndimage import convolve
    k = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
    nbr = convolve(sm.astype(int), k, mode="constant")
    edge = sm & (nbr < 8)
    result["gap_fraction_edge"] = float(np.mean(is_gap[:, edge])) if edge.any() else None
    result["gap_fraction_interior"] = float(np.mean(is_gap[:, sm & ~edge]))
    result["n_edge_cells"] = int(edge.sum())

    # ARIMA predictions (from cache) if present
    cache = os.path.join(OUT_DIR, "arima_full_pred.npz")
    if os.path.exists(cache):
        z = np.load(cache)
        au = z["pred_u"].astype(np.float64); av = z["pred_v"].astype(np.float64)
        # only evaluate where ARIMA produced finite predictions
        finite = np.isfinite(au)
        au = np.where(finite, au, true_u)   # placeholder; masked out below
        gapA = is_gap & sm[None] & finite
        obsA = (~is_gap) & sm[None] & finite
        result["arima"] = {
            "rmse_u_obs": rmse(z["pred_u"][obsA], true_u[obsA]),
            "rmse_v_obs": rmse(z["pred_v"][obsA], true_v[obsA]),
            "rmse_u_gap": rmse(z["pred_u"][gapA], true_u[gapA]),
            "rmse_v_gap": rmse(z["pred_v"][gapA], true_v[gapA]),
            "n_obs": int(obsA.sum()), "n_gap": int(gapA.sum())}

    np.savez(os.path.join(OUT_DIR, "gap_fraction_map.npz"),
             gap_frac=gap_frac, edge=edge, sea_mask=sm)

    print(f"  test-target gap fraction (sea): {result['overall_gap_fraction_sea']:.1%}")
    print(f"    edge {result['gap_fraction_edge']:.1%} vs interior "
          f"{result['gap_fraction_interior']:.1%}")
    print(f"  persistence  obs U={pers_split['rmse_u_obs']:.2f}  "
          f"gap U={pers_split['rmse_u_gap']:.2f}  "
          f"obs V={pers_split['rmse_v_obs']:.2f}  gap V={pers_split['rmse_v_gap']:.2f}")
    if "arima" in result:
        a = result["arima"]
        print(f"  ARIMA        obs U={a['rmse_u_obs']:.2f}  gap U={a['rmse_u_gap']:.2f}  "
              f"obs V={a['rmse_v_obs']:.2f}  gap V={a['rmse_v_gap']:.2f}")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# TASK: vector (speed / vector / direction error metrics)
# ─────────────────────────────────────────────────────────────────────────────

def _vector_metrics(pu, pv, tu, tv, speed_thresh=10.0):
    """pu,pv,tu,tv are 1-D arrays of matched predictions/truth (cm/s)."""
    pspd = np.sqrt(pu ** 2 + pv ** 2)
    tspd = np.sqrt(tu ** 2 + tv ** 2)
    speed_rmse = float(np.sqrt(np.mean((pspd - tspd) ** 2)))
    speed_bias = float(np.mean(pspd - tspd))
    vec_rmse = float(np.sqrt(np.mean((pu - tu) ** 2 + (pv - tv) ** 2)))
    # direction error only where the true current is non-trivial
    m = tspd > speed_thresh
    if m.sum() > 0:
        dang = np.degrees(np.arctan2(pv[m], pu[m]) - np.arctan2(tv[m], tu[m]))
        dang = (dang + 180) % 360 - 180     # wrap to [-180,180]
        dir_mae = float(np.mean(np.abs(dang)))
        dir_rmse = float(np.sqrt(np.mean(dang ** 2)))
    else:
        dir_mae = dir_rmse = None
    return {"speed_rmse": speed_rmse, "speed_bias": speed_bias,
            "vector_rmse": vec_rmse, "direction_mae_deg": dir_mae,
            "direction_rmse_deg": dir_rmse,
            "n_dir": int(m.sum()), "speed_thresh": speed_thresh}


def task_vector(data, workers):
    print("\n" + "=" * 70)
    print("VECTOR — speed / vector / direction error (persistence + ARIMA)")
    print("=" * 70)
    U, V = data["U"], data["V"]
    iv, te = data["idx_test"], data["test_end"]
    sm = data["sea_mask"]
    true_u = U[iv:te][:, sm].ravel()
    true_v = V[iv:te][:, sm].ravel()

    out = {}
    pers_u = U[iv - 1:te - 1][:, sm].ravel()
    pers_v = V[iv - 1:te - 1][:, sm].ravel()
    out["persistence"] = _vector_metrics(pers_u, pers_v, true_u, true_v)

    cache = os.path.join(OUT_DIR, "arima_full_pred.npz")
    if os.path.exists(cache):
        z = np.load(cache)
        au = z["pred_u"]; av = z["pred_v"]
        finite = np.isfinite(au[:, sm]).all(axis=0)  # cells with full finite preds
        cols = np.where(sm)
        pu = au[:, sm]; pv = av[:, sm]
        fmask = np.isfinite(pu) & np.isfinite(pv)
        out["arima"] = _vector_metrics(pu[fmask], pv[fmask],
                                       U[iv:te][:, sm][fmask],
                                       V[iv:te][:, sm][fmask])
    for k, v in out.items():
        print(f"  {k:12s} speed_RMSE={v['speed_rmse']:.2f}  vec_RMSE={v['vector_rmse']:.2f}  "
              f"dir_MAE={v['direction_mae_deg']:.1f} deg")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# TASK: knn_full (temporal kNN on all 291 cells, full test period)
# ─────────────────────────────────────────────────────────────────────────────

def task_knn_full(data, workers, k=5, T=3):
    """Per-cell temporal kNN (k=5, T=3 lookback) on all sea cells over the full
    test window — the manuscript only reported a 30-cell/720-step estimate."""
    print("\n" + "=" * 70)
    print("KNN-FULL — temporal kNN (k=5, T=3) all 291 cells, full test period")
    print("=" * 70)
    from sklearn.neighbors import KNeighborsRegressor
    U, V = data["U"], data["V"]
    it, iv, te = data["idx_train"], data["idx_test"], data["test_end"]
    sm = data["sea_mask"]

    def knn_cell(series):
        # lookback feature matrix
        X = np.stack([series[i:i + T] for i in range(len(series) - T)])
        y = series[T:]
        Xtr, ytr = X[:it - T], y[:it - T]
        Xte = X[iv - T:te - T]
        yte = y[iv - T:te - T]
        kr = KNeighborsRegressor(n_neighbors=k).fit(Xtr, ytr)
        return kr.predict(Xte), yte

    ue, ve = [], []
    for (yi, xi) in sea_cells(data):
        pu, tu = knn_cell(U[:te, yi, xi]); ue.append(np.mean((pu - tu) ** 2))
        pv, tv = knn_cell(V[:te, yi, xi]); ve.append(np.mean((pv - tv) ** 2))
    ru = float(np.sqrt(np.mean(ue))); rv = float(np.sqrt(np.mean(ve)))
    # skill vs persistence
    pers_u = U[iv:te][:, sm]; pers_v = V[iv:te][:, sm]
    lag_u = U[iv - 1:te - 1][:, sm]; lag_v = V[iv - 1:te - 1][:, sm]
    ss_u = 1 - ru ** 2 / np.mean((lag_u - pers_u) ** 2)
    ss_v = 1 - rv ** 2 / np.mean((lag_v - pers_v) ** 2)
    out = {"rmse_u": ru, "rmse_v": rv, "ss_u": float(ss_u), "ss_v": float(ss_v),
           "n_cells": len(ue), "n_test_steps": int(te - iv),
           "submitted_rmse_u": 14.62, "submitted_rmse_v": 18.48}
    print(f"  FULL kNN: U={ru:.2f} (SS={ss_u:.3f})  V={rv:.2f} (SS={ss_v:.3f})  "
          f"(submitted 14.62 / 18.48)")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# TASK: lb_order (Ljung-Box / normality for a specific ARIMA order)
# ─────────────────────────────────────────────────────────────────────────────

def _lb_worker(task):
    comp, series, it, order = task
    return comp, _ljungbox_normality(series, it, order)


def task_lb_order(data, workers, order=(2, 0, 2)):
    print("\n" + "=" * 70)
    print(f"LB-ORDER — Ljung-Box / normality of ARIMA{order} residuals (30 cells)")
    print("=" * 70)
    cells = sample_cells(data, N_SAMPLE)
    U, V = data["U"], data["V"]
    it = data["idx_train"]
    tasks = []
    for (yi, xi) in cells:
        for comp, arr in (("U", U), ("V", V)):
            tasks.append((comp, arr[:it, yi, xi].copy(), it, order))
    with Pool(workers) as pool:
        recs = pool.map(_lb_worker, tasks)
    lb = {"U": [], "V": []}
    for comp, r in recs:
        if r:
            lb[comp].append(r)
    summ = {"order": list(order)}
    for comp in ("U", "V"):
        recs_c = lb[comp]
        summ[comp] = {
            "n": len(recs_c),
            "median_lb_p_lag20": float(np.median([r["lb_p_lag20"] for r in recs_c])),
            "frac_resid_white_lag20": float(np.mean([r["lb_resid_white_lag20"] for r in recs_c])),
            "median_jb_p": float(np.median([r["jb_pvalue"] for r in recs_c])),
            "median_resid_kurtosis": float(np.median([r["resid_kurtosis"] for r in recs_c]))}
    print(f"  ARIMA{order}: U frac-white(lag20)={summ['U']['frac_resid_white_lag20']:.2f} "
          f"median p={summ['U']['median_lb_p_lag20']:.3f}; "
          f"V frac-white={summ['V']['frac_resid_white_lag20']:.2f}")
    return summ


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

TASKS = {
    "validate": task_validate,
    "arima_full": task_arima_full,
    "diagnostics": task_diagnostics,
    "ets": task_ets,
    "gap": task_gap,
    "vector": task_vector,
    "knn_full": task_knn_full,
    "lb_order": task_lb_order,
}
# arima_full must run before gap / vector (they read its cached predictions)
ORDER = ["validate", "arima_full", "diagnostics", "ets", "gap", "vector"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="all",
                    choices=["all"] + list(TASKS))
    ap.add_argument("--workers", type=int, default=min(12, os.cpu_count()))
    ap.add_argument("--order", default=None,
                    help="ARIMA order as p,d,q (e.g. 2,0,2) for arima_full/lb_order")
    args = ap.parse_args()

    if args.order:
        global ARIMA_ORDER
        ARIMA_ORDER = tuple(int(x) for x in args.order.split(","))
        print(f"ARIMA order set to {ARIMA_ORDER}")

    data = load_data()
    todo = ORDER if args.task == "all" else [args.task]

    results = {}
    res_path = os.path.join(OUT_DIR, "reviewer_results.json")
    if os.path.exists(res_path):
        results = json.load(open(res_path))

    for name in todo:
        t0 = time.time()
        results[name] = TASKS[name](data, args.workers)
        results[name]["_runtime_s"] = round(time.time() - t0, 1)
        with open(res_path, "w") as f:
            json.dump(results, f, indent=2, default=float)
        print(f"  [{name}] saved ({results[name]['_runtime_s']}s)")

    print(f"\nAll results -> {res_path}")


if __name__ == "__main__":
    main()
