#!/usr/bin/env python3
"""R2-3b: ARIMA tidal-vs-residual skill decomposition (ASCMO referee 2).

The referee asked us to TEST, not assert, that ARIMA's one-step skill comes from
tidal autocorrelation rather than from modelling sub-tidal dynamics. We test it
directly and non-circularly: for each sea cell we split the observed series into a
UTide tidal component (fitted on the training period) and a non-tidal residual,
then fit the SAME ARIMA(1,0,1) used in the manuscript to the RESIDUAL series and
score one-step-ahead predictions over the full 5,761-step test period.

Logic:
  * Tidal band: the tide is so strongly autocorrelated at a 1 h lag that
    persistence already tracks it (persistence tidal RMSE is tiny). We report the
    tidal variance fraction and the persistence tidal-tracking error -- no ARIMA
    fit is needed (and an ARMA(1,1) on a pure multi-sinusoid is ill-posed).
  * Residual band: if ARIMA's skill were due to modelling sub-tidal dynamics, an
    ARIMA fitted to the residual would beat persistence-on-the-residual. We show
    it does NOT (skill ~ 0), so ARIMA's total skill (SS ~ 0.13/0.12, cached
    full-domain run) derives wholly from the tidal autocorrelation.

Identical data/mask/splits/order as reviewer_response.py. CPU only.
"""
import os
import json
import warnings
from datetime import datetime
from multiprocessing import Pool

import numpy as np

warnings.filterwarnings("ignore")

_ROOT = os.environ.get("RADAR_ROOT", "/mnt/hdd/rocket/radarMaritim")
DATA_PATH = os.path.join(_ROOT, "data/processed/BADA_hourly_qc.nc")
OUT = os.path.join(_ROOT, "paper12/output/reviewer/arima_tidal_decomp.json")
RAW_CACHE = os.path.join(_ROOT, "paper12/output/reviewer/arima_full_pred.npz")
TRAIN_END = datetime(2025, 1, 1)
VAL_END = datetime(2025, 7, 1)
TEST_END_IDX = 28321
ORDER = (1, 0, 1)
CONSTIT = ["M2", "S2", "K1", "O1", "N2", "K2", "P1", "M4"]
NPROC = int(os.environ.get("NPROC", "12"))


def load_data():
    import xarray as xr
    ds = xr.open_dataset(DATA_PATH)
    U = ds["U"].values.astype(np.float32)
    V = ds["V"].values.astype(np.float32)
    times = ds["time"].values
    ds.close()
    sea = np.mean(np.isfinite(U), axis=0) > 0.5
    U = np.nan_to_num(U, nan=0.0)
    V = np.nan_to_num(V, nan=0.0)
    it = int(np.searchsorted(times, np.datetime64(TRAIN_END)))
    iv = int(np.searchsorted(times, np.datetime64(VAL_END)))
    te = min(TEST_END_IDX, len(times))
    return U, V, times, sea, it, iv, te


def _arima_resid_onestep(series, it, iv, te):
    """Fit ARIMA on residual[:it]; one-step-ahead over [iv:te], fixed params."""
    from statsmodels.tsa.arima.model import ARIMA
    try:
        fit = ARIMA(series[:it], order=ORDER,
                    enforce_stationarity=False,
                    enforce_invertibility=False).fit(method_kwargs={"maxiter": 50})
        res = fit.apply(series[:te])
        return np.asarray(res.predict(start=iv, end=te - 1), dtype=np.float64)
    except Exception:
        return None


def worker(task):
    (yi, xi), u_full, v_full, train_mpl, full_mpl, it, iv, te = task
    from utide import solve, reconstruct
    out = {}
    for comp, s in (("u", u_full), ("v", v_full)):
        valid = np.isfinite(s[:it])
        if valid.sum() < 1000:
            out[comp] = None
            continue
        try:
            coef = solve(train_mpl[valid], s[:it][valid].astype(np.float64),
                         lat=-6.0, constit=CONSTIT, method="ols",
                         conf_int="none", verbose=False)
            tide = np.asarray(reconstruct(full_mpl, coef, verbose=False).h, dtype=np.float64)
        except Exception:
            out[comp] = None
            continue
        obs = s[:te].astype(np.float64)
        resid = obs - tide
        rp = _arima_resid_onestep(resid, it, iv, te)
        if rp is None:
            out[comp] = None
            continue
        # test-window slices
        obs_t = obs[iv:te]
        tide_t, tide_p = tide[iv:te], tide[iv - 1:te - 1]
        res_t, res_p = resid[iv:te], resid[iv - 1:te - 1]
        m = np.isfinite(rp)
        out[comp] = {
            "n": int(m.sum()),
            "sse_arima_resid": float(np.sum((rp[m] - res_t[m]) ** 2)),
            "sse_pers_resid": float(np.sum((res_p[m] - res_t[m]) ** 2)),
            "sse_pers_tide": float(np.sum((tide_p[m] - tide_t[m]) ** 2)),
            "var_tide": float(np.sum((tide_t[m] - tide_t[m].mean()) ** 2)),
            "var_resid": float(np.sum((res_t[m] - res_t[m].mean()) ** 2)),
            "var_obs": float(np.sum((obs_t[m] - obs_t[m].mean()) ** 2)),
        }
    return (yi, xi), out


def raw_arima_rmse(U, V, sea, iv, te):
    """Aggregate raw-ARIMA test RMSE from the cached full-domain predictions."""
    if not os.path.exists(RAW_CACHE):
        return None
    d = np.load(RAW_CACHE)
    pu, pv = d["pred_u"], d["pred_v"]          # (5761, 21, 21)
    tu, tv = U[iv:te][:, sea], V[iv:te][:, sea]
    pu, pv = pu[:, sea], pv[:, sea]
    return {"u": float(np.sqrt(np.nanmean((pu - tu) ** 2))),
            "v": float(np.sqrt(np.nanmean((pv - tv) ** 2)))}


def main():
    import matplotlib.dates as mdates
    U, V, times, sea, it, iv, te = load_data()
    print(f"sea cells {int(sea.sum())}  train[:{it}] test[{iv}:{te}]={te - iv} steps")

    train_mpl = mdates.date2num([np.datetime64(t, "s").astype(datetime) for t in times[:it]])
    full_mpl = mdates.date2num([np.datetime64(t, "s").astype(datetime) for t in times[:te]])
    cells = list(zip(*np.where(sea)))
    tasks = [((yi, xi), U[:te, yi, xi].copy(), V[:te, yi, xi].copy(),
              train_mpl, full_mpl, it, iv, te) for (yi, xi) in cells]
    print(f"dispatching {len(tasks)} cells on {NPROC} procs (ARIMA on residual only)...")

    keys = ["n", "sse_arima_resid", "sse_pers_resid", "sse_pers_tide",
            "var_tide", "var_resid", "var_obs"]
    agg = {c: {k: 0.0 for k in keys} for c in ("u", "v")}
    done = 0
    with Pool(NPROC) as pool:
        for (yi, xi), res in pool.imap_unordered(worker, tasks, chunksize=2):
            done += 1
            if done % 40 == 0:
                print(f"  {done}/{len(tasks)}")
            for c in ("u", "v"):
                r = res.get(c)
                if not r:
                    continue
                for k in keys:
                    agg[c][k] += r[k]

    raw = raw_arima_rmse(U, V, sea, iv, te)
    out = {"order": list(ORDER), "constituents": CONSTIT,
           "test_steps": te - iv, "n_cells": int(sea.sum()),
           "raw_arima_total_rmse": raw}
    for c in ("u", "v"):
        a = agg[c]
        n = a["n"]
        out[c] = {
            "tidal_variance_fraction": round(a["var_tide"] / (a["var_tide"] + a["var_resid"]), 4),
            "persistence_tidal_rmse": round((a["sse_pers_tide"] / n) ** 0.5, 3),
            "residual_rms": round((a["var_resid"] / n) ** 0.5, 3),
            "arima_residual_rmse": round((a["sse_arima_resid"] / n) ** 0.5, 3),
            "persistence_residual_rmse": round((a["sse_pers_resid"] / n) ** 0.5, 3),
            "arima_residual_skill_vs_persistence": round(1 - a["sse_arima_resid"] / a["sse_pers_resid"], 4),
            "arima_residual_frac_var_explained": round(1 - a["sse_arima_resid"] / a["var_resid"], 4),
            "n": int(n),
        }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
