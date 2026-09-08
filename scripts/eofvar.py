#!/usr/bin/env python3
"""
Paper 1.2 (ASCMO-2026-18) — Reduced-rank spatio-temporal statistical baseline.

Referee 1 (comment 6) asked for a benchmark against a *thoughtful spatio-temporal
statistical method* that accounts for spatial correlation, rather than only the
pointwise per-cell time-series models (ARIMA, kNN). We implement an
EOF / principal-component vector autoregression (PC-VAR), a standard reduced-rank
dynamic spatio-temporal model (Cressie & Wikle, 2011; Wikle et al., 2019):

  1. Stack the (U,V) field over the 291 sea cells into a data matrix; centre on the
     training mean.
  2. Empirical orthogonal functions (EOFs) via truncated SVD of the training matrix;
     keep the leading K modes (spatial covariance structure).
  3. Fit a vector autoregression VAR(p) on the K principal-component time series
     (joint temporal dynamics across modes), order p by AIC.
  4. One-step-ahead forecast of the PCs over the test period using the fitted VAR and
     the true lagged PCs (analogous to the one-step protocol for the other methods),
     reconstruct the field, and score RMSE over the 291 sea cells.

This captures spatial correlation (shared EOFs) and temporal dynamics (VAR) jointly,
unlike the per-cell ARIMA. Same data, sea mask, splits, and test window
([22560:28321], 5,761 steps) as reviewer_response.py.

Output -> paper12/output/reviewer/eofvar_results.json
"""
import json
import os

import numpy as np
import xarray as xr
from statsmodels.tsa.api import VAR

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
DATA = os.path.join(REPO, "data/processed/BADA_hourly_qc.nc")
OUT = os.path.join(REPO, "paper12/output/reviewer")
TRAIN_END = np.datetime64("2025-01-01")
VAL_END = np.datetime64("2025-07-01")
TEST_END_IDX = 28321


def load():
    ds = xr.open_dataset(DATA)
    U = np.nan_to_num(ds["U"].values.astype(np.float64), nan=0.0)
    V = np.nan_to_num(ds["V"].values.astype(np.float64), nan=0.0)
    t = ds["time"].values
    ds.close()
    sea = np.mean(np.isfinite(xr.open_dataset(DATA)["U"].values), axis=0) > 0.5
    it = int(np.searchsorted(t, TRAIN_END))
    iv = int(np.searchsorted(t, VAL_END))
    return U, V, sea, it, iv


def run(K_var=0.95, maxlags=6):
    U, V, sea, it, iv = load()
    te = TEST_END_IDX
    ncell = int(sea.sum())
    # field matrix X: (T, 2*ncell) = [U over sea cells | V over sea cells]
    Xall = np.concatenate([U[:, sea], V[:, sea]], axis=1)      # (T, 582)
    mean = Xall[:it].mean(axis=0)
    Xc = Xall - mean

    # EOFs from training period via SVD
    Xtr = Xc[:it]
    # economy SVD: Xtr = Us S Vt ; columns of Vt.T are EOFs (spatial), Us*S are PCs
    U_s, S, Vt = np.linalg.svd(Xtr, full_matrices=False)
    var_explained = np.cumsum(S ** 2) / np.sum(S ** 2)
    if K_var < 1:
        K = int(np.searchsorted(var_explained, K_var) + 1)
    else:
        K = int(K_var)
    EOF = Vt[:K].T                                             # (582, K)

    # project the FULL series onto the leading EOFs -> PC time series
    PC = Xc @ EOF                                              # (T, K)

    # fit VAR(p) on training PCs, order by AIC
    model = VAR(PC[:it])
    sel = model.select_order(maxlags=maxlags)
    p = int(sel.aic) if sel.aic and sel.aic >= 1 else 1
    res = model.fit(p)

    # one-step-ahead forecast of PCs over the test window using TRUE lagged PCs
    pred_pc = np.empty((te - iv, K))
    coefs = res.coefs                # (p, K, K)
    intercept = res.intercept        # (K,)
    for i, t_idx in enumerate(range(iv, te)):
        x = intercept.copy()
        for lag in range(1, p + 1):
            x = x + coefs[lag - 1] @ PC[t_idx - lag]
        pred_pc[i] = x

    # reconstruct field and score
    Xrec = pred_pc @ EOF.T + mean                             # (n_test, 582)
    Xtrue = Xall[iv:te]
    half = ncell
    ru = float(np.sqrt(np.mean((Xrec[:, :half] - Xtrue[:, :half]) ** 2)))
    rv = float(np.sqrt(np.mean((Xrec[:, half:] - Xtrue[:, half:]) ** 2)))

    # persistence reference on same window/cells
    Xpers = Xall[iv - 1:te - 1]
    mse_pu = np.mean((Xpers[:, :half] - Xtrue[:, :half]) ** 2)
    mse_pv = np.mean((Xpers[:, half:] - Xtrue[:, half:]) ** 2)
    ss_u = 1 - ru ** 2 / mse_pu
    ss_v = 1 - rv ** 2 / mse_pv

    out = {"method": "EOF-VAR (PC-VAR)", "n_eofs": K,
           "var_explained_by_K": float(var_explained[K - 1]),
           "var_order_p": p, "n_sea": ncell, "n_test_steps": int(te - iv),
           "rmse_u": ru, "rmse_v": rv, "ss_u": float(ss_u), "ss_v": float(ss_v)}
    print(f"  EOF-VAR: K={K} EOFs ({var_explained[K-1]*100:.1f}% var), VAR(p={p})")
    print(f"  RMSE U={ru:.2f} (SS={ss_u:.3f})  V={rv:.2f} (SS={ss_v:.3f})  cm/s")
    print(f"  cf. ARIMA(1,0,1) 14.95/18.95, CNN-GRU 11.33/15.44, persistence 16.03/20.22")

    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "eofvar_results.json"), "w") as f:
        json.dump(out, f, indent=2, default=float)
    return out


if __name__ == "__main__":
    run()
