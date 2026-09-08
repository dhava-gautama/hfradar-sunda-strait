#!/usr/bin/env python3
"""
Figures for the ASCMO-2026-18 revision (referee response).

  fig_gap_fraction.png  — per-cell gap-filled fraction of test targets (R2-4)
  fig_acf_pacf.png      — ACF/PACF of representative cell series (R1-1)
  fig_arima_spatial.png — ARIMA vs CNN-GRU per-cell RMSE maps, same pattern (R2-minor2)

Reads cached arrays from paper12/output/reviewer/ and paper12/output/extras_results.json.
Writes 600-dpi PNGs to paper12/figure/.
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import xarray as xr

HERE = os.path.dirname(os.path.abspath(__file__))
P12 = os.path.dirname(HERE)
REPO = os.path.dirname(P12)
REV = os.path.join(P12, "output/reviewer")
FIGDIR = os.path.join(P12, "figure")
os.makedirs(FIGDIR, exist_ok=True)

ds = xr.open_dataset(os.path.join(REPO, "data/processed/BADA_hourly_qc.nc"))
lon = ds["longitude"].values
lat = ds["latitude"].values
ds.close()


def _mask(a, sea):
    return np.where(sea, a, np.nan)


def fig_gap_fraction():
    z = np.load(os.path.join(REV, "gap_fraction_map.npz"))
    gf, sea = z["gap_frac"], z["sea_mask"].astype(bool)
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    pm = ax.pcolormesh(lon, lat, _mask(gf, sea) * 100, cmap="YlOrRd",
                       vmin=0, vmax=100, shading="auto")
    cb = fig.colorbar(pm, ax=ax, label="gap-filled fraction of test targets (\\%)")
    ax.set_xlabel("Longitude ($^\\circ$E)"); ax.set_ylabel("Latitude ($^\\circ$S)")
    ax.set_title("Test-period gap-filled fraction per cell")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "fig_gap_fraction.png"), dpi=600)
    plt.close(fig)
    print("wrote fig_gap_fraction.png  (overall %.1f%%)" % (np.nanmean(_mask(gf, sea)) * 100))


def fig_acf_pacf():
    z = np.load(os.path.join(REV, "acf_pacf.npz"))
    # one representative cell, U and V
    keyU = [k for k in z.files if k.startswith("U_")][0]
    keyV = "V_" + keyU[2:]
    lags = np.arange(z[keyU].shape[1])
    fig, axes = plt.subplots(2, 2, figsize=(8, 5), sharex=True)
    conf = 1.96 / np.sqrt(18216)
    for col, (comp, key) in enumerate([("U", keyU), ("V", keyV)]):
        acf, pacf = z[key][0], z[key][1]
        for row, (name, arr) in enumerate([("ACF", acf), ("PACF", pacf)]):
            ax = axes[row, col]
            ax.bar(lags, arr, width=0.4, color="C0")
            ax.axhline(0, color="k", lw=0.6)
            ax.axhline(conf, color="r", ls="--", lw=0.6)
            ax.axhline(-conf, color="r", ls="--", lw=0.6)
            ax.set_title(f"{comp} component {name}")
            if row == 1:
                ax.set_xlabel("lag (h)")
    axes[0, 0].set_ylabel("correlation"); axes[1, 0].set_ylabel("partial corr.")
    fig.suptitle(f"Autocorrelation structure, representative cell ({keyU[2:]})")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "fig_acf_pacf.png"), dpi=600)
    plt.close(fig)
    print("wrote fig_acf_pacf.png")


def fig_arima_spatial():
    z = np.load(os.path.join(REV, "arima_full_pred.npz"))
    extras = json.load(open(os.path.join(P12, "output/extras_results.json")))
    sea = np.isfinite(z["cell_rmse_v"])
    a_v = z["cell_rmse_v"]
    d_v = np.array(extras["CNN-GRU"]["cell_rmse_v"])
    vmax = np.nanpercentile(np.concatenate([a_v[sea], d_v[sea]]), 98)
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, data, title in [(axes[0], a_v, "ARIMA(1,0,1)"),
                            (axes[1], d_v, "CNN-GRU")]:
        pm = ax.pcolormesh(lon, lat, _mask(data, sea), cmap="viridis",
                           vmin=0, vmax=vmax, shading="auto")
        ax.set_title(f"{title}: per-cell RMSE$_V$")
        ax.set_xlabel("Longitude ($^\\circ$E)")
        fig.colorbar(pm, ax=ax, label="RMSE$_V$ (cm\\,s$^{-1}$)")
    axes[0].set_ylabel("Latitude ($^\\circ$S)")
    # correlation annotation
    r = np.corrcoef(a_v[sea], d_v[sea])[0, 1]
    fig.suptitle(f"Spatial error pattern: ARIMA vs CNN-GRU (r = {r:.2f})")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "fig_arima_spatial.png"), dpi=600)
    plt.close(fig)
    print("wrote fig_arima_spatial.png  (r=%.3f)" % r)


if __name__ == "__main__":
    fig_gap_fraction()
    fig_acf_pacf()
    fig_arima_spatial()
    print("done ->", FIGDIR)
