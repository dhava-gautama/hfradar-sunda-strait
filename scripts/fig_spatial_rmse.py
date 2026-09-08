#!/usr/bin/env python3
"""Generate spatial RMSE map figure for paper12 with proper geographic coordinates."""

import json
import numpy as np
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PAPER_DIR = os.path.dirname(SCRIPT_DIR)
PROJECT_DIR = os.path.dirname(PAPER_DIR)
FIG_DIR = os.path.join(PAPER_DIR, "figure")
RESULTS_FILE = os.path.join(PAPER_DIR, "output", "extras_results.json")
DATA_FILE = os.path.join(PROJECT_DIR, "data", "processed", "BADA_hourly_qc.nc")
DPI = 600

# Load coordinates
ds = xr.open_dataset(DATA_FILE)
lon = ds["longitude"].values
lat = ds["latitude"].values
U = ds["U"].values
ds.close()

land_mask = np.all(np.isnan(U), axis=0)
sea_mask = ~land_mask

# Create 2D coordinate arrays for pcolormesh (need edges, not centers)
dlon = np.mean(np.diff(lon))
dlat = np.mean(np.diff(lat))
lon_edges = np.concatenate([lon - dlon/2, [lon[-1] + dlon/2]])
lat_edges = np.concatenate([lat - dlat/2, [lat[-1] + dlat/2]])

with open(RESULTS_FILE) as f:
    data = json.load(f)

models = ["CNN", "GRU", "CNN-GRU"]

# ── Spatial RMSE figure ──────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 3, figsize=(12, 8))

for col, name in enumerate(models):
    for row, comp in enumerate(["u", "v"]):
        ax = axes[row, col]
        rmse = np.array(data[name][f"cell_rmse_{comp}"])
        rmse_masked = np.ma.masked_where(~sea_mask, rmse)

        vmin = 5 if comp == "u" else 8
        vmax = 20 if comp == "u" else 25

        im = ax.pcolormesh(lon_edges, lat_edges, rmse_masked,
                           cmap="YlOrRd", vmin=vmin, vmax=vmax, shading="flat")
        ax.set_aspect("equal")

        comp_label = "$U$" if comp == "u" else "$V$"
        if row == 0:
            ax.set_title(f"{name}", fontsize=12)
        if col == 0:
            ax.set_ylabel(f"{comp_label} component\nLatitude (°S)", fontsize=10)
        else:
            ax.set_yticklabels([])

        if row == 1:
            ax.set_xlabel("Longitude (°E)", fontsize=10)
        else:
            ax.set_xticklabels([])

        ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        ax.tick_params(labelsize=8)

        # Reduce tick density
        ax.set_xticks(lon[::5])
        ax.set_yticks(lat[::5])

        cb = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
        cb.set_label("RMSE (cm s$^{-1}$)", fontsize=9)

for idx, ax in enumerate(axes.flat):
    letter = chr(ord('a') + idx)
    ax.text(0.03, 0.95, f"({letter})", transform=ax.transAxes,
            fontsize=10, fontweight="bold", va="top",
            bbox=dict(boxstyle="round,pad=0.15", fc="white", alpha=0.7, ec="none"))

plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "fig_spatial_rmse.png"), dpi=DPI, bbox_inches="tight")
plt.close()
print("Saved fig_spatial_rmse.png")

# ── Spatial correlation figure ───────────────────────────────────────────────

fig2, axes2 = plt.subplots(2, 3, figsize=(12, 8))

for col, name in enumerate(models):
    for row, comp in enumerate(["u", "v"]):
        ax = axes2[row, col]
        corr = np.array(data[name][f"cell_corr_{comp}"])
        corr_masked = np.ma.masked_where(~sea_mask, corr)

        im = ax.pcolormesh(lon_edges, lat_edges, corr_masked,
                           cmap="RdYlGn", vmin=0.85, vmax=1.0, shading="flat")
        ax.set_aspect("equal")

        comp_label = "$U$" if comp == "u" else "$V$"
        if row == 0:
            ax.set_title(f"{name}", fontsize=12)
        if col == 0:
            ax.set_ylabel(f"{comp_label} component\nLatitude (°S)", fontsize=10)
        else:
            ax.set_yticklabels([])

        if row == 1:
            ax.set_xlabel("Longitude (°E)", fontsize=10)
        else:
            ax.set_xticklabels([])

        ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        ax.tick_params(labelsize=8)
        ax.set_xticks(lon[::5])
        ax.set_yticks(lat[::5])

        cb = fig2.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
        cb.set_label("Correlation ($r$)", fontsize=9)

for idx, ax in enumerate(axes2.flat):
    letter = chr(ord('a') + idx)
    ax.text(0.03, 0.95, f"({letter})", transform=ax.transAxes,
            fontsize=10, fontweight="bold", va="top",
            bbox=dict(boxstyle="round,pad=0.15", fc="white", alpha=0.7, ec="none"))

plt.tight_layout()
fig2.savefig(os.path.join(FIG_DIR, "fig_spatial_corr.png"), dpi=DPI, bbox_inches="tight")
plt.close()
print("Saved fig_spatial_corr.png")
