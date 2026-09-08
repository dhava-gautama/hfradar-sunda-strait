#!/usr/bin/env python3
"""
Generate all figures for Paper 1.2: Combined one-step + 6h nowcasting.
Reads phase1_results.json, phase2_lookback.json, phase3_multistep.json, and BADA_hourly_qc.nc.
"""

import numpy as np
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import os

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════════════

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PAPER_DIR = os.path.dirname(SCRIPT_DIR)
PROJECT_DIR = os.path.dirname(PAPER_DIR)

P1_FILE = os.path.join(PAPER_DIR, "output", "phase1_results.json")
P2_FILE = os.path.join(PAPER_DIR, "output", "phase2_lookback.json")
P3_FILE = os.path.join(PAPER_DIR, "output", "phase3_multistep.json")
DATA_FILE = os.path.join(PROJECT_DIR, "data", "processed", "BADA_hourly_qc.nc")
FIG_DIR = os.path.join(PAPER_DIR, "figures")
DPI = 600

os.makedirs(FIG_DIR, exist_ok=True)

# Colors
CATCOLORS = {
    "baseline": "#888888",
    "classical": "#5B9BD5",
    "shallow": "#ED7D31",
    "deep": "#70AD47",
}
P3_COLORS = {
    "Persistence": "#888888",
    "ConvLSTM-ED": "#1f77b4",
    "BiEF": "#ff7f0e",
    "CNN-GRU-MS": "#d62728",
}
P3_MARKERS = {
    "Persistence": "s",
    "ConvLSTM-ED": "o",
    "BiEF": "^",
    "CNN-GRU-MS": "v",
}


def load_json(path):
    with open(path) as f:
        return json.load(f)


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 1: STUDY AREA MAP
# ═══════════════════════════════════════════════════════════════════════════════

def fig_study_area():
    """Two-panel study area map."""
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        import xarray as xr
    except ImportError:
        print("  Cartopy/xarray not available, skipping study area map")
        return

    ds = xr.open_dataset(DATA_FILE)
    lon = ds["longitude"].values
    lat = ds["latitude"].values
    ds.close()

    # Load domain masks for proper land/sea/blind classification
    domain_file = os.path.join(PROJECT_DIR, "data", "processed", "paper3_domain.nc")
    dd = xr.open_dataset(domain_file)
    mask_obs = dd["mask_observed"].values.astype(bool)   # 291 observed sea
    mask_blind = dd["mask_target"].values.astype(bool)   # 137 blind zone (sea)
    mask_land = dd["mask_land"].values.astype(bool)      # 13 land
    dd.close()

    lon2d, lat2d = np.meshgrid(lon, lat)

    fig = plt.figure(figsize=(10, 4.5))

    # Panel (a): regional
    ax1 = fig.add_subplot(1, 2, 1, projection=ccrs.PlateCarree())
    ax1.set_extent([104.5, 107.0, -7.0, -5.5], crs=ccrs.PlateCarree())
    ax1.add_feature(cfeature.LAND, facecolor="#d2b48c")
    ax1.add_feature(cfeature.OCEAN, facecolor="#cce5ff")
    ax1.add_feature(cfeature.COASTLINE, linewidth=0.5)
    gl = ax1.gridlines(draw_labels=True, linewidth=0.3, alpha=0.5)
    gl.top_labels = gl.right_labels = False

    rect_lon = [lon.min(), lon.max(), lon.max(), lon.min(), lon.min()]
    rect_lat = [lat.min(), lat.min(), lat.max(), lat.max(), lat.min()]
    ax1.plot(rect_lon, rect_lat, 'r-', linewidth=2, transform=ccrs.PlateCarree())

    ax1.text(105.0, -5.65, "Sumatra", fontsize=9, fontstyle="italic",
             transform=ccrs.PlateCarree())
    ax1.text(106.2, -6.7, "Java", fontsize=9, fontstyle="italic",
             transform=ccrs.PlateCarree())
    ax1.text(105.1, -6.8, "Sunda Strait", fontsize=8, color="navy",
             transform=ccrs.PlateCarree())
    ax1.text(0.01, 0.01, "Map data: Natural Earth", fontsize=5, color="grey",
             transform=ax1.transAxes, va="bottom", ha="left")
    ax1.set_title("(a) Regional setting", fontsize=10)

    # Panel (b): radar grid
    ax2 = fig.add_subplot(1, 2, 2, projection=ccrs.PlateCarree())
    extent = [lon.min() - 0.02, lon.max() + 0.02, lat.min() - 0.02, lat.max() + 0.02]
    ax2.set_extent(extent, crs=ccrs.PlateCarree())
    ax2.add_feature(cfeature.LAND, facecolor="#d2b48c")
    ax2.add_feature(cfeature.OCEAN, facecolor="#cce5ff")
    ax2.add_feature(cfeature.COASTLINE, linewidth=0.5)

    ax2.scatter(lon2d[mask_obs], lat2d[mask_obs], s=8, c="blue", alpha=0.6,
                transform=ccrs.PlateCarree(), label=f"Observed ({mask_obs.sum()})")
    ax2.scatter(lon2d[mask_blind], lat2d[mask_blind], s=8, c="orange", alpha=0.5,
                marker="s", transform=ccrs.PlateCarree(), label=f"Blind zone ({mask_blind.sum()})")
    ax2.scatter(lon2d[mask_land], lat2d[mask_land], s=8, c="brown", alpha=0.4,
                marker="x", transform=ccrs.PlateCarree(), label=f"Land ({mask_land.sum()})")

    gl2 = ax2.gridlines(draw_labels=True, linewidth=0.3, alpha=0.5)
    gl2.top_labels = gl2.right_labels = False
    ax2.legend(loc="lower right", fontsize=7, markerscale=1.5)
    ax2.set_title("(b) HF radar grid (21×21)", fontsize=10)

    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig_study_area.png"), dpi=DPI, bbox_inches="tight")
    plt.close()
    print("  fig_study_area.png")


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 2: PHASE 1 MODEL COMPARISON (grouped bar)
# ═══════════════════════════════════════════════════════════════════════════════

# Phase-1 results — values match Table 2 (tab:phase1) in paper.tex EXACTLY.
# ARIMA is shown at the parsimonious (1,0,1) order (the headline specification);
# these are the corrected FULL-DOMAIN values that supersede the earlier 30-cell
# subsample. All twelve methods are included. Keep in sync with Table 2.
PHASE1_TABLE2 = [
    # label,          category,    rmse_u, rmse_v,  ss_u,   ss_v
    ("Persistence",   "baseline",  16.03,  20.22,   0.000,  0.000),
    ("UTide",         "baseline",  36.43,  35.90,  -4.167, -2.151),
    ("Moving Avg",    "classical", 24.35,  27.42,  -1.309, -0.839),
    ("SES",           "classical", 16.03,  20.09,   0.000,  0.013),
    ("ARIMA",         "classical", 14.95,  18.95,   0.129,  0.122),
    ("EOF-VAR",       "ststat",    12.50,  16.27,   0.392,  0.353),
    ("Temporal kNN",  "shallow",   14.00,  19.08,   0.237,  0.110),
    ("Perceptron",    "shallow",   12.26,  16.62,   0.415,  0.324),
    ("MLP",           "shallow",   12.22,  16.42,   0.418,  0.340),
    ("CNN",           "deep",      11.31,  15.65,   0.502,  0.401),
    ("GRU",           "deep",      11.56,  15.63,   0.480,  0.403),
    ("CNN-GRU",       "deep",      11.33,  15.44,   0.500,  0.417),
]
CAT_SPANS = [(0, 1, "Baseline", "#888888"), (2, 4, "Classical", "#5B9BD5"),
             (5, 5, "ST-stat", "#7030A0"), (6, 8, "Shallow ML", "#ED7D31"),
             (9, 11, "Deep Learning", "#70AD47")]


def fig_model_comparison():
    labels = [m[0] for m in PHASE1_TABLE2]
    u_vals = [m[2] for m in PHASE1_TABLE2]
    v_vals = [m[3] for m in PHASE1_TABLE2]

    fig, ax = plt.subplots(figsize=(11, 4.5))
    x = np.arange(len(labels))
    w = 0.38

    # Cap display at ylim but plot full bars (clipped); annotate over-cap bars
    y_cap = 42
    bars_u = ax.bar(x - w/2, u_vals, w, label="U (zonal)", color="#5B9BD5", edgecolor="white")
    bars_v = ax.bar(x + w/2, v_vals, w, label="V (meridional)", color="#ED7D31", edgecolor="white")

    # Highlight best deep-learning bar per component
    dl = [i for i, m in enumerate(PHASE1_TABLE2) if m[1] == "deep"]
    bu = min(dl, key=lambda i: u_vals[i])
    bv = min(dl, key=lambda i: v_vals[i])
    bars_u[bu].set_edgecolor("black"); bars_u[bu].set_linewidth(2)
    bars_v[bv].set_edgecolor("black"); bars_v[bv].set_linewidth(2)

    ax.set_ylabel("RMSE (cm s$^{-1}$)", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=9)
    ax.legend(fontsize=10)
    ax.set_ylim(0, y_cap)

    for i, (u, v) in enumerate(zip(u_vals, v_vals)):
        if u > y_cap - 2:
            ax.text(x[i] - w/2, y_cap - 0.5, f"{u:.1f}", ha="center", va="top",
                    fontsize=7, fontweight="bold", color="#2060A0")
        if v > y_cap - 2:
            ax.text(x[i] + w/2, y_cap - 0.5, f"{v:.1f}", ha="center", va="top",
                    fontsize=7, fontweight="bold", color="#C05010")

    for s, e, label, color in CAT_SPANS:
        ax.axvspan(s - 0.5, e + 0.5, alpha=0.08, color=color)
        ax.text((s + e) / 2, y_cap * 0.96, label, ha="center", fontsize=8,
                color=color, fontweight="bold")

    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig_model_comparison.png"), dpi=DPI, bbox_inches="tight")
    plt.close()
    print("  fig_model_comparison.png")


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 3: SKILL SCORES
# ═══════════════════════════════════════════════════════════════════════════════

def fig_skill_scores():
    labels = [m[0] for m in PHASE1_TABLE2]
    ss_u = [m[4] for m in PHASE1_TABLE2]
    ss_v = [m[5] for m in PHASE1_TABLE2]

    fig, ax = plt.subplots(figsize=(11, 4.5))
    x = np.arange(len(labels))
    w = 0.38

    y_floor = -4.5

    ax.bar(x - w/2, ss_u, w, label="SS$_U$", color="#5B9BD5", edgecolor="white")
    ax.bar(x + w/2, ss_v, w, label="SS$_V$", color="#ED7D31", edgecolor="white")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Skill Score (MSE-based)", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=9)
    ax.legend(fontsize=10)
    ax.set_ylim(y_floor, 1.0)
    ax.grid(axis="y", alpha=0.3)

    # Annotate truncated bars
    for i, (u, v) in enumerate(zip(ss_u, ss_v)):
        if u < y_floor + 0.3:
            ax.text(x[i] - w/2, y_floor + 0.1, f"{u:.1f}", ha="center", va="bottom",
                    fontsize=7, fontweight="bold", color="#2060A0")
        if v < y_floor + 0.3:
            ax.text(x[i] + w/2, y_floor + 0.1, f"{v:.1f}", ha="center", va="bottom",
                    fontsize=7, fontweight="bold", color="#C05010")

    for s, e, label, color in CAT_SPANS:
        ax.axvspan(s - 0.5, e + 0.5, alpha=0.08, color=color)

    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig_skill_scores.png"), dpi=DPI, bbox_inches="tight")
    plt.close()
    print("  fig_skill_scores.png")


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 4: LOOKBACK SENSITIVITY (Phase 2)
# ═══════════════════════════════════════════════════════════════════════════════

def fig_lookback():
    r = load_json(P2_FILE)
    models = ["CNN", "GRU", "CNN-GRU"]
    lookbacks = ["T=3", "T=6", "T=12"]
    colors = {"CNN": "#1f77b4", "GRU": "#ff7f0e", "CNN-GRU": "#d62728"}
    markers = {"CNN": "o", "GRU": "s", "CNN-GRU": "^"}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4), sharey=False)

    for m in models:
        u_vals = [r[t][m]["rmse_u"] for t in lookbacks]
        v_vals = [r[t][m]["rmse_v"] for t in lookbacks]
        ax1.plot([3, 6, 12], u_vals, marker=markers[m], color=colors[m],
                 label=m, linewidth=2, markersize=8)
        ax2.plot([3, 6, 12], v_vals, marker=markers[m], color=colors[m],
                 label=m, linewidth=2, markersize=8)

    for ax, comp in [(ax1, "U"), (ax2, "V")]:
        ax.set_xlabel("Lookback T (hours)", fontsize=11)
        ax.set_ylabel(f"RMSE$_{comp}$ (cm s$^{{-1}}$)", fontsize=11)
        ax.set_xticks([3, 6, 12])
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        ax.set_title(f"({['a', 'b'][['U', 'V'].index(comp)]}) {comp} component", fontsize=10)

    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig_lookback.png"), dpi=DPI, bbox_inches="tight")
    plt.close()
    print("  fig_lookback.png")


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 5: LEAD-TIME RMSE DEGRADATION (Phase 3)
# ═══════════════════════════════════════════════════════════════════════════════

def fig_leadtime():
    r = load_json(P3_FILE)
    models = ["Persistence", "ConvLSTM-ED", "BiEF", "CNN-GRU-MS"]
    leads = ["t+1", "t+2", "t+3", "t+4", "t+5", "t+6"]
    x = np.arange(1, 7)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5), sharey=False)

    for m in models:
        u_vals = [r[m]["per_lead"][t]["rmse_u"] for t in leads]
        v_vals = [r[m]["per_lead"][t]["rmse_v"] for t in leads]
        kw = dict(marker=P3_MARKERS[m], color=P3_COLORS[m], label=m,
                  linewidth=2, markersize=7)
        ax1.plot(x, u_vals, **kw)
        ax2.plot(x, v_vals, **kw)

    for ax, comp in [(ax1, "U"), (ax2, "V")]:
        ax.set_xlabel("Lead time (hours)", fontsize=11)
        ax.set_ylabel(f"RMSE$_{comp}$ (cm s$^{{-1}}$)", fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels([f"t+{i}" for i in x])
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(alpha=0.3)
        ax.set_title(f"({['a', 'b'][['U', 'V'].index(comp)]}) {comp} component", fontsize=10)

    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig_leadtime.png"), dpi=DPI, bbox_inches="tight")
    plt.close()
    print("  fig_leadtime.png")


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 6: LEAD-TIME SKILL SCORES (Phase 3)
# ═══════════════════════════════════════════════════════════════════════════════

def fig_leadtime_skill():
    r = load_json(P3_FILE)
    models = ["ConvLSTM-ED", "BiEF", "CNN-GRU-MS"]
    leads = ["t+1", "t+2", "t+3", "t+4", "t+5", "t+6"]
    x = np.arange(1, 7)

    # Compute skill scores per lead time
    pers_u = [r["Persistence"]["per_lead"][t]["rmse_u"] for t in leads]
    pers_v = [r["Persistence"]["per_lead"][t]["rmse_v"] for t in leads]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)

    for m in models:
        u_vals = [r[m]["per_lead"][t]["rmse_u"] for t in leads]
        v_vals = [r[m]["per_lead"][t]["rmse_v"] for t in leads]
        ss_u = [1 - (u**2 / p**2) for u, p in zip(u_vals, pers_u)]
        ss_v = [1 - (v**2 / p**2) for v, p in zip(v_vals, pers_v)]
        kw = dict(marker=P3_MARKERS[m], color=P3_COLORS[m], label=m,
                  linewidth=2, markersize=7)
        ax1.plot(x, ss_u, **kw)
        ax2.plot(x, ss_v, **kw)

    for ax, comp in [(ax1, "U"), (ax2, "V")]:
        ax.set_xlabel("Lead time (hours)", fontsize=11)
        ax.set_ylabel(f"Skill Score (SS$_{comp}$)", fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels([f"t+{i}" for i in x])
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        ax.set_ylim(0, 1)
        ax.set_title(f"({['a', 'b'][['U', 'V'].index(comp)]}) {comp} component", fontsize=10)

    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig_leadtime_skill.png"), dpi=DPI, bbox_inches="tight")
    plt.close()
    print("  fig_leadtime_skill.png")


# ═══════════════════════════════════════════════════════════════════════════════
# FIG 7: DIURNAL RMSE PATTERN
# ═══════════════════════════════════════════════════════════════════════════════

def fig_hourly_rmse():
    r = load_json(P1_FILE)
    models = ["Perceptron", "MLP", "CNN", "GRU", "CNN-GRU"]
    colors = {"Perceptron": "#aaaaaa", "MLP": "#999999",
              "CNN": "#1f77b4", "GRU": "#ff7f0e", "CNN-GRU": "#d62728"}

    hours = np.arange(24)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), sharey=False)

    for m in models:
        lw = 2.5 if m in ("CNN", "GRU", "CNN-GRU") else 1.2
        alpha = 1.0 if m in ("CNN", "GRU", "CNN-GRU") else 0.5
        ax1.plot(hours, r[m]["hourly_rmse_u"], label=m, color=colors[m],
                 linewidth=lw, alpha=alpha)
        ax2.plot(hours, r[m]["hourly_rmse_v"], label=m, color=colors[m],
                 linewidth=lw, alpha=alpha)

    for ax, comp in [(ax1, "U"), (ax2, "V")]:
        # Shade high-error period (06-18 UTC = 13-01 LT)
        ax.axvspan(6, 18, alpha=0.1, color="red", label="High-error (13–01 LT)")
        ax.set_xlabel("Hour (UTC)", fontsize=11)
        ax.set_ylabel(f"RMSE$_{comp}$ (cm s$^{{-1}}$)", fontsize=11)
        ax.set_xticks(np.arange(0, 24, 3))
        ax.legend(fontsize=7, loc="upper left")
        ax.grid(alpha=0.3)
        ax.xaxis.set_minor_locator(MultipleLocator(1))
        ax.set_title(f"({['a', 'b'][['U', 'V'].index(comp)]}) {comp} component", fontsize=10)

    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig_hourly_rmse.png"), dpi=DPI, bbox_inches="tight")
    plt.close()
    print("  fig_hourly_rmse.png")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Generating Paper 1.2 figures...")
    fig_study_area()
    fig_model_comparison()
    fig_skill_scores()
    fig_lookback()
    fig_leadtime()
    fig_leadtime_skill()
    fig_hourly_rmse()
    print("Done!")
