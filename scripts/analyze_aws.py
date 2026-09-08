#!/usr/bin/env python3
"""
Analyze AWS wind data from Merak, Ciwandan, Bakauheni ports.
Compute diurnal wind speed pattern and correlate with model RMSE.
"""

import numpy as np
import pandas as pd
import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PAPER_DIR = os.path.dirname(SCRIPT_DIR)
AWS_DIR = "/mnt/hdd/rocket/awsMaritim"
FIG_DIR = os.path.join(PAPER_DIR, "figures")
OUTPUT_DIR = os.path.join(PAPER_DIR, "output")
P1_FILE = os.path.join(OUTPUT_DIR, "phase1_results.json")
DPI = 600

# Test period: Jul 2025 - Feb 2026
TEST_START = "2025-07-01"
TEST_END = "2026-03-01"

stations = {
    "Merak": os.path.join(AWS_DIR, "AWS_Maritim_Dermaga7_Merak_20250101_20260401.csv"),
    "Ciwandan": os.path.join(AWS_DIR, "AWS_Maritim_Ciwandan_20250101_20260401.csv"),
    "Bakauheni": os.path.join(AWS_DIR, "AWS_Maritim_Bakauheni_20250101_20260401.csv"),
}


def load_aws(path, name):
    """Load and QC AWS data."""
    print(f"  Loading {name}...")
    df = pd.read_csv(path, parse_dates=["twaktu"], dayfirst=True)
    df = df.rename(columns={"twaktu": "time"})
    df = df.set_index("time")

    # Basic QC
    n_raw = len(df)
    df.loc[df["windspeed"] < 0, "windspeed"] = np.nan
    df.loc[df["windspeed"] > 40, "windspeed"] = np.nan
    df.loc[df["winddir"] < 0, "winddir"] = np.nan
    df.loc[df["winddir"] > 360, "winddir"] = np.nan
    n_valid = df["windspeed"].notna().sum()
    print(f"    Raw: {n_raw}, Valid wind: {n_valid} ({100*n_valid/n_raw:.1f}%)")

    # Resample to hourly
    df_h = df.resample("1h").mean()
    print(f"    Hourly: {len(df_h)} steps, wind valid: {df_h['windspeed'].notna().sum()}")

    return df_h


def main():
    print("=" * 60)
    print("AWS Wind Analysis for Paper 1.2")
    print("=" * 60)

    # Load all stations
    hourly = {}
    for name, path in stations.items():
        hourly[name] = load_aws(path, name)

    # Filter to test period
    test_data = {}
    for name, df in hourly.items():
        mask = (df.index >= TEST_START) & (df.index < TEST_END)
        test_data[name] = df.loc[mask]
        print(f"  {name} test period: {test_data[name]['windspeed'].notna().sum()} valid hours")

    # ══════════════════════════════════════════════════════════════
    # Diurnal wind speed pattern (by hour UTC)
    # ══════════════════════════════════════════════════════════════
    print("\nDiurnal wind speed (test period, UTC):")
    diurnal = {}
    for name, df in test_data.items():
        hourly_mean = df.groupby(df.index.hour)["windspeed"].mean()
        hourly_std = df.groupby(df.index.hour)["windspeed"].std()
        diurnal[name] = {"mean": hourly_mean, "std": hourly_std}
        print(f"  {name}: min={hourly_mean.min():.2f} m/s (hour {hourly_mean.idxmin()}), "
              f"max={hourly_mean.max():.2f} m/s (hour {hourly_mean.idxmax()})")

    # Average across all 3 stations
    all_means = pd.DataFrame({n: d["mean"] for n, d in diurnal.items()})
    avg_wind = all_means.mean(axis=1)
    print(f"\n  3-station average: min={avg_wind.min():.2f} (hour {avg_wind.idxmin()}), "
          f"max={avg_wind.max():.2f} (hour {avg_wind.idxmax()})")

    # Local time conversion: UTC+7
    print("\n  In local time (UTC+7):")
    for h in range(24):
        lt = (h + 7) % 24
        print(f"    UTC {h:02d} = LT {lt:02d}: avg wind = {avg_wind[h]:.2f} m/s")

    # ══════════════════════════════════════════════════════════════
    # Load model RMSE by hour
    # ══════════════════════════════════════════════════════════════
    with open(P1_FILE) as f:
        p1 = json.load(f)

    cnn_gru_u = np.array(p1["CNN-GRU"]["hourly_rmse_u"])
    cnn_gru_v = np.array(p1["CNN-GRU"]["hourly_rmse_v"])

    # ══════════════════════════════════════════════════════════════
    # Correlation: wind speed vs RMSE by hour
    # ══════════════════════════════════════════════════════════════
    print("\nCorrelation (24 hourly bins):")
    r_u, p_u = stats.pearsonr(avg_wind.values, cnn_gru_u)
    r_v, p_v = stats.pearsonr(avg_wind.values, cnn_gru_v)
    print(f"  Wind vs RMSE_U: r = {r_u:.3f}, p = {p_u:.4f}")
    print(f"  Wind vs RMSE_V: r = {r_v:.3f}, p = {p_v:.4f}")

    # Spearman (more robust)
    rs_u, ps_u = stats.spearmanr(avg_wind.values, cnn_gru_u)
    rs_v, ps_v = stats.spearmanr(avg_wind.values, cnn_gru_v)
    print(f"  Spearman Wind vs RMSE_U: r_s = {rs_u:.3f}, p = {ps_u:.4f}")
    print(f"  Spearman Wind vs RMSE_V: r_s = {rs_v:.3f}, p = {ps_v:.4f}")

    # ══════════════════════════════════════════════════════════════
    # Figure: Diurnal wind + RMSE overlay
    # ══════════════════════════════════════════════════════════════
    hours = np.arange(24)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

    # Panel (a): U component
    ax1_wind = ax1.twinx()
    for name in stations:
        ax1_wind.plot(hours, diurnal[name]["mean"].values, '--', alpha=0.3,
                      color='#aec7e8', linewidth=0.8)
    ax1_wind.plot(hours, avg_wind.values, '--', color='#1f77b4', linewidth=2,
                  label='Wind speed (3-stn avg)', zorder=1)
    ax1_wind.set_ylabel("Wind speed (m s$^{-1}$)", fontsize=11, color='#1f77b4')
    ax1_wind.tick_params(axis='y', labelcolor='#1f77b4')

    ax1.plot(hours, cnn_gru_u, 'o-', color='#d62728', linewidth=2,
             markersize=5, label='CNN-GRU RMSE$_U$', zorder=5)
    ax1.axvspan(6, 18, alpha=0.08, color='red')
    ax1.set_xlabel("Hour (UTC)", fontsize=11)
    ax1.set_ylabel("RMSE$_U$ (cm s$^{-1}$)", fontsize=11, color='#d62728')
    ax1.tick_params(axis='y', labelcolor='#d62728')
    ax1.set_xticks(np.arange(0, 24, 3))
    ax1.set_title(f"(a) $U$ component ($r_s$ = {rs_u:.2f}, $p$ = {ps_u:.3f})", fontsize=10)

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_wind.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc='upper left')

    # Panel (b): V component
    ax2_wind = ax2.twinx()
    for name in stations:
        ax2_wind.plot(hours, diurnal[name]["mean"].values, '--', alpha=0.3,
                      color='#aec7e8', linewidth=0.8)
    ax2_wind.plot(hours, avg_wind.values, '--', color='#1f77b4', linewidth=2,
                  label='Wind speed (3-stn avg)', zorder=1)
    ax2_wind.set_ylabel("Wind speed (m s$^{-1}$)", fontsize=11, color='#1f77b4')
    ax2_wind.tick_params(axis='y', labelcolor='#1f77b4')

    ax2.plot(hours, cnn_gru_v, 'o-', color='#d62728', linewidth=2,
             markersize=5, label='CNN-GRU RMSE$_V$', zorder=5)
    ax2.axvspan(6, 18, alpha=0.08, color='red')
    ax2.set_xlabel("Hour (UTC)", fontsize=11)
    ax2.set_ylabel("RMSE$_V$ (cm s$^{-1}$)", fontsize=11, color='#d62728')
    ax2.tick_params(axis='y', labelcolor='#d62728')
    ax2.set_xticks(np.arange(0, 24, 3))
    ax2.set_title(f"(b) $V$ component ($r_s$ = {rs_v:.2f}, $p$ = {ps_v:.3f})", fontsize=10)

    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_wind.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc='upper left')

    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig_wind_rmse.png"), dpi=DPI, bbox_inches="tight")
    plt.close()
    print("\n  Saved fig_wind_rmse.png")

    # ══════════════════════════════════════════════════════════════
    # Save results
    # ══════════════════════════════════════════════════════════════
    results = {
        "stations": list(stations.keys()),
        "test_period": f"{TEST_START} to {TEST_END}",
        "diurnal_wind_avg": {str(h): round(float(avg_wind[h]), 3) for h in range(24)},
        "diurnal_wind_by_station": {
            name: {str(h): round(float(d["mean"][h]), 3) for h in range(24)}
            for name, d in diurnal.items()
        },
        "correlation": {
            "pearson_U": {"r": round(r_u, 4), "p": round(p_u, 5)},
            "pearson_V": {"r": round(r_v, 4), "p": round(p_v, 5)},
            "spearman_U": {"r_s": round(rs_u, 4), "p": round(ps_u, 5)},
            "spearman_V": {"r_s": round(rs_v, 4), "p": round(ps_v, 5)},
        },
    }
    with open(os.path.join(OUTPUT_DIR, "aws_wind_analysis.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("  Saved aws_wind_analysis.json")

    print("\nDone!")


if __name__ == "__main__":
    main()
