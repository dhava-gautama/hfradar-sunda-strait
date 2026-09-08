#!/usr/bin/env python3
"""R2-minor-3: along-/cross-strait wind-projection analysis (ASCMO referee 2).

The submitted diurnal analysis correlated component RMSE against wind SPEED only,
which left the negative V-error/wind-speed relationship unresolved. A scalar speed
cannot distinguish the cross-strait sea breeze (which should load on the zonal U
current) from along-strait wind (which would be the candidate driver of the
meridional V current). Here we (1) fix the strait axis from the principal axis of
the observed current variability, (2) decompose the three-station AWS wind into
along- and cross-strait components, and (3) correlate each current component's
diurnal RMSE against the matching wind component. This directly tests whether the
V error is driven by along-strait wind or is, as we argue, tidal-phase controlled.

Reuses the diurnal CNN-GRU RMSE from phase1_results.json and the AWS loader
convention from analyze_aws.py. Fast, CPU only.
"""
import os
import json

import numpy as np
import pandas as pd
from scipy import stats

ROOT = os.environ.get("RADAR_ROOT", "/mnt/hdd/rocket/radarMaritim")
AWS_DIR = "/mnt/hdd/rocket/awsMaritim"
DATA_PATH = os.path.join(ROOT, "data/processed/BADA_hourly_qc.nc")
OUT = os.path.join(ROOT, "paper12/output/wind_projection.json")
P1 = os.path.join(ROOT, "paper12/output/phase1_results.json")

TEST_START, TEST_END = "2025-07-01", "2026-03-01"
STATIONS = {
    "Merak": f"{AWS_DIR}/AWS_Maritim_Dermaga7_Merak_20250101_20260401.csv",
    "Ciwandan": f"{AWS_DIR}/AWS_Maritim_Ciwandan_20250101_20260401.csv",
    "Bakauheni": f"{AWS_DIR}/AWS_Maritim_Bakauheni_20250101_20260401.csv",
}


def load_aws(path):
    df = pd.read_csv(path, parse_dates=["twaktu"], dayfirst=True).rename(
        columns={"twaktu": "time"}).set_index("time")
    df.loc[(df["windspeed"] < 0) | (df["windspeed"] > 40), "windspeed"] = np.nan
    df.loc[(df["winddir"] < 0) | (df["winddir"] > 360), "winddir"] = np.nan
    # meteorological "from" convention -> eastward/northward components
    d = np.deg2rad(df["winddir"].values)
    df["u_w"] = -df["windspeed"].values * np.sin(d)   # eastward
    df["v_w"] = -df["windspeed"].values * np.cos(d)   # northward
    return df.resample("1h").mean()


def strait_axis():
    """Principal axis of observed (U,V) variability = along-strait direction."""
    import xarray as xr
    ds = xr.open_dataset(DATA_PATH)
    U = ds["U"].values.astype(np.float32)
    V = ds["V"].values.astype(np.float32)
    ds.close()
    sea = np.mean(np.isfinite(U), axis=0) > 0.5
    u = U[:, sea].ravel()
    v = V[:, sea].ravel()
    m = np.isfinite(u) & np.isfinite(v)
    cov = np.cov(np.vstack([u[m], v[m]]))           # [east, north]
    w, vec = np.linalg.eigh(cov)
    major = vec[:, np.argmax(w)]                      # (east, north) of major axis
    alpha = np.arctan2(major[1], major[0])           # angle from east
    bearing = (90 - np.rad2deg(alpha)) % 180          # compass bearing of axis
    return float(alpha), float(bearing)


def main():
    alpha, bearing = strait_axis()
    ca, sa = np.cos(alpha), np.sin(alpha)
    print(f"Strait (current principal) axis: alpha={np.rad2deg(alpha):.1f} deg from E, "
          f"bearing {bearing:.1f} deg (N-S=0/180, E-W=90)")

    # per-station hourly components over test period, projected onto strait axes
    diurnal = {"along": [], "cross": [], "speed": []}
    per_station = {}
    for name, path in STATIONS.items():
        df = load_aws(path)
        df = df.loc[(df.index >= TEST_START) & (df.index < TEST_END)]
        along = df["u_w"] * ca + df["v_w"] * sa
        cross = -df["u_w"] * sa + df["v_w"] * ca
        spd = df["windspeed"]
        by = lambda s: s.groupby(df.index.hour).mean()
        a, c, p = by(along), by(cross), by(spd)
        diurnal["along"].append(a)
        diurnal["cross"].append(c)
        diurnal["speed"].append(p)
        per_station[name] = {"along_amp": float(a.max() - a.min()),
                             "cross_amp": float(c.max() - c.min())}

    along = pd.concat(diurnal["along"], axis=1).mean(axis=1).reindex(range(24))
    cross = pd.concat(diurnal["cross"], axis=1).mean(axis=1).reindex(range(24))
    speed = pd.concat(diurnal["speed"], axis=1).mean(axis=1).reindex(range(24))

    with open(P1) as f:
        p1 = json.load(f)
    rmse_u = np.array(p1["CNN-GRU"]["hourly_rmse_u"])
    rmse_v = np.array(p1["CNN-GRU"]["hourly_rmse_v"])

    def corr(x, y):
        rp, pp = stats.pearsonr(x, y)
        rs, ps = stats.spearmanr(x, y)
        return {"pearson_r": round(float(rp), 3), "pearson_p": round(float(pp), 4),
                "spearman_r": round(float(rs), 3), "spearman_p": round(float(ps), 4)}

    res = {
        "strait_axis_deg_from_east": round(np.rad2deg(alpha), 1),
        "strait_axis_compass_bearing": round(bearing, 1),
        "note_axis": "along-strait approx NE-SW (principal current axis, ~43 deg bearing); cross-strait approx NW-SE",
        "diurnal_wind_amp_mps": {
            "along_strait": round(float(np.nanmax(along) - np.nanmin(along)), 3),
            "cross_strait": round(float(np.nanmax(cross) - np.nanmin(cross)), 3),
            "speed": round(float(np.nanmax(speed) - np.nanmin(speed)), 3),
        },
        "cross_strait_peak_hour_utc": int(np.nanargmax(cross.values)),
        "along_strait_peak_hour_utc": int(np.nanargmax(along.values)),
        "RMSE_U_vs": {
            "cross_strait_wind": corr(cross.values, rmse_u),
            "along_strait_wind": corr(along.values, rmse_u),
            "wind_speed": corr(speed.values, rmse_u),
        },
        "RMSE_V_vs": {
            "cross_strait_wind": corr(cross.values, rmse_v),
            "along_strait_wind": corr(along.values, rmse_v),
            "wind_speed": corr(speed.values, rmse_v),
        },
        "diurnal": {
            "along_strait": [round(float(x), 3) for x in along.values],
            "cross_strait": [round(float(x), 3) for x in cross.values],
            "speed": [round(float(x), 3) for x in speed.values],
            "rmse_u": [round(float(x), 3) for x in rmse_u],
            "rmse_v": [round(float(x), 3) for x in rmse_v],
        },
        "per_station_amp": per_station,
    }
    with open(OUT, "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps({k: v for k, v in res.items() if k != "diurnal"}, indent=2))
    print(f"\nsaved {OUT}")


if __name__ == "__main__":
    main()
