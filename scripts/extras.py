#!/usr/bin/env python3
"""
Paper 1.2 extras: SARIMA baseline, spatial RMSE maps, correlation coefficients.
Run on server: python3 -u scripts/extras.py
"""

import numpy as np
import json
import os
import time
import torch
import torch.nn as nn
import xarray as xr
from collections import OrderedDict

# ══════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════

DATA_FILE = "data/processed/BADA_hourly_qc.nc"
CKPT_DIR = "paper12/output/checkpoints"
OUT_DIR = "paper12/output"
os.makedirs(OUT_DIR, exist_ok=True)

# Splits (same as train_paper12.py)
TRAIN_END = 18216
VAL_END = 22560
# Test: 22560 to end

T = 3  # Phase 1 lookback


# ══════════════════════════════════════════════════════════════════
# LOAD DATA
# ══════════════════════════════════════════════════════════════════

print("Loading data...")
ds = xr.open_dataset(DATA_FILE)
U = ds["U"].values  # (time, lat, lon)
V = ds["V"].values
ds.close()

NT, NY, NX = U.shape
print(f"  Shape: {NT} x {NY} x {NX}")

# Sea mask
land_mask = np.all(np.isnan(U), axis=0)
sea_mask = ~land_mask
n_sea = sea_mask.sum()
print(f"  Sea cells: {n_sea}")

# Replace NaN with 0
U = np.nan_to_num(U, nan=0.0)
V = np.nan_to_num(V, nan=0.0)

# Min-max normalization from training set
U_train = U[:TRAIN_END]
V_train = V[:TRAIN_END]
U_MIN, U_MAX = U_train.min(), U_train.max()
V_MIN, V_MAX = V_train.min(), V_train.max()
print(f"  U range: [{U_MIN:.1f}, {U_MAX:.1f}]")
print(f"  V range: [{V_MIN:.1f}, {V_MAX:.1f}]")

U_norm = (U - U_MIN) / (U_MAX - U_MIN)
V_norm = (V - V_MIN) / (V_MAX - V_MIN)

# Test set
test_start = VAL_END
U_test_raw = U[test_start:]  # original scale
V_test_raw = V[test_start:]
n_test = len(U_test_raw)
print(f"  Test: {n_test} steps")


# ══════════════════════════════════════════════════════════════════
# MODEL DEFINITIONS (same as train_paper12.py)
# ══════════════════════════════════════════════════════════════════

class CNNModel(nn.Module):
    def __init__(self, T=3, ny=21, nx=21):
        super().__init__()
        in_ch = T * 2
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1), nn.ELU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.ELU(),
            nn.MaxPool2d(2), nn.Dropout(0.2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ELU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ELU(),
            nn.MaxPool2d(2), nn.Dropout(0.2),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, in_ch, ny, nx)
            flat = self.conv(dummy).view(1, -1).shape[1]
        self.fc = nn.Sequential(
            nn.Linear(flat, 512), nn.ReLU(),
            nn.Linear(512, 2 * ny * nx), nn.Sigmoid(),
        )
        self.ny, self.nx = ny, nx

    def forward(self, x):
        B = x.size(0)
        x = x.reshape(B, -1, self.ny, self.nx)  # (B, T*2, ny, nx)
        h = self.conv(x).view(B, -1)
        return self.fc(h).view(B, 2, self.ny, self.nx)


class GRUModel(nn.Module):
    def __init__(self, T=3, ny=21, nx=21):
        super().__init__()
        self.T = T
        self.input_size = 2 * ny * nx
        self.gru = nn.GRU(self.input_size, 256, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(256, 2 * ny * nx), nn.Sigmoid(),
        )
        self.ny, self.nx = ny, nx

    def forward(self, x):
        B = x.shape[0]
        seq = x.view(B, self.T, -1)
        _, h = self.gru(seq)
        return self.fc(h.squeeze(0)).view(B, 2, self.ny, self.nx)


class CNNGRUModel(nn.Module):
    def __init__(self, T=3, ny=21, nx=21):
        super().__init__()
        self.T = T
        self.conv = nn.Sequential(
            nn.Conv2d(2, 32, 3, padding=1), nn.ELU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.ELU(),
            nn.MaxPool2d(2), nn.Dropout(0.2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ELU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ELU(),
            nn.MaxPool2d(2), nn.Dropout(0.2),
        )
        h, w = ny // 4, nx // 4
        cnn_out = 64 * h * w
        self.gru = nn.GRU(cnn_out, 256, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(256, 2 * ny * nx), nn.Sigmoid(),
        )
        self.ny, self.nx = ny, nx

    def forward(self, x):
        B = x.shape[0]
        feats = []
        for t in range(x.size(1)):
            f = self.conv(x[:, t]).view(B, -1)
            feats.append(f)
        seq = torch.stack(feats, dim=1)
        _, h = self.gru(seq)
        return self.fc(h.squeeze(0)).view(B, 2, self.ny, self.nx)


# ══════════════════════════════════════════════════════════════════
# #3: SARIMA BASELINE
# ══════════════════════════════════════════════════════════════════

def run_sarima():
    """Fit SARIMA(1,0,1)(1,0,1)_12 per cell and predict on test set."""
    print("\n" + "=" * 60)
    print("#3: SARIMA(1,0,1)(1,0,1)_12 baseline")
    print("=" * 60)

    try:
        from statsmodels.tsa.statespace.sarimax import SARIMAX
        import warnings
        warnings.filterwarnings("ignore")
    except ImportError:
        print("  statsmodels not available, skipping SARIMA")
        return None

    # Use last 3000 training points for speed (covers ~125 days, plenty for seasonal)
    FIT_LEN = 3000
    fit_start = VAL_END - FIT_LEN

    sarima_pred_u = np.zeros((n_test, NY, NX))
    sarima_pred_v = np.zeros((n_test, NY, NX))

    sea_ys, sea_xs = np.where(sea_mask)
    n_cells = len(sea_ys)
    print(f"  Fitting {n_cells} sea cells (last {FIT_LEN} training points)...")

    t0 = time.time()
    n_done = 0
    n_fail = 0

    for idx, (iy, ix) in enumerate(zip(sea_ys, sea_xs)):
        if idx % 30 == 0:
            elapsed = time.time() - t0
            eta = (elapsed / max(idx, 1)) * (n_cells - idx)
            print(f"    Cell {idx}/{n_cells} ({elapsed:.0f}s, ETA {eta:.0f}s)")

        u_series = U[fit_start:VAL_END, iy, ix]
        v_series = V[fit_start:VAL_END, iy, ix]

        try:
            model_u = SARIMAX(u_series, order=(1, 0, 1),
                             seasonal_order=(1, 0, 1, 12),
                             enforce_stationarity=False,
                             enforce_invertibility=False,
                             simple_differencing=True)
            res_u = model_u.fit(disp=False, maxiter=30, method='lbfgs')

            model_v = SARIMAX(v_series, order=(1, 0, 1),
                             seasonal_order=(1, 0, 1, 12),
                             enforce_stationarity=False,
                             enforce_invertibility=False,
                             simple_differencing=True)
            res_v = model_v.fit(disp=False, maxiter=30, method='lbfgs')

            # One-step forecast (only need 1 step for fair comparison)
            fc_u = res_u.get_forecast(steps=n_test)
            fc_v = res_v.get_forecast(steps=n_test)

            sarima_pred_u[:, iy, ix] = fc_u.predicted_mean
            sarima_pred_v[:, iy, ix] = fc_v.predicted_mean
            n_done += 1

        except Exception:
            sarima_pred_u[:, iy, ix] = U_test_raw[:, iy, ix]
            sarima_pred_v[:, iy, ix] = V_test_raw[:, iy, ix]
            n_fail += 1

    elapsed = time.time() - t0
    print(f"  Done: {n_done} OK, {n_fail} failed ({elapsed:.0f}s)")

    # Compute RMSE
    diff_u = (sarima_pred_u - U_test_raw) ** 2
    diff_v = (sarima_pred_v - V_test_raw) ** 2

    # Overall RMSE (sea cells only)
    rmse_u = np.sqrt(np.mean(diff_u[:, sea_mask]))
    rmse_v = np.sqrt(np.mean(diff_v[:, sea_mask]))

    # Persistence RMSE for skill score
    pers_u = U[test_start - 1: test_start - 1 + n_test]
    pers_v = V[test_start - 1: test_start - 1 + n_test]
    pers_rmse_u = np.sqrt(np.mean((pers_u - U_test_raw)[:, sea_mask] ** 2))
    pers_rmse_v = np.sqrt(np.mean((pers_v - V_test_raw)[:, sea_mask] ** 2))

    ss_u = 1 - (rmse_u ** 2) / (pers_rmse_u ** 2)
    ss_v = 1 - (rmse_v ** 2) / (pers_rmse_v ** 2)

    print(f"  SARIMA RMSE_U = {rmse_u:.2f}, SS_U = {ss_u:.3f}")
    print(f"  SARIMA RMSE_V = {rmse_v:.2f}, SS_V = {ss_v:.3f}")
    print(f"  (Persistence: RMSE_U = {pers_rmse_u:.2f}, RMSE_V = {pers_rmse_v:.2f})")

    # Per-cell RMSE for spatial map
    cell_rmse_u = np.sqrt(np.mean(diff_u, axis=0))
    cell_rmse_v = np.sqrt(np.mean(diff_v, axis=0))

    return {
        "rmse_u": float(rmse_u), "rmse_v": float(rmse_v),
        "ss_u": float(ss_u), "ss_v": float(ss_v),
        "cell_rmse_u": cell_rmse_u, "cell_rmse_v": cell_rmse_v,
    }


# ══════════════════════════════════════════════════════════════════
# #2 & #4: SPATIAL RMSE + CORRELATIONS from checkpoints
# ══════════════════════════════════════════════════════════════════

def load_model_and_predict(model_class, ckpt_path, T_val, data_u, data_v):
    """Load model checkpoint and run inference on test set."""
    model = model_class(T=T_val, ny=NY, nx=NX)

    # Load checkpoint (handle IPEX-trained models)
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    # Strip 'module.' prefix if present
    new_state = OrderedDict()
    for k, v in state.items():
        name = k.replace("module.", "")
        new_state[name] = v
    model.load_state_dict(new_state, strict=False)
    model.eval()

    # Build test sequences
    preds_u = []
    preds_v = []

    with torch.no_grad():
        batch_size = 256
        for i in range(T_val, len(data_u)):
            if i < test_start:
                continue
            # Lookback
            u_lb = data_u[i - T_val:i]  # (T, ny, nx)
            v_lb = data_v[i - T_val:i]

            # Build input matching training layout: (1, T, 2, ny, nx)
            # Training: X[i, :, 0] = U, X[i, :, 1] = V
            frames = np.stack([u_lb, v_lb], axis=1)  # (T, 2, ny, nx)
            x = torch.FloatTensor(frames).unsqueeze(0)  # (1, T, 2, ny, nx)

            out = model(x)  # (1, 2, ny, nx)
            pred = out.numpy()[0]

            # Denormalize
            pred_u = pred[0] * (U_MAX - U_MIN) + U_MIN
            pred_v = pred[1] * (V_MAX - V_MIN) + V_MIN
            preds_u.append(pred_u)
            preds_v.append(pred_v)

    preds_u = np.array(preds_u)
    preds_v = np.array(preds_v)
    return preds_u, preds_v


def compute_spatial_metrics():
    """Compute per-cell RMSE and correlation for DL models."""
    print("\n" + "=" * 60)
    print("#2 & #4: Spatial RMSE + Correlation Coefficients")
    print("=" * 60)

    models = {
        "CNN": (CNNModel, os.path.join(CKPT_DIR, "CNN_T3_onestep.pt"), 3),
        "GRU": (GRUModel, os.path.join(CKPT_DIR, "GRU_T3_onestep.pt"), 3),
        "CNN-GRU": (CNNGRUModel, os.path.join(CKPT_DIR, "CNN_GRU_T3_onestep.pt"), 3),
    }

    results = {}

    # Persistence predictions (for reference)
    pers_u = U_test_raw[:-1]  # shifted
    pers_v = V_test_raw[:-1]
    obs_u = U_test_raw[1:]  # target at t+1
    obs_v = V_test_raw[1:]

    for name, (cls, ckpt, T_val) in models.items():
        if not os.path.exists(ckpt):
            print(f"  {name}: checkpoint not found, skipping")
            continue

        print(f"\n  {name} (T={T_val})...")
        t0 = time.time()
        preds_u, preds_v = load_model_and_predict(cls, ckpt, T_val, U_norm, V_norm)
        elapsed = time.time() - t0
        print(f"    Inference: {elapsed:.1f}s, predictions: {preds_u.shape}")

        # Align with test targets
        # Predictions start at i=test_start (lookback uses [test_start-T:test_start])
        # So preds[0] predicts for time test_start = U_test_raw[0]
        n_pred = len(preds_u)
        target_u = U_test_raw[:n_pred]
        target_v = V_test_raw[:n_pred]

        # Per-cell RMSE
        cell_rmse_u = np.sqrt(np.mean((preds_u - target_u) ** 2, axis=0))
        cell_rmse_v = np.sqrt(np.mean((preds_v - target_v) ** 2, axis=0))

        # Overall RMSE (sea cells only)
        rmse_u = np.sqrt(np.mean((preds_u - target_u)[:, sea_mask] ** 2))
        rmse_v = np.sqrt(np.mean((preds_v - target_v)[:, sea_mask] ** 2))

        # Per-cell correlation (#4)
        cell_corr_u = np.zeros((NY, NX))
        cell_corr_v = np.zeros((NY, NX))
        for iy in range(NY):
            for ix in range(NX):
                if sea_mask[iy, ix]:
                    r_u = np.corrcoef(preds_u[:, iy, ix], target_u[:, iy, ix])[0, 1]
                    r_v = np.corrcoef(preds_v[:, iy, ix], target_v[:, iy, ix])[0, 1]
                    cell_corr_u[iy, ix] = r_u if np.isfinite(r_u) else 0
                    cell_corr_v[iy, ix] = r_v if np.isfinite(r_v) else 0

        # Overall correlation (sea cells)
        flat_pred_u = preds_u[:, sea_mask].flatten()
        flat_obs_u = target_u[:, sea_mask].flatten()
        flat_pred_v = preds_v[:, sea_mask].flatten()
        flat_obs_v = target_v[:, sea_mask].flatten()
        overall_r_u = np.corrcoef(flat_pred_u, flat_obs_u)[0, 1]
        overall_r_v = np.corrcoef(flat_pred_v, flat_obs_v)[0, 1]

        # Mean per-cell correlation
        mean_cell_r_u = np.mean(cell_corr_u[sea_mask])
        mean_cell_r_v = np.mean(cell_corr_v[sea_mask])

        print(f"    RMSE: U={rmse_u:.2f}, V={rmse_v:.2f}")
        print(f"    Overall r: U={overall_r_u:.4f}, V={overall_r_v:.4f}")
        print(f"    Mean cell r: U={mean_cell_r_u:.4f}, V={mean_cell_r_v:.4f}")

        results[name] = {
            "rmse_u": float(rmse_u), "rmse_v": float(rmse_v),
            "overall_r_u": float(overall_r_u), "overall_r_v": float(overall_r_v),
            "mean_cell_r_u": float(mean_cell_r_u), "mean_cell_r_v": float(mean_cell_r_v),
            "cell_rmse_u": cell_rmse_u.tolist(),
            "cell_rmse_v": cell_rmse_v.tolist(),
            "cell_corr_u": cell_corr_u.tolist(),
            "cell_corr_v": cell_corr_v.tolist(),
        }

    return results


# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    all_results = {}

    # #3: SARIMA — check for cached results first
    sarima_cache = os.path.join(OUT_DIR, "sarima_cache.json")
    if os.path.exists(sarima_cache):
        print("Loading cached SARIMA results...")
        with open(sarima_cache) as f:
            all_results["SARIMA"] = json.load(f)
        print(f"  SARIMA RMSE_U = {all_results['SARIMA']['rmse_u']:.2f}")
        print(f"  SARIMA RMSE_V = {all_results['SARIMA']['rmse_v']:.2f}")
    else:
        sarima = run_sarima()
        if sarima:
            all_results["SARIMA"] = {
                "rmse_u": sarima["rmse_u"],
                "rmse_v": sarima["rmse_v"],
                "ss_u": sarima["ss_u"],
                "ss_v": sarima["ss_v"],
                "cell_rmse_u": sarima["cell_rmse_u"].tolist(),
                "cell_rmse_v": sarima["cell_rmse_v"].tolist(),
            }
            # Cache SARIMA results
            with open(sarima_cache, "w") as f:
                json.dump(all_results["SARIMA"], f, indent=2)
            print(f"  Cached SARIMA results to {sarima_cache}")

    # #2 & #4: Spatial RMSE + correlations
    spatial = compute_spatial_metrics()
    all_results.update(spatial)

    # Save results
    # Convert numpy arrays for JSON
    out_file = os.path.join(OUT_DIR, "extras_results.json")
    with open(out_file, "w") as f:
        json.dump(all_results, f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else x)
    print(f"\nSaved to {out_file}")
    print("Done!")
