#!/usr/bin/env python3
"""
Paper 1.2 Phase B: Additional experiments for reviewer robustness.

  B1: Multi-seed uncertainty (5 seeds × CNN, GRU, CNN-GRU, CNN-GRU-MS)
  B2: Capacity-matched CNN-GRU-MS-Small (~465K params, GRU hidden=90)
  B3: Seasonal/monsoon error stratification (existing predictions)
  B4: Tidal-residual error decomposition (existing predictions)

Usage:
  python phase_b.py --task seed       # B1: multi-seed
  python phase_b.py --task capacity   # B2: capacity-matched
  python phase_b.py --task seasonal   # B3: seasonal analysis
  python phase_b.py --task tidal      # B4: tidal decomposition
  python phase_b.py --task all        # all tasks
"""

import argparse
import json
import os
import time
import warnings
from datetime import datetime

import numpy as np

warnings.filterwarnings("ignore")

DATA_PATH = os.path.expanduser("~/radarMaritim/data/processed/BADA_hourly_qc.nc")
OUTPUT_DIR = os.path.expanduser("~/radarMaritim/paper12/output")
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")

TRAIN_END = datetime(2025, 1, 1)
VAL_END = datetime(2025, 7, 1)
SEEDS = [42, 123, 456, 789, 1024]

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


def load_data():
    import xarray as xr
    print("Loading data...")
    ds = xr.open_dataset(DATA_PATH)
    U = ds["U"].values.astype(np.float32)
    V = ds["V"].values.astype(np.float32)
    times = ds["time"].values
    lat = ds["latitude"].values
    lon = ds["longitude"].values
    ds.close()

    valid_frac = np.mean(np.isfinite(U), axis=0)
    sea_mask = valid_frac > 0.5
    n_sea = int(sea_mask.sum())
    print(f"Grid: {U.shape[1]}x{U.shape[2]}, sea cells: {n_sea}")

    U = np.nan_to_num(U, nan=0.0)
    V = np.nan_to_num(V, nan=0.0)

    idx_train_end = int(np.searchsorted(times, np.datetime64(TRAIN_END)))
    idx_val_end = int(np.searchsorted(times, np.datetime64(VAL_END)))

    u_min = U[:idx_train_end][:, sea_mask].min()
    u_max = U[:idx_train_end][:, sea_mask].max()
    v_min = V[:idx_train_end][:, sea_mask].min()
    v_max = V[:idx_train_end][:, sea_mask].max()

    U_norm = (U - u_min) / (u_max - u_min)
    V_norm = (V - v_min) / (v_max - v_min)

    return {
        "U": U, "V": V, "U_norm": U_norm, "V_norm": V_norm,
        "times": times, "lat": lat, "lon": lon,
        "sea_mask": sea_mask, "n_sea": n_sea,
        "u_min": float(u_min), "u_max": float(u_max),
        "v_min": float(v_min), "v_max": float(v_max),
        "idx_train_end": idx_train_end, "idx_val_end": idx_val_end,
        "ny": U.shape[1], "nx": U.shape[2],
    }


def rmse_sea(pred, true, sea_mask):
    diff = pred[:, sea_mask] - true[:, sea_mask]
    return float(np.sqrt(np.mean(diff ** 2)))


def denorm_u(x, data):
    return x * (data["u_max"] - data["u_min"]) + data["u_min"]

def denorm_v(x, data):
    return x * (data["v_max"] - data["v_min"]) + data["v_min"]


def build_onestep_sequences(U, V, T):
    N = len(U) - T
    ny, nx = U.shape[1], U.shape[2]
    X = np.zeros((N, T, 2, ny, nx), dtype=np.float32)
    Y = np.zeros((N, 2, ny, nx), dtype=np.float32)
    for i in range(N):
        X[i, :, 0] = U[i:i + T]
        X[i, :, 1] = V[i:i + T]
        Y[i, 0] = U[i + T]
        Y[i, 1] = V[i + T]
    return X, Y


def build_multistep_sequences(U, V, T, H):
    N = len(U) - T - H + 1
    ny, nx = U.shape[1], U.shape[2]
    X = np.zeros((N, T, 2, ny, nx), dtype=np.float32)
    Y = np.zeros((N, H, 2, ny, nx), dtype=np.float32)
    for i in range(N):
        X[i, :, 0] = U[i:i + T]
        X[i, :, 1] = V[i:i + T]
        for h in range(H):
            Y[i, h, 0] = U[i + T + h]
            Y[i, h, 1] = V[i + T + h]
    return X, Y


# ═══════════════════════════════════════════════════════════════════════════════
# B1: MULTI-SEED TRAINING
# ═══════════════════════════════════════════════════════════════════════════════

def train_with_seed(data, seed, model_name, T=3, phase="onestep"):
    """Train a single model with a given seed. Returns RMSE dict."""
    import torch
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader

    torch.manual_seed(seed)
    np.random.seed(seed)

    try:
        import intel_extension_for_pytorch as ipex
        use_ipex = True
    except ImportError:
        use_ipex = False

    device = torch.device("cpu")
    ny, nx = data["ny"], data["nx"]
    sea_mask = data["sea_mask"]

    if phase == "onestep":
        X, Y = build_onestep_sequences(data["U_norm"], data["V_norm"], T)
        H = 1
    else:
        H = 6
        X, Y = build_multistep_sequences(data["U_norm"], data["V_norm"], T, H)

    idx_train = data["idx_train_end"] - T
    idx_val = data["idx_val_end"] - T
    X_train, Y_train = X[:idx_train], Y[:idx_train]
    X_val, Y_val = X[idx_train:idx_val], Y[idx_train:idx_val]
    X_test, Y_test = X[idx_val:], Y[idx_val:]

    # ── Model definitions ─────────────────────────────────────────────
    class CNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(T * 2, 32, 3, padding=1), nn.ELU(),
                nn.Conv2d(32, 32, 3, padding=1), nn.ELU(),
                nn.MaxPool2d(2), nn.Dropout(0.2),
                nn.Conv2d(32, 64, 3, padding=1), nn.ELU(),
                nn.Conv2d(64, 64, 3, padding=1), nn.ELU(),
                nn.MaxPool2d(2), nn.Dropout(0.2),
            )
            h, w = ny // 4, nx // 4
            self.fc = nn.Sequential(
                nn.Linear(64 * h * w, 512), nn.ReLU(),
                nn.Linear(512, 2 * ny * nx), nn.Sigmoid(),
            )
        def forward(self, x):
            x_cat = x.reshape(x.size(0), -1, ny, nx)
            return self.fc(self.conv(x_cat).reshape(x.size(0), -1))

    class GRUModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.gru = nn.GRU(2 * ny * nx, 256, batch_first=True)
            self.fc = nn.Sequential(
                nn.Linear(256, 2 * ny * nx), nn.Sigmoid(),
            )
        def forward(self, x):
            b = x.size(0)
            seq = x.reshape(b, x.size(1), -1)
            _, h = self.gru(seq)
            return self.fc(h.squeeze(0))

    class CNNGRU(nn.Module):
        def __init__(self, gru_hidden=256):
            super().__init__()
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
            self.gru = nn.GRU(cnn_out, gru_hidden, batch_first=True)
            self.fc = nn.Sequential(
                nn.Linear(gru_hidden, 2 * ny * nx), nn.Sigmoid(),
            )
        def forward(self, x):
            b = x.size(0)
            feats = []
            for t in range(x.size(1)):
                feats.append(self.conv(x[:, t]).reshape(b, -1))
            seq = torch.stack(feats, dim=1)
            _, h = self.gru(seq)
            return self.fc(h.squeeze(0))

    class CNNGRU_MS(nn.Module):
        def __init__(self, gru_hidden=256):
            super().__init__()
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
            self.gru = nn.GRU(cnn_out, gru_hidden, batch_first=True)
            self.fc = nn.Sequential(
                nn.Linear(gru_hidden, H * 2 * ny * nx), nn.Sigmoid(),
            )
        def forward(self, x):
            b = x.size(0)
            feats = []
            for t in range(x.size(1)):
                feats.append(self.conv(x[:, t]).reshape(b, -1))
            seq = torch.stack(feats, dim=1)
            _, h = self.gru(seq)
            out = self.fc(h.squeeze(0))
            return out.reshape(b, H, 2, ny, nx)

    # Select model
    model_map = {
        "CNN": CNN, "GRU": GRUModel, "CNN-GRU": CNNGRU,
        "CNN-GRU-MS": CNNGRU_MS,
        "CNN-GRU-MS-Small": lambda: CNNGRU_MS(gru_hidden=90),
    }

    if model_name in ("CNN-GRU-MS-Small",):
        model = model_map[model_name]().to(device)
    else:
        model = model_map[model_name]().to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"    {model_name} seed={seed}, params={n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    if use_ipex:
        model, optimizer = ipex.optimize(model, optimizer=optimizer)

    if phase == "onestep":
        batch_size, patience, max_epochs = 64, 10, 50
    else:
        batch_size, patience, max_epochs = 32, 5, 20

    train_ds = TensorDataset(torch.from_numpy(X_train),
                             torch.from_numpy(Y_train.reshape(len(Y_train), -1)))
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    val_ds = TensorDataset(torch.from_numpy(X_val),
                           torch.from_numpy(Y_val.reshape(len(Y_val), -1)))
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    best_val = float("inf")
    best_state = None
    wait = 0

    for epoch in range(max_epochs):
        model.train()
        total_loss = 0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = criterion(pred.reshape(pred.size(0), -1), yb)
            optimizer.zero_grad()
            loss.backward()
            if phase == "multistep":
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * xb.size(0)

        model.eval()
        val_loss_sum = 0
        val_n = 0
        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                val_loss_sum += criterion(pred.reshape(pred.size(0), -1), yb).item() * xb.size(0)
                val_n += xb.size(0)
        val_loss = val_loss_sum / val_n

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"      Early stop epoch {epoch + 1}")
                break

    # Eval with fresh model (IPEX workaround)
    if model_name == "CNN-GRU-MS-Small":
        eval_model = CNNGRU_MS(gru_hidden=90).to(device)
    else:
        eval_model = model_map[model_name]().to(device)
    try:
        eval_model.load_state_dict(best_state)
    except RuntimeError:
        clean = {k.replace("_orig_mod.", ""): v for k, v in best_state.items()}
        eval_model.load_state_dict(clean)
    eval_model.eval()

    # Inference
    test_ds = TensorDataset(torch.from_numpy(X_test))
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    preds = []
    with torch.no_grad():
        for (xb,) in test_dl:
            preds.append(eval_model(xb.to(device)).cpu().numpy())
    pred_norm = np.concatenate(preds, axis=0)

    if phase == "onestep":
        pred_flat = pred_norm.reshape(len(pred_norm), 2, ny, nx)
        pred_u = denorm_u(pred_flat[:, 0], data)
        pred_v = denorm_v(pred_flat[:, 1], data)
        true_u = denorm_u(Y_test[:, 0], data)
        true_v = denorm_v(Y_test[:, 1], data)
        return {
            "rmse_u": rmse_sea(pred_u, true_u, sea_mask),
            "rmse_v": rmse_sea(pred_v, true_v, sea_mask),
            "n_params": n_params,
            "best_epoch": epoch + 1 - wait,
        }
    else:
        # Multi-step: per lead time
        lead_results = {}
        for h_step in range(H):
            pu = denorm_u(pred_norm[:, h_step, 0], data)
            pv = denorm_v(pred_norm[:, h_step, 1], data)
            tu = denorm_u(Y_test[:, h_step, 0], data)
            tv = denorm_v(Y_test[:, h_step, 1], data)
            lead_results[f"t+{h_step + 1}"] = {
                "rmse_u": rmse_sea(pu, tu, sea_mask),
                "rmse_v": rmse_sea(pv, tv, sea_mask),
            }
        avg_u = float(np.mean([v["rmse_u"] for v in lead_results.values()]))
        avg_v = float(np.mean([v["rmse_v"] for v in lead_results.values()]))
        return {
            "avg_rmse_u": avg_u, "avg_rmse_v": avg_v,
            "per_lead": lead_results, "n_params": n_params,
            "best_epoch": epoch + 1 - wait,
        }


def run_multiseed(data):
    """B1: Multi-seed training for DL models."""
    print("\n" + "=" * 70)
    print("B1: MULTI-SEED UNCERTAINTY QUANTIFICATION")
    print("=" * 70)

    results = {}

    # Phase 1 DL models (T=3, one-step)
    for model_name in ["CNN", "GRU", "CNN-GRU"]:
        results[model_name] = {"seeds": {}}
        for seed in SEEDS:
            print(f"\n  {model_name} seed={seed}")
            r = train_with_seed(data, seed, model_name, T=3, phase="onestep")
            results[model_name]["seeds"][str(seed)] = r
            print(f"    RMSE_U={r['rmse_u']:.3f}, RMSE_V={r['rmse_v']:.3f}")

        # Compute mean/std
        rmse_u = [r["rmse_u"] for r in results[model_name]["seeds"].values()]
        rmse_v = [r["rmse_v"] for r in results[model_name]["seeds"].values()]
        results[model_name]["mean_rmse_u"] = float(np.mean(rmse_u))
        results[model_name]["std_rmse_u"] = float(np.std(rmse_u))
        results[model_name]["mean_rmse_v"] = float(np.mean(rmse_v))
        results[model_name]["std_rmse_v"] = float(np.std(rmse_v))
        print(f"\n  {model_name}: U={np.mean(rmse_u):.3f}±{np.std(rmse_u):.3f}, "
              f"V={np.mean(rmse_v):.3f}±{np.std(rmse_v):.3f}")

    # Phase 3: CNN-GRU-MS (T=12, multi-step) — most expensive
    model_name = "CNN-GRU-MS"
    results[model_name] = {"seeds": {}}
    for seed in SEEDS:
        print(f"\n  {model_name} seed={seed}")
        r = train_with_seed(data, seed, model_name, T=12, phase="multistep")
        results[model_name]["seeds"][str(seed)] = r
        print(f"    Avg RMSE_U={r['avg_rmse_u']:.3f}, Avg RMSE_V={r['avg_rmse_v']:.3f}")

    avg_u = [r["avg_rmse_u"] for r in results[model_name]["seeds"].values()]
    avg_v = [r["avg_rmse_v"] for r in results[model_name]["seeds"].values()]
    results[model_name]["mean_avg_rmse_u"] = float(np.mean(avg_u))
    results[model_name]["std_avg_rmse_u"] = float(np.std(avg_u))
    results[model_name]["mean_avg_rmse_v"] = float(np.mean(avg_v))
    results[model_name]["std_avg_rmse_v"] = float(np.std(avg_v))
    print(f"\n  CNN-GRU-MS: U={np.mean(avg_u):.3f}±{np.std(avg_u):.3f}, "
          f"V={np.mean(avg_v):.3f}±{np.std(avg_v):.3f}")

    out_path = os.path.join(OUTPUT_DIR, "multiseed_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\nSaved to {out_path}")
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# B2: CAPACITY-MATCHED CNN-GRU-MS-SMALL
# ═══════════════════════════════════════════════════════════════════════════════

def run_capacity_matched(data):
    """B2: CNN-GRU-MS-Small with ~465K params to match BiEF."""
    print("\n" + "=" * 70)
    print("B2: CAPACITY-MATCHED CNN-GRU-MS-SMALL (gru_hidden=90)")
    print("=" * 70)

    results = {"seeds": {}}
    for seed in SEEDS[:3]:  # 3 seeds for speed
        print(f"\n  CNN-GRU-MS-Small seed={seed}")
        r = train_with_seed(data, seed, "CNN-GRU-MS-Small", T=12, phase="multistep")
        results["seeds"][str(seed)] = r
        print(f"    Avg RMSE_U={r['avg_rmse_u']:.3f}, Avg RMSE_V={r['avg_rmse_v']:.3f}")

    avg_u = [r["avg_rmse_u"] for r in results["seeds"].values()]
    avg_v = [r["avg_rmse_v"] for r in results["seeds"].values()]
    results["mean_avg_rmse_u"] = float(np.mean(avg_u))
    results["std_avg_rmse_u"] = float(np.std(avg_u))
    results["mean_avg_rmse_v"] = float(np.mean(avg_v))
    results["std_avg_rmse_v"] = float(np.std(avg_v))
    results["n_params"] = results["seeds"][str(SEEDS[0])]["n_params"]
    print(f"\n  CNN-GRU-MS-Small ({results['n_params']:,} params): "
          f"U={np.mean(avg_u):.3f}±{np.std(avg_u):.3f}, "
          f"V={np.mean(avg_v):.3f}±{np.std(avg_v):.3f}")

    out_path = os.path.join(OUTPUT_DIR, "capacity_matched_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\nSaved to {out_path}")
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# B3: SEASONAL / MONSOON STRATIFICATION
# ═══════════════════════════════════════════════════════════════════════════════

def run_seasonal(data):
    """B3: Stratify existing Phase 1 results by monsoon season."""
    print("\n" + "=" * 70)
    print("B3: SEASONAL / MONSOON ERROR STRATIFICATION")
    print("=" * 70)

    import torch
    import torch.nn as nn

    T = 3
    ny, nx = data["ny"], data["nx"]
    sea_mask = data["sea_mask"]

    X, Y = build_onestep_sequences(data["U_norm"], data["V_norm"], T)
    idx_val = data["idx_val_end"] - T
    X_test, Y_test = X[idx_val:], Y[idx_val:]
    test_times = data["times"][data["idx_val_end"]:][:len(X_test)]

    # Season classification
    months = np.array([t.astype("datetime64[M]").astype(int) % 12 + 1 for t in test_times])
    seasons = {}
    seasons["SE monsoon (Jul-Aug)"] = (months >= 7) & (months <= 8)
    seasons["Transition (Sep-Nov)"] = (months >= 9) & (months <= 11)
    seasons["NW monsoon (Dec-Feb)"] = (months == 12) | (months <= 2)

    # Persistence predictions
    pers_u = data["U"][data["idx_val_end"] - 1:][:len(X_test)]
    pers_v = data["V"][data["idx_val_end"] - 1:][:len(X_test)]
    true_u = data["U"][data["idx_val_end"]:][:len(X_test)]
    true_v = data["V"][data["idx_val_end"]:][:len(X_test)]

    results = {"Persistence": {}}
    for sname, smask in seasons.items():
        if smask.sum() == 0:
            continue
        results["Persistence"][sname] = {
            "n_steps": int(smask.sum()),
            "rmse_u": rmse_sea(pers_u[smask], true_u[smask], sea_mask),
            "rmse_v": rmse_sea(pers_v[smask], true_v[smask], sea_mask),
        }

    # Load DL checkpoints and evaluate per season
    device = torch.device("cpu")

    class CNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(T * 2, 32, 3, padding=1), nn.ELU(),
                nn.Conv2d(32, 32, 3, padding=1), nn.ELU(),
                nn.MaxPool2d(2), nn.Dropout(0.2),
                nn.Conv2d(32, 64, 3, padding=1), nn.ELU(),
                nn.Conv2d(64, 64, 3, padding=1), nn.ELU(),
                nn.MaxPool2d(2), nn.Dropout(0.2),
            )
            h, w = ny // 4, nx // 4
            self.fc = nn.Sequential(
                nn.Linear(64 * h * w, 512), nn.ReLU(),
                nn.Linear(512, 2 * ny * nx), nn.Sigmoid(),
            )
        def forward(self, x):
            return self.fc(self.conv(x.reshape(x.size(0), -1, ny, nx)).reshape(x.size(0), -1))

    class GRUModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.gru = nn.GRU(2 * ny * nx, 256, batch_first=True)
            self.fc = nn.Sequential(nn.Linear(256, 2 * ny * nx), nn.Sigmoid())
        def forward(self, x):
            seq = x.reshape(x.size(0), x.size(1), -1)
            _, h = self.gru(seq)
            return self.fc(h.squeeze(0))

    class CNNGRU(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(2, 32, 3, padding=1), nn.ELU(),
                nn.Conv2d(32, 32, 3, padding=1), nn.ELU(),
                nn.MaxPool2d(2), nn.Dropout(0.2),
                nn.Conv2d(32, 64, 3, padding=1), nn.ELU(),
                nn.Conv2d(64, 64, 3, padding=1), nn.ELU(),
                nn.MaxPool2d(2), nn.Dropout(0.2),
            )
            h, w = ny // 4, nx // 4
            self.gru = nn.GRU(64 * h * w, 256, batch_first=True)
            self.fc = nn.Sequential(nn.Linear(256, 2 * ny * nx), nn.Sigmoid())
        def forward(self, x):
            b = x.size(0)
            feats = [self.conv(x[:, t]).reshape(b, -1) for t in range(x.size(1))]
            _, h = self.gru(torch.stack(feats, 1))
            return self.fc(h.squeeze(0))

    model_classes = {"CNN": CNN, "GRU": GRUModel, "CNN-GRU": CNNGRU}

    for mname, ModelClass in model_classes.items():
        ckpt_name = f"{mname.replace('-', '_')}_T3_onestep.pt"
        ckpt_path = os.path.join(CHECKPOINT_DIR, ckpt_name)
        if not os.path.exists(ckpt_path):
            print(f"  Skipping {mname}: checkpoint not found at {ckpt_path}")
            continue

        model = ModelClass().to(device)
        state = torch.load(ckpt_path, map_location=device, weights_only=True)
        try:
            model.load_state_dict(state)
        except RuntimeError:
            clean = {k.replace("_orig_mod.", ""): v for k, v in state.items()}
            model.load_state_dict(clean)
        model.eval()

        # Full test prediction
        from torch.utils.data import TensorDataset, DataLoader
        test_ds = TensorDataset(torch.from_numpy(X_test))
        test_dl = DataLoader(test_ds, batch_size=64, shuffle=False)
        preds = []
        with torch.no_grad():
            for (xb,) in test_dl:
                preds.append(model(xb.to(device)).cpu().numpy())
        pred_norm = np.concatenate(preds, 0).reshape(-1, 2, ny, nx)
        pred_u = denorm_u(pred_norm[:, 0], data)
        pred_v = denorm_v(pred_norm[:, 1], data)

        results[mname] = {}
        for sname, smask in seasons.items():
            if smask.sum() == 0:
                continue
            results[mname][sname] = {
                "n_steps": int(smask.sum()),
                "rmse_u": rmse_sea(pred_u[smask], true_u[smask], sea_mask),
                "rmse_v": rmse_sea(pred_v[smask], true_v[smask], sea_mask),
            }

    # Print summary
    print("\nSeasonal RMSE summary:")
    print(f"{'Model':15s}", end="")
    for sname in seasons:
        print(f"  {sname:25s}", end="")
    print()
    for mname, mresults in results.items():
        print(f"{mname:15s}", end="")
        for sname in seasons:
            if sname in mresults:
                r = mresults[sname]
                print(f"  U={r['rmse_u']:5.2f} V={r['rmse_v']:5.2f} n={r['n_steps']:4d}", end="")
            else:
                print(f"  {'N/A':25s}", end="")
        print()

    out_path = os.path.join(OUTPUT_DIR, "seasonal_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\nSaved to {out_path}")
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# B4: TIDAL-RESIDUAL ERROR DECOMPOSITION
# ═══════════════════════════════════════════════════════════════════════════════

def run_tidal_decomposition(data):
    """B4: Decompose errors into tidal and residual components."""
    print("\n" + "=" * 70)
    print("B4: TIDAL-RESIDUAL ERROR DECOMPOSITION")
    print("=" * 70)

    import torch
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader

    try:
        from utide import solve, reconstruct
    except ImportError:
        print("ERROR: utide not installed. pip install utide")
        return None

    T = 3
    ny, nx = data["ny"], data["nx"]
    sea_mask = data["sea_mask"]
    idx_train = data["idx_train_end"]
    idx_test = data["idx_val_end"]
    times = data["times"]

    # 1. Fit UTide on training data, predict for test period
    print("  Fitting UTide on training data per cell...")
    import matplotlib.dates as mdates

    train_times_dt = [np.datetime64(t, "s").astype(datetime) for t in times[:idx_train]]
    test_times_dt = [np.datetime64(t, "s").astype(datetime) for t in times[idx_test:]]
    train_mpl = mdates.date2num(train_times_dt)
    test_mpl = mdates.date2num(test_times_dt)

    n_test = len(times) - idx_test
    tidal_u = np.zeros((n_test, ny, nx), dtype=np.float32)
    tidal_v = np.zeros((n_test, ny, nx), dtype=np.float32)

    sea_yx = list(zip(*np.where(sea_mask)))
    for ci, (yi, xi) in enumerate(sea_yx):
        u_train = data["U"][:idx_train, yi, xi]
        v_train = data["V"][:idx_train, yi, xi]

        valid = np.isfinite(u_train) & np.isfinite(v_train)
        if valid.sum() < 1000:
            continue

        try:
            coef_u = solve(train_mpl[valid], u_train[valid], lat=-6.0,
                          constit=["M2", "S2", "K1", "O1", "N2", "K2", "P1", "M4"])
            coef_v = solve(train_mpl[valid], v_train[valid], lat=-6.0,
                          constit=["M2", "S2", "K1", "O1", "N2", "K2", "P1", "M4"])
            rec_u = reconstruct(test_mpl, coef_u)
            rec_v = reconstruct(test_mpl, coef_v)
            tidal_u[:, yi, xi] = rec_u.h.astype(np.float32)
            tidal_v[:, yi, xi] = rec_v.h.astype(np.float32)
        except Exception as e:
            if ci < 3:
                print(f"    Cell ({yi},{xi}) failed: {e}")

        if (ci + 1) % 50 == 0:
            print(f"    UTide: {ci + 1}/{len(sea_yx)} cells")

    print(f"  UTide fitted at {len(sea_yx)} cells")

    # 2. Compute observed residuals
    obs_u = data["U"][idx_test:]
    obs_v = data["V"][idx_test:]
    resid_u = obs_u - tidal_u
    resid_v = obs_v - tidal_v

    # 3. Load DL model predictions and decompose
    X, Y = build_onestep_sequences(data["U_norm"], data["V_norm"], T)
    idx_val_seq = data["idx_val_end"] - T
    X_test_seq = X[idx_val_seq:]
    # Align: model predictions start at test_time[T], so skip first T tidal entries
    # Actually model test starts at idx_val_end (first predicted timestep)
    # The sequences are offset by T, so predicted time index = idx_val_end + i for sample i
    n_model_test = len(X_test_seq)

    device = torch.device("cpu")

    # Redefine models (same as seasonal)
    class CNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(T * 2, 32, 3, padding=1), nn.ELU(),
                nn.Conv2d(32, 32, 3, padding=1), nn.ELU(),
                nn.MaxPool2d(2), nn.Dropout(0.2),
                nn.Conv2d(32, 64, 3, padding=1), nn.ELU(),
                nn.Conv2d(64, 64, 3, padding=1), nn.ELU(),
                nn.MaxPool2d(2), nn.Dropout(0.2),
            )
            h, w = ny // 4, nx // 4
            self.fc = nn.Sequential(nn.Linear(64 * h * w, 512), nn.ReLU(),
                                    nn.Linear(512, 2 * ny * nx), nn.Sigmoid())
        def forward(self, x):
            return self.fc(self.conv(x.reshape(x.size(0), -1, ny, nx)).reshape(x.size(0), -1))

    class CNNGRU(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(2, 32, 3, padding=1), nn.ELU(),
                nn.Conv2d(32, 32, 3, padding=1), nn.ELU(),
                nn.MaxPool2d(2), nn.Dropout(0.2),
                nn.Conv2d(32, 64, 3, padding=1), nn.ELU(),
                nn.Conv2d(64, 64, 3, padding=1), nn.ELU(),
                nn.MaxPool2d(2), nn.Dropout(0.2),
            )
            h, w = ny // 4, nx // 4
            self.gru = nn.GRU(64 * h * w, 256, batch_first=True)
            self.fc = nn.Sequential(nn.Linear(256, 2 * ny * nx), nn.Sigmoid())
        def forward(self, x):
            b = x.size(0)
            feats = [self.conv(x[:, t]).reshape(b, -1) for t in range(x.size(1))]
            _, h = self.gru(torch.stack(feats, 1))
            return self.fc(h.squeeze(0))

    results = {}
    model_classes = {"CNN": CNN, "CNN-GRU": CNNGRU}

    # Also evaluate persistence and UTide baseline
    # Model predictions align with target times: idx_val_end, idx_val_end+1, ...
    # obs_u/tidal_u index 0 = idx_val_end, so use direct indexing
    # Persistence for target time i: obs at i-1; need i >= 1
    n_eval = n_model_test - 1  # skip first to allow persistence lag
    true_u_aligned = obs_u[1:1 + n_eval]
    true_v_aligned = obs_v[1:1 + n_eval]
    tidal_true_u = tidal_u[1:1 + n_eval]
    tidal_true_v = tidal_v[1:1 + n_eval]
    resid_true_u = true_u_aligned - tidal_true_u
    resid_true_v = true_v_aligned - tidal_true_v

    pers_pred_u = obs_u[:n_eval]
    pers_pred_v = obs_v[:n_eval]
    # Persistence tidal/residual
    pers_tidal_pred_u = tidal_u[:n_eval]
    pers_tidal_pred_v = tidal_v[:n_eval]
    pers_resid_pred_u = pers_pred_u - pers_tidal_pred_u
    pers_resid_pred_v = pers_pred_v - pers_tidal_pred_v

    results["Persistence"] = {
        "total_rmse_u": rmse_sea(pers_pred_u, true_u_aligned, sea_mask),
        "total_rmse_v": rmse_sea(pers_pred_v, true_v_aligned, sea_mask),
        "tidal_rmse_u": rmse_sea(pers_tidal_pred_u, tidal_true_u, sea_mask),
        "tidal_rmse_v": rmse_sea(pers_tidal_pred_v, tidal_true_v, sea_mask),
        "resid_rmse_u": rmse_sea(pers_resid_pred_u, resid_true_u, sea_mask),
        "resid_rmse_v": rmse_sea(pers_resid_pred_v, resid_true_v, sea_mask),
    }

    # UTide baseline: pred = tidal at target time
    results["UTide"] = {
        "total_rmse_u": rmse_sea(tidal_true_u, true_u_aligned, sea_mask),
        "total_rmse_v": rmse_sea(tidal_true_v, true_v_aligned, sea_mask),
        "tidal_rmse_u": 0.0,  # UTide perfectly predicts tidal component by definition
        "tidal_rmse_v": 0.0,
        "resid_rmse_u": rmse_sea(np.zeros_like(resid_true_u), resid_true_u, sea_mask),
        "resid_rmse_v": rmse_sea(np.zeros_like(resid_true_v), resid_true_v, sea_mask),
    }

    for mname, ModelClass in model_classes.items():
        ckpt_name = f"{mname.replace('-', '_')}_T3_onestep.pt"
        ckpt_path = os.path.join(CHECKPOINT_DIR, ckpt_name)
        if not os.path.exists(ckpt_path):
            print(f"  Skipping {mname}: no checkpoint")
            continue

        model = ModelClass().to(device)
        state = torch.load(ckpt_path, map_location=device, weights_only=True)
        try:
            model.load_state_dict(state)
        except RuntimeError:
            clean = {k.replace("_orig_mod.", ""): v for k, v in state.items()}
            model.load_state_dict(clean)
        model.eval()

        test_ds = TensorDataset(torch.from_numpy(X_test_seq))
        test_dl = DataLoader(test_ds, batch_size=64, shuffle=False)
        preds = []
        with torch.no_grad():
            for (xb,) in test_dl:
                preds.append(model(xb.to(device)).cpu().numpy())
        pred_norm = np.concatenate(preds, 0).reshape(-1, 2, ny, nx)
        pred_u = denorm_u(pred_norm[:, 0], data)
        pred_v = denorm_v(pred_norm[:, 1], data)

        # Align DL predictions to the same n_eval window (skip first prediction)
        pred_u_al = pred_u[1:1 + n_eval]
        pred_v_al = pred_v[1:1 + n_eval]

        # Decompose prediction: model_tidal = tidal at predicted time (known)
        # model_resid = model_pred - tidal at predicted time
        model_resid_u = pred_u_al - tidal_true_u
        model_resid_v = pred_v_al - tidal_true_v

        results[mname] = {
            "total_rmse_u": rmse_sea(pred_u_al, true_u_aligned, sea_mask),
            "total_rmse_v": rmse_sea(pred_v_al, true_v_aligned, sea_mask),
            "tidal_rmse_u": 0.0,  # DL model doesn't explicitly predict tidal component
            "tidal_rmse_v": 0.0,
            "resid_rmse_u": rmse_sea(model_resid_u, resid_true_u, sea_mask),
            "resid_rmse_v": rmse_sea(model_resid_v, resid_true_v, sea_mask),
        }

    # Print summary
    print("\nTidal-Residual Decomposition:")
    print(f"{'Model':15s} {'Total_U':>8s} {'Tidal_U':>8s} {'Resid_U':>8s} "
          f"{'Total_V':>8s} {'Tidal_V':>8s} {'Resid_V':>8s}")
    for mname, r in results.items():
        print(f"{mname:15s} {r['total_rmse_u']:8.2f} {r['tidal_rmse_u']:8.2f} "
              f"{r['resid_rmse_u']:8.2f} {r['total_rmse_v']:8.2f} "
              f"{r['tidal_rmse_v']:8.2f} {r['resid_rmse_v']:8.2f}")

    out_path = os.path.join(OUTPUT_DIR, "tidal_decomposition.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\nSaved to {out_path}")
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Paper 1.2 Phase B experiments")
    parser.add_argument("--task", type=str, default="all",
                        choices=["seed", "capacity", "seasonal", "tidal", "all"])
    args = parser.parse_args()

    data = load_data()

    if args.task in ("seasonal", "all"):
        run_seasonal(data)

    if args.task in ("tidal", "all"):
        run_tidal_decomposition(data)

    if args.task in ("capacity", "all"):
        run_capacity_matched(data)

    if args.task in ("seed", "all"):
        run_multiseed(data)

    print("\nAll Phase B tasks complete!")
