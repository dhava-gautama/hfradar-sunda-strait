#!/usr/bin/env python3
"""
Paper 1.2 (ASCMO-2026-18) — Parameter-matched Phase-3 comparison (R2-6).

Referee 2 (comment 6) noted that CNN-GRU-MS (2.85 M params) is ~6x larger than its
ConvLSTM competitor BiEF (0.47 M), so the "direct multi-step beats autoregressive"
conclusion is confounded with capacity. The submitted reduced variant
(CNN-GRU-MS-Small, 1.00 M) is still ~2x BiEF. Here we train a CNN-GRU-MS with the GRU
hidden size reduced to 40, giving ~0.48 M parameters --- essentially matched to BiEF
(0.47 M) --- and test whether the direct-prediction advantage survives at equal capacity.

Same data, splits, and Phase-3 protocol as train_paper12.py (T=12, H=6, Adam 1e-3,
batch 32, early stopping patience 5, gradient clipping 1.0). Compares against the
published BiEF (19.43 / 22.99) and CNN-GRU-MS full (18.67 / 22.53) and Small
(19.21 / 23.05) average RMSE.

Output -> paper12/output/reviewer/capacity_matched_b.json
"""
import json
import os
import time

import numpy as np
import xarray as xr
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

torch.set_num_threads(min(12, os.cpu_count()))

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
DATA = os.path.join(REPO, "data/processed/BADA_hourly_qc.nc")
OUT = os.path.join(REPO, "paper12/output/reviewer")
os.makedirs(OUT, exist_ok=True)

TRAIN_END = np.datetime64("2025-01-01")
VAL_END = np.datetime64("2025-07-01")
T, H = 12, 6
LR, BATCH, EPOCHS, PATIENCE, SEED = 1e-3, 32, 20, 5, 42
GRU_HIDDEN = 40   # -> ~0.48 M params, matched to BiEF (0.47 M)

PUBLISHED = {"BiEF (0.47M)": (19.43, 22.99),
             "CNN-GRU-MS-Small (1.00M)": (19.21, 23.05),
             "CNN-GRU-MS full (2.85M)": (18.67, 22.53)}


def load():
    ds = xr.open_dataset(DATA)
    U = ds["U"].values.astype(np.float32); V = ds["V"].values.astype(np.float32)
    t = ds["time"].values; ds.close()
    sea = np.mean(np.isfinite(U), axis=0) > 0.5
    U = np.nan_to_num(U, nan=0.0); V = np.nan_to_num(V, nan=0.0)
    it = int(np.searchsorted(t, TRAIN_END)); iv = int(np.searchsorted(t, VAL_END))
    umin = U[:it][:, sea].min(); umax = U[:it][:, sea].max()
    vmin = V[:it][:, sea].min(); vmax = V[:it][:, sea].max()
    return dict(U=U, V=V, sea=sea, it=it, iv=iv, ny=U.shape[1], nx=U.shape[2],
                umin=umin, umax=umax, vmin=vmin, vmax=vmax,
                Un=(U - umin) / (umax - umin), Vn=(V - vmin) / (vmax - vmin))


def build_ms(Un, Vn, T, H):
    N = len(Un) - T - H + 1
    ny, nx = Un.shape[1], Un.shape[2]
    X = np.zeros((N, T, 2, ny, nx), np.float32)
    Y = np.zeros((N, H, 2, ny, nx), np.float32)
    for i in range(N):
        X[i, :, 0] = Un[i:i + T]; X[i, :, 1] = Vn[i:i + T]
        for h in range(H):
            Y[i, h, 0] = Un[i + T + h]; Y[i, h, 1] = Vn[i + T + h]
    return X, Y


class CNNGRU_MS(nn.Module):
    def __init__(self, ny, nx, gru_hidden=GRU_HIDDEN):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(2, 32, 3, padding=1), nn.ELU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.ELU(),
            nn.MaxPool2d(2), nn.Dropout(0.2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ELU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ELU(),
            nn.MaxPool2d(2), nn.Dropout(0.2))
        h, w = ny // 4, nx // 4
        self.gru = nn.GRU(64 * h * w, gru_hidden, batch_first=True)
        self.fc = nn.Sequential(nn.Linear(gru_hidden, H * 2 * ny * nx), nn.Sigmoid())
        self.ny, self.nx = ny, nx

    def forward(self, x):
        b = x.size(0)
        feats = [self.conv(x[:, t]).reshape(b, -1) for t in range(x.size(1))]
        _, h = self.gru(torch.stack(feats, 1))
        return self.fc(h.squeeze(0)).reshape(b, H, 2, self.ny, self.nx)


def main():
    torch.manual_seed(SEED); np.random.seed(SEED)
    d = load()
    ny, nx, sea = d["ny"], d["nx"], d["sea"]
    print(f"Building multi-step sequences (T={T}, H={H}) ...")
    X, Y = build_ms(d["Un"], d["Vn"], T, H)
    it, iv = d["it"] - T, d["iv"] - T   # sequence-index offsets
    Xtr, Ytr = X[:it], Y[:it]
    Xva, Yva = X[it:iv], Y[it:iv]
    Xte, Yte = X[iv:], Y[iv:]
    print(f"  seqs {X.shape}  train {len(Xtr)} val {len(Xva)} test {len(Xte)}")

    model = CNNGRU_MS(ny, nx, GRU_HIDDEN)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  CNN-GRU-MS(gru_hidden={GRU_HIDDEN}): {n_params:,} params "
          f"({n_params/1e6:.2f} M)  vs BiEF 0.47 M")

    opt = torch.optim.Adam(model.parameters(), lr=LR)
    crit = nn.MSELoss()
    tdl = DataLoader(TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(Ytr)),
                     batch_size=BATCH, shuffle=True)
    Xva_t = torch.from_numpy(Xva); Yva_t = torch.from_numpy(Yva).reshape(len(Xva), -1)
    best, best_state, wait = 1e9, None, 0
    t0 = time.time()
    for ep in range(EPOCHS):
        model.train()
        for xb, yb in tdl:
            pred = model(xb)
            loss = crit(pred.reshape(yb.size(0), -1), yb.reshape(yb.size(0), -1))
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            vl = crit(model(Xva_t).reshape(len(Xva), -1), Yva_t).item()
        print(f"    epoch {ep+1:2d}: val={vl:.6f}  ({time.time()-t0:.0f}s)")
        if vl < best:
            best, best_state, wait = vl, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            wait += 1
            if wait >= PATIENCE:
                print(f"    early stop at epoch {ep+1}"); break
    model.load_state_dict(best_state); model.eval()

    # test predictions, per-lead-time RMSE over sea cells (denormalised)
    preds = []
    with torch.no_grad():
        for (xb,) in DataLoader(TensorDataset(torch.from_numpy(Xte)), batch_size=BATCH):
            preds.append(model(xb).numpy())
    P = np.concatenate(preds, 0)               # (n_test, H, 2, ny, nx)
    du, dv = d["umax"] - d["umin"], d["vmax"] - d["vmin"]
    Pu = P[:, :, 0] * du + d["umin"]; Pv = P[:, :, 1] * dv + d["vmin"]
    Tu = Yte[:, :, 0] * du + d["umin"]; Tv = Yte[:, :, 1] * dv + d["vmin"]
    rmse_u_lead = [float(np.sqrt(np.mean((Pu[:, h][:, sea] - Tu[:, h][:, sea]) ** 2))) for h in range(H)]
    rmse_v_lead = [float(np.sqrt(np.mean((Pv[:, h][:, sea] - Tv[:, h][:, sea]) ** 2))) for h in range(H)]
    avg_u, avg_v = float(np.mean(rmse_u_lead)), float(np.mean(rmse_v_lead))

    out = {"model": f"CNN-GRU-MS(gru_hidden={GRU_HIDDEN})", "n_params": int(n_params),
           "params_M": round(n_params / 1e6, 3), "seed": SEED,
           "avg_rmse_u": avg_u, "avg_rmse_v": avg_v,
           "rmse_u_per_lead": rmse_u_lead, "rmse_v_per_lead": rmse_v_lead,
           "epochs": ep + 1, "published": PUBLISHED}
    with open(os.path.join(OUT, "capacity_matched_b.json"), "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"\n  PARAM-MATCHED CNN-GRU-MS ({n_params/1e6:.2f} M): "
          f"avg RMSE U={avg_u:.2f}  V={avg_v:.2f} cm/s")
    for k, (u, v) in PUBLISHED.items():
        print(f"    cf. {k}: {u} / {v}")
    print(f"  -> {'BEATS' if avg_u < 19.43 else 'does NOT beat'} BiEF on U at matched capacity")


if __name__ == "__main__":
    main()
