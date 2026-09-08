#!/usr/bin/env python3
"""
Paper 1.2 (ASCMO-2026-18) — DL vector & gap-split metrics.

The submitted Phase-1 deep-learning checkpoints live on the (currently
unreachable) IPEX server, so this script retrains the three Phase-1 models
(CNN, GRU, CNN-GRU) locally with fixed seeds, verifies they reproduce the
published Table 2 RMSE, and then computes the metrics the referees asked for
but which require per-timestep predictions:

  * vector-aware error (speed RMSE, complex/vector RMSE, direction MAE) -> R2.minor1
  * RMSE split by gap-filled vs genuinely-observed test targets         -> R2.4

Uses the SAME data, sea mask, chronological splits, and min-max
normalisation as train_paper12.py, and evaluates on the manuscript test
window [22560:28321] (5,761 steps) for direct comparability with the
persistence/ARIMA numbers from reviewer_response.py.

Outputs -> paper12/output/reviewer/dl_vector_gap.json
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
DATA_PATH = os.path.join(REPO, "data/processed/BADA_hourly_qc.nc")
OUT_DIR = os.path.join(REPO, "paper12/output/reviewer")
os.makedirs(OUT_DIR, exist_ok=True)

TRAIN_END = np.datetime64("2025-01-01")
VAL_END = np.datetime64("2025-07-01")
TEST_END_IDX = 28321          # manuscript test window end (matches ARIMA run)
T = 3
LR = 1e-3
BATCH = 64
EPOCHS = 50
PATIENCE = 10
SEED = 42
PUBLISHED = {"CNN": (11.31, 15.65), "GRU": (11.56, 15.63), "CNN-GRU": (11.33, 15.44)}


def load_data():
    ds = xr.open_dataset(DATA_PATH)
    U = ds["U"].values.astype(np.float32)
    V = ds["V"].values.astype(np.float32)
    qc = ds["qc_flag"].values
    times = ds["time"].values
    ds.close()
    valid_frac = np.mean(np.isfinite(U), axis=0)
    sea = valid_frac > 0.5
    U = np.nan_to_num(U, nan=0.0); V = np.nan_to_num(V, nan=0.0)
    it = int(np.searchsorted(times, TRAIN_END))
    iv = int(np.searchsorted(times, VAL_END))
    umin = U[:it][:, sea].min(); umax = U[:it][:, sea].max()
    vmin = V[:it][:, sea].min(); vmax = V[:it][:, sea].max()
    Un = (U - umin) / (umax - umin)
    Vn = (V - vmin) / (vmax - vmin)
    return dict(U=U, V=V, Un=Un, Vn=Vn, qc=qc, times=times, sea=sea,
                it=it, iv=iv, ny=U.shape[1], nx=U.shape[2],
                umin=umin, umax=umax, vmin=vmin, vmax=vmax)


def seqs(Un, Vn, T):
    # build (N, T, 2, ny, nx) lookback tensors and (N, 2, ny, nx) targets
    n = Un.shape[0]
    N = n - T
    ny, nx = Un.shape[1], Un.shape[2]
    X = np.empty((N, T, 2, ny, nx), np.float32)
    Y = np.empty((N, 2, ny, nx), np.float32)
    for i in range(N):
        for t in range(T):
            X[i, t, 0] = Un[i + t]; X[i, t, 1] = Vn[i + t]
        Y[i, 0] = Un[i + T]; Y[i, 1] = Vn[i + T]
    return X, Y


class CNN(nn.Module):
    def __init__(self, ny, nx):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(T * 2, 32, 3, padding=1), nn.ELU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.ELU(),
            nn.MaxPool2d(2), nn.Dropout(0.2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ELU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ELU(),
            nn.MaxPool2d(2), nn.Dropout(0.2))
        h, w = ny // 4, nx // 4
        self.fc = nn.Sequential(nn.Linear(64 * h * w, 512), nn.ReLU(),
                                nn.Linear(512, 2 * ny * nx), nn.Sigmoid())
        self.ny, self.nx = ny, nx

    def forward(self, x):
        b = x.size(0)
        x = x.reshape(b, -1, self.ny, self.nx)
        return self.fc(self.conv(x).reshape(b, -1))


class GRU(nn.Module):
    def __init__(self, ny, nx, hidden=256):
        super().__init__()
        self.gru = nn.GRU(2 * ny * nx, hidden, batch_first=True)
        self.fc = nn.Sequential(nn.Linear(hidden, 2 * ny * nx), nn.Sigmoid())
        self.ny, self.nx = ny, nx

    def forward(self, x):
        b = x.size(0)
        x = x.reshape(b, T, -1)
        _, h = self.gru(x)
        return self.fc(h.squeeze(0))


class CNNGRU(nn.Module):
    def __init__(self, ny, nx, gru_hidden=256):
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
        self.fc = nn.Sequential(nn.Linear(gru_hidden, 2 * ny * nx), nn.Sigmoid())
        self.ny, self.nx = ny, nx

    def forward(self, x):
        b = x.size(0)
        feats = [self.conv(x[:, t]).reshape(b, -1) for t in range(x.size(1))]
        _, h = self.gru(torch.stack(feats, 1))
        return self.fc(h.squeeze(0))


def vector_metrics(pu, pv, tu, tv, thr=10.0):
    pspd = np.sqrt(pu ** 2 + pv ** 2); tspd = np.sqrt(tu ** 2 + tv ** 2)
    m = tspd > thr
    dang = np.degrees(np.arctan2(pv[m], pu[m]) - np.arctan2(tv[m], tu[m]))
    dang = (dang + 180) % 360 - 180
    return {"speed_rmse": float(np.sqrt(np.mean((pspd - tspd) ** 2))),
            "speed_bias": float(np.mean(pspd - tspd)),
            "vector_rmse": float(np.sqrt(np.mean((pu - tu) ** 2 + (pv - tv) ** 2))),
            "direction_mae_deg": float(np.mean(np.abs(dang))),
            "direction_rmse_deg": float(np.sqrt(np.mean(dang ** 2))),
            "n_dir": int(m.sum())}


def train_eval(name, ModelClass, d, X, Y):
    torch.manual_seed(SEED); np.random.seed(SEED)
    ny, nx, sea = d["ny"], d["nx"], d["sea"]
    it, iv = d["it"] - T, d["iv"] - T
    Xtr, Ytr = X[:it], Y[:it]
    Xva, Yva = X[it:iv], Y[it:iv]
    Xte, Yte = X[iv:], Y[iv:]

    model = ModelClass(ny, nx)
    n_params = sum(p.numel() for p in model.parameters())
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
            loss = crit(pred, yb.reshape(yb.size(0), -1))
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            vl = crit(model(Xva_t), Yva_t).item()
        if vl < best:
            best, best_state, wait = vl, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            wait += 1
            if wait >= PATIENCE:
                break
    model.load_state_dict(best_state); model.eval()

    # predictions over the full available test set, then clip to manuscript window
    preds = []
    with torch.no_grad():
        for (xb,) in DataLoader(TensorDataset(torch.from_numpy(Xte)),
                                batch_size=BATCH, shuffle=False):
            preds.append(model(xb).numpy())
    pn = np.concatenate(preds, 0).reshape(-1, 2, ny, nx)
    Up = pn[:, 0] * (d["umax"] - d["umin"]) + d["umin"]
    Vp = pn[:, 1] * (d["vmax"] - d["vmin"]) + d["vmin"]
    Ut = Yte[:, 0] * (d["umax"] - d["umin"]) + d["umin"]
    Vt = Yte[:, 1] * (d["vmax"] - d["vmin"]) + d["vmin"]

    # raw time index of test sample i is iv_full + i = d['iv'] + i
    n_keep = TEST_END_IDX - d["iv"]
    Up, Vp, Ut, Vt = Up[:n_keep], Vp[:n_keep], Ut[:n_keep], Vt[:n_keep]

    smb = sea
    ru = float(np.sqrt(np.mean((Up[:, smb] - Ut[:, smb]) ** 2)))
    rv = float(np.sqrt(np.mean((Vp[:, smb] - Vt[:, smb]) ** 2)))

    # vector metrics over sea cells
    vec = vector_metrics(Up[:, smb].ravel(), Vp[:, smb].ravel(),
                         Ut[:, smb].ravel(), Vt[:, smb].ravel())

    # gap split (target step t+1 is raw index d['iv']+i+ ... actually target is
    # the predicted timestep = raw index d['iv'] + i for sample i in test)
    qc_t = d["qc"][d["iv"]:TEST_END_IDX]      # (n_keep, ny, nx) flag of target step
    is_gap = qc_t != 0
    gap = is_gap & smb[None]; obs = (~is_gap) & smb[None]
    gapm = {
        "rmse_u_obs": float(np.sqrt(np.mean((Up[obs] - Ut[obs]) ** 2))),
        "rmse_v_obs": float(np.sqrt(np.mean((Vp[obs] - Vt[obs]) ** 2))),
        "rmse_u_gap": float(np.sqrt(np.mean((Up[gap] - Ut[gap]) ** 2))),
        "rmse_v_gap": float(np.sqrt(np.mean((Vp[gap] - Vt[gap]) ** 2))),
        "n_obs": int(obs.sum()), "n_gap": int(gap.sum())}

    pub = PUBLISHED.get(name, (None, None))
    print(f"  {name:8s} RMSE U={ru:.2f} V={rv:.2f} (published {pub[0]}/{pub[1]})  "
          f"dirMAE={vec['direction_mae_deg']:.1f}deg  "
          f"gapU={gapm['rmse_u_gap']:.1f}/obsU={gapm['rmse_u_obs']:.1f}  "
          f"[{time.time()-t0:.0f}s, {n_params/1e6:.2f}M]")
    return {"rmse_u": ru, "rmse_v": rv, "published": pub, "n_params": n_params,
            "vector": vec, "gap": gapm}


def main():
    print(f"Loading data, building T={T} sequences ...")
    d = load_data()
    X, Y = seqs(d["Un"], d["Vn"], T)
    print(f"  sea {int(d['sea'].sum())}  splits it={d['it']} iv={d['iv']} "
          f"test_end={TEST_END_IDX}  seqs {X.shape}")
    out = {}
    for name, M in [("CNN", CNN), ("GRU", GRU), ("CNN-GRU", CNNGRU)]:
        out[name] = train_eval(name, M, d, X, Y)
        with open(os.path.join(OUT_DIR, "dl_vector_gap.json"), "w") as f:
            json.dump(out, f, indent=2, default=float)
    print(f"\nSaved -> {os.path.join(OUT_DIR, 'dl_vector_gap.json')}")


if __name__ == "__main__":
    main()
