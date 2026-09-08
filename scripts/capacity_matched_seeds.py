#!/usr/bin/env python3
"""Multi-seed parameter-matched CNN-GRU-MS (R2-6 strengthening).

Re-runs the 0.48 M capacity-matched CNN-GRU-MS over five seeds (matching the
full CNN-GRU-MS protocol) so the architectural-effect claim at exact capacity
parity with BiEF (0.47 M) is seed-robust rather than single-seed. Reuses the
exact data pipeline, model, and train/eval loop from capacity_matched.py.
CPU only.
"""
import os
import sys
import json
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from capacity_matched import (  # noqa: E402
    load, build_ms, CNNGRU_MS, T, H, LR, BATCH, EPOCHS, PATIENCE,
    GRU_HIDDEN, OUT, PUBLISHED)

SEEDS = [42, 123, 456, 789, 1024]
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_seed(seed, d, X, Y, it, iv):
    torch.manual_seed(seed)
    np.random.seed(seed)
    ny, nx, sea = d["ny"], d["nx"], d["sea"]
    Xtr, Ytr = X[:it], Y[:it]
    Xva, Yva = X[it:iv], Y[it:iv]
    Xte, Yte = X[iv:], Y[iv:]
    model = CNNGRU_MS(ny, nx, GRU_HIDDEN).to(DEV)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    crit = nn.MSELoss()
    tdl = DataLoader(TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(Ytr)),
                     batch_size=BATCH, shuffle=True)
    Xva_t = torch.from_numpy(Xva).to(DEV)
    Yva_t = torch.from_numpy(Yva).reshape(len(Xva), -1).to(DEV)
    best, best_state, wait = 1e9, None, 0
    for ep in range(EPOCHS):
        model.train()
        for xb, yb in tdl:
            xb, yb = xb.to(DEV), yb.to(DEV)
            pred = model(xb)
            loss = crit(pred.reshape(yb.size(0), -1), yb.reshape(yb.size(0), -1))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            vl = crit(model(Xva_t).reshape(len(Xva), -1), Yva_t).item()
        if vl < best:
            best, best_state, wait = vl, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            wait += 1
            if wait >= PATIENCE:
                break
    model.load_state_dict(best_state)
    model.eval()
    preds = []
    with torch.no_grad():
        for (xb,) in DataLoader(TensorDataset(torch.from_numpy(Xte)), batch_size=BATCH):
            preds.append(model(xb.to(DEV)).cpu().numpy())
    P = np.concatenate(preds, 0)
    du, dv = d["umax"] - d["umin"], d["vmax"] - d["vmin"]
    Pu = P[:, :, 0] * du + d["umin"]
    Pv = P[:, :, 1] * dv + d["vmin"]
    Tu = Yte[:, :, 0] * du + d["umin"]
    Tv = Yte[:, :, 1] * dv + d["vmin"]
    ru = [float(np.sqrt(np.mean((Pu[:, h][:, sea] - Tu[:, h][:, sea]) ** 2))) for h in range(H)]
    rv = [float(np.sqrt(np.mean((Pv[:, h][:, sea] - Tv[:, h][:, sea]) ** 2))) for h in range(H)]
    return float(np.mean(ru)), float(np.mean(rv)), ep + 1, n_params


def main():
    d = load()
    X, Y = build_ms(d["Un"], d["Vn"], T, H)
    it, iv = d["it"] - T, d["iv"] - T
    aus, avs, npar = [], [], 0
    for s in SEEDS:
        t0 = time.time()
        au, av, eps, npar = run_seed(s, d, X, Y, it, iv)
        aus.append(au)
        avs.append(av)
        print(f"seed {s}: U={au:.3f} V={av:.3f}  ({eps} ep, {time.time()-t0:.0f}s)", flush=True)
    aus, avs = np.array(aus), np.array(avs)
    bi_u, bi_v = PUBLISHED["BiEF (0.47M)"]
    out = {
        "model": f"CNN-GRU-MS(gru_hidden={GRU_HIDDEN})", "n_params": int(npar),
        "params_M": round(npar / 1e6, 3), "seeds": SEEDS,
        "avg_rmse_u_mean": float(aus.mean()), "avg_rmse_u_std": float(aus.std(ddof=0)),
        "avg_rmse_v_mean": float(avs.mean()), "avg_rmse_v_std": float(avs.std(ddof=0)),
        "per_seed_u": aus.tolist(), "per_seed_v": avs.tolist(),
        "beats_bief_u_all_seeds": bool((aus < bi_u).all()),
        "beats_bief_v_all_seeds": bool((avs < bi_v).all()),
        "published": PUBLISHED,
    }
    with open(os.path.join(OUT, "phase3_matched_seeds.json"), "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(json.dumps({k: v for k, v in out.items() if not k.startswith("per_seed")}, indent=2))
    print(f"saved {OUT}/phase3_matched_seeds.json")


if __name__ == "__main__":
    main()
