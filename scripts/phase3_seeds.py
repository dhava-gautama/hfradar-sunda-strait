#!/usr/bin/env python3
"""Same-platform, multi-seed Phase-3 comparison (R2-6, apples-to-apples).

Re-runs every trainable Phase-3 multi-step model on ONE platform (GPU) with the
SAME five seeds and the SAME training config, so Table 3 is fully consistent and
the direct-vs-autoregressive / capacity-matched comparison is not confounded by
platform or single-seed noise. Architectures are exact copies of train_paper12.py;
data pipeline/config reused from capacity_matched.py.

Writes results incrementally to paper12/output/reviewer/phase3_controlled_seeds.json after
every (model, seed) so a contended box can be resumed without losing progress.
GPU if available, else CPU.
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
from capacity_matched import load, build_ms, T, H, LR, BATCH, EPOCHS, PATIENCE, OUT  # noqa: E402

BATCH = 32  # uniform across ALL models. On a >=24 GB GPU the autoregressive BiEF/ConvLSTM
            # fit at this batch without OOM, so every Phase-3 model is trained with an
            # identical config (data, splits, batch, epochs, patience, seeds, platform) ->
            # a fully apples-to-apples direct-vs-autoregressive comparison.
SEEDS = [42, 123, 456, 789, 1024]
SEEDS_BY_MODEL = {}  # every model gets all five seeds
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULT = os.path.join(OUT, "phase3_controlled_seeds.json")


# ── architectures (exact copies of train_paper12.py; ny/nx/H explicit) ──────
class ConvLSTMCell(nn.Module):
    def __init__(self, in_ch, hid_ch, kernel=3):
        super().__init__()
        self.hid_ch = hid_ch
        self.gates = nn.Conv2d(in_ch + hid_ch, 4 * hid_ch, kernel, padding=kernel // 2)

    def forward(self, x, state):
        h, c = state
        g = self.gates(torch.cat([x, h], dim=1))
        i, f, o, g_ = g.chunk(4, dim=1)
        i, f, o = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o)
        c = f * c + i * torch.tanh(g_)
        return o * torch.tanh(c), c

    def init_state(self, b, ny, nx, device):
        return (torch.zeros(b, self.hid_ch, ny, nx, device=device),
                torch.zeros(b, self.hid_ch, ny, nx, device=device))


class ConvLSTMED(nn.Module):
    def __init__(self, ny, nx, H, hid=64):
        super().__init__()
        self.enc = ConvLSTMCell(2, hid); self.dec = ConvLSTMCell(2, hid)
        self.out = nn.Conv2d(hid, 2, 1); self.ny, self.nx, self.H = ny, nx, H

    def forward(self, x):
        b, Tin = x.size(0), x.size(1)
        state = self.enc.init_state(b, self.ny, self.nx, x.device)
        for t in range(Tin):
            state = self.enc(x[:, t], state)
        outs, inp = [], x[:, -1]
        for _ in range(self.H):
            state = self.dec(inp, state)
            inp = torch.sigmoid(self.out(state[0])); outs.append(inp)
        return torch.stack(outs, dim=1)


class BiEF(nn.Module):
    def __init__(self, ny, nx, H, hid=64):
        super().__init__()
        self.enc_fwd = ConvLSTMCell(2, hid); self.enc_bwd = ConvLSTMCell(2, hid)
        self.merge = nn.Conv2d(2 * hid, hid, 1)
        self.dec = ConvLSTMCell(2, hid); self.out = nn.Conv2d(hid, 2, 1)
        self.ny, self.nx, self.H = ny, nx, H

    def forward(self, x):
        b, Tin = x.size(0), x.size(1)
        sf = self.enc_fwd.init_state(b, self.ny, self.nx, x.device)
        for t in range(Tin):
            sf = self.enc_fwd(x[:, t], sf)
        sb = self.enc_bwd.init_state(b, self.ny, self.nx, x.device)
        for t in range(Tin - 1, -1, -1):
            sb = self.enc_bwd(x[:, t], sb)
        state = (self.merge(torch.cat([sf[0], sb[0]], 1)), self.merge(torch.cat([sf[1], sb[1]], 1)))
        outs, inp = [], x[:, -1]
        for _ in range(self.H):
            state = self.dec(inp, state)
            inp = torch.sigmoid(self.out(state[0])); outs.append(inp)
        return torch.stack(outs, dim=1)


class CNNGRU_MS(nn.Module):
    def __init__(self, ny, nx, H, gru_hidden=256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(2, 32, 3, padding=1), nn.ELU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.ELU(), nn.MaxPool2d(2), nn.Dropout(0.2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ELU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ELU(), nn.MaxPool2d(2), nn.Dropout(0.2))
        h, w = ny // 4, nx // 4
        self.gru = nn.GRU(64 * h * w, gru_hidden, batch_first=True)
        self.fc = nn.Sequential(nn.Linear(gru_hidden, H * 2 * ny * nx), nn.Sigmoid())
        self.ny, self.nx, self.H = ny, nx, H

    def forward(self, x):
        b = x.size(0)
        feats = [self.conv(x[:, t]).reshape(b, -1) for t in range(x.size(1))]
        _, h = self.gru(torch.stack(feats, 1))
        return self.fc(h.squeeze(0)).reshape(b, self.H, 2, self.ny, self.nx)


def make_models(ny, nx):
    # ordered so the R2-6-critical, fast direct models finish first;
    # BiEF (the autoregressive comparator) next; ConvLSTM-ED (slowest, least
    # relevant to the capacity claim) last.
    return {
        "CNN-GRU-MS": lambda: CNNGRU_MS(ny, nx, H, 256),
        "CNN-GRU-MS-Matched": lambda: CNNGRU_MS(ny, nx, H, 40),
        "CNN-GRU-MS-Small": lambda: CNNGRU_MS(ny, nx, H, 90),
        "BiEF": lambda: BiEF(ny, nx, H, 64),
        "ConvLSTM-ED": lambda: ConvLSTMED(ny, nx, H, 64),
    }


def train_eval(make, seed, d, X, Y, it, iv):
    torch.manual_seed(seed); np.random.seed(seed)
    sea = d["sea"]
    Xtr, Ytr = X[:it], Y[:it]; Xva, Yva = X[it:iv], Y[it:iv]; Xte, Yte = X[iv:], Y[iv:]
    model = make().to(DEV)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=LR); crit = nn.MSELoss()
    tdl = DataLoader(TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(Ytr)),
                     batch_size=BATCH, shuffle=True)
    Xva_t = torch.from_numpy(Xva).to(DEV); Yva_t = torch.from_numpy(Yva).reshape(len(Xva), -1).to(DEV)
    best, best_state, wait = 1e9, None, 0
    for ep in range(EPOCHS):
        model.train()
        for xb, yb in tdl:
            xb, yb = xb.to(DEV), yb.to(DEV)
            loss = crit(model(xb).reshape(yb.size(0), -1), yb.reshape(yb.size(0), -1))
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        model.eval()
        with torch.no_grad():
            vl = crit(model(Xva_t).reshape(len(Xva), -1), Yva_t).item()
        if vl < best:
            best, best_state, wait = vl, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            wait += 1
            if wait >= PATIENCE:
                break
    model.load_state_dict(best_state); model.eval()
    preds = []
    with torch.no_grad():
        for (xb,) in DataLoader(TensorDataset(torch.from_numpy(Xte)), batch_size=BATCH):
            preds.append(model(xb.to(DEV)).cpu().numpy())
    P = np.concatenate(preds, 0)
    du, dv = d["umax"] - d["umin"], d["vmax"] - d["vmin"]
    Pu = P[:, :, 0] * du + d["umin"]; Pv = P[:, :, 1] * dv + d["vmin"]
    Tu = Yte[:, :, 0] * du + d["umin"]; Tv = Yte[:, :, 1] * dv + d["vmin"]
    au = float(np.mean([np.sqrt(np.mean((Pu[:, h][:, sea] - Tu[:, h][:, sea]) ** 2)) for h in range(H)]))
    av = float(np.mean([np.sqrt(np.mean((Pv[:, h][:, sea] - Tv[:, h][:, sea]) ** 2)) for h in range(H)]))
    return au, av, ep + 1, n_params


def main():
    d = load()
    ny, nx = d["ny"], d["nx"]
    X, Y = build_ms(d["Un"], d["Vn"], T, H)
    it, iv = d["it"] - T, d["iv"] - T
    print(f"device={DEV}  seqs {X.shape}  models x seeds = {5*len(SEEDS)}", flush=True)
    res = json.load(open(RESULT)) if os.path.exists(RESULT) else {"platform": str(DEV), "seeds": SEEDS, "config": {"T": T, "H": H, "lr": LR, "batch": BATCH, "epochs": EPOCHS, "patience": PATIENCE}, "models": {}}
    for name, make in make_models(ny, nx).items():
        m = res["models"].setdefault(name, {"per_seed_u": {}, "per_seed_v": {}})
        for s in SEEDS_BY_MODEL.get(name, SEEDS):
            if str(s) in m["per_seed_u"]:
                continue  # resume
            t0 = time.time()
            au, av, eps, npar = train_eval(make, s, d, X, Y, it, iv)
            m["per_seed_u"][str(s)] = au; m["per_seed_v"][str(s)] = av; m["n_params"] = int(npar)
            print(f"{name} seed {s}: U={au:.3f} V={av:.3f} ({eps} ep, {time.time()-t0:.0f}s)", flush=True)
            json.dump(res, open(RESULT, "w"), indent=2)
        u = np.array(list(m["per_seed_u"].values())); v = np.array(list(m["per_seed_v"].values()))
        m["u_mean"], m["u_std"] = float(u.mean()), float(u.std())
        m["v_mean"], m["v_std"] = float(v.mean()), float(v.std())
        json.dump(res, open(RESULT, "w"), indent=2)
        print(f"  => {name}: U={m['u_mean']:.2f}±{m['u_std']:.2f} V={m['v_mean']:.2f}±{m['v_std']:.2f} ({m['n_params']/1e6:.2f}M)", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
