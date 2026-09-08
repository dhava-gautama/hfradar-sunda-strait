#!/usr/bin/env python3
"""
Paper 1.2 (ASCMO-2026-18) — minor-revision analyses (round 2).

Addresses the editor's first point and Reviewer 1's remaining comment:
the AIC order search for the per-cell ARIMA baseline was restricted to
p, q <= 2 and the optimum (2,0,2) lies on the boundary of that set, so
higher-order and seasonal specifications must be explored (or the
low-order baseline explicitly qualified).

Stages (run sequentially, one script):
  grid      Extended AIC grid: ARIMA(p,0,q), p,q in {0..4}, d = 0
            (stationarity established by ADF in round 1), on the same 60
            cell-series (30 representative cells x U,V) used for the
            round-1 diagnostics. Reports the distribution of AIC-best
            orders, how many series select an order outside the original
            {0,1,2}^2 boundary, and the AIC gain over (2,0,2).
  forecast  One-step-ahead test RMSE (full 5,761-step test period, fixed
            train-estimated parameters) on the same 60 series for:
            (1,0,1), (2,0,2), and the per-series extended-AIC-best order.
            Shows whether higher orders MATERIALLY improve forecast skill.
  seasonal  Seasonal SARIMA candidates with period m = 12 h (semidiurnal
            tide) on a 10-cell subset: (2,0,2)(1,0,0)12, (2,0,2)(2,0,0)12,
            (1,0,1)(1,0,1)12, (2,0,2)(1,0,1)12. Reports AIC vs the
            non-seasonal (2,0,2), one-step test RMSE, and Ljung-Box
            residual diagnostics (does seasonality whiten residuals?).

Output: paper12/output/reviewer/arima_extended.json

Usage:
  python arima_extended.py [--workers 10] [--stage all]
"""

import argparse
import json
import os
import signal
import sys
import time
import warnings
from multiprocessing import Pool

import numpy as np


class _FitTimeout(Exception):
    pass


def _alarm_handler(signum, frame):
    raise _FitTimeout()


def _with_timeout(seconds, fn, *args, **kwargs):
    """Run fn with a SIGALRM deadline; raises _FitTimeout on expiry.
    Works in Pool workers (unix processes, main thread)."""
    signal.signal(signal.SIGALRM, _alarm_handler)
    signal.alarm(seconds)
    try:
        return fn(*args, **kwargs)
    finally:
        signal.alarm(0)

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from reviewer_response import (  # noqa: E402
    load_data, sample_cells, sea_cells, N_SAMPLE, SAMPLE_SEED,
)

OUT_JSON = os.path.join(
    os.path.dirname(HERE), "paper12/output/reviewer/arima_extended.json")

GRID_PQ = range(0, 5)          # extended grid p, q in {0..4}, d = 0
N_SEAS_CELLS = 10              # cells for the seasonal-SARIMA stage
FIT_TIMEOUT = 120              # s per non-seasonal ML fit (then: non-converged)
SEAS_TIMEOUT = 420             # s per seasonal ML fit
SEASONAL_SPECS = [
    ((2, 0, 2), (1, 0, 0, 12)),
    ((2, 0, 2), (2, 0, 0, 12)),
    ((1, 0, 1), (1, 0, 1, 12)),
    ((2, 0, 2), (1, 0, 1, 12)),
]


# ─────────────────────────────────────────────────────────────────────────────
# Workers (module-level for multiprocessing)
# ─────────────────────────────────────────────────────────────────────────────

def _aic_grid_series(task):
    """Extended AIC grid for one cell-series. Returns {order: aic}.
    Fits that fail to converge within FIT_TIMEOUT s are recorded as None
    (a non-converged AIC is not meaningful for order selection)."""
    comp, yi, xi, series, idx_train = task
    from statsmodels.tsa.arima.model import ARIMA
    grid = {}
    for p in GRID_PQ:
        for q in GRID_PQ:
            if p == 0 and q == 0:
                continue
            try:
                aic = _with_timeout(
                    FIT_TIMEOUT,
                    lambda: ARIMA(series[:idx_train], order=(p, 0, q)).fit().aic)
                grid[f"{p}0{q}"] = float(aic)
            except _FitTimeout:
                grid[f"{p}0{q}"] = None
            except Exception:
                continue
    return {"comp": comp, "yi": yi, "xi": xi, "grid": grid}


def _forecast_series(task):
    """One-step test forecasts for several non-seasonal orders of one series."""
    comp, yi, xi, series, idx_train, idx_test, test_end, orders = task
    from statsmodels.tsa.arima.model import ARIMA
    out = {"comp": comp, "yi": yi, "xi": xi, "rmse": {}}
    truth = series[idx_test:test_end]
    for order in orders:
        key = "%d0%d" % (order[0], order[2])
        try:
            def _run():
                fit = ARIMA(series[:idx_train], order=order).fit()
                res = fit.apply(series[:test_end])
                pred = np.asarray(res.predict(start=idx_test, end=test_end - 1))
                return float(np.sqrt(np.mean((pred - truth) ** 2)))
            out["rmse"][key] = _with_timeout(FIT_TIMEOUT * 2, _run)
        except Exception:
            out["rmse"][key] = None
    return out


def _seasonal_series(task):
    """Seasonal-SARIMA candidates for one cell-series: AIC, test RMSE,
    Ljung-Box on residuals. Also fits non-seasonal (2,0,2) as reference."""
    comp, yi, xi, series, idx_train, idx_test, test_end = task
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.stats.diagnostic import acorr_ljungbox
    truth = series[idx_test:test_end]
    out = {"comp": comp, "yi": yi, "xi": xi, "specs": {}}

    def _eval(label, order, seasonal=None):
        try:
            kw = dict(order=order)
            if seasonal is not None:
                kw["seasonal_order"] = seasonal
            timeout = SEAS_TIMEOUT if seasonal is not None else FIT_TIMEOUT

            def _run():
                fit = ARIMA(series[:idx_train], **kw).fit()
                res = fit.apply(series[:test_end])
                pred = np.asarray(res.predict(start=idx_test, end=test_end - 1))
                resid = np.asarray(fit.resid)[50:]  # drop init transient
                lb = acorr_ljungbox(resid, lags=[20], return_df=True)
                return {
                    "aic": float(fit.aic),
                    "rmse": float(np.sqrt(np.mean((pred - truth) ** 2))),
                    "lb_p_lag20": float(lb["lb_pvalue"].iloc[0]),
                }
            out["specs"][label] = _with_timeout(timeout, _run)
        except _FitTimeout:
            out["specs"][label] = {"error": f"timeout>{timeout}s (non-converged)"}
        except Exception as e:
            out["specs"][label] = {"error": str(e)[:120]}

    _eval("202_noseas", (2, 0, 2))
    for order, seas in SEASONAL_SPECS:
        label = "%d0%d_%d0%d12" % (order[0], order[2], seas[0], seas[2])
        _eval(label, order, seas)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Stages
# ─────────────────────────────────────────────────────────────────────────────

def _series_tasks(data, cells):
    U, V = data["U"], data["V"]
    it = data["idx_train"]
    for (yi, xi) in cells:
        yield ("U", yi, xi, U[:, yi, xi].copy(), it)
        yield ("V", yi, xi, V[:, yi, xi].copy(), it)


def stage_grid(data, cells, workers):
    print("\n" + "=" * 70)
    print("GRID — extended AIC search p,q in {0..4}, d=0 (60 series)")
    print("=" * 70)
    t0 = time.time()
    with Pool(workers) as pool:
        results = pool.map(_aic_grid_series, list(_series_tasks(data, cells)))
    print(f"  [{time.time()-t0:.0f}s]")

    summary = {}
    best_orders = {}
    n_timeouts = 0
    for r in results:
        key = f"{r['comp']}({r['yi']},{r['xi']})"
        g = {k: v for k, v in r["grid"].items() if v is not None}
        n_timeouts += sum(1 for v in r["grid"].values() if v is None)
        if not g:
            continue
        best = min(g, key=g.get)
        best_orders[key] = best
        aic_best = g[best]
        aic_202 = g.get("202")
        summary[key] = {
            "best_order": best,
            "aic_best": aic_best,
            "aic_202": aic_202,
            "delta_aic_202_minus_best": (aic_202 - aic_best)
            if aic_202 is not None else None,
            "outside_original_grid": best[0] > "2" or best[2] > "2",
            "n_timeout": sum(1 for v in r["grid"].values() if v is None),
        }
    outside = sum(1 for s in summary.values() if s["outside_original_grid"])
    deltas = [s["delta_aic_202_minus_best"] for s in summary.values()
              if s["delta_aic_202_minus_best"] is not None]
    print(f"  best order outside original {{0,1,2}}^2 grid: "
          f"{outside}/{len(summary)} series")
    print(f"  fits timed out (non-converged in {FIT_TIMEOUT}s): {n_timeouts}")
    print(f"  AIC(2,0,2) - AIC(best): median {np.median(deltas):.1f}, "
          f"max {np.max(deltas):.1f}")
    from collections import Counter
    print("  best-order distribution:", dict(Counter(best_orders.values())))
    return {"per_series": summary, "n_outside_original_grid": outside,
            "n_fit_timeouts": n_timeouts,
            "delta_aic_median": float(np.median(deltas)),
            "delta_aic_max": float(np.max(deltas))}


def stage_forecast(data, cells, workers, grid):
    print("\n" + "=" * 70)
    print("FORECAST — one-step test RMSE: (1,0,1) vs (2,0,2) vs extended best")
    print("=" * 70)
    U, V = data["U"], data["V"]
    it, iv, te = data["idx_train"], data["idx_test"], data["test_end"]
    tasks = []
    for (yi, xi) in cells:
        for comp, arr in (("U", U), ("V", V)):
            key = f"{comp}({yi},{xi})"
            best = grid["per_series"].get(key, {}).get("best_order", "202")
            orders = {(1, 0, 1), (2, 0, 2),
                      (int(best[0]), 0, int(best[2]))}
            tasks.append((comp, yi, xi, arr[:te, yi, xi].copy(),
                          it, iv, te, sorted(orders)))
    t0 = time.time()
    with Pool(workers) as pool:
        results = pool.map(_forecast_series, tasks)
    print(f"  [{time.time()-t0:.0f}s]")

    agg = {}
    per_series = {}
    for r in results:
        key = f"{r['comp']}({r['yi']},{r['xi']})"
        per_series[key] = r["rmse"]
        for order_key, val in r["rmse"].items():
            if val is not None:
                agg.setdefault(order_key, []).append(val)
    pooled = {k: {"mean_rmse": float(np.mean(v)),
                  "median_rmse": float(np.median(v)), "n": len(v)}
              for k, v in sorted(agg.items())}
    for k, s in pooled.items():
        print(f"  ARIMA({k[0]},0,{k[2]}): mean {s['mean_rmse']:.2f}  "
              f"median {s['median_rmse']:.2f} cm/s  (n={s['n']})")

    # per-series extended best vs (2,0,2)
    gains = []
    for key, rm in per_series.items():
        best = grid["per_series"].get(key, {}).get("best_order", "202")
        if rm.get("202") is not None and rm.get(best) is not None:
            gains.append(rm["202"] - rm[best])
    print(f"  RMSE gain of extended best over (2,0,2): "
          f"mean {np.mean(gains):+.3f} cm/s, "
          f"median {np.median(gains):+.3f}, max {np.max(gains):+.3f}")
    return {"pooled_rmse": pooled, "per_series": per_series,
            "gain_over_202_mean": float(np.mean(gains)),
            "gain_over_202_median": float(np.median(gains)),
            "gain_over_202_max": float(np.max(gains))}


def stage_seasonal(data, cells, workers):
    print("\n" + "=" * 70)
    print("SEASONAL — SARIMA with period 12 on a cell subset")
    print("=" * 70)
    U, V = data["U"], data["V"]
    it, iv, te = data["idx_train"], data["idx_test"], data["test_end"]
    sub = cells[:N_SEAS_CELLS]
    tasks = []
    for (yi, xi) in sub:
        tasks.append(("U", yi, xi, U[:te, yi, xi].copy(), it, iv, te))
        tasks.append(("V", yi, xi, V[:te, yi, xi].copy(), it, iv, te))
    t0 = time.time()
    with Pool(workers) as pool:
        results = pool.map(_seasonal_series, tasks)
    print(f"  [{time.time()-t0:.0f}s]")

    per_series = {f"{r['comp']}({r['yi']},{r['xi']})": r["specs"]
                  for r in results}
    labels = sorted({lab for r in results for lab in r["specs"]})
    pooled = {}
    for lab in labels:
        aics, rmses, lbs = [], [], []
        for r in results:
            s = r["specs"].get(lab)
            if s and "aic" in s:
                aics.append(s["aic"])
                rmses.append(s["rmse"])
                lbs.append(s["lb_p_lag20"])
        if aics:
            pooled[lab] = {
                "mean_aic": float(np.mean(aics)),
                "mean_rmse": float(np.mean(rmses)),
                "n": len(aics),
                "lb_white_frac": float(np.mean([p > 0.05 for p in lbs])),
            }
            print(f"  {lab}: mean AIC {np.mean(aics):.0f}, "
                  f"mean test RMSE {np.mean(rmses):.2f} cm/s, "
                  f"white residuals in {pooled[lab]['lb_white_frac']:.0%} "
                  f"of series")
    # AIC preference: seasonal vs non-seasonal per series
    wins = 0
    total = 0
    for specs in per_series.values():
        ref = specs.get("202_noseas")
        if not ref or "aic" not in ref:
            continue
        total += 1
        best_seas = min((s["aic"] for k, s in specs.items()
                         if k != "202_noseas" and "aic" in s),
                        default=np.inf)
        if best_seas < ref["aic"]:
            wins += 1
    print(f"  seasonal preferred by AIC in {wins}/{total} series")
    return {"pooled": pooled, "per_series": per_series,
            "seasonal_aic_wins": wins, "n_series": total}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--stage", default="all",
                    choices=["all", "grid", "forecast", "seasonal"])
    args = ap.parse_args()

    data = load_data()
    cells = sample_cells(data, N_SAMPLE, SAMPLE_SEED)  # same 30 cells as round 1

    out = {"config": {
        "grid_pq": [min(GRID_PQ), max(GRID_PQ)], "d": 0,
        "n_sample_cells": len(cells), "sample_seed": SAMPLE_SEED,
        "seasonal_specs": [list(o) + [list(s)] for o, s in SEASONAL_SPECS],
        "n_seas_cells": N_SEAS_CELLS,
        "fit_timeout_s": FIT_TIMEOUT, "seas_timeout_s": SEAS_TIMEOUT,
        "test_steps": data["test_end"] - data["idx_test"],
    }}

    def _checkpoint():
        prev = {}
        if os.path.exists(OUT_JSON):
            prev = json.load(open(OUT_JSON))
        prev.update(out)
        with open(OUT_JSON, "w") as f:
            json.dump(prev, f, indent=1)

    grid = None
    if args.stage in ("all", "grid"):
        grid = stage_grid(data, cells, args.workers)
        out["grid"] = grid
        _checkpoint()
    if args.stage in ("all", "forecast"):
        if grid is None and os.path.exists(OUT_JSON):
            grid = json.load(open(OUT_JSON)).get("grid")
        if grid is None:
            raise SystemExit("forecast stage needs grid results; run grid first")
        out["forecast"] = stage_forecast(data, cells, args.workers, grid)
        _checkpoint()
    if args.stage in ("all", "seasonal"):
        out["seasonal"] = stage_seasonal(data, cells, args.workers)
        _checkpoint()

    print(f"\nSaved {OUT_JSON}")


if __name__ == "__main__":
    main()
