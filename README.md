# Analysis code for "Comparative evaluation of statistical and deep learning methods for high-frequency radar surface current forecasting in a narrow tropical strait"

This archive contains the Python code that reproduces every table, figure, and
reported number in the manuscript (Gautama & Putra, submitted to *Advances in
Statistical Climatology, Meteorology and Oceanography*, ASCMO-2026-18). It is
provided for full reproducibility even though the raw HF-radar and AWS
observations are access-controlled (see **Data availability**).

## Authors
- Dhava Gautama — Indonesia Meteorology, Climatology, and Geophysical Agency (BMKG), Jakarta, Indonesia — dhava.gautama@bmkg.go.id
- Alifficionaldo A. Putra — BMKG, Jakarta, Indonesia; Department of Meteorology, University of Reading, UK

## Contents

### Core analysis (per-cell statistical models, deep learning, diagnostics)

| Script | Produces | Description |
|--------|----------|-------------|
| `scripts/phase_b.py` | Tables 2 (seeds), 3, 5, 6 | Deep-learning model definitions and training; multi-seed Phase-1 runs, capacity-matched Phase-3, seasonal stratification, tidal–residual error decomposition |
| `scripts/reviewer_response.py` | Table 2, diagnostics | Full-domain ARIMA(1,0,1) and AIC-optimal ARIMA(2,0,2); ADF stationarity, AIC order grid, Ljung–Box residual tests; simple exponential smoothing (SES); gap-filled vs observed split; vector error metrics |
| `scripts/arima_extended.py` | Sect. 3.2 (extended order search) | Extended ARIMA AIC grid (p, q ≤ 4) with out-of-sample verification and seasonal SARIMA (period 12 h) evaluation: shows higher-order/seasonal specifications do not materially improve one-step forecasts |
| `scripts/eofvar.py` | Table 2 (EOF-VAR row) | Reduced-rank EOF-VAR dynamic spatio-temporal statistical baseline (EOFs + vector autoregression on the leading PCs) |
| `scripts/capacity_matched.py` | Tables 3–4 | Parameter-matched CNN-GRU-MS (0.48 M, matched to BiEF) for the direct-vs-autoregressive comparison |
| `scripts/phase3_seeds.py` | Tables 3–4 | Controlled same-platform, 5-seed Phase-3 sweep of all five multi-step models (ConvLSTM-ED, BiEF, CNN-GRU-MS full / Small / Matched) on a single GPU under one identical config — the apples-to-apples direct-vs-autoregressive and capacity comparison |
| `scripts/capacity_matched_seeds.py` | Tables 3–4 | Multi-seed capacity-matched CNN-GRU-MS variant of the controlled sweep |
| `scripts/dl_vector_gap.py` | Tables 7–8 | Deep-learning vector (speed / complex / direction-MAE) and gap-split metrics |
| `scripts/arima_tidal_decomp.py` | Sect. 4.6 | ARIMA tidal-vs-residual skill decomposition (UTide split + ARIMA re-fit to the residual): the direct test that ARIMA's skill is tidal autocorrelation |
| `scripts/extras.py` | spatial maps, misc. | Per-cell spatial error fields and supporting statistics |

### Figures and wind analysis

| Script | Produces | Description |
|--------|----------|-------------|
| `scripts/generate_figures.py` | Figs 1, 9, 10, 11, 12, 13, 14 | Study-area map; Phase-1 RMSE and skill-score bar charts (all twelve methods, from Table 2 values); lookback sensitivity; lead-time RMSE/skill; diurnal RMSE |
| `scripts/fig_phase1_arch.py`, `scripts/fig_phase3_arch.py`, `scripts/fig_architecture.py` | Figs 3–8 | Network architecture diagrams (CNN, GRU, CNN-GRU, ConvLSTM-ED, BiEF, CNN-GRU-MS) |
| `scripts/analyze_aws.py` | Fig 15 | Diurnal wind / RMSE overlay from the three AWS stations |
| `scripts/wind_projection.py` | Sect. 4.4 | Along-/cross-strait wind projection and regression of component errors (resolves the negative V–wind correlation) |
| `scripts/inference_latency_bench.py` | Table 4 | Single-CPU inference-latency benchmark (direct vs autoregressive forecast time per step) |
| `scripts/fig_spatial_rmse.py` | Fig 16 | Per-cell spatial RMSE maps |
| `scripts/fig_reviewer.py` | Figs 2, 17, 18 | ACF/PACF; ARIMA-vs-DL spatial error map; gap-fill fraction map |

### Precomputed results (`output/`)
The small JSON result files used to populate the tables and figures are included
so that reported numbers can be verified without re-running the full pipeline
(`phase1_results.json`, `phase2_lookback.json`, `phase3_multistep.json`,
`seasonal_results.json`, `multiseed_results.json`, `capacity_matched*.json`,
`tidal_decomposition.json`, `wind_projection.json`, and the `reviewer/` set:
`reviewer_results.json`, `eofvar_results.json`, `dl_vector_gap.json`,
`arima_tidal_decomp.json`, and the controlled Phase-3 sweep outputs
`phase3_controlled_seeds.json`, `phase3_controlled_timing.json`, `phase3_matched_seeds.json`).

## Requirements
See `requirements.txt`. Tested with Python 3.12 (PyTorch on CPU is sufficient;
Intel Extension for PyTorch was used for acceleration but is optional).

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

## Data availability
The scripts read the following inputs, which are **not** included in this archive:

| Input | Source / access |
|-------|-----------------|
| `data/processed/BADA_hourly_qc.nc` | Quality-controlled BADA HF-radar surface currents (Dec 2022 – Feb 2026, 21×21 grid). Owned and archived by BMKG; available on reasonable request to the corresponding author. |
| `data/processed/paper3_domain.nc` | Domain masks (observed / blind / land) + BATNAS v1.6 bathymetry (BATNAS openly available from the Indonesian Geospatial Information Agency, BIG: https://batnas.big.go.id). |
| `awsMaritim/AWS_Maritim_*.csv` | 1-minute wind speed/direction at Merak, Ciwandan, and Bakauheni (BMKG automatic weather stations); available on reasonable request to the corresponding author. |

The BADA HF-radar and AWS observations are owned and archived by BMKG. Under the
Indonesian national regulation governing access to meteorological, climatological,
and geophysical data (Peraturan BMKG Nomor 4 Tahun 2022,
https://jdih.bmkg.go.id/dokumen/detail/4193), these datasets cannot be released
unconditionally in open, FAIR-aligned repositories; access for scientific
validation and non-commercial research is granted upon reasonable request to the
corresponding author, subject to BMKG's formal approval procedures and
data-sharing agreements. The BATNAS bathymetry is openly available (link above).

Paths are relative to a repository root; set the `RADAR_ROOT` environment
variable (or edit the path constants at the top of each script) to point at your
local data directory.

## Reproduction order
1. `phase_b.py` (and the original Phase-1/2/3 training) → model checkpoints + `phase*_results.json`, `multiseed_results.json`
2. `reviewer_response.py` → full-domain ARIMA/SES/kNN, diagnostics, gap, vector → `reviewer/reviewer_results.json`
3. `eofvar.py` → `reviewer/eofvar_results.json`
4. `capacity_matched.py`, `phase3_seeds.py`, `capacity_matched_seeds.py`, `dl_vector_gap.py` → controlled 5-seed Phase-3 sweep + vector/gap metrics → Tables 3–4, 7–8 (`reviewer/phase3_controlled_seeds.json`, `phase3_matched_seeds.json`); `inference_latency_bench.py` → `reviewer/phase3_controlled_timing.json`
5. `arima_tidal_decomp.py` → `reviewer/arima_tidal_decomp.json` (Sect. 4.6)
6. `analyze_aws.py`, `wind_projection.py` → diurnal/wind analysis (Sect. 4.4, Fig 15)
7. `generate_figures.py`, `fig_*.py` → all manuscript figures

> Note: `generate_figures.py` and `fig_*.py` write descriptively-named PNGs; the
> mapping to the numbered manuscript figures (`fig01.png`–`fig18.png`) is given in
> `figure/FIGURE_MAP.md` of the source repository.

## License
Code released under the MIT License (see `LICENSE`).
