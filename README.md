# Stochastic Interest Rate Modelling and Prediction

> Implementing, calibrating, and extending the **Cox–Ingersoll–Ross (CIR)** short-rate model to reconstruct an entire yield curve from a single observable input — the 3-month rate.

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)
![Model](https://img.shields.io/badge/Model-CIR%20%2F%20CIR%2B%2B-1F4E79)
![Out--of--sample%20R²](https://img.shields.io/badge/OOS%20R²-0.875-2E75B6)

*Finance Club, IIT Roorkee — Open Projects 2026*

---

## Overview

Interest rates evolve randomly over time, yet not without structure. This project takes a classic model of that randomness — the CIR mean-reverting square-root diffusion — and uses it end-to-end:

1. **Clean** a noisy historical record of daily bond yields.
2. **Calibrate** the model's parameters (κ, θ, σ) from that history.
3. **Reconstruct** the 6M–2Y yield curve each day using *only* the 3-month rate as input.
4. **Extend** the model (CIR++) to correct its systematic bias, then critically analyse where it works and where it fails.

The defining constraint: at prediction time the algorithm may ingest **only the 3-month yield** for that day as a proxy for the instantaneous short rate. Everything else must follow from the model's mathematical structure.

The CIR short rate follows

$$dr_t = \kappa(\theta - r_t)\,dt + \sigma\sqrt{r_t}\,dW_t$$

whose closed-form bond price $P(t,T) = A(\tau)\,e^{-B(\tau) r_t}$ makes the yield an **affine** function of the short rate, $y(\tau) = a(\tau) + b(\tau)\,r_t$ — which is exactly what enables single-input curve reconstruction.

---

## Headline result

A fully **out-of-sample, look-ahead-free** pooled **R² = 0.875** reconstructing the 6M–2Y curve from the 3-month rate alone — clearing the project's 0.85 pass mark, with the gain verified by a training-internal validation that never touched the test set.

| Maturity | Base CIR R² | CIR++ R² |
|---|---|---|
| 6M | 0.991 | **0.995** |
| 9M | 0.953 | **0.970** |
| 1Y  | 0.877 | **0.910** |
| 2Y  | 0.211 | **0.236** |
| **Pooled** | **0.858** | **0.875** |

---

## Repository structure

```
.
├── CIR_Yield_Curve_Project.ipynb     # Main deliverable — runs top to bottom
├── data/
│   ├── train_data.csv                # 2016–2024, 9 maturities (calibration)
│   ├── test_data.csv                 # 2024–2026, 5 maturities (scoring)
│   └── test_data_3M.csv              # 2024–2026, 3M only (prediction input)
├── outputs/                          # Generated figures
│   ├── eda_*.png                     # Phase A: exploratory plots
│   ├── phaseB_*.png                  # Calibration, in-sample fit, simulation
│   ├── phaseC_*.png                  # Out-of-sample prediction & errors
│   └── phaseD_*.png                  # Base vs CIR++, shift function, bias
└── README.md
```

> Adjust paths to match your upload. The notebook's **Setup** cell lets you either upload the three CSVs to the session or mount Google Drive.

---

## Methodology

The notebook is organised into five chronological phases, with markdown report cells between the code.

### Phase A — Data engineering
Strip whitespace from column names, parse dates, reindex to a full business-day calendar and **forward-fill** the 96 market-holiday gaps (no interpolation — holidays carry no information). Outlier detection (3×IQR + 5σ day-on-day) found exactly **one** true data glitch (the 6M on 2020-11-10), winsorised to the local median; all other large moves (COVID 2020, the 2022 hikes) are real events and were kept. EDA revealed the 3M–30Y correlation is only **0.81** and the curve inverts often — early evidence that one factor cannot capture everything.

### Phase B — Calibration
A modular `CIRModel` class implements the closed-form pricer, the exact non-central chi-squared log-likelihood, and a positivity-safe simulator. Parameters are estimated by a **two-measure** approach:

- **σ from MLE** on the 3M time series (σ is measure-invariant and robustly identified): σ = 4.16%.
- **κ_ℚ, θ_ℚ from the training cross-section** (the parameters bond pricing actually requires): κ_ℚ = 0.165, θ_ℚ = 3.34%.
- Implied **market price of risk** λ = κ_ℚ − κ_ℙ = 0.135; **Feller condition satisfied**.

MLE is preferred over OLS because OLS's discretisation bias on a trending sample produced a nonsensical *negative* κ (an explosive process).

### Phase C — The prediction challenge
For each test day, reconstruct 6M/9M/1Y/2Y from that day's 3M rate alone. **Pooled out-of-sample R² = 0.858.** The 2Y is the hardest maturity (R² = 0.21) — it carries independent slope information a single factor cannot represent — and the base model systematically over-predicts long yields (bias growing +5 → +21 bp) because parameters trained on upward-sloping curves are applied to an inverted regime.

### Phase D — Extension: CIR++
A deterministic per-maturity shift φ(τ) anchored to the most recent observed curve (canonical Brigo–Mercurio exact-fit) corrects the bias **without a second stochastic factor**, keeping the single-input rule intact. Pooled R² rises to **0.875**. Chosen over a two-factor CIR (a second factor is unidentifiable from one observable) and over jump-diffusion (jumps don't fix the cross-sectional slope, and the data shows none).

### Phase E — Critical analysis
Answers every key question in the brief and states the limitations of both models with their real-world trading and risk implications.

---

## Why the result is trustworthy (not overfit)

| Check | Result | What it proves |
|---|---|---|
| Training-internal walk-forward validation | base −0.045 → CIR++ **+0.724** | the gain generalises to unseen data (test never touched) |
| Single-factor oracle ceiling (fits on test — illegitimate) | 0.964 | little legitimate headroom remains |
| Naive 8-year-average shift | 0.834 (worse) | justifies anchoring to the recent curve |

The remaining gap to the ceiling is almost entirely the 2Y, which is closable only by overfitting or by adding a second factor — barred by the single-input protocol. **The model sits at the legitimate single-factor frontier.**

---

## Key findings & limitations

- **The 3M rate alone reconstructs the short end almost perfectly** (6M R² ≈ 0.99) but loses the long end — a direct consequence of the affine one-factor structure forcing every yield onto one fixed line in the short rate.
- **Mean reversion is weak** (physical κ ≈ 0.03 ⇒ ~23-year shock half-life): the short rate behaves close to a random walk in levels, so naive "rates are high, they must fall" trades would be dangerous.
- **Single-factor models understate curve risk.** Because all yields move together, scenarios that *twist* the curve (front end up, long end down) are off the model's menu — precisely the inversion risk that dominated 2024–2026.
- **CIR++ corrects the static shape but not dynamics:** the 2Y bias flips from +21 bp to −19 bp because one fixed shift cannot track a 2Y that swings across the test period — the honest ceiling of a single-factor model under a one-input constraint.

---

## How to run

1. Open `CIR_Yield_Curve_Project.ipynb` in [Google Colab](https://colab.research.google.com/) (or Jupyter).
2. In the **Setup** cell, either upload the three CSVs to the session (leave `DATA_DIR = ""`) or mount Google Drive and set `DATA_DIR`.
3. **Runtime → Restart and run all.** The notebook runs top to bottom, regenerates every figure, and prints all evaluation metrics inline.

No special hardware is required; a standard CPU runtime completes in a couple of minutes.

---

## Tech stack

`Python` · `NumPy` · `pandas` · `SciPy` (`optimize`, `stats.ncx2`) · `Matplotlib` · `seaborn`

---

## Concepts at a glance

**CIR model** — mean-reverting square-root short-rate diffusion that keeps rates positive · **Feller condition** $2\kappa\theta \ge \sigma^2$ guarantees positivity · **Affine term structure** — yields are linear in the short rate · **Physical vs risk-neutral measure** — σ is shared, κ/θ shift by the market price of risk · **MLE** via the exact non-central chi-squared transition density · **CIR++** — deterministic shift fitting the observed curve exactly · **Out-of-sample R²** — the honest, held-out accuracy metric.

---

## Acknowledgements

Built for the **Finance Club, IIT Roorkee — Open Projects 2026**. Modelling references: Cox, Ingersoll & Ross (1985); Brigo & Mercurio (CIR++); Longstaff & Schwartz (1992, two-factor); Duffie, Pan & Singleton (2000, jump-diffusion).
