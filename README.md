# BullionSpread: MCX Gold Futures Relative-Value Intelligence Platform

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Exchange](https://img.shields.io/badge/Exchange-MCX%20India-gold.svg)](https://www.mcxindia.com/)
[![Dashboard](https://img.shields.io/badge/Dashboard-Streamlit-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An institutional relative-value commodity intelligence and statistical arbitrage platform for Multi Commodity Exchange of India (MCX) gold futures (`GOLDM`, `GOLDTEN`, `GOLDGUINEA`, `GOLDPETAL`). 

BullionSpread standardizes fragmented contract specifications into an equivalent **1-gram 999-purity benchmark**, models rolling spread dynamics with dynamic Z-scores, enforces liquidity safety gates against thin settlement prints, and simulates walk-forward trades with realistic transaction friction attribution.

---

## System Architecture

```text
mcx-bullionspread/
├── data/                    # Storage for raw and cleaned Bhavcopy CSVs
│   ├── raw_bhavcopy_sample.csv
│   └── cleaned_bhavcopy.csv
├── src/
│   ├── __init__.py          # Package initialization
│   ├── config.py            # Centralized contract specs, statutory fees & default thresholds
│   ├── ingest.py            # Bhavcopy downloader, exchange cleaner & mock generator
│   ├── normalize.py         # Per-gram 999-purity conversion engine & pair aligner
│   ├── signals.py           # Spread calculation, rolling Z-score & liquidity gates
│   └── backtest.py          # Walk-forward simulation & cost attribution engine
├── tests/
│   ├── __init__.py
│   └── test_bug_fixes.py    # 20-point verification test suite
├── app.py                   # Trader-facing Streamlit dashboard
├── requirements.txt         # Core dependencies (pandas, numpy, requests, streamlit, plotly)
└── README.md                # Platform documentation & mathematical specifications
```

---

## Core Quantitative Methodology

### 1. Per-Gram 999-Purity Normalization
MCX gold contracts trade in distinct lot sizes, quotation units, and purity standards. BullionSpread standardizes every contract settlement price to the effective price per **1 gram of 999 fineness (24 Karat)**:

| Contract | Trading Unit | Quotation Unit | Base Purity | Formula for 1g 999 Gold |
| :--- | :---: | :---: | :---: | :--- |
| **GOLDM** (Gold Mini) | 100 grams | 10 grams | 995 fineness | `(Close / 10) * (999 / 995)` |
| **GOLDTEN** (Gold Ten) | 10 grams | 10 grams | 999 fineness | `Close / 10` *(evaluated 2025 onwards)* |
| **GOLDGUINEA** | 8 grams | 8 grams | 999 fineness | `Close / 8` |
| **GOLDPETAL** | 1 gram | 1 gram | 999 fineness | `Close / 1` |

### 2. Relative Spread & Dynamic Z-Score
For contracts sharing similar delivery horizons:
$$\text{Spread}_t = P_{\text{norm}, A, t} - P_{\text{norm}, B, t}$$
$$\mu_t = \text{RollingMean}_{15\text{D}}(\text{Spread}_t), \quad \sigma_t = \text{RollingStd}_{15\text{D}}(\text{Spread}_t)$$
$$Z_t = \frac{\text{Spread}_t - \mu_t}{\sigma_t}$$

### 3. Dynamic Liquidity Gate
Official exchange settlement prices can be misleading on thin contracts where executable liquidity does not exist. Signals are suppressed unless both legs exceed configurable expanding/rolling liquidity percentiles (default 20th percentile) in both **Volume** and **Open Interest**:
$$\text{Liquidity Pass} = (\text{Vol}_A \ge Q_{\text{vol}, A}) \land (\text{OI}_A \ge Q_{\text{oi}, A}) \land (\text{Vol}_B \ge Q_{\text{vol}, B}) \land (\text{OI}_B \ge Q_{\text{oi}, B})$$

### 4. Walk-Forward Simulation & Tender Period Exit
- **Entry**: $|Z_t| \ge 2.0\sigma$ with `Liquidity Pass == True`.
- **Mean-Reversion Exit**: $Z_t \to 0$ or $|Z_t| \le 0.25\sigma$.
- **Mandatory Expiry Guardrail**: Positions are liquidated **5 trading days prior to contract expiry** to prevent physical tender period margin surges and delivery risk.
- **Cost Attribution**: Accounts for exchange turnover charges, SEBI turnover fees, Commodities Transaction Tax (CTT), stamp duty, brokerage, and bid-ask slippage buffer (configurable, default 5 bps/leg).
- **Market Neutrality**: Alpha is decomposed from underlying gold price movement, confirming $\beta \approx 0.0$.

---

## Quickstart & Installation

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Comprehensive Test Suite
```bash
python -m unittest discover tests
```

### 3. Run Ingestion / Synthetic Data Pipeline
```bash
python -m src.ingest
```

### 4. Launch Streamlit Dashboard
```bash
streamlit run app.py --server.port 8501
```

Access the interface in your browser at `http://localhost:8501`.

---

## Dashboard Features
1. **Executive RV Overview**: Top KPI cards (Normalized Gold Price, Active Spread, Dynamic Z-Score, Signal Status, Liquidity Gate status).
2. **Interactive Charting**: Multi-line Plotly visualization of normalized prices across all active contracts.
3. **Signal Engine & Statistical Bands**: Dynamic spread plotted against rolling mean and $\pm 2\sigma$ bands, with overlaid buy/sell signals and liquidity suppressed alerts.
4. **Volume & OI Depth Profile**: Subplots detailing liquidity depth and gate thresholds.
5. **Walk-Forward Performance & Market Drift Attribution**:
   - Strategy Net PnL curve vs Raw Gold Buy & Hold drift.
   - Cost attribution waterfall (Gross PnL vs Frictional Costs vs Net Alpha).
   - Underwater drawdown curve.
   - Complete chronological trade log with granular trade-by-trade metrics.
