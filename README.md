# Hyperliquid × Fear & Greed — Market Sentiment & Trader Behavior Analysis

## Overview
This project analyzes how the **Crypto Fear & Greed Index** relates to trader behavior and performance on **Hyperliquid**, a decentralized perpetuals exchange. Using 35,554 merged trades across 452 days (May 2023 – May 2025), we identify actionable patterns for smarter trading.

---

## Setup

### Requirements
```
Python 3.8+
pandas
numpy
matplotlib
seaborn
scikit-learn
```

All are standard libraries — no pip install needed if you have a standard Anaconda/PyPI env.

### File Structure
```
├── data/
│   ├── fear_greed_index.csv      ← Fear & Greed dataset
│   └── historical_data.csv       ← Hyperliquid trade history
├── analysis.py                   ← Full standalone analysis script
├── hyperliquid_analysis.ipynb    ← Jupyter notebook version
├── outputs/                      ← Generated figures & tables
│   ├── fig1_overview_dashboard.png
│   ├── fig2_performance_deepdive.png
│   ├── fig3_behavioral_analysis.png
│   ├── fig4_segmentation.png
│   ├── fig5_clustering.png
│   ├── fig6_predictive_model.png
│   └── summary_stats_table.csv
└── README.md
```

### How to Run

**Option 1 — Python script (recommended):**
```bash
python analysis.py
```

**Option 2 — Jupyter Notebook:**
```bash
Jupyter Notebook hyperliquid_analysis.ipynb
```
> Before running the notebook, comment out `matplotlib.use('Agg')` for interactive plots.

---

## Methodology

### Data Sources
| Dataset | Rows | Columns | Date Range |
|---------|------|---------|-----------|
| Fear & Greed Index | 2,644 | 4 | Feb 2018 – present |
| Hyperliquid Trades | 211,224 | 16 | May 2023 – May 2025 |
| **Merged** | **35,554** | **+4** | **May 2023 – May 2025** |

### Timestamp Fix
The `Timestamp` column in the trading data has truncated millisecond precision — all values resolve to a ~6-hour window per day. We instead parse `Timestamp IST` (`DD-MM-YYYY HH: MM`), which gives correct daily granularity, yielding **452 matching dates** (vs 6 with the raw `Timestamp`).

### Sentiment Buckets
- **Fear** = "Fear" + "Extreme Fear" (FG index ≤ ~40)
- **Greed** = "Greed" + "Extreme Greed" (FG index ≥ ~60)
- **Neutral** = all others

### Key Metrics Computed
| Metric | Description |
|--------|-------------|
| `win_rate` | % of trades with positive Closed PnL |
| `pnl_per_trade` | Mean Closed PnL per individual trade |
| `intensity` | Trades per active trader per day |
| `long_ratio` | BUY trades / total trades |
| `consistency` | win_rate minus normalized PnL std dev |
| `fear_ratio` | % of a trader's trades executed on Fear days |

---

## Key Findings

### Insight 1 — Fear Days: Fewer Trades, Higher Stakes
Fear days average **$7,599 total PnL/day** vs $4,543 on Greed days — despite a *lower* win rate (33.4% vs 38.3%). This paradox is explained by **position sizing**: traders take larger positions on Fear days (avg $7,442 vs $5,942), and the occasional winner is a much bigger win. Fear days are characterized by concentrated, high-conviction bets.

### Insight 2 — Greed Days: Volume Over Precision
On Greed days, trade intensity rises to **18.4 trades/trader/day** vs 11.5 on Fear days — a 60% increase. Traders become more active, churn more positions, and collect more small wins. This is "momentum mode": profitable when the trend is clear, dangerous when it reverses.

### Insight 3 — Sentiment Does NOT Strongly Drive Long Bias
The regression of the FG index vs long ratio shows a near-zero slope (~0.0). Despite intuition, traders on Hyperliquid do **not** become significantly more long-biased on Greed days. The long ratio stays near 46–50% regardless of sentiment, suggesting most activity is either hedging or indifferent to directionality.

### Insight 4 (Segmentation) — High-Size Traders Capture Fear Alpha
"High Size" traders (top tercile by avg trade size) generate **positive avg PnL on Fear days** while other segments lose money or break even. Large-position traders appear to have conviction-based strategies that outperform in volatile, fearful markets.

### Insight 5 (Clustering) — 4 Distinct Archetypes
K-Means (k=4) identifies: **Disciplined Winners** (high win rate, low volatility), **Whale Traders** (large positions, variable outcomes), **HFT/Bot** accounts (extreme trade frequency), and **Casual Traders** (sporadic, break-even on average). Each archetype responds differently to sentiment.

---

## Strategy Recommendations

### Strategy 1: "Fear Day Concentration" (for High-Size traders)
> **On Fear days (FG ≤ 40):** Reduce trade frequency. Increase per-trade size only on high-conviction setups. Do NOT try to win-rate chase — expect ~33% wins but make them count. Cap maximum drawdown per day at 2× the normal level (because PnL variance is 25K on Fear vs 24K on Greed — Fear is NOT lower risk). Target mean-reverting coins (those with high fear-day trade share in the sentiment heatmap).

### Strategy 2: "Greed Day Momentum" (for Frequent traders)
> **On Greed days (FG ≥ 60):** Increase trade frequency — your win rate will be higher (~38% vs 33%). Keep individual position sizes **smaller** than your Fear-day baseline ($5,942 vs $7,442). Use momentum-following entries. **Reduce exposure when FG > 80** (Extreme Greed) — the heatmap shows PnL drops sharply in that regime on Fridays and Weekends, historically coinciding with sentiment reversals.

---

## Figures Summary
| Figure | Contents |
|--------|----------|
| fig1_overview_dashboard | Trade distribution pie, FG history, 5 core metrics by sentiment |
| fig2_performance_deepdive | PnL violins, time series, win rate boxplots, volatility, fee drag |
| fig3_behavioral_analysis | Long ratio timeline, intensity, FG vs long scatter, size histogram, DoW heatmap |
| fig4_segmentation | 3 segment types × sentiment performance |
| fig5_clustering | K-Means archetypes — scatter, PnL ranking, behavioral profiles |
| fig6_predictive_model | Random Forest next-day profitability predictor + feature importances |
