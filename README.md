# market-intel

AI-powered market intelligence dashboard for stocks and crypto.
Produces a daily morning report to help decide what to buy, watch, or avoid on IBKR and Binance.
**No automated trading. No order placement. Just a report you read with your coffee.**

---

## Web Dashboard

Launch the interactive web dashboard:

```bash
uv run streamlit run app.py
```

Opens at `http://localhost:8501`. Shows the same data as the terminal report
with a browser UI — tabs for each screener section, expandable signal cards,
and a price chart with SMA overlays and volume.

| Tab | Contents |
|-----|----------|
| 📈 Movers | Top gainers and losers side by side |
| 🔴 Oversold | RSI < 30 symbols with volume filter |
| 🟠 Overbought | RSI > 70 symbols — potential exit signals |
| 🟢 Breakouts | Volume ≥ 3× average |
| 🎯 High-Conviction | Expandable cards with score + reasoning |
| 📊 All Symbols | Searchable table of all 156 composite scores |
| 📉 Chart | 90-day price line + SMA 20/50 + volume subplot |

**Buttons:**
- **🔄 Refresh View** — rerender from cache (instant)
- **📥 Pull Fresh Data** — re-pull from APIs then rerender (~30s)

Data is cached for 5 minutes (`@st.cache_data(ttl=300)`). The web app calls
the same `src/dashboard/data.build_dashboard_data()` function as the terminal
report, so both interfaces always show identical numbers.

---

## Daily Usage

```bash
uv run python -m scripts.daily_report
```

Pulls the latest ~400 days of data, runs six screeners across **152 symbols**
(106 stocks + 46 crypto pairs), applies ML scoring to the 10 core symbols with
trained models, and prints today's report.

Report also saved to `data/reports/YYYY-MM-DD.md`.

### CLI flags

| Flag | Effect |
|------|--------|
| `--no-refresh` | Skip data pull (use cached prices — fastest) |
| `--full-refresh` | Re-pull full history from 2020 (use weekly) |
| `--symbols AAPL,MSFT` | Limit to specific symbols |
| `--save-only` | Write markdown file without printing |
| `--ml-threshold 8` | Raise ML signal threshold (default: 7/10) |

---

## Report Sections

| Section | What it shows |
|---------|---------------|
| ⚡ Today's Movers | Top 5 gainers and losers (≥5% move) |
| 🔴 Oversold Watch | RSI < 30 with normal volume — potential bounce setups |
| 🟢 Volume Breakouts | Volume ≥ 3× 20-day average — unusual activity |
| 🆕 New Listings | Symbols with < 90 days of history — limited analytics |
| 🎯 ML Signals | Composite score ≥ 7/10 — highest-conviction setups |
| 📊 Summary | Counts at a glance |

**Honest language:** "Oversold setups historically bounce ~50% of the time" —
not predictions. The screeners identify conditions, not outcomes.

---

## Composite Score (0–10)

For the 10 core symbols with trained ML models:

| Component | Points | Logic |
|-----------|--------|-------|
| ML probability | 0–4 | Calibrated P(+5% in 20 days): ≥70%=4, ≥65%=3, ≥60%=2, ≥55%=1 |
| RSI | 0–2 | <30 oversold=2, neutral=1, >70 overbought=0 |
| MACD | 0–2 | Bullish cross=2, above signal=1, below=0 |
| Volume | 0–2 | >1.5× avg=2, >1.0×=1, else=0 |

For all other symbols: technical-only (RSI + MACD + Volume), scaled to 0–10.

---

## Universe

**106 stocks:** SPY, QQQ, AAPL, MSFT, NVDA, JPM (core ML) + S&P 100 large-caps
+ high-beta growth (PLTR, COIN, SNOW, CRWD, …) + crypto-adjacent (MARA, RIOT, IBIT, …)

**46 crypto pairs (Binance):** BTC, ETH, SOL, BNB (core ML) + top liquid alts
(XRP, ADA, DOGE, AVAX, …) + DeFi, gaming, meme coins

Configure in `config/universe.yaml`. Adjust screener thresholds in `config/settings.py`.

---

## Initial Setup (one-time)

```bash
# 1. Install dependencies
uv sync

# 2. Fetch full historical data (2020 → today, ~10 min)
uv run python -m scripts.pull_data

# 3. Compute features + triple-barrier labels
uv run python -m scripts.build_dataset

# 4. Train per-symbol XGBoost models (walk-forward, 4 folds)
uv run python -m scripts.train_per_symbol

# 5. First report
uv run python -m scripts.daily_report --no-refresh
```

### libomp fix (macOS without Homebrew)

XGBoost requires libomp. If it's not at `/opt/homebrew/opt/libomp/lib/`:

```bash
# Find libomp (e.g. from R installation)
find /Library -name "libomp.dylib" 2>/dev/null

# Patch the dylib to point at it (replace <PATH> with your path)
install_name_tool -change @rpath/libomp.dylib <PATH>/libomp.dylib \
  .venv/lib/python3.12/site-packages/xgboost/lib/libxgboost.dylib
```

---

## Project Structure

```
market-intel/
├── config/
│   ├── settings.py          # All paths, thresholds, rate limits
│   └── universe.yaml        # Stock + crypto symbol list
├── src/
│   ├── ingestion/           # yfinance (stocks) + ccxt/Binance (crypto)
│   ├── features/            # 22 technical/volatility/volume features
│   ├── labels/              # Triple-barrier labels (+5%/-3%/20d)
│   ├── models/              # XGBoost, LightGBM, calibration, registry
│   │   └── registry.py      # Production model persistence
│   ├── backtest/            # Walk-forward engine, costs, metrics
│   └── dashboard/
│       ├── screener.py      # Six screener functions
│       ├── scoring.py       # 0-10 composite score
│       └── report.py        # Terminal + markdown report generator
├── scripts/
│   ├── daily_report.py      # ← main entry point
│   ├── pull_data.py         # Data ingestion
│   ├── build_dataset.py     # Feature + label pipeline
│   ├── train_per_symbol.py  # Per-symbol model training
│   └── run_backtest.py      # Phase 4 backtest
├── tests/                   # 75 tests (ingestion, leakage, models, backtest, dashboard)
└── data/
    ├── raw/                 # Cached OHLCV parquet files
    ├── features/            # Feature + label files (core 10 symbols)
    ├── models/production/   # Production model registry
    └── reports/             # Daily markdown reports
```

---

## Phase History

| Phase | Description |
|-------|-------------|
| 1 | Data ingestion (yfinance + ccxt), OHLCV schema, Parquet caching |
| 2 | Feature engineering (22 indicators), triple-barrier labels, purged walk-forward CV |
| 3 | XGBoost + LightGBM, isotonic calibration, global vs per-symbol comparison |
| 4 | Walk-forward backtest engine, realistic costs, t+1 entry, metrics |
| 4.5 | Universe filtering, cooldown logic, buy-and-hold benchmark |
| 5 | Daily dashboard: 152-symbol screener + ML scoring + morning report |

**Backtest verdict (Phase 4.5):** ML system Sharpe=0.82 at best operating point
(threshold=0.70, cooldown=5d, WINNERS universe). Buy-and-hold of same 3 symbols
Sharpe=1.06 over the same period. NVDA +546% return dominated passive holding.
Strategy needs Phase 5+ improvements before live deployment.
