# Portfolio Analyzer

A Streamlit app that turns a manually-entered portfolio into a full quant report — live prices + historical data via `yfinance`, all metrics computed from actual return series, benchmarked against Nifty 50.

## Setup

```bash
pip install -r requirements.txt
streamlit run portfolio_analyzer.py
```

The app opens at `http://localhost:8501`.

## How to use

1. **Edit the holdings table** (or upload a CSV with columns `Ticker, Shares, Buy Price`).
   - NSE stocks need `.NS` suffix → `RELIANCE.NS`, `INFY.NS`
   - BSE stocks need `.BO` suffix
   - US stocks → plain ticker like `AAPL`
2. Pick a **lookback period** and **risk-free rate** in the sidebar.
3. Click **Analyze portfolio**.

## What it computes

**From actual daily returns** (not user-input proxies):
- CAGR, annualized volatility, Sharpe, Sortino, Calmar
- Beta and alpha via OLS regression vs Nifty 50 (with R²)
- Jensen's alpha (CAPM), Treynor ratio
- Max drawdown with peak/trough dates
- VaR and CVaR at 95% confidence
- Skewness, kurtosis, best/worst day

**Visualizations:**
- Growth-of-₹100 vs benchmark
- Rolling Sharpe ratio
- Monthly returns heatmap
- Drawdown curves
- Return distribution histogram with VaR line
- Security Market Line (CAPM) scatter with all holdings + portfolio + benchmark plotted
- Allocation pie charts (by stock and by sector)
- Per-holding return bar with benchmark CAGR reference
- Correlation matrix heatmap

**Exports:** Holdings, metrics, and full price history as CSV.

## CSV upload format

```csv
Ticker,Shares,Buy Price
RELIANCE.NS,10,2400
TCS.NS,5,3500
HDFCBANK.NS,15,1500
```

## Notes

- yfinance is free but rate-limited and occasionally has stale or missing metadata (especially `beta`). The app falls back to its own regression-computed beta, which is usually more reliable anyway.
- Data is cached for 15 minutes (prices) / 1 hour (metadata). Restart Streamlit or clear cache to force-refresh.
- This is an educational tool, not investment advice.
