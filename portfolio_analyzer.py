"""
Quant Portfolio Analyzer
========================
A Streamlit app that takes a manually-entered Indian-equity portfolio,
pulls live + historical data from yfinance, and computes a full suite of
risk-adjusted performance metrics benchmarked against the Nifty 50.

Run with:
    pip install streamlit yfinance pandas numpy plotly scipy
    streamlit run portfolio_analyzer.py
"""

from __future__ import annotations

import io
from datetime import date, timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from scipy import stats
from scipy.optimize import brentq, newton

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Portfolio Analyzer",
    page_icon="📈",
    layout="wide",
)

TRADING_DAYS = 252
BENCHMARK = "^NSEI"  # Nifty 50
BENCHMARK_NAME = "Nifty 50"

EMPTY_PORTFOLIO = pd.DataFrame(columns=["Ticker", "Shares", "Buy Price", "Buy Date"])

# ---------------------------------------------------------------------------
# Data layer
# ---------------------------------------------------------------------------
@st.cache_data(ttl=60 * 15, show_spinner=False)
def fetch_prices(tickers: list[str], start: date, end: date) -> pd.DataFrame:
    """Download adjusted close prices for the given tickers."""
    data = yf.download(
        tickers,
        start=start,
        end=end + timedelta(days=1),
        auto_adjust=True,
        progress=False,
        group_by="column",
    )
    if isinstance(data.columns, pd.MultiIndex):
        # Standard layout when downloading multiple tickers
        prices = data["Close"].copy()
    else:
        prices = data[["Close"]].copy()
        prices.columns = tickers
    prices = prices.dropna(how="all")
    return prices


@st.cache_data(ttl=60 * 10, show_spinner=False)
def search_tickers(query: str, limit: int = 15) -> list[dict]:
    """Search Yahoo Finance for tickers matching a company name or symbol."""
    if not query or len(query.strip()) < 1:
        return []
    try:
        # yfinance >= 0.2.40 ships a Search class
        results = yf.Search(query.strip(), max_results=limit).quotes or []
    except Exception:
        try:
            # Fallback: direct query to Yahoo's search endpoint
            import requests
            r = requests.get(
                "https://query2.finance.yahoo.com/v1/finance/search",
                params={"q": query.strip(), "quotesCount": limit, "newsCount": 0},
                headers={"User-Agent": "Mozilla/5.0"},
                timeout=5,
            )
            results = r.json().get("quotes", [])
        except Exception:
            return []

    out = []
    for q in results:
        sym = q.get("symbol")
        if not sym:
            continue
        out.append({
            "symbol": sym,
            "name": q.get("shortname") or q.get("longname") or sym,
            "exchange": q.get("exchDisp") or q.get("exchange") or "",
            "type": q.get("quoteType", ""),
        })
    # Equities first, then ETFs, then everything else
    rank = {"EQUITY": 0, "ETF": 1, "MUTUALFUND": 2, "INDEX": 3}
    out.sort(key=lambda x: rank.get(x["type"], 9))
    return out


@st.cache_data(ttl=86400, show_spinner=False)
def isin_to_ticker(isin: str) -> dict | None:
    """
    Resolve an ISIN to its best yfinance ticker via Yahoo search.
    Returns dict with symbol, name, exchange — or None if no match.
    Prefers NSE equity > NSE anything > BSE > whatever's first.
    """
    if not isin or len(str(isin).strip()) < 10:
        return None
    isin_clean = str(isin).strip().upper()
    try:
        results = search_tickers(isin_clean, limit=8)
    except Exception:
        return None
    if not results:
        return None

    # Ranking: prefer NSE equity, then NSE anything, then BSE, then anything
    def score(r):
        sym = r.get("symbol", "")
        typ = r.get("type", "")
        if sym.endswith(".NS") and typ == "EQUITY": return 0
        if sym.endswith(".NS"):                     return 1
        if sym.endswith(".BO"):                     return 2
        return 3
    results.sort(key=score)
    return results[0]


@st.cache_data(ttl=86400, show_spinner=False)
def resolve_portfolio_tickers(
    isins: list[str | None], fallback_symbols: list[str | None]
) -> list[dict]:
    """Batch-resolve a list of ISINs to tickers, falling back to symbol+.NS."""
    out = []
    for isin, fb in zip(isins, fallback_symbols):
        resolved = isin_to_ticker(isin) if isin else None
        if resolved:
            out.append({
                "ticker": resolved["symbol"],
                "name": resolved["name"],
                "source": "ISIN",
            })
        else:
            sym = (str(fb).strip().upper() if fb else "")
            if sym and "." not in sym and "^" not in sym:
                sym = sym + ".NS"
            out.append({
                "ticker": sym,
                "name": fb or sym,
                "source": "Symbol" if sym else "—",
            })
    return out


@st.cache_data(ttl=60 * 60, show_spinner=False)
def fetch_day_ohlc(ticker: str, target_date: date) -> dict | None:
    """Fetch OHLC for the trading day on (or nearest before) target_date."""
    try:
        start = target_date - timedelta(days=7)
        end = target_date + timedelta(days=2)
        hist = yf.Ticker(ticker).history(start=start, end=end, auto_adjust=False)
        if hist.empty:
            return None
        hist.index = pd.to_datetime(hist.index).date
        if target_date in hist.index:
            bar = hist.loc[target_date]
        else:
            earlier = [d for d in hist.index if d <= target_date]
            if not earlier:
                return None
            bar = hist.loc[max(earlier)]
            return {
                "date": max(earlier),
                "open": float(bar["Open"]), "high": float(bar["High"]),
                "low": float(bar["Low"]),   "close": float(bar["Close"]),
                "is_exact": False,
            }
        return {
            "date": target_date,
            "open": float(bar["Open"]), "high": float(bar["High"]),
            "low": float(bar["Low"]),   "close": float(bar["Close"]),
            "is_exact": True,
        }
    except Exception:
        return None


@st.cache_data(ttl=60 * 60, show_spinner=False)
def fetch_info(ticker: str) -> dict:
    """Pull metadata (beta, sector, name, etc.) for a single ticker."""
    try:
        info = yf.Ticker(ticker).info or {}
    except Exception:
        info = {}
    return {
        "name": info.get("shortName") or info.get("longName") or ticker,
        "sector": info.get("sector", "Unknown"),
        "industry": info.get("industry", "Unknown"),
        "beta": info.get("beta"),
        "market_cap": info.get("marketCap"),
    }


# ---------------------------------------------------------------------------
# Metric engine
# ---------------------------------------------------------------------------
def annualized_return(daily_returns: pd.Series) -> float:
    dr = daily_returns.dropna()
    if dr.empty:
        return 0.0
    total = (1 + dr).prod()
    years = len(dr) / TRADING_DAYS
    return total ** (1 / years) - 1 if years > 0 else 0.0


def annualized_vol(daily_returns: pd.Series) -> float:
    return daily_returns.std() * np.sqrt(TRADING_DAYS)


def downside_vol(daily_returns: pd.Series, mar: float = 0.0) -> float:
    downside = daily_returns[daily_returns < mar]
    if downside.empty:
        return 0.0
    return downside.std() * np.sqrt(TRADING_DAYS)


def max_drawdown(equity_curve: pd.Series) -> tuple[float, pd.Timestamp, pd.Timestamp]:
    running_max = equity_curve.cummax()
    drawdown = (equity_curve - running_max) / running_max
    trough = drawdown.idxmin()
    peak = equity_curve.loc[:trough].idxmax()
    return drawdown.min(), peak, trough


def regress_beta_alpha(asset_ret: pd.Series, bench_ret: pd.Series) -> tuple[float, float, float]:
    """OLS regression of asset on benchmark. Returns (beta, alpha_daily, r_squared)."""
    df = pd.concat([asset_ret, bench_ret], axis=1).dropna()
    if len(df) < 30:
        return np.nan, np.nan, np.nan
    # Force plain numpy float64 — some pandas/scipy version combos (e.g. on
    # Streamlit Cloud) produce nullable/extension dtypes that scipy's
    # linregress can't handle internally.
    x = np.asarray(df.iloc[:, 1], dtype=np.float64)
    y = np.asarray(df.iloc[:, 0], dtype=np.float64)
    slope, intercept, r, _, _ = stats.linregress(x, y)
    return slope, intercept, r ** 2


def value_at_risk(daily_returns: pd.Series, confidence: float = 0.95) -> float:
    if daily_returns.empty:
        return 0.0
    return np.percentile(daily_returns, (1 - confidence) * 100)


def conditional_var(daily_returns: pd.Series, confidence: float = 0.95) -> float:
    var = value_at_risk(daily_returns, confidence)
    tail = daily_returns[daily_returns <= var]
    return tail.mean() if not tail.empty else 0.0


def xirr(cashflows: list[tuple]) -> float | None:
    """
    Extended IRR for irregular cashflows.
    cashflows: list of (date, amount) — buys negative, sells/current value positive.
    Returns annualized rate as a decimal (0.18 = 18%), or None if it doesn't converge.
    """
    if not cashflows or len(cashflows) < 2:
        return None
    # Must have both positive and negative flows
    amounts = [cf[1] for cf in cashflows]
    if not (any(a > 0 for a in amounts) and any(a < 0 for a in amounts)):
        return None

    dates = [pd.Timestamp(cf[0]) for cf in cashflows]
    t0 = min(dates)
    years = np.array([(d - t0).days / 365.0 for d in dates])
    amounts = np.array(amounts, dtype=float)

    def npv(rate):
        return float(np.sum(amounts / np.power(1 + rate, years)))

    # Try brentq with wide bracket
    try:
        return brentq(npv, -0.9999, 100.0, xtol=1e-7, maxiter=200)
    except (ValueError, RuntimeError):
        pass
    # Fallback to Newton
    try:
        return float(newton(npv, x0=0.1, maxiter=200, tol=1e-7))
    except Exception:
        return None


def build_portfolio_timeseries(
    portfolio_df: pd.DataFrame,
    asset_prices: pd.DataFrame,
    bench_prices: pd.Series,
) -> pd.DataFrame:
    """
    Build a daily time series of:
      - portfolio_value: market value of holdings, each starting on its buy date
      - invested: cumulative ₹ invested up to that date
      - nifty_equivalent: value if the same ₹ were invested in Nifty on each buy date
    Only includes dates from the earliest buy date onward.
    """
    if portfolio_df.empty:
        return pd.DataFrame()

    # Drop rows missing a buy date
    df = portfolio_df.dropna(subset=["Buy Date"])
    if df.empty:
        return pd.DataFrame()

    buy_dates = pd.to_datetime(df["Buy Date"])
    earliest = buy_dates.min()
    idx = asset_prices.index[asset_prices.index >= earliest]
    if len(idx) == 0:
        return pd.DataFrame()

    port_value = pd.Series(0.0, index=idx)
    invested = pd.Series(0.0, index=idx)
    nifty_equiv = pd.Series(0.0, index=idx)

    for ticker, row in df.iterrows():
        if ticker not in asset_prices.columns:
            continue
        buy_dt = pd.Timestamp(row["Buy Date"])
        # First trading day on/after buy date in our index
        mask = idx >= buy_dt
        if not mask.any():
            continue
        active_idx = idx[mask]

        # Holding value over time
        prices_for_t = asset_prices[ticker].reindex(active_idx).ffill()
        port_value.loc[active_idx] += prices_for_t * row["Shares"]

        # Cumulative invested (step function)
        invested.loc[active_idx] += row["Shares"] * row["Buy Price"]

        # Nifty equivalent: hypothetically invest the same ₹ in Nifty on buy date
        # Find Nifty close on/closest to buy_dt
        bench_on_or_after = bench_prices[bench_prices.index >= buy_dt]
        if bench_on_or_after.empty:
            continue
        bench_buy_price = bench_on_or_after.iloc[0]
        invested_amt = row["Shares"] * row["Buy Price"]
        nifty_units = invested_amt / bench_buy_price
        bench_for_t = bench_prices.reindex(active_idx).ffill()
        nifty_equiv.loc[active_idx] += nifty_units * bench_for_t

    return pd.DataFrame({
        "Portfolio Value": port_value,
        "Cumulative Invested": invested,
        "Nifty Equivalent": nifty_equiv,
    })


def compute_metrics(
    daily_returns: pd.Series,
    bench_returns: pd.Series,
    rf_annual: float,
) -> dict:
    rf_daily = (1 + rf_annual) ** (1 / TRADING_DAYS) - 1

    ann_ret = annualized_return(daily_returns)
    ann_vol = annualized_vol(daily_returns)
    bench_ret_ann = annualized_return(bench_returns)
    bench_vol_ann = annualized_vol(bench_returns)

    beta, alpha_daily, r2 = regress_beta_alpha(daily_returns, bench_returns)

    # Jensen's alpha via CAPM
    jensen = ann_ret - (rf_annual + beta * (bench_ret_ann - rf_annual)) if pd.notna(beta) else np.nan

    sharpe = (ann_ret - rf_annual) / ann_vol if ann_vol > 0 else 0.0
    sortino_den = downside_vol(daily_returns, rf_daily)
    sortino = (ann_ret - rf_annual) / sortino_den if sortino_den > 0 else 0.0
    treynor = (ann_ret - rf_annual) / beta if beta and beta != 0 else np.nan

    equity = (1 + daily_returns).cumprod()
    mdd, peak, trough = max_drawdown(equity)
    calmar = ann_ret / abs(mdd) if mdd < 0 else np.nan

    return {
        "CAGR": ann_ret,
        "Volatility": ann_vol,
        "Beta": beta,
        "Jensen's Alpha": jensen,
        "Sharpe": sharpe,
        "Sortino": sortino,
        "Treynor": treynor,
        "Max Drawdown": mdd,
        "Calmar": calmar,
        "R²": r2,
        "VaR 95% (daily)": value_at_risk(daily_returns, 0.95),
        "CVaR 95% (daily)": conditional_var(daily_returns, 0.95),
        "Best Day": daily_returns.max(),
        "Worst Day": daily_returns.min(),
        "Skew": daily_returns.skew(),
        "Kurtosis": daily_returns.kurt(),
        "_peak": peak,
        "_trough": trough,
    }


# ---------------------------------------------------------------------------
# Formatters
# ---------------------------------------------------------------------------
def pct(x, d=2):
    return f"{x*100:.{d}f}%" if pd.notna(x) else "—"


def num(x, d=2):
    return f"{x:.{d}f}" if pd.notna(x) else "—"


def inr(x):
    return f"₹{x:,.0f}" if pd.notna(x) else "—"


def inr_signed(x):
    """Signed currency for metric deltas — Streamlit needs the sign as the first char."""
    if pd.isna(x):
        return "—"
    return f"-₹{abs(x):,.0f}" if x < 0 else f"+₹{x:,.0f}"


def pct_signed(x, d=2):
    if pd.isna(x):
        return "—"
    return f"{x*100:+.{d}f}%"


def num_signed(x, d=2):
    if pd.isna(x):
        return "—"
    return f"{x:+.{d}f}"


# ---------------------------------------------------------------------------
# Sidebar — inputs
# ---------------------------------------------------------------------------
st.sidebar.title("⚙️ Settings")

period_choice = st.sidebar.selectbox(
    "Lookback period",
    ["3 months", "6 months", "1 year", "2 years", "3 years", "5 years",
     "10 years", "15 years", "Max"],
    index=2,
)
period_map = {
    "3 months": 90, "6 months": 180, "1 year": 365, "2 years": 730,
    "3 years": 1095, "5 years": 1825, "10 years": 3650, "15 years": 5475,
    "Max": 365 * 30,  # 30 years — yfinance returns whatever it actually has
}
end_date = date.today()
start_date = end_date - timedelta(days=period_map[period_choice])

st.sidebar.caption(
    f"Window: **{start_date} → {end_date}**  \n"
    f"_yfinance returns whatever history is available — earliest data varies per ticker._"
)

rf_annual = st.sidebar.number_input(
    "Risk-free rate (annual %)",
    value=6.5, min_value=0.0, max_value=20.0, step=0.1,
) / 100

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Ticker tips**\n"
    "- NSE stocks: append `.NS` (e.g. `RELIANCE.NS`)\n"
    "- BSE stocks: append `.BO`\n"
    "- US stocks: plain ticker (`AAPL`)"
)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("📈 Portfolio Analyzer")
st.caption(
    f"Live data via yfinance · Benchmarked against **{BENCHMARK_NAME}** · "
    f"Window: {start_date} → {end_date}"
)

# ---------------------------------------------------------------------------
# Portfolio editor
# ---------------------------------------------------------------------------
st.subheader("1. Build your portfolio")

if "portfolio" not in st.session_state:
    st.session_state["portfolio"] = EMPTY_PORTFOLIO.copy()

# --- CSV upload (optional) -------------------------------------------------
with st.expander("📤 Bulk import from CSV (broker format supported)"):
    st.caption(
        "Upload directly from Zerodha / Groww / ICICI Direct etc. "
        "**ISIN codes are preferred for accuracy** — if your CSV has an `ISIN` "
        "column, each one is resolved to the correct yfinance ticker via Yahoo's "
        "lookup. Otherwise falls back to Symbol/Company Name with `.NS` appended. "
        "Scrip Code (BSE) also supported as a fallback. Buy dates optional."
    )
    uploaded = st.file_uploader("Choose CSV", type=["csv"], key="csv_upload")

    if uploaded is not None:
        try:
            df_in = pd.read_csv(uploaded)

            # Normalize column names: lowercase, strip
            df_in.columns = [str(c).strip() for c in df_in.columns]
            lower_map = {c.lower(): c for c in df_in.columns}

            def pick(candidates):
                for cand in candidates:
                    if cand.lower() in lower_map:
                        return lower_map[cand.lower()]
                return None

            ticker_col = pick(["Ticker", "Symbol", "Company Name", "Scrip",
                                "Instrument", "Stock", "Tradingsymbol"])
            shares_col = pick(["Shares", "Quantity", "Qty", "Total Quantity",
                                "Holdings", "Units"])
            price_col  = pick(["Buy Price", "Avg Price", "Avg Trading Price",
                                "Average Price", "Avg Cost", "Avg. Cost",
                                "Average Cost", "Purchase Price"])
            date_col   = pick(["Buy Date", "Date", "Purchase Date",
                                "Trade Date", "Transaction Date"])
            isin_col   = pick(["Isin", "ISIN", "Isin Code", "Isin Number"])
            scrip_col  = pick(["Scrip Code", "Bse Code", "Scripcode"])

            missing = []
            if not (ticker_col or isin_col or scrip_col):
                missing.append("ticker/ISIN/scrip code")
            if not shares_col: missing.append("shares/quantity")
            if not price_col:  missing.append("buy/avg price")

            if missing:
                st.error(
                    f"Could not find columns for: {', '.join(missing)}. "
                    f"Found columns: {list(df_in.columns)}"
                )
            else:
                # Build base frame
                norm = pd.DataFrame({
                    "Shares":    pd.to_numeric(df_in[shares_col], errors="coerce"),
                    "Buy Price": pd.to_numeric(df_in[price_col],  errors="coerce"),
                })

                # Resolve tickers — priority: ISIN > Scrip Code (BSE) > Symbol
                isins = (df_in[isin_col].tolist() if isin_col
                          else [None] * len(df_in))
                fallback_syms = (df_in[ticker_col].tolist() if ticker_col
                                  else [None] * len(df_in))

                # If only scrip code is available, use it as fallback (BSE numeric → .BO)
                if not ticker_col and scrip_col:
                    fallback_syms = [
                        f"{int(c)}.BO" if pd.notna(c) and str(c).strip().isdigit() else None
                        for c in df_in[scrip_col]
                    ]

                with st.spinner(f"Resolving {len(df_in)} tickers via ISIN lookup…"):
                    resolved = resolve_portfolio_tickers(isins, fallback_syms)

                norm["Ticker"] = [r["ticker"] for r in resolved]
                norm["_source"] = [r["source"] for r in resolved]
                norm["_name"]   = [r["name"]   for r in resolved]

                # Buy Date column
                if date_col:
                    norm["Buy Date"] = pd.to_datetime(
                        df_in[date_col], errors="coerce", dayfirst=True
                    ).dt.date
                else:
                    norm["Buy Date"] = pd.NaT

                # Drop bad rows
                norm = norm.dropna(subset=["Ticker", "Shares", "Buy Price"])
                norm = norm[norm["Ticker"].str.len() > 0]
                norm = norm[(norm["Shares"] > 0) & (norm["Buy Price"] > 0)]

                if norm.empty:
                    st.error("No valid rows after cleaning. Check the file content.")
                else:
                    # Show resolution report so the user can verify
                    isin_resolved = (norm["_source"] == "ISIN").sum()
                    sym_resolved  = (norm["_source"] == "Symbol").sum()
                    with st.expander(
                        f"🔎 Resolved {len(norm)} tickers "
                        f"({isin_resolved} via ISIN, {sym_resolved} via symbol) — "
                        "click to verify", expanded=(isin_resolved > 0)
                    ):
                        report = norm[["Ticker", "_name", "_source",
                                        "Shares", "Buy Price"]].rename(
                            columns={"_name": "Resolved Name", "_source": "Lookup"}
                        )
                        st.dataframe(report, use_container_width=True, hide_index=True)
                        st.caption(
                            "If any ticker is wrong, edit it directly in the holdings "
                            "table below before clicking Analyze."
                        )

                    # Strip debug columns before saving to state
                    final = norm[["Ticker", "Shares", "Buy Price", "Buy Date"]].copy()
                    st.session_state["portfolio"] = final.reset_index(drop=True)

                    n_with_date = final["Buy Date"].notna().sum()
                    msg = f"Loaded {len(final)} holdings."
                    if n_with_date < len(final):
                        msg += (f" ⚠️ {len(final) - n_with_date} have no buy date — "
                                "fill them below or skip the XIRR tab.")
                    st.success(msg)
        except Exception as e:
            st.error(f"Could not parse CSV: {e}")

    # Bulk date helper — appears after upload if any rows are missing dates
    if (not st.session_state["portfolio"].empty
            and "Buy Date" in st.session_state["portfolio"].columns):
        n_missing = st.session_state["portfolio"]["Buy Date"].isna().sum()
        if n_missing > 0:
            st.markdown("---")
            st.markdown(f"**🗓️ Bulk-fill buy dates** ({n_missing} holdings missing dates)")
            bd_col1, bd_col2 = st.columns([2, 1])
            with bd_col1:
                bulk_date = st.date_input(
                    "Apply this date to all holdings missing a date",
                    value=date.today() - timedelta(days=365),
                    min_value=date(1990, 1, 1), max_value=date.today(),
                    key="bulk_date_input",
                )
            with bd_col2:
                st.write("")
                st.write("")
                if st.button("Apply to missing dates", use_container_width=True):
                    p = st.session_state["portfolio"].copy()
                    mask = p["Buy Date"].isna()
                    p.loc[mask, "Buy Date"] = bulk_date
                    st.session_state["portfolio"] = p
                    st.success(f"Set buy date to {bulk_date} for {mask.sum()} holdings.")
                    st.rerun()
            st.caption(
                "💡 Tip: edit individual buy dates in the holdings table below — "
                "the bulk fill is just a starting point for rough XIRR estimates."
            )

# --- Live ticker search ----------------------------------------------------
st.markdown("**🔍 Search for a stock**")
search_col, _ = st.columns([3, 1])
with search_col:
    query = st.text_input(
        "Company name or symbol",
        placeholder="e.g. Reliance, Tata, Infosys, Apple…",
        label_visibility="collapsed",
        key="ticker_search",
    )

picked_symbol = None
if query and len(query.strip()) >= 1:
    with st.spinner("Searching…"):
        matches = search_tickers(query, limit=12)
    if not matches:
        st.warning("No matches found. Try a different keyword.")
    else:
        options = [f"{m['symbol']} — {m['name']} · {m['exchange']} ({m['type']})"
                    for m in matches]
        selected = st.selectbox("Suggestions", options, key="suggestion_pick",
                                 label_visibility="collapsed")
        picked_symbol = matches[options.index(selected)]["symbol"]

# --- Add-holding form ------------------------------------------------------
add_c1, add_c2, add_c3, add_c4 = st.columns([1.8, 0.9, 1.1, 1.2])
with add_c1:
    manual_ticker = st.text_input(
        "Ticker", value=picked_symbol or "",
        placeholder="Pick from suggestions or type manually",
        key="manual_ticker",
    )
with add_c2:
    shares_in = st.number_input("Shares", min_value=0, value=0, step=1, key="shares_in")
with add_c3:
    buy_date_in = st.date_input(
        "Buy date", value=date.today(),
        min_value=date(1990, 1, 1), max_value=date.today(), key="buy_date_in",
    )
with add_c4:
    buy_in = st.number_input("Buy price (₹)", min_value=0.0, value=0.0,
                              step=0.01, format="%.2f", key="buy_in")

# Live OHLC hint for that day
ohlc_hint = None
if manual_ticker and buy_date_in:
    ohlc_hint = fetch_day_ohlc(manual_ticker.strip().upper(), buy_date_in)
if ohlc_hint:
    label = "on " + str(ohlc_hint["date"])
    if not ohlc_hint["is_exact"]:
        label = f"(nearest trading day: {ohlc_hint['date']})"
    st.caption(
        f"📊 **{manual_ticker.upper()}** {label} — "
        f"Open ₹{ohlc_hint['open']:.2f} · "
        f"High ₹{ohlc_hint['high']:.2f} · "
        f"Low ₹{ohlc_hint['low']:.2f} · "
        f"Close ₹{ohlc_hint['close']:.2f}"
    )
elif manual_ticker:
    st.caption(f"⚠️ Could not fetch OHLC for {manual_ticker.upper()} on {buy_date_in}.")

add_clicked = st.button("➕ Add to portfolio", use_container_width=True)

if add_clicked:
    sym = (manual_ticker or "").strip().upper()
    err = None
    if not sym:
        err = "Pick a ticker first."
    elif shares_in <= 0:
        err = "Shares must be greater than 0."
    elif buy_in <= 0:
        err = "Buy price must be greater than 0."
    elif buy_date_in > date.today():
        err = "Buy date can't be in the future."
    else:
        # Validate price is within that day's high/low
        ohlc = fetch_day_ohlc(sym, buy_date_in)
        if ohlc is None:
            err = (f"Could not fetch market data for {sym} on {buy_date_in}. "
                    "Check the ticker symbol or pick a different date.")
        elif buy_in < ohlc["low"] or buy_in > ohlc["high"]:
            err = (f"₹{buy_in:.2f} is outside the day's range "
                    f"₹{ohlc['low']:.2f} – ₹{ohlc['high']:.2f} "
                    f"(on {ohlc['date']}). Adjust the price or the date.")

    if err:
        st.error(err)
    else:
        new_row = pd.DataFrame([{
            "Ticker": sym, "Shares": shares_in,
            "Buy Price": buy_in, "Buy Date": buy_date_in,
        }])
        st.session_state["portfolio"] = pd.concat(
            [st.session_state["portfolio"], new_row], ignore_index=True
        )
        st.success(
            f"Added {shares_in} × {sym} @ ₹{buy_in:.2f} on {buy_date_in} "
            f"(within day's ₹{ohlc['low']:.2f}–₹{ohlc['high']:.2f} range ✓)"
        )
        st.rerun()

# --- Editable holdings table ----------------------------------------------
st.markdown("**Current holdings**")
if st.session_state["portfolio"].empty:
    st.info("No holdings yet. Search and add a stock above to get started.")
else:
    edited = st.data_editor(
        st.session_state["portfolio"],
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "Ticker": st.column_config.TextColumn(required=True),
            "Shares": st.column_config.NumberColumn(min_value=0, step=1, format="%d"),
            "Buy Price": st.column_config.NumberColumn(min_value=0.0, step=0.01, format="%.2f"),
            "Buy Date": st.column_config.DateColumn(
                min_value=date(1990, 1, 1), max_value=date.today()
            ),
        },
        key="portfolio_editor",
    )
    st.session_state["portfolio"] = edited

    c_a, c_b = st.columns([1, 5])
    if c_a.button("🗑️ Clear all"):
        st.session_state["portfolio"] = EMPTY_PORTFOLIO.copy()
        st.rerun()

run = st.button(
    "🚀 Analyze portfolio", type="primary", use_container_width=True,
    disabled=st.session_state["portfolio"].empty,
)

if not run:
    st.stop()

# Clean the input
portfolio = st.session_state["portfolio"].dropna(subset=["Ticker"]).copy()
portfolio["Ticker"] = portfolio["Ticker"].astype(str).str.strip().str.upper()
portfolio = portfolio[portfolio["Shares"] > 0]
if portfolio.empty:
    st.error("No valid holdings. Add at least one ticker with shares > 0.")
    st.stop()

tickers = portfolio["Ticker"].tolist()

# Extend fetch window back to earliest buy date if older than the lookback window.
# Also add a 90-day warm-up buffer so rolling-window stats (Sharpe, etc.) can be
# computed from the actual start of the user's lookback without a startup gap.
warmup_start = start_date - timedelta(days=90)
if "Buy Date" in portfolio.columns and portfolio["Buy Date"].notna().any():
    earliest_buy = pd.to_datetime(portfolio["Buy Date"]).min().date()
    fetch_start = min(warmup_start, earliest_buy - timedelta(days=5))
else:
    fetch_start = warmup_start

# ---------------------------------------------------------------------------
# Fetch
# ---------------------------------------------------------------------------
with st.spinner(f"Fetching prices for {len(tickers)} tickers + {BENCHMARK_NAME}…"):
    prices = fetch_prices(tickers + [BENCHMARK], fetch_start, end_date)

if prices.empty:
    st.error("No price data returned. Check your tickers (NSE stocks need `.NS`).")
    st.stop()

# Drop tickers that returned no data at all
missing = [t for t in tickers if t not in prices.columns or prices[t].dropna().empty]
if missing:
    st.warning(f"No data for: {', '.join(missing)} — these will be excluded.")
    portfolio = portfolio[~portfolio["Ticker"].isin(missing)]
    tickers = portfolio["Ticker"].tolist()
    if not tickers:
        st.error("No tickers left after exclusions. Cannot continue.")
        st.stop()
    # Remove dead columns from prices so they don't wipe rows during dropna
    prices = prices[[c for c in prices.columns if c not in missing]]

# Subset to just the columns we need, then ffill — but DON'T drop rows where
# only some tickers are missing (e.g. recent IPOs). We only require the
# benchmark to have data, since it's our anchor.
keep_cols = [c for c in tickers + [BENCHMARK] if c in prices.columns]
prices = prices[keep_cols].ffill()
prices = prices.dropna(subset=[BENCHMARK])

if prices.empty:
    st.error(
        "Benchmark data unavailable for this window. Try a shorter lookback."
    )
    st.stop()

# Report any recent IPOs so the user knows their full history can't be shown
ipo_notes = []
for t in tickers:
    if t in prices.columns:
        first_date = prices[t].first_valid_index()
        if first_date is not None and first_date > prices.index[0] + timedelta(days=10):
            ipo_notes.append(f"`{t}` data starts {first_date.date()}")
if ipo_notes:
    st.info("📅 " + " · ".join(ipo_notes) +
             " — earlier history shown for other holdings and the benchmark.")

# Normalize index to tz-naive for safe comparisons with user-entered dates
if prices.index.tz is not None:
    prices.index = prices.index.tz_localize(None)
asset_prices = prices[tickers]
bench_prices = prices[BENCHMARK]

# Per-ticker metadata
info_map = {t: fetch_info(t) for t in tickers}

# Current value & P&L
portfolio = portfolio.set_index("Ticker")
portfolio["Current Price"] = asset_prices.iloc[-1]
portfolio["Invested"] = portfolio["Shares"] * portfolio["Buy Price"]
portfolio["Value"] = portfolio["Shares"] * portfolio["Current Price"]
portfolio["P&L"] = portfolio["Value"] - portfolio["Invested"]
portfolio["Return %"] = portfolio["P&L"] / portfolio["Invested"] * 100
portfolio["Weight"] = portfolio["Value"] / portfolio["Value"].sum()
portfolio["Beta (yf)"] = [info_map[t]["beta"] for t in portfolio.index]
portfolio["Sector"] = [info_map[t]["sector"] for t in portfolio.index]
portfolio["Name"] = [info_map[t]["name"] for t in portfolio.index]

# Holding period & annualized return per holding
if "Buy Date" in portfolio.columns:
    today = pd.Timestamp(date.today())
    portfolio["Days Held"] = portfolio["Buy Date"].apply(
        lambda d: (today - pd.Timestamp(d)).days if pd.notna(d) else np.nan
    )
    portfolio["Annualized %"] = portfolio.apply(
        lambda r: ((r["Current Price"] / r["Buy Price"]) ** (365 / r["Days Held"]) - 1) * 100
        if pd.notna(r["Days Held"]) and r["Days Held"] > 0 else np.nan,
        axis=1,
    )

# Daily returns
# We DON'T drop rows where some tickers are NaN (e.g. recent IPOs) so that
# the benchmark and older stocks can still show their full history. The
# portfolio aggregator below handles NaN by renormalizing weights each day
# across whichever holdings exist on that date.
asset_returns_full = asset_prices.pct_change()
bench_returns_full = bench_prices.pct_change().dropna()

# Use the user-selected lookback for risk metrics (sharpe, beta, etc.)
window_start = pd.Timestamp(start_date)
asset_returns = asset_returns_full[asset_returns_full.index >= window_start]
bench_returns = bench_returns_full[bench_returns_full.index >= window_start]
if asset_returns.empty:
    asset_returns = asset_returns_full
    bench_returns = bench_returns_full


def aggregate_portfolio_returns(returns_df: pd.DataFrame, weights: np.ndarray) -> pd.Series:
    """
    Compute weighted portfolio daily returns, gracefully handling missing data
    (e.g. recent IPOs). On each date, weights are renormalized over the
    holdings that have data — so a portfolio with one not-yet-IPO'd stock
    still produces sensible returns from the other holdings.
    """
    w_series = pd.Series(weights, index=returns_df.columns)
    mask = returns_df.notna().astype(float)
    daily_w = mask.multiply(w_series, axis=1)
    row_sum = daily_w.sum(axis=1)
    daily_w = daily_w.div(row_sum.where(row_sum > 0), axis=0).fillna(0)
    return (returns_df.fillna(0) * daily_w).sum(axis=1)

# Weighted portfolio returns (using current weights, fine for analytics view).
# Dynamic renormalization handles holdings with shorter history (e.g. recent IPOs).
weights = portfolio["Weight"].reindex(asset_returns.columns).fillna(0).values
portfolio_returns = aggregate_portfolio_returns(asset_returns, weights)
# Full series (with warm-up buffer) for rolling/monthly views — no startup gap
portfolio_returns_full = aggregate_portfolio_returns(asset_returns_full, weights)

# Compute metrics
port_metrics = compute_metrics(portfolio_returns, bench_returns, rf_annual)
bench_metrics = compute_metrics(bench_returns, bench_returns, rf_annual)

# Per-ticker beta from regression
for t in tickers:
    b, _, r2 = regress_beta_alpha(asset_returns[t], bench_returns)
    portfolio.loc[t, "Beta (regr)"] = b
    portfolio.loc[t, "R²"] = r2

# ---------------------------------------------------------------------------
# Headline cards
# ---------------------------------------------------------------------------
total_inv = portfolio["Invested"].sum()
total_val = portfolio["Value"].sum()
total_pnl = total_val - total_inv
total_ret = total_pnl / total_inv

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Portfolio value", inr(total_val))
c2.metric("Invested", inr(total_inv))
c3.metric("Unrealized P&L", inr(total_pnl), delta=pct_signed(total_ret))
c4.metric("CAGR", pct(port_metrics["CAGR"]),
          delta=pct_signed(port_metrics["CAGR"] - bench_metrics["CAGR"]))
c5.metric("Sharpe", num(port_metrics["Sharpe"]),
          delta=num_signed(port_metrics["Sharpe"] - bench_metrics["Sharpe"]))

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_metrics, tab_inception, tab_perf, tab_risk, tab_holdings, tab_data = st.tabs(
    ["📊 Metrics", "📅 Since Inception (XIRR)", "📈 Performance",
     "⚠️ Risk", "🔎 Holdings", "📥 Data"]
)

# ---------------- Metrics tab ----------------
with tab_metrics:
    st.subheader("Portfolio vs Nifty 50 — head to head")

    rows = [
        ("CAGR",              pct(port_metrics["CAGR"]),          pct(bench_metrics["CAGR"])),
        ("Volatility (ann.)", pct(port_metrics["Volatility"]),    pct(bench_metrics["Volatility"])),
        ("Sharpe",            num(port_metrics["Sharpe"]),        num(bench_metrics["Sharpe"])),
        ("Sortino",           num(port_metrics["Sortino"]),       num(bench_metrics["Sortino"])),
        ("Treynor",           num(port_metrics["Treynor"]),       "—"),
        ("Beta",              num(port_metrics["Beta"]),          "1.00"),
        ("Jensen's Alpha",    pct(port_metrics["Jensen's Alpha"]),"—"),
        ("Max Drawdown",      pct(port_metrics["Max Drawdown"]),  pct(bench_metrics["Max Drawdown"])),
        ("Calmar",            num(port_metrics["Calmar"]),        num(bench_metrics["Calmar"])),
        ("VaR 95% (daily)",   pct(port_metrics["VaR 95% (daily)"]), pct(bench_metrics["VaR 95% (daily)"])),
        ("CVaR 95% (daily)",  pct(port_metrics["CVaR 95% (daily)"]),pct(bench_metrics["CVaR 95% (daily)"])),
        ("Skew",              num(port_metrics["Skew"]),          num(bench_metrics["Skew"])),
        ("Kurtosis",          num(port_metrics["Kurtosis"]),      num(bench_metrics["Kurtosis"])),
        ("Best Day",          pct(port_metrics["Best Day"]),      pct(bench_metrics["Best Day"])),
        ("Worst Day",         pct(port_metrics["Worst Day"]),     pct(bench_metrics["Worst Day"])),
    ]
    metrics_df = pd.DataFrame(rows, columns=["Metric", "Portfolio", BENCHMARK_NAME])
    st.dataframe(metrics_df, use_container_width=True, hide_index=True)

# ---------------- Since Inception tab ----------------
with tab_inception:
    st.subheader("Since-inception view — staggered buy dates, real cashflows")
    st.caption(
        "This view treats each holding's actual purchase date as its inception. "
        "The Nifty-equivalent line shows what the same ₹ would be worth had it been "
        "invested in Nifty 50 on the same day. XIRR annualizes the irregular cashflows."
    )

    if "Buy Date" not in portfolio.columns or portfolio["Buy Date"].isna().all():
        st.warning(
            "No buy dates available. Add buy dates to your holdings — either "
            "individually in the holdings table, or use the bulk-fill helper "
            "in the CSV upload section."
        )
    else:
        # Filter to holdings that have buy dates
        dated = portfolio.dropna(subset=["Buy Date"])
        skipped = len(portfolio) - len(dated)
        if skipped > 0:
            st.warning(
                f"⚠️ {skipped} holding(s) have no buy date and are excluded from "
                f"this view. {len(dated)} holding(s) included."
            )

        ts = build_portfolio_timeseries(dated, asset_prices, bench_prices)

        if ts.empty:
            st.warning("Could not build the time series — check your buy dates.")
        else:
            # Build cashflows for XIRR
            today_ts = pd.Timestamp(date.today())
            port_cashflows, nifty_cashflows = [], []
            for ticker, row in dated.iterrows():
                buy_dt = pd.Timestamp(row["Buy Date"])
                amt = row["Shares"] * row["Buy Price"]
                port_cashflows.append((buy_dt, -amt))
                nifty_cashflows.append((buy_dt, -amt))

            current_port_value = float(ts["Portfolio Value"].iloc[-1])
            current_nifty_value = float(ts["Nifty Equivalent"].iloc[-1])
            port_cashflows.append((today_ts, current_port_value))
            nifty_cashflows.append((today_ts, current_nifty_value))

            port_xirr = xirr(port_cashflows)
            nifty_xirr = xirr(nifty_cashflows)

            # Headline cards
            x1, x2, x3, x4 = st.columns(4)
            x1.metric("Portfolio XIRR",
                       pct(port_xirr) if port_xirr is not None else "—")
            x2.metric("Nifty 50 XIRR (same dates)",
                       pct(nifty_xirr) if nifty_xirr is not None else "—")
            if port_xirr is not None and nifty_xirr is not None:
                edge = port_xirr - nifty_xirr
                x3.metric("XIRR edge", pct(edge),
                           delta=("Beating Nifty" if edge >= 0 else "Lagging Nifty"),
                           delta_color=("normal" if edge >= 0 else "inverse"))
            else:
                x3.metric("XIRR edge", "—")
            invested_total = float(ts["Cumulative Invested"].iloc[-1])
            x4.metric("If invested in Nifty instead", inr(current_nifty_value),
                       delta=inr_signed(current_nifty_value - invested_total))

            # Time series chart
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=ts.index, y=ts["Portfolio Value"], name="Portfolio value",
                line=dict(color="#7F77DD", width=2.5),
            ))
            fig.add_trace(go.Scatter(
                x=ts.index, y=ts["Nifty Equivalent"], name="Nifty equivalent",
                line=dict(color="#888780", width=2, dash="dash"),
            ))
            fig.add_trace(go.Scatter(
                x=ts.index, y=ts["Cumulative Invested"], name="Cumulative invested",
                line=dict(color="#E24B4A", width=1.5, dash="dot"),
            ))
            # Mark each buy date
            for ticker, row in dated.iterrows():
                fig.add_vline(
                    x=pd.Timestamp(row["Buy Date"]),
                    line=dict(color="rgba(127,119,221,0.25)", width=1),
                    annotation_text=f"Bought {ticker}",
                    annotation_position="top",
                    annotation=dict(font_size=9, textangle=-90),
                )
            fig.update_layout(
                title="Portfolio value vs Nifty-equivalent vs cumulative invested",
                yaxis_title="Value (₹)", xaxis_title="",
                hovermode="x unified", height=460, template="plotly_white",
                legend=dict(orientation="h", y=-0.15),
            )
            st.plotly_chart(fig, use_container_width=True)

            # Per-holding XIRR table
            st.markdown("**Per-holding XIRR breakdown**")
            rows_data = []
            for ticker, row in dated.iterrows():
                if ticker not in asset_prices.columns:
                    continue
                buy_dt = pd.Timestamp(row["Buy Date"])
                amt = row["Shares"] * row["Buy Price"]
                curr_val = row["Shares"] * float(asset_prices[ticker].iloc[-1])

                # Stock XIRR
                stock_xirr = xirr([(buy_dt, -amt), (today_ts, curr_val)])

                # Nifty alternative for this holding
                bench_on_or_after = bench_prices[bench_prices.index >= buy_dt]
                if not bench_on_or_after.empty:
                    nifty_units = amt / bench_on_or_after.iloc[0]
                    nifty_val_h = nifty_units * float(bench_prices.iloc[-1])
                    nifty_xirr_h = xirr([(buy_dt, -amt), (today_ts, nifty_val_h)])
                else:
                    nifty_val_h, nifty_xirr_h = np.nan, None

                rows_data.append({
                    "Ticker": ticker,
                    "Buy Date": row["Buy Date"],
                    "Invested": amt,
                    "Current Value": curr_val,
                    "Nifty Value": nifty_val_h,
                    "Stock XIRR": stock_xirr * 100 if stock_xirr is not None else np.nan,
                    "Nifty XIRR": nifty_xirr_h * 100 if nifty_xirr_h is not None else np.nan,
                    "Edge (pp)": (stock_xirr - nifty_xirr_h) * 100
                                  if (stock_xirr is not None and nifty_xirr_h is not None)
                                  else np.nan,
                })

            breakdown = pd.DataFrame(rows_data)
            st.dataframe(
                breakdown.style.format({
                    "Invested": "₹{:,.0f}", "Current Value": "₹{:,.0f}",
                    "Nifty Value": "₹{:,.0f}",
                    "Stock XIRR": "{:+.2f}%", "Nifty XIRR": "{:+.2f}%",
                    "Edge (pp)": "{:+.2f}",
                }, na_rep="—"),
                use_container_width=True, hide_index=True,
            )

            with st.expander("ℹ️ What is XIRR and why use it here?"):
                st.markdown(
                    "**XIRR (Extended IRR)** is the annualized return that makes the "
                    "present value of all your cashflows equal to zero. Unlike CAGR — "
                    "which assumes a single lumpsum at the start — XIRR correctly "
                    "handles money invested on different dates.\n\n"
                    "Each buy is a *negative* cashflow on its date; the current portfolio "
                    "value is the *positive* cashflow today. The **Nifty XIRR** is "
                    "computed the same way but assumes each ₹ had bought Nifty 50 on "
                    "its buy date instead — a true apples-to-apples benchmark."
                )

# ---------------- Performance tab ----------------
with tab_perf:
    # Growth curves
    port_equity = (1 + portfolio_returns).cumprod() * 100
    bench_equity = (1 + bench_returns.reindex(portfolio_returns.index)).cumprod() * 100

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=port_equity.index, y=port_equity, name="Portfolio",
                              line=dict(color="#7F77DD", width=2)))
    fig.add_trace(go.Scatter(x=bench_equity.index, y=bench_equity, name=BENCHMARK_NAME,
                              line=dict(color="#888780", width=2, dash="dash")))
    fig.update_layout(
        title=f"Growth of ₹100 — {period_choice}",
        yaxis_title="Value (₹)", xaxis_title="",
        hovermode="x unified", height=400, template="plotly_white",
    )
    st.plotly_chart(fig, use_container_width=True)

    # Monthly returns heatmap
    monthly = portfolio_returns.resample("M").apply(lambda r: (1 + r).prod() - 1)
    if len(monthly) > 1:
        m_df = monthly.to_frame("ret")
        m_df["Year"] = m_df.index.year.astype(str)  # string so axis is categorical
        m_df["Month"] = m_df.index.month_name().str[:3]
        pivot = m_df.pivot_table(index="Year", columns="Month", values="ret")
        month_order = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
        pivot = pivot.reindex(columns=[m for m in month_order if m in pivot.columns])
        fig3 = px.imshow(
            pivot * 100, text_auto=".1f", aspect="auto",
            color_continuous_scale="RdYlGn", zmin=-10, zmax=10,
            labels=dict(color="Return %"),
        )
        fig3.update_layout(title="Monthly returns heatmap (%)", height=300)
        fig3.update_yaxes(type="category")
        st.plotly_chart(fig3, use_container_width=True)

# ---------------- Risk tab ----------------
with tab_risk:
    # Drawdown
    port_equity = (1 + portfolio_returns).cumprod()
    bench_eq = (1 + bench_returns.reindex(portfolio_returns.index)).cumprod()
    port_dd = (port_equity / port_equity.cummax() - 1) * 100
    bench_dd = (bench_eq / bench_eq.cummax() - 1) * 100

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=port_dd.index, y=port_dd, name="Portfolio",
                              fill="tozeroy", line=dict(color="#E24B4A")))
    fig.add_trace(go.Scatter(x=bench_dd.index, y=bench_dd, name=BENCHMARK_NAME,
                              line=dict(color="#888780", dash="dash")))
    fig.update_layout(title="Drawdown curve (%)", height=350, template="plotly_white",
                       hovermode="x unified", yaxis_title="Drawdown %")
    st.plotly_chart(fig, use_container_width=True)

    # Security Market Line
    betas = portfolio["Beta (regr)"].values
    rets = portfolio["Return %"].values / 100  # using holding-period return as proxy
    # Convert holding returns to annual-equivalent for SML to make sense:
    rets_ann = np.array([annualized_return(asset_returns[t]) for t in portfolio.index])
    max_b = max(np.nanmax(betas), 1.5) + 0.3 if len(betas) else 2
    sml_x = np.array([0, max_b])
    sml_y = rf_annual + sml_x * (bench_metrics["CAGR"] - rf_annual)

    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=sml_x, y=sml_y * 100, mode="lines",
                               name="SML (CAPM)", line=dict(color="gray", dash="dash")))
    fig3.add_trace(go.Scatter(
        x=betas, y=rets_ann * 100, mode="markers+text",
        text=portfolio.index, textposition="top center",
        marker=dict(size=12, color="#7F77DD"), name="Holdings",
    ))
    fig3.add_trace(go.Scatter(
        x=[port_metrics["Beta"]], y=[port_metrics["CAGR"] * 100],
        mode="markers+text", text=["Portfolio"], textposition="top center",
        marker=dict(size=18, color="#1D9E75", symbol="diamond"), name="Portfolio",
    ))
    fig3.add_trace(go.Scatter(
        x=[1], y=[bench_metrics["CAGR"] * 100],
        mode="markers+text", text=[BENCHMARK_NAME], textposition="top center",
        marker=dict(size=18, color="#888780", symbol="triangle-up"), name=BENCHMARK_NAME,
    ))
    fig3.update_layout(title="Security Market Line — beta vs annualized return",
                       xaxis_title="Beta", yaxis_title="Annualized return (%)",
                       height=420, template="plotly_white")
    st.plotly_chart(fig3, use_container_width=True)

# ---------------- Holdings tab ----------------
with tab_holdings:
    col1, col2 = st.columns([2, 3])
    with col1:
        fig = px.pie(
            portfolio.reset_index(), values="Value", names="Ticker",
            title="Allocation by value", hole=0.4,
        )
        fig.update_traces(textposition="outside", textinfo="label+percent")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        sector_df = portfolio.groupby("Sector")["Value"].sum().reset_index()
        fig2 = px.pie(sector_df, values="Value", names="Sector",
                       title="Allocation by sector", hole=0.4)
        st.plotly_chart(fig2, use_container_width=True)

    display_cols = ["Ticker", "Name", "Sector", "Shares"]
    if "Buy Date" in portfolio.columns:
        display_cols.append("Buy Date")
    display_cols += ["Buy Price", "Current Price", "Invested", "Value", "P&L",
                      "Return %"]
    if "Days Held" in portfolio.columns:
        display_cols += ["Days Held", "Annualized %"]
    display_cols += ["Weight", "Beta (yf)", "Beta (regr)", "R²"]
    display = portfolio.reset_index()[display_cols].copy()
    display["Weight"] = display["Weight"] * 100
    fmt_dict = {
        "Buy Price": "₹{:,.2f}", "Current Price": "₹{:,.2f}",
        "Invested": "₹{:,.0f}", "Value": "₹{:,.0f}", "P&L": "₹{:,.0f}",
        "Return %": "{:+.2f}%", "Weight": "{:.2f}%",
        "Beta (yf)": "{:.2f}", "Beta (regr)": "{:.2f}", "R²": "{:.2f}",
    }
    if "Days Held" in display.columns:
        fmt_dict["Days Held"] = "{:.0f}"
        fmt_dict["Annualized %"] = "{:+.2f}%"
    st.dataframe(
        display.style.format(fmt_dict, na_rep="—"),
        use_container_width=True,
    )

    # Per-holding return bar
    bar_df = portfolio.reset_index().copy()
    bar_df["color"] = np.where(bar_df["Return %"] >= 0, "#1D9E75", "#E24B4A")
    fig3 = go.Figure()
    fig3.add_trace(go.Bar(x=bar_df["Ticker"], y=bar_df["Return %"],
                           marker_color=bar_df["color"], name="Holding"))
    fig3.add_hline(y=bench_metrics["CAGR"] * 100, line_dash="dash",
                   line_color="gray",
                   annotation_text=f"{BENCHMARK_NAME} CAGR")
    fig3.update_layout(title="Return per holding vs benchmark CAGR (%)",
                       height=350, template="plotly_white")
    st.plotly_chart(fig3, use_container_width=True)

# ---------------- Data tab ----------------
with tab_data:
    st.subheader("Download raw data")

    csv_holdings = portfolio.reset_index().to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Holdings (CSV)", csv_holdings,
                        "portfolio_holdings.csv", "text/csv")

    metrics_export = pd.DataFrame({
        "Portfolio": {k: v for k, v in port_metrics.items() if not k.startswith("_")},
        BENCHMARK_NAME: {k: v for k, v in bench_metrics.items() if not k.startswith("_")},
    })
    csv_metrics = metrics_export.to_csv().encode("utf-8")
    st.download_button("⬇️ Metrics (CSV)", csv_metrics,
                        "portfolio_metrics.csv", "text/csv")

    csv_prices = prices.to_csv().encode("utf-8")
    st.download_button("⬇️ Price history (CSV)", csv_prices,
                        "prices.csv", "text/csv")

    with st.expander("View price history"):
        st.dataframe(prices.tail(50), use_container_width=True)

st.markdown("---")
st.caption(
    "⚠️ Educational tool. Not investment advice. yfinance data can have gaps, "
    "stale betas, or missing ticker metadata — always cross-check before acting."
)