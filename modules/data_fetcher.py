"""
data_fetcher.py
Fetches historical price data from yfinance and synthesizes returns for hardcoded assets.
Caches fetched data for 24 hours to avoid redundant API calls.
"""

import os
import pickle
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

# ── Hardcoded Assets (non-tradeable / limited yfinance coverage) ─────────────
HARDCODED_ASSETS = {
    "SBI_FD": {
        "label": "SBI Fixed Deposit",
        "category": "fd",
        "sector": "Fixed Income",
        "annual_return": 0.070,
        "annual_vol": 0.0005,
        "equity_corr": 0.00,
    },
    "HDFC_FD": {
        "label": "HDFC Fixed Deposit",
        "category": "fd",
        "sector": "Fixed Income",
        "annual_return": 0.0725,
        "annual_vol": 0.0005,
        "equity_corr": 0.00,
    },
    "ICICI_FD": {
        "label": "ICICI Bank Fixed Deposit",
        "category": "fd",
        "sector": "Fixed Income",
        "annual_return": 0.075,
        "annual_vol": 0.0005,
        "equity_corr": 0.00,
    },
    "POST_OFFICE_TD": {
        "label": "Post Office Term Deposit",
        "category": "fd",
        "sector": "Fixed Income",
        "annual_return": 0.075,
        "annual_vol": 0.0005,
        "equity_corr": 0.00,
    },
    # Indian Bonds
    "INDIA_GOVT_10Y": {
        "label": "India 10Y Govt Bond",
        "category": "bond",
        "sector": "Fixed Income",
        "annual_return": 0.072,
        "annual_vol": 0.030,
        "equity_corr": -0.15,
    },
    "INDIA_CORP_AAA": {
        "label": "India AAA Corp Bond",
        "category": "bond",
        "sector": "Fixed Income",
        "annual_return": 0.080,
        "annual_vol": 0.040,
        "equity_corr": -0.10,
    },
    # International Bonds
    "US_TREASURY_10Y": {
        "label": "US Treasury 10Y",
        "category": "bond",
        "sector": "Fixed Income",
        "annual_return": 0.045,
        "annual_vol": 0.050,
        "equity_corr": -0.20,
    },
    "US_CORP_IG": {
        "label": "US Corp IG Bond",
        "category": "bond",
        "sector": "Fixed Income",
        "annual_return": 0.055,
        "annual_vol": 0.060,
        "equity_corr": -0.10,
    },
    # REITs (hardcoded — limited yfinance India REIT coverage)
    "EMBASSY_REIT": {
        "label": "Embassy Office Parks REIT",
        "category": "reit",
        "sector": "Real Estate",
        "annual_return": 0.080,
        "annual_vol": 0.150,
        "equity_corr": 0.45,
    },
    "MINDSPACE_REIT": {
        "label": "Mindspace Business Parks REIT",
        "category": "reit",
        "sector": "Real Estate",
        "annual_return": 0.075,
        "annual_vol": 0.140,
        "equity_corr": 0.42,
    },
    "BROOKFIELD_REIT": {
        "label": "Brookfield India REIT",
        "category": "reit",
        "sector": "Real Estate",
        "annual_return": 0.085,
        "annual_vol": 0.160,
        "equity_corr": 0.48,
    },
}

# ── Live Asset Universe (fetched from yfinance) ───────────────────────────────
LIVE_ASSETS = {
    # Broad Market
    "^NSEI": {"label": "Nifty 50", "category": "equity", "sector": "Broad Market"},
    "^NSEBANK": {"label": "Nifty Bank", "category": "equity", "sector": "Finance"},
    # Tech / IT
    "INFY.NS": {"label": "Infosys", "category": "equity", "sector": "Technology"},
    "TCS.NS": {"label": "TCS", "category": "equity", "sector": "Technology"},
    "WIPRO.NS": {"label": "Wipro", "category": "equity", "sector": "Technology"},
    # Healthcare
    "SUNPHARMA.NS": {"label": "Sun Pharma", "category": "equity", "sector": "Healthcare"},
    "DRREDDY.NS": {"label": "Dr. Reddy's", "category": "equity", "sector": "Healthcare"},
    # Finance
    "HDFCBANK.NS": {"label": "HDFC Bank", "category": "equity", "sector": "Finance"},
    "ICICIBANK.NS": {"label": "ICICI Bank", "category": "equity", "sector": "Finance"},
    "SBIN.NS": {"label": "SBI", "category": "equity", "sector": "Finance"},
    # Energy
    "RELIANCE.NS": {"label": "Reliance Industries", "category": "equity", "sector": "Energy"},
    "ONGC.NS": {"label": "ONGC", "category": "equity", "sector": "Energy"},
    # Consumer / FMCG
    "HINDUNILVR.NS": {"label": "Hindustan Unilever", "category": "equity", "sector": "Consumer"},
    "MARUTI.NS": {"label": "Maruti Suzuki", "category": "equity", "sector": "Consumer"},
    # US Equities
    "SPY": {"label": "S&P 500 ETF", "category": "equity", "sector": "US Market"},
    "QQQ": {"label": "NASDAQ ETF", "category": "equity", "sector": "US Market"},
    # Gold
    "GOLDBEES.NS": {"label": "Gold BeES ETF", "category": "commodity", "sector": "Commodities"},
    "GLD": {"label": "SPDR Gold ETF", "category": "commodity", "sector": "Commodities"},
}

# Sector → equity/debt/alt classification for constraint mapping
CATEGORY_CLASS = {
    "equity": "equity",
    "commodity": "alt",
    "reit": "alt",
    "bond": "debt",
    "fd": "debt",
}

CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
CACHE_TTL_HOURS = 24


def _cache_path(key: str) -> str:
    os.makedirs(CACHE_DIR, exist_ok=True)
    return os.path.join(CACHE_DIR, f"{key}.pkl")


def _load_cache(key: str):
    path = _cache_path(key)
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        ts, data = pickle.load(f)
    if datetime.now() - ts > timedelta(hours=CACHE_TTL_HOURS):
        return None
    return data


def _save_cache(key: str, data):
    with open(_cache_path(key), "wb") as f:
        pickle.dump((datetime.now(), data), f)


def get_asset_universe(preferences: dict) -> dict:
    """
    Returns filtered asset dict based on user preferences.
    Always includes broad market + sector picks. Adds bonds/gold/intl/reits per prefs.
    """
    sectors = preferences.get("sectors", [])
    sector_map = {
        "Technology": ["INFY.NS", "TCS.NS", "WIPRO.NS"],
        "Healthcare": ["SUNPHARMA.NS", "DRREDDY.NS"],
        "Finance": ["HDFCBANK.NS", "ICICIBANK.NS", "^NSEBANK"],
        "Energy": ["RELIANCE.NS", "ONGC.NS"],
        "Consumer": ["HINDUNILVR.NS", "MARUTI.NS"],
    }

    selected_live = {"^NSEI", "HDFCBANK.NS", "RELIANCE.NS"}  # always included base
    for s in sectors:
        for ticker in sector_map.get(s, []):
            selected_live.add(ticker)

    selected_hardcoded = set()

    if preferences.get("interested_international", False):
        selected_live.update(["SPY", "QQQ"])
        selected_hardcoded.update(["US_TREASURY_10Y", "US_CORP_IG"])

    if preferences.get("interested_commodities", False):
        selected_live.add("GOLDBEES.NS")

    if preferences.get("willing_bonds", True):
        selected_hardcoded.update(["INDIA_GOVT_10Y", "INDIA_CORP_AAA"])
        # FDs
        selected_hardcoded.update(["SBI_FD", "HDFC_FD", "POST_OFFICE_TD"])

    if preferences.get("open_reits", False):
        selected_hardcoded.update(["EMBASSY_REIT", "MINDSPACE_REIT", "BROOKFIELD_REIT"])

    live = {t: LIVE_ASSETS[t] for t in selected_live if t in LIVE_ASSETS}
    hardcoded = {k: HARDCODED_ASSETS[k] for k in selected_hardcoded}
    return {"live": live, "hardcoded": hardcoded}


def fetch_price_data(tickers: list, period: str = "5y") -> pd.DataFrame:
    """Download adjusted close prices from yfinance with caching."""
    key = "prices_" + "_".join(sorted(tickers))
    cached = _load_cache(key)
    if cached is not None:
        return cached

    failed = []
    frames = {}
    for ticker in tickers:
        try:
            data = yf.download(ticker, period=period, auto_adjust=True, progress=False)
            if data.empty:
                failed.append(ticker)
                continue
            close = data["Close"]
            if isinstance(close, pd.DataFrame):
                close = close.squeeze()
            frames[ticker] = close
        except Exception:
            failed.append(ticker)

    if not frames:
        return pd.DataFrame()

    prices = pd.DataFrame(frames).dropna(how="all")
    prices = prices.fillna(method="ffill").dropna()
    _save_cache(key, prices)
    return prices


def compute_monthly_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Resample daily prices to monthly and compute simple returns."""
    monthly = prices.resample("ME").last()
    returns = monthly.pct_change().dropna()
    return returns


def synthesize_returns(hardcoded: dict, n_months: int, equity_monthly_returns: pd.Series = None, seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic monthly returns for hardcoded assets.
    Correlates with equity market returns where equity_corr != 0.
    """
    rng = np.random.default_rng(seed)
    synth = {}

    for key, cfg in hardcoded.items():
        monthly_mean = cfg["annual_return"] / 12
        monthly_std = cfg["annual_vol"] / np.sqrt(12)
        corr = cfg.get("equity_corr", 0.0)

        if corr != 0.0 and equity_monthly_returns is not None and len(equity_monthly_returns) == n_months:
            eq = equity_monthly_returns.values
            eq_std = eq.std() if eq.std() > 0 else 1e-6
            idio_std = monthly_std * np.sqrt(max(0, 1 - corr**2))
            idio = rng.normal(0, idio_std, n_months)
            systematic = corr * (monthly_std / eq_std) * (eq - eq.mean())
            returns = monthly_mean + systematic + idio
        else:
            returns = rng.normal(monthly_mean, monthly_std, n_months)

        synth[key] = returns

    return pd.DataFrame(synth)


def build_combined_returns(asset_universe: dict, progress_cb=None) -> tuple:
    """
    Main entry point: returns (combined_returns_df, asset_meta_df).
    asset_meta_df has columns: label, category, sector, asset_class
    """
    live_tickers = list(asset_universe["live"].keys())
    hardcoded = asset_universe["hardcoded"]

    if progress_cb:
        progress_cb("Fetching historical prices from yfinance…", 0.2)

    prices = fetch_price_data(live_tickers) if live_tickers else pd.DataFrame()

    live_returns = pd.DataFrame()
    if not prices.empty:
        live_returns = compute_monthly_returns(prices)
        # Only keep tickers that actually returned data
        available = [t for t in live_tickers if t in live_returns.columns]
        live_returns = live_returns[available]

    n_months = len(live_returns) if not live_returns.empty else 60

    # Equity proxy for correlation: mean of equity assets
    equity_cols = [t for t in live_returns.columns
                   if asset_universe["live"].get(t, {}).get("category") == "equity"]
    eq_proxy = live_returns[equity_cols].mean(axis=1) if equity_cols else None

    if progress_cb:
        progress_cb("Synthesizing hardcoded asset returns…", 0.5)

    synth_returns = synthesize_returns(hardcoded, n_months, eq_proxy)
    synth_returns.index = live_returns.index if not live_returns.empty else range(n_months)

    # Combine
    combined = pd.concat([live_returns, synth_returns], axis=1).dropna()

    # Build metadata
    meta_rows = []
    for t, info in asset_universe["live"].items():
        if t in combined.columns:
            meta_rows.append({
                "ticker": t,
                "label": info["label"],
                "category": info["category"],
                "sector": info["sector"],
                "asset_class": CATEGORY_CLASS.get(info["category"], "equity"),
                "hardcoded": False,
            })
    for k, info in hardcoded.items():
        if k in combined.columns:
            meta_rows.append({
                "ticker": k,
                "label": info["label"],
                "category": info["category"],
                "sector": info["sector"],
                "asset_class": CATEGORY_CLASS.get(info["category"], "debt"),
                "hardcoded": True,
            })

    meta = pd.DataFrame(meta_rows).set_index("ticker")
    meta = meta.loc[combined.columns]  # align order

    if progress_cb:
        progress_cb("Computing statistics…", 0.8)

    return combined, meta
