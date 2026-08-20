"""
data_fetcher.py
Fetches historical price data from yfinance and synthesizes returns for hardcoded assets.
Caches fetched data for 24 hours to avoid redundant API calls.
"""

import os
import psycopg2.extras
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from modules.db import get_db_connection

def get_asset_universe(preferences: dict) -> dict:
    """
    Returns filtered asset dict based on user preferences by querying the Postgres assets table.
    """
    selected_tickers = {"^NSEI", "HDFCBANK.NS", "RELIANCE.NS"}  # always included base
    
    sectors = preferences.get("sectors", [])
    sector_map = {
        "Technology": ["INFY.NS", "TCS.NS", "WIPRO.NS"],
        "Healthcare": ["SUNPHARMA.NS", "DRREDDY.NS"],
        "Finance": ["HDFCBANK.NS", "ICICIBANK.NS", "^NSEBANK"],
        "Energy": ["RELIANCE.NS", "ONGC.NS"],
        "Consumer": ["HINDUNILVR.NS", "MARUTI.NS"],
    }
    for s in sectors:
        selected_tickers.update(sector_map.get(s, []))

    if preferences.get("interested_international", False):
        selected_tickers.update(["SPY", "QQQ", "EEM", "IVE", "IVW", "USMV", "US_TREASURY_10Y", "US_CORP_IG"])

    if preferences.get("interested_commodities", False):
        selected_tickers.update(["GOLDBEES.NS", "SILVERBEES.NS", "BTC-USD", "ETH-USD", "SOL-USD"])

    if preferences.get("willing_bonds", True):
        selected_tickers.update([
            "INDIA_GOVT_10Y", "INDIA_CORP_AAA", "INFLATION_LINKED_BOND",
            "SBI_FD", "HDFC_FD", "POST_OFFICE_TD"
        ])

    if preferences.get("open_reits", False):
        selected_tickers.update(["EMBASSY_REIT", "MINDSPACE_REIT", "BROOKFIELD_REIT"])
        
    live = {}
    hardcoded = {}
    
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT ticker, is_synthetic, annual_return, annual_volatility, equity_corr, asset_class, sector FROM assets WHERE ticker = ANY(%s)", (list(selected_tickers),))
                rows = cur.fetchall()
                for row in rows:
                    ticker, is_syn, ret, vol, corr, asset_class, sector = row
                    # Currency defaults to INR for live assets except specific ones, but for synthetic we just need the stats.
                    # We will handle USD/INR in fetch_price_data checking the ticker name or we can just hardcode for USD assets.
                    if is_syn:
                        hardcoded[ticker] = {
                            "label": ticker,
                            "category": asset_class,
                            "sector": sector,
                            "annual_return": float(ret) if ret else 0.0,
                            "annual_vol": float(vol) if vol else 0.0,
                            "equity_corr": float(corr) if corr else 0.0
                        }
                    else:
                        live[ticker] = {
                            "label": ticker,
                            "category": asset_class,
                            "sector": sector,
                            "currency": "USD" if ticker in ["SPY", "QQQ", "EEM", "IVE", "IVW", "USMV", "GLD", "BTC-USD", "ETH-USD", "SOL-USD"] else "INR"
                        }
    except Exception as e:
        print(f"Error fetching asset universe from DB: {e}")
        
    return {"live": live, "hardcoded": hardcoded}

CACHE_TTL_HOURS = 24

def _get_stale_or_missing_tickers(tickers: list) -> list:
    """Returns a list of tickers that either have no data or data older than 24 hours."""
    needs_fetch = []
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                for ticker in tickers:
                    cur.execute("SELECT MAX(fetched_at) FROM prices WHERE ticker = %s", (ticker,))
                    row = cur.fetchone()
                    if row and row[0]:
                        last_fetched = row[0]
                        if datetime.now() - last_fetched > timedelta(hours=CACHE_TTL_HOURS):
                            needs_fetch.append(ticker)
                    else:
                        needs_fetch.append(ticker)
    except Exception as e:
        print(f"DB Error checking cache status: {e}")
        return tickers  # Fetch all if DB fails
    return needs_fetch

def _save_prices_to_db(ticker: str, df: pd.Series, currency: str):
    if df.empty:
        return
    data = []
    for date, price in df.items():
        if pd.isna(price): continue
        data.append((ticker, date.date(), float(price), currency))
        
    query = """
        INSERT INTO prices (ticker, price_date, close_price, currency, fetched_at)
        VALUES %s
        ON CONFLICT (ticker, price_date) DO UPDATE 
        SET close_price = EXCLUDED.close_price, 
            currency = EXCLUDED.currency, 
            fetched_at = EXCLUDED.fetched_at
    """
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                psycopg2.extras.execute_values(cur, query, data, template="(%s, %s, %s, %s, NOW())")
            conn.commit()
    except Exception as e:
        print(f"DB Error saving {ticker}: {e}")

def _load_prices_from_db(tickers: list) -> pd.DataFrame:
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                query = "SELECT ticker, price_date, close_price FROM prices WHERE ticker = ANY(%s) ORDER BY price_date"
                cur.execute(query, (tickers,))
                rows = cur.fetchall()
                if not rows:
                    return pd.DataFrame()
                
                df = pd.DataFrame(rows, columns=['ticker', 'price_date', 'close_price'])
                df['close_price'] = df['close_price'].astype(float)
                df['price_date'] = pd.to_datetime(df['price_date'])
                pivot_df = df.pivot(index='price_date', columns='ticker', values='close_price')
                return pivot_df
    except Exception as e:
        print(f"DB Error loading prices: {e}")
        return pd.DataFrame()


def fetch_price_data(tickers: list, asset_universe: dict, period: str = "5y") -> pd.DataFrame:
    """Download adjusted close prices from yfinance and cache to Postgres."""
    if not tickers:
        return pd.DataFrame()

    needs_fetch = _get_stale_or_missing_tickers(tickers)

    if needs_fetch:
        # Fetch USDINR if needed
        has_usd = any(asset_universe["live"].get(t, {}).get("currency") == "USD" for t in needs_fetch)
        usdinr = pd.Series(dtype=float)
        if has_usd:
            try:
                usdinr_data = yf.download("USDINR=X", period=period, auto_adjust=True, progress=False)["Close"]
                if isinstance(usdinr_data, pd.DataFrame):
                    usdinr = usdinr_data.iloc[:, 0]
                else:
                    usdinr = usdinr_data
            except Exception:
                pass

        for ticker in needs_fetch:
            try:
                data = yf.download(ticker, period=period, auto_adjust=True, progress=False)
                if data.empty:
                    continue
                close = data["Close"]
                if isinstance(close, pd.DataFrame):
                    close = close.iloc[:, 0]
                
                currency = asset_universe["live"].get(ticker, {}).get("currency", "INR")
                
                # Convert to INR if asset is in USD
                if currency == "USD" and not usdinr.empty:
                    common_idx = close.index.intersection(usdinr.index)
                    close = close.loc[common_idx] * usdinr.loc[common_idx]
                    currency = "INR" # after conversion it's stored as INR

                close.name = ticker
                _save_prices_to_db(ticker, close, currency)
            except Exception as e:
                print(f"Error processing {ticker}: {e}")

    prices = _load_prices_from_db(tickers)
    if not prices.empty:
        prices = prices.ffill().dropna()
        
    return prices


def compute_monthly_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Resample daily prices to monthly and compute Log Returns."""
    monthly = prices.resample("ME").last()
    # Log returns: log(p_t / p_{t-1})
    returns = np.log(monthly / monthly.shift(1)).dropna()
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

    prices = fetch_price_data(live_tickers, asset_universe) if live_tickers else pd.DataFrame()

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

    # We will let portfolio_optimizer query metadata via JOIN.
    # However, for compatibility, we just return a dataframe containing the tickers as index
    # and their asset classes/sectors fetched from DB.
    meta_rows = []
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT ticker, asset_class, sector FROM assets WHERE ticker = ANY(%s)", (list(combined.columns),))
                rows = cur.fetchall()
                for r in rows:
                    meta_rows.append({"ticker": r[0], "asset_class": r[1], "sector": r[2], "category": r[1], "label": r[0]})
    except Exception as e:
        print(f"Error fetching meta from DB: {e}")
        
    meta = pd.DataFrame(meta_rows).set_index("ticker")
    # ensure alignment
    available_tickers = [t for t in combined.columns if t in meta.index]
    combined = combined[available_tickers]
    meta = meta.loc[combined.columns]

    if progress_cb:
        progress_cb("Computing statistics…", 0.8)

    return combined, meta
