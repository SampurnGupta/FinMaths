import psycopg2.extras
from modules.db import get_db_connection

# ── Hardcoded Assets (non-tradeable / limited yfinance coverage) ─────────────
HARDCODED_ASSETS = {
    "SBI_FD": {"label": "SBI Fixed Deposit", "category": "fd", "sector": "Fixed Income", "annual_return": 0.070, "annual_vol": 0.0005, "equity_corr": 0.00},
    "HDFC_FD": {"label": "HDFC Fixed Deposit", "category": "fd", "sector": "Fixed Income", "annual_return": 0.0725, "annual_vol": 0.0005, "equity_corr": 0.00},
    "ICICI_FD": {"label": "ICICI Bank Fixed Deposit", "category": "fd", "sector": "Fixed Income", "annual_return": 0.075, "annual_vol": 0.0005, "equity_corr": 0.00},
    "POST_OFFICE_TD": {"label": "Post Office Term Deposit", "category": "fd", "sector": "Fixed Income", "annual_return": 0.075, "annual_vol": 0.0005, "equity_corr": 0.00},
    "INDIA_GOVT_10Y": {"label": "India 10Y Govt Bond", "category": "bond", "sector": "Fixed Income", "annual_return": 0.072, "annual_vol": 0.030, "equity_corr": -0.15},
    "INDIA_CORP_AAA": {"label": "India AAA Corp Bond", "category": "bond", "sector": "Fixed Income", "annual_return": 0.080, "annual_vol": 0.040, "equity_corr": -0.10},
    "US_TREASURY_10Y": {"label": "US Treasury 10Y", "category": "bond", "sector": "Fixed Income", "annual_return": 0.045, "annual_vol": 0.050, "equity_corr": -0.20},
    "US_CORP_IG": {"label": "US Corp IG Bond", "category": "bond", "sector": "Fixed Income", "annual_return": 0.055, "annual_vol": 0.060, "equity_corr": -0.10},
    "EMBASSY_REIT": {"label": "Embassy Office Parks REIT", "category": "reit", "sector": "Real Estate", "annual_return": 0.080, "annual_vol": 0.150, "equity_corr": 0.45},
    "MINDSPACE_REIT": {"label": "Mindspace Business Parks REIT", "category": "reit", "sector": "Real Estate", "annual_return": 0.075, "annual_vol": 0.140, "equity_corr": 0.42},
    "BROOKFIELD_REIT": {"label": "Brookfield India REIT", "category": "reit", "sector": "Real Estate", "annual_return": 0.085, "annual_vol": 0.160, "equity_corr": 0.48},
    "INFLATION_LINKED_BOND": {"label": "Inflation Linked Bond", "category": "bond", "sector": "Fixed Income", "annual_return": 0.065, "annual_vol": 0.040, "equity_corr": -0.05}
}

LIVE_ASSETS = {
    "^NSEI": {"label": "Nifty 50", "category": "equity", "sector": "Broad Market", "currency": "INR"},
    "^NSEBANK": {"label": "Nifty Bank", "category": "equity", "sector": "Finance", "currency": "INR"},
    "INFY.NS": {"label": "Infosys", "category": "equity", "sector": "Technology", "currency": "INR"},
    "TCS.NS": {"label": "TCS", "category": "equity", "sector": "Technology", "currency": "INR"},
    "WIPRO.NS": {"label": "Wipro", "category": "equity", "sector": "Technology", "currency": "INR"},
    "SUNPHARMA.NS": {"label": "Sun Pharma", "category": "equity", "sector": "Healthcare", "currency": "INR"},
    "DRREDDY.NS": {"label": "Dr. Reddy's", "category": "equity", "sector": "Healthcare", "currency": "INR"},
    "HDFCBANK.NS": {"label": "HDFC Bank", "category": "equity", "sector": "Finance", "currency": "INR"},
    "ICICIBANK.NS": {"label": "ICICI Bank", "category": "equity", "sector": "Finance", "currency": "INR"},
    "SBIN.NS": {"label": "SBI", "category": "equity", "sector": "Finance", "currency": "INR"},
    "RELIANCE.NS": {"label": "Reliance Industries", "category": "equity", "sector": "Energy", "currency": "INR"},
    "ONGC.NS": {"label": "ONGC", "category": "equity", "sector": "Energy", "currency": "INR"},
    "HINDUNILVR.NS": {"label": "Hindustan Unilever", "category": "equity", "sector": "Consumer", "currency": "INR"},
    "MARUTI.NS": {"label": "Maruti Suzuki", "category": "equity", "sector": "Consumer", "currency": "INR"},
    "SPY": {"label": "S&P 500 ETF", "category": "equity", "sector": "US Market", "currency": "USD"},
    "QQQ": {"label": "NASDAQ ETF", "category": "equity", "sector": "US Market", "currency": "USD"},
    "EEM": {"label": "Emerging Markets ETF", "category": "equity", "sector": "International", "currency": "USD"},
    "IVE": {"label": "Value ETF (S&P 500)", "category": "equity", "sector": "US Market", "currency": "USD"},
    "IVW": {"label": "Growth ETF (S&P 500)", "category": "equity", "sector": "US Market", "currency": "USD"},
    "USMV": {"label": "Low Volatility ETF", "category": "equity", "sector": "US Market", "currency": "USD"},
    "GOLDBEES.NS": {"label": "Gold BeES ETF", "category": "commodity", "sector": "Commodities", "currency": "INR"},
    "SILVERBEES.NS": {"label": "Silver BeES ETF", "category": "commodity", "sector": "Commodities", "currency": "INR"},
    "GLD": {"label": "SPDR Gold ETF", "category": "commodity", "sector": "Commodities", "currency": "USD"},
    "BTC-USD": {"label": "Bitcoin", "category": "commodity", "sector": "Crypto", "currency": "USD"},
    "ETH-USD": {"label": "Ethereum", "category": "commodity", "sector": "Crypto", "currency": "USD"},
    "SOL-USD": {"label": "Solana", "category": "commodity", "sector": "Crypto", "currency": "USD"}
}

CATEGORY_CLASS = {
    "equity": "equity",
    "commodity": "alt",
    "reit": "alt",
    "bond": "debt",
    "fd": "debt",
}

def seed_assets():
    data = []
    
    # Process LIVE_ASSETS
    for ticker, info in LIVE_ASSETS.items():
        asset_class = CATEGORY_CLASS.get(info["category"], "equity")
        data.append((
            ticker, 
            asset_class, 
            info["sector"], 
            None, # annual_return
            None, # annual_volatility
            None, # equity_corr
            False # is_synthetic
        ))
        
    # Process HARDCODED_ASSETS
    for ticker, info in HARDCODED_ASSETS.items():
        asset_class = CATEGORY_CLASS.get(info["category"], "debt")
        data.append((
            ticker,
            asset_class,
            info["sector"],
            info.get("annual_return"),
            info.get("annual_vol"),
            info.get("equity_corr"),
            True # is_synthetic
        ))

    query = """
        INSERT INTO assets (ticker, asset_class, sector, annual_return, annual_volatility, equity_corr, is_synthetic)
        VALUES %s
        ON CONFLICT (ticker) DO UPDATE
        SET asset_class = EXCLUDED.asset_class,
            sector = EXCLUDED.sector,
            annual_return = EXCLUDED.annual_return,
            annual_volatility = EXCLUDED.annual_volatility,
            equity_corr = EXCLUDED.equity_corr,
            is_synthetic = EXCLUDED.is_synthetic
    """
    
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                psycopg2.extras.execute_values(cur, query, data)
            conn.commit()
        print(f"Successfully seeded {len(data)} assets into the database.")
    except Exception as e:
        print(f"Error seeding assets: {e}")

if __name__ == "__main__":
    seed_assets()
