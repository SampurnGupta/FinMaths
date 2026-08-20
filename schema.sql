-- FinMaths PostgreSQL Schema

-- Phase 1: Prices Table
CREATE TABLE IF NOT EXISTS prices (
    ticker TEXT,
    price_date DATE,
    close_price NUMERIC(14,4),
    currency TEXT DEFAULT 'INR',
    fetched_at TIMESTAMP DEFAULT now(),
    PRIMARY KEY (ticker, price_date)
);

CREATE INDEX IF NOT EXISTS idx_prices_ticker_date ON prices(ticker, price_date DESC);

-- Phase 2: Assets Table
CREATE TABLE IF NOT EXISTS assets (
    ticker TEXT PRIMARY KEY,
    asset_class TEXT,
    sector TEXT,
    annual_return NUMERIC(6,4),
    annual_volatility NUMERIC(6,4),
    equity_corr NUMERIC(4,3),
    is_synthetic BOOLEAN DEFAULT false
);

-- Phase 3: Portfolio Runs
CREATE TABLE IF NOT EXISTS portfolio_runs (
    run_id SERIAL PRIMARY KEY,
    risk_profile TEXT,
    weights JSONB,
    expected_return NUMERIC(6,4),
    sharpe_ratio NUMERIC(6,4),
    created_at TIMESTAMP DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_portfolio_runs_created_at ON portfolio_runs(created_at DESC);
