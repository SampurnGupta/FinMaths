"""
portfolio_optimizer.py
Efficient Frontier + Monte Carlo portfolio optimization using scipy.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from modules.db import get_db_connection

RISK_FREE_RATE_ANNUAL = 0.065  # RBI repo-rate approximation
INFLATION_RATE = 0.060         # Indian CPI average approximation
RF_MONTHLY = RISK_FREE_RATE_ANNUAL / 12


def _portfolio_stats(weights: np.ndarray, mean_returns: np.ndarray, cov_matrix: np.ndarray, meta: pd.DataFrame = None, prev_weights: np.ndarray = None):
    """
    Return (annual_return, annual_volatility, sharpe).
    Incorporates tax adjustment and turnover penalty if meta/prev_weights provided.
    """
    raw_ret = np.dot(weights, mean_returns) * 12
    
    # Tax Adjustment (Approximate blended rate)
    tax_adj_ret = raw_ret
    if meta is not None:
        # Equity LTCG ~12.5%, Debt ~30%
        equity_mask = (meta["asset_class"] == "equity").values
        eq_w = weights[equity_mask].sum()
        debt_w = 1 - eq_w
        blended_tax = (eq_w * 0.125) + (debt_w * 0.30)
        tax_adj_ret = raw_ret * (1 - blended_tax)
    
    # Turnover Penalty (Rebalancing cost ~0.5%)
    penalty = 0
    if prev_weights is not None:
        turnover = np.sum(np.abs(weights - prev_weights)) / 2
        penalty = turnover * 0.005 # 0.5% rebalancing cost
    
    final_ret = (tax_adj_ret - penalty) - INFLATION_RATE
    vol = np.sqrt(weights @ cov_matrix @ weights) * np.sqrt(12)
    sharpe = (final_ret - RISK_FREE_RATE_ANNUAL) / vol if vol > 0 else 0
    return final_ret, vol, sharpe

def get_return_explanation():
    return """
    **How we calculate Expected Returns:**
    1. **Pre-Tax Return**: Calculated as the weighted average of individual asset historical returns (or synthesized returns for FDs/Bonds).
    2. **Tax Adjustment**: We apply a blended tax rate based on your portfolio composition:
       - **Equity (12.5%)**: Long-term Capital Gains (LTCG) approximation.
       - **Debt/FD (30%)**: Flat tax rate for interest/gains.
    3. **Rebalancing Cost**: A 0.5% penalty is applied to turnover to account for brokerage and slippage.
    4. **Inflation**: An assumed inflation rate of 6.0% p.a. is subtracted to show 'Real Returns' (purchasing power).
    """


def build_constraints(risk_profile: dict, meta_fallback: pd.DataFrame):
    """
    Construct scipy constraints and bounds for optimization.
    Pulls sector/asset_class via a JOIN against the prices table from Postgres.
    """
    tickers = list(meta_fallback.index)
    n = len(tickers)

    query = """
        SELECT DISTINCT p.ticker, a.asset_class, a.sector
        FROM prices p
        JOIN assets a ON p.ticker = a.ticker
        WHERE p.ticker = ANY(%s)
    """
    
    meta_dict = {}
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, (tickers,))
                for row in cur.fetchall():
                    meta_dict[row[0]] = {"asset_class": row[1], "sector": row[2]}
    except Exception as e:
        print(f"Error executing JOIN in build_constraints: {e}")

    # Fallback to provided dataframe if DB fetch fails or tickers haven't been inserted into prices yet
    for t in tickers:
        if t not in meta_dict:
            meta_dict[t] = {
                "asset_class": meta_fallback.loc[t, "asset_class"] if t in meta_fallback.index else "equity",
                "sector": meta_fallback.loc[t, "sector"] if t in meta_fallback.index else "Unknown"
            }

    equity_idx = [i for i, t in enumerate(tickers) if meta_dict[t]["asset_class"] == "equity"]
    debt_idx   = [i for i, t in enumerate(tickers) if meta_dict[t]["asset_class"] == "debt"]
    alt_idx    = [i for i, t in enumerate(tickers) if meta_dict[t]["asset_class"] == "alt"]

    e_lb, e_ub = risk_profile["equity_bounds"]
    d_lb, d_ub = risk_profile["debt_bounds"]
    a_lb, a_ub = risk_profile["alt_bounds"]

    constraints = [
        {"type": "eq", "fun": lambda w: np.sum(w) - 1},
    ]

    if equity_idx:
        constraints += [
            {"type": "ineq", "fun": lambda w, idx=equity_idx: sum(w[i] for i in idx) - e_lb},
            {"type": "ineq", "fun": lambda w, idx=equity_idx: e_ub - sum(w[i] for i in idx)},
        ]
    if debt_idx:
        constraints += [
            {"type": "ineq", "fun": lambda w, idx=debt_idx: sum(w[i] for i in idx) - d_lb},
            {"type": "ineq", "fun": lambda w, idx=debt_idx: d_ub - sum(w[i] for i in idx)},
        ]
    if alt_idx:
        constraints += [
            {"type": "ineq", "fun": lambda w, idx=alt_idx: sum(w[i] for i in idx) - a_lb},
            {"type": "ineq", "fun": lambda w, idx=alt_idx: a_ub - sum(w[i] for i in idx)},
        ]

    # Sector constraints: max 25% per sector
    sectors = {info["sector"] for info in meta_dict.values()}
    for sector in sectors:
        s_idx = [i for i, t in enumerate(tickers) if meta_dict[t]["sector"] == sector]
        if len(s_idx) > 1:
            constraints.append(
                {"type": "ineq", "fun": lambda w, idx=s_idx: 0.25 - sum(w[i] for i in idx)}
            )

    # Per-asset max 15%
    bounds = tuple((0.0, 0.15) for _ in range(n))

    return constraints, bounds


def _initial_weights(n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    w = rng.dirichlet(np.ones(n))
    return w


def monte_carlo_portfolios(
    mean_returns: pd.Series,
    cov_matrix: pd.DataFrame,
    risk_profile: dict,
    meta: pd.DataFrame,
    n: int = 10000,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate n random feasible portfolios via Monte Carlo.
    Returns DataFrame with columns: [ticker weights..., return, volatility, sharpe]
    """
    rng = np.random.default_rng(seed)
    tickers = list(mean_returns.index)
    n_assets = len(tickers)
    mu = mean_returns.values
    sigma = cov_matrix.values

    e_lb, e_ub = risk_profile["equity_bounds"]
    d_lb, d_ub = risk_profile["debt_bounds"]
    a_lb, a_ub = risk_profile["alt_bounds"]

    equity_idx = [i for i, t in enumerate(tickers) if meta.loc[t, "asset_class"] == "equity"]
    debt_idx   = [i for i, t in enumerate(tickers) if meta.loc[t, "asset_class"] == "debt"]
    alt_idx    = [i for i, t in enumerate(tickers) if meta.loc[t, "asset_class"] == "alt"]

    results = []
    attempts = 0
    max_attempts = n * 20

    while len(results) < n and attempts < max_attempts:
        attempts += 1
        w = rng.dirichlet(np.ones(n_assets) * 0.5)

        # Clip to 15% per asset and renormalize
        w = np.clip(w, 0, 0.15)
        if w.sum() == 0:
            continue
        w = w / w.sum()

        # Check asset class bounds
        eq_w = sum(w[i] for i in equity_idx)
        dt_w = sum(w[i] for i in debt_idx)
        al_w = sum(w[i] for i in alt_idx)

        if equity_idx and not (e_lb - 0.05 <= eq_w <= e_ub + 0.05):
            continue
        if debt_idx and not (d_lb - 0.05 <= dt_w <= d_ub + 0.05):
            continue
        if alt_idx and not (a_lb - 0.05 <= al_w <= a_ub + 0.05):
            continue

        ret, vol, sharpe = _portfolio_stats(w, mu, sigma, meta)
        row = dict(zip(tickers, w))
        row["return"] = ret
        row["volatility"] = vol
        row["sharpe"] = sharpe
        results.append(row)

    df = pd.DataFrame(results)
    if df.empty: return df
    
    # Identify Top 3 distinct strategies
    df["strategy"] = "Random"
    
    # 1. Balanced (Min Risk)
    idx_balanced = df["volatility"].idxmin()
    df.loc[idx_balanced, "strategy"] = "Balanced (Min Risk)"
    
    # 2. Optimal (Max Sharpe)
    sharpe_sorted = df.sort_values("sharpe", ascending=False)
    idx_optimal = sharpe_sorted.index[0]
    # If it overlaps with Balanced, pick the next best Sharpe that is at least slightly different
    i = 0
    while idx_optimal == idx_balanced and i < len(sharpe_sorted) - 1:
        i += 1
        idx_optimal = sharpe_sorted.index[i]
    df.loc[idx_optimal, "strategy"] = "Optimal (Max Sharpe)"
    
    # 3. Growth (Higher Return)
    ret_sorted = df.sort_values("return", ascending=False)
    idx_growth = ret_sorted.index[0]
    i = 0
    while (idx_growth == idx_balanced or idx_growth == idx_optimal) and i < len(ret_sorted) - 1:
        i += 1
        idx_growth = ret_sorted.index[i]
    df.loc[idx_growth, "strategy"] = "Growth (Higher Return)"
    
    return df
    
    return df


def efficient_frontier(
    mean_returns: pd.Series,
    cov_matrix: pd.DataFrame,
    risk_profile: dict,
    meta: pd.DataFrame,
    n_points: int = 20,
) -> pd.DataFrame:
    """
    Solve for minimum variance at n_points evenly spaced target returns.
    Returns DataFrame with frontier portfolios.
    """
    tickers = list(mean_returns.index)
    n_assets = len(tickers)
    mu = mean_returns.values
    sigma = cov_matrix.values

    constraints, bounds = build_constraints(risk_profile, meta)

    annual_returns_range = mu * 12
    r_min = max(annual_returns_range.min(), RISK_FREE_RATE_ANNUAL * 0.5)
    r_max = annual_returns_range.max() * 0.95
    target_returns = np.linspace(r_min, r_max, n_points)

    frontier_rows = []
    w0 = np.array([1 / n_assets] * n_assets)

    for target_r in target_returns:
        cons = constraints + [
            {"type": "eq", "fun": lambda w, tr=target_r: np.dot(w, mu) * 12 - tr}
        ]
        result = minimize(
            lambda w: w @ sigma @ w,
            w0,
            method="SLSQP",
            bounds=bounds,
            constraints=cons,
            options={"maxiter": 500, "ftol": 1e-9},
        )
        if result.success:
            w = result.x
            ret, vol, sharpe = _portfolio_stats(w, mu, sigma, meta)
            row = dict(zip(tickers, w))
            row["return"] = ret
            row["volatility"] = vol
            row["sharpe"] = sharpe
            frontier_rows.append(row)
            w0 = w  # warm-start next iteration

    return pd.DataFrame(frontier_rows)


def find_max_sharpe(portfolios_df: pd.DataFrame) -> pd.Series:
    """Return portfolio row with maximum Sharpe ratio."""
    return portfolios_df.loc[portfolios_df["sharpe"].idxmax()]


def find_gmvp(portfolios_df: pd.DataFrame) -> pd.Series:
    """Return Global Minimum Variance Portfolio."""
    return portfolios_df.loc[portfolios_df["volatility"].idxmin()]


def optimize_max_sharpe(
    mean_returns: pd.Series,
    cov_matrix: pd.DataFrame,
    risk_profile: dict,
    meta: pd.DataFrame,
) -> pd.Series:
    """
    Direct scipy optimization for max Sharpe portfolio.
    Falls back to MC result if optimization fails.
    """
    tickers = list(mean_returns.index)
    n_assets = len(tickers)
    mu = mean_returns.values
    sigma = cov_matrix.values

    constraints, bounds = build_constraints(risk_profile, meta)

    def neg_sharpe(w):
        ret = np.dot(w, mu) * 12
        vol = np.sqrt(w @ sigma @ w) * np.sqrt(12)
        return -(ret - RISK_FREE_RATE_ANNUAL) / (vol + 1e-10)

    best = None
    best_sharpe = -np.inf
    for seed in range(5):
        w0 = _initial_weights(n_assets, seed)
        w0 = w0 / w0.sum()
        res = minimize(neg_sharpe, w0, method="SLSQP", bounds=bounds, constraints=constraints,
                       options={"maxiter": 1000, "ftol": 1e-10})
        if res.success:
            ret, vol, sharpe = _portfolio_stats(res.x, mu, sigma, meta)
            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best = res.x

    if best is None:
        return None

    ret, vol, sharpe = _portfolio_stats(best, mu, sigma, meta)
    row = dict(zip(tickers, best))
    row["return"] = ret
    row["volatility"] = vol
    row["sharpe"] = sharpe
    return pd.Series(row)


def equal_weight_portfolio(mean_returns: pd.Series, cov_matrix: pd.DataFrame) -> pd.Series:
    """Naive equal-weight portfolio stats."""
    n = len(mean_returns)
    w = np.array([1 / n] * n)
    mu = mean_returns.values
    sigma = cov_matrix.values
    ret, vol, sharpe = _portfolio_stats(w, mu, sigma)
    row = dict(zip(mean_returns.index, w))
    row["return"] = ret
    row["volatility"] = vol
    row["sharpe"] = sharpe
    return pd.Series(row)


def compute_asset_individual_stats(mean_returns: pd.Series, cov_matrix: pd.DataFrame) -> pd.DataFrame:
    """Per-asset statistics for individual scatter points on the frontier plot."""
    rows = []
    for t in mean_returns.index:
        std = np.sqrt(cov_matrix.loc[t, t]) * np.sqrt(12)
        ret = mean_returns[t] * 12
        sharpe = (ret - RISK_FREE_RATE_ANNUAL) / std if std > 0 else 0
        rows.append({"ticker": t, "return": ret, "volatility": std, "sharpe": sharpe})
    return pd.DataFrame(rows).set_index("ticker")


def diversification_score(weights: np.ndarray, cov_matrix: np.ndarray) -> float:
    """
    Effective number of assets (inverse HHI).
    Ranges from 1 (fully concentrated) to n (perfectly diversified).
    Saturation Metric: We consider a portfolio with 10+ effective assets to be 'perfectly' diversified (10/10).
    """
    hhi = np.sum(weights**2)
    effective_n = 1 / hhi if hhi > 0 else 1
    # Saturation at 10 assets
    score = (effective_n / 10.0) * 10.0
    return round(min(10.0, score), 2)


def get_diversification_advantages(weights: np.ndarray, cov_matrix: np.ndarray, asset_stats: pd.DataFrame, meta: pd.DataFrame):
    """
    Returns a dictionary of benefits and benchmark comparisons.
    """
    # 1. Volatility Reduction (Correlation Benefit)
    # The difference between the weighted average volatility and the actual portfolio volatility.
    tickers = asset_stats.index.tolist()
    # We need to make sure weights align with asset_stats index
    # (Assuming they already do from the calling context in app.py)
    individual_vols = asset_stats["volatility"].values
    weighted_avg_vol = np.dot(weights, individual_vols)
    
    # Portfolio vol calculation (annualized)
    portfolio_vol = np.sqrt(weights @ cov_matrix @ weights) * np.sqrt(12)
    
    # Risk reduction is the 'free lunch'
    risk_reduction = max(0, weighted_avg_vol - portfolio_vol)
    
    # 2. Benchmarks
    # Market Proxy (Nifty 50 or highest return equity)
    market_ticker = "^NSEI" if "^NSEI" in asset_stats.index else None
    if not market_ticker:
        # Fallback to asset with highest return if no Nifty
        market_ticker = asset_stats["return"].idxmax()
    market_stats = asset_stats.loc[market_ticker]
    
    # Debt Proxy (SBI FD or lowest volatility)
    debt_ticker = "SBI_FD" if "SBI_FD" in asset_stats.index else None
    if not debt_ticker:
        debt_ticker = asset_stats["volatility"].idxmin()
    debt_stats = asset_stats.loc[debt_ticker]
    
    # Adjust benchmarks for tax and inflation (Real Returns)
    # Market (Equity): LTCG ~12.5%
    m_cat = meta.loc[market_ticker, "asset_class"] if market_ticker in meta.index else "equity"
    m_tax = 0.125 if m_cat == "equity" else 0.30
    m_adj_ret = (market_stats["return"] * (1 - m_tax)) - INFLATION_RATE
    m_sharpe = (m_adj_ret - RISK_FREE_RATE_ANNUAL) / market_stats["volatility"] if market_stats["volatility"] > 0 else 0
    
    # Debt: Flat ~30%
    d_cat = meta.loc[debt_ticker, "asset_class"] if debt_ticker in meta.index else "debt"
    d_tax = 0.30 if d_cat == "debt" else 0.125
    d_adj_ret = (debt_stats["return"] * (1 - d_tax)) - INFLATION_RATE
    d_sharpe = (d_adj_ret - RISK_FREE_RATE_ANNUAL) / debt_stats["volatility"] if debt_stats["volatility"] > 0 else 0

    # Update stats for return to real returns
    m_stats_adj = market_stats.copy()
    m_stats_adj["return"] = m_adj_ret
    m_stats_adj["sharpe"] = m_sharpe

    d_stats_adj = debt_stats.copy()
    d_stats_adj["return"] = d_adj_ret
    d_stats_adj["sharpe"] = d_sharpe
    
    return {
        "risk_reduction": risk_reduction,
        "weighted_avg_vol": weighted_avg_vol,
        "portfolio_vol": portfolio_vol,
        "market": m_stats_adj,
        "debt": d_stats_adj,
        "market_label": market_ticker,
        "debt_label": debt_ticker
    }
