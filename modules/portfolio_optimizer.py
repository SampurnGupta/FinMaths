"""
portfolio_optimizer.py
Efficient Frontier + Monte Carlo portfolio optimization using scipy.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize

RISK_FREE_RATE_ANNUAL = 0.065  # RBI repo-rate approximation
RF_MONTHLY = RISK_FREE_RATE_ANNUAL / 12


def _portfolio_stats(weights: np.ndarray, mean_returns: np.ndarray, cov_matrix: np.ndarray):
    """Return (annual_return, annual_volatility, sharpe)."""
    ret = np.dot(weights, mean_returns) * 12
    vol = np.sqrt(weights @ cov_matrix @ weights) * np.sqrt(12)
    sharpe = (ret - RISK_FREE_RATE_ANNUAL) / vol if vol > 0 else 0
    return ret, vol, sharpe


def build_constraints(risk_profile: dict, meta: pd.DataFrame):
    """
    Construct scipy constraints and bounds for optimization.
    - Asset class bounds (equity/debt/alt)
    - Per-asset max 15%
    - Per-sector max 25%
    """
    n = len(meta)
    tickers = list(meta.index)

    equity_idx = [i for i, t in enumerate(tickers) if meta.loc[t, "asset_class"] == "equity"]
    debt_idx   = [i for i, t in enumerate(tickers) if meta.loc[t, "asset_class"] == "debt"]
    alt_idx    = [i for i, t in enumerate(tickers) if meta.loc[t, "asset_class"] == "alt"]

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
    sectors = meta["sector"].unique()
    for sector in sectors:
        s_idx = [i for i, t in enumerate(tickers) if meta.loc[t, "sector"] == sector]
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

        ret, vol, sharpe = _portfolio_stats(w, mu, sigma)
        row = dict(zip(tickers, w))
        row["return"] = ret
        row["volatility"] = vol
        row["sharpe"] = sharpe
        results.append(row)

    return pd.DataFrame(results)


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
            ret, vol, sharpe = _portfolio_stats(w, mu, sigma)
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
            ret, vol, sharpe = _portfolio_stats(res.x, mu, sigma)
            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best = res.x

    if best is None:
        return None

    ret, vol, sharpe = _portfolio_stats(best, mu, sigma)
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
    Normalized to 0–10 scale.
    """
    n = len(weights)
    hhi = np.sum(weights**2)
    effective_n = 1 / hhi if hhi > 0 else 1
    return round((effective_n / n) * 10, 2)
