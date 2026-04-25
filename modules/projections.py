"""
projections.py
SIP future value calculations and Monte Carlo future portfolio path simulations.
"""

import numpy as np
import pandas as pd


def sip_future_value(
    initial: float,
    monthly_sip: float,
    annual_return: float,
    years: int,
) -> dict:
    """
    Calculate portfolio growth over time with lump-sum + SIP.
    Returns yearly breakdown dict.
    """
    r = annual_return / 12  # monthly rate
    n_months = years * 12

    year_data = []
    portfolio_value = initial
    total_invested = initial

    for month in range(1, n_months + 1):
        portfolio_value = portfolio_value * (1 + r) + monthly_sip
        total_invested += monthly_sip

        if month % 12 == 0:
            year = month // 12
            gains = portfolio_value - total_invested
            year_data.append({
                "year": year,
                "portfolio_value": portfolio_value,
                "total_invested": total_invested,
                "total_gains": gains,
                "return_pct": (gains / total_invested * 100) if total_invested > 0 else 0,
            })

    return {
        "final_value": portfolio_value,
        "total_invested": total_invested,
        "total_gains": portfolio_value - total_invested,
        "year_by_year": pd.DataFrame(year_data),
        "cagr": annual_return,
    }


def monte_carlo_future_paths(
    initial: float,
    monthly_sip: float,
    annual_return: float,
    annual_vol: float,
    years: int,
    n_paths: int = 1000,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Simulate n_paths portfolio growth paths using monthly returns.
    Returns (summary_df, worst_case_drawdown_value)
    """
    rng = np.random.default_rng(seed)
    monthly_mean = annual_return / 12
    monthly_std = annual_vol / np.sqrt(12)
    n_months = years * 12

    # Shape: (n_paths, n_months)
    monthly_returns = rng.normal(monthly_mean, monthly_std, size=(n_paths, n_months))

    paths = np.zeros((n_paths, n_months + 1))
    paths[:, 0] = initial

    for m in range(n_months):
        paths[:, m + 1] = paths[:, m] * (1 + monthly_returns[:, m]) + monthly_sip

    # Extract yearly snapshots
    year_indices = list(range(0, n_months + 1, 12))
    yearly_paths = paths[:, year_indices]  # shape: (n_paths, years+1)

    rows = []
    for i, y in enumerate(range(years + 1)):
        vals = yearly_paths[:, i]
        rows.append({
            "year": y,
            "p5": np.percentile(vals, 5),
            "p25": np.percentile(vals, 25),
            "p50": np.percentile(vals, 50),
            "p75": np.percentile(vals, 75),
            "p95": np.percentile(vals, 95),
        })

    # Worst-case drawdown across all paths
    all_drawdowns = []
    for i in range(n_paths):
        path = paths[i, :]
        peak = np.maximum.accumulate(path + 1) # add 1 to avoid div by zero if balance is zero
        drawdown = (peak - path) / peak
        all_drawdowns.append(np.max(drawdown))
    worst_case = np.percentile(all_drawdowns, 95) 

    return pd.DataFrame(rows), worst_case


def goal_sip_required(
    target: float,
    initial: float,
    annual_return: float,
    years: int,
) -> float:
    """
    Back-calculate required monthly SIP to reach target corpus.
    Uses standard FV formula: FV = initial*(1+r)^n + SIP*(((1+r)^n-1)/r)
    """
    r = annual_return / 12
    n = years * 12
    growth_factor = (1 + r) ** n
    lump_sum_fv = initial * growth_factor
    remaining = target - lump_sum_fv

    if remaining <= 0:
        return 0.0

    sip_annuity_factor = (growth_factor - 1) / r if r > 0 else n
    return max(0.0, remaining / sip_annuity_factor)


def goal_achievement_probability(
    paths_df: pd.DataFrame,
    target: float,
    years: int,
) -> float:
    """
    Estimate probability of achieving target by given year.
    Uses the p-value distribution embedded in paths_df p50 column as reference.
    Approximates using the p5/p50/p95 percentile spread.
    """
    row = paths_df[paths_df["year"] == years]
    if row.empty:
        return 0.0

    p5  = float(row["p5"].values[0])
    p50 = float(row["p50"].values[0])
    p95 = float(row["p95"].values[0])

    # Linear interpolation in log-normal space
    if target <= p5:
        return 0.97
    elif target >= p95:
        return 0.03
    elif target <= p50:
        # Between p5 and p50: probability between 50%–95%
        frac = (p50 - target) / (p50 - p5 + 1e-9)
        return 0.50 + frac * 0.45
    else:
        # Between p50 and p95: probability between 5%–50%
        frac = (p95 - target) / (p95 - p50 + 1e-9)
        return 0.05 + frac * 0.45
