"""
risk_profiler.py
Calculates risk tolerance and allocation bounds based on age, horizon, and preferences.
"""

RISK_PROFILES = {
    "Aggressive": {
        "equity_bounds": (0.60, 0.90),
        "debt_bounds": (0.05, 0.30),
        "alt_bounds": (0.05, 0.20),
        "color": "#EF4444",
        "badge": "🔴",
        "description": "You seek maximum growth and can weather significant market swings.",
        "suitable_for": "Long horizon (10+ yrs), high income stability",
    },
    "Moderate": {
        "equity_bounds": (0.40, 0.70),
        "debt_bounds": (0.20, 0.50),
        "alt_bounds": (0.05, 0.15),
        "color": "#F59E0B",
        "badge": "🟡",
        "description": "You balance growth with stability, accepting moderate fluctuations.",
        "suitable_for": "Medium horizon (5–10 yrs), steady income",
    },
    "Conservative": {
        "equity_bounds": (0.10, 0.40),
        "debt_bounds": (0.50, 0.80),
        "alt_bounds": (0.05, 0.15),
        "color": "#3B82F6",
        "badge": "🔵",
        "description": "You prioritize capital preservation with modest growth expectations.",
        "suitable_for": "Short–medium horizon (3–7 yrs), near retirement",
    },
    "Very Conservative": {
        "equity_bounds": (0.05, 0.20),
        "debt_bounds": (0.70, 0.90),
        "alt_bounds": (0.00, 0.10),
        "color": "#6B7280",
        "badge": "⚪",
        "description": "Safety and income are your top priorities.",
        "suitable_for": "Retired or near-retirement, capital protection focused",
    },
}

PROFILES_LIST = ["Very Conservative", "Conservative", "Moderate", "Aggressive"]


def get_base_profile_name(age: int, horizon: int) -> str:
    """Determine base risk profile from age and investment horizon."""
    if age < 30:
        idx = 3  # Aggressive
    elif age < 45:
        idx = 2  # Moderate
    elif age < 60:
        idx = 1  # Conservative
    else:
        idx = 0  # Very Conservative

    # Horizon nudge
    if horizon >= 15:
        idx = min(idx + 1, 3)
    elif horizon <= 3:
        idx = max(idx - 1, 0)

    return PROFILES_LIST[idx]


def calculate_risk_profile(age: int, horizon: int, override: str = None) -> dict:
    """Return full risk profile dict. Override accepted from user."""
    name = override if override in PROFILES_LIST else get_base_profile_name(age, horizon)
    profile = RISK_PROFILES[name].copy()
    profile["name"] = name
    return profile


def adjust_for_preferences(base_profile: dict, preferences: dict) -> dict:
    """Fine-tune allocation bounds based on investment preference answers."""
    profile = base_profile.copy()
    e_lb, e_ub = profile["equity_bounds"]
    d_lb, d_ub = profile["debt_bounds"]
    a_lb, a_ub = profile["alt_bounds"]

    if not preferences.get("willing_bonds", True):
        d_lb = max(0.0, d_lb - 0.10)
        d_ub = max(0.05, d_ub - 0.15)
        e_lb = min(0.95, e_lb + 0.05)
        e_ub = min(0.95, e_ub + 0.10)

    if not preferences.get("interested_commodities", True) and not preferences.get("open_reits", True):
        a_lb, a_ub = 0.0, 0.05

    profile["equity_bounds"] = (round(e_lb, 2), round(e_ub, 2))
    profile["debt_bounds"] = (round(d_lb, 2), round(d_ub, 2))
    profile["alt_bounds"] = (round(a_lb, 2), round(a_ub, 2))
    return profile


def get_risk_score(profile_name: str) -> int:
    """Returns a 0–100 risk score for display."""
    return {"Aggressive": 80, "Moderate": 55, "Conservative": 30, "Very Conservative": 10}.get(profile_name, 50)


# Simplified tax slabs (FY 2024–25, post-budget)
TAX_SLABS = {
    "equity_stcg": 0.20,   # <1 yr holding
    "equity_ltcg": 0.125,  # >1 yr, gains above ₹1.25L exempt (updated Budget 2024)
    "equity_ltcg_exemption": 125000,
    "debt_flat": 0.30,     # Added to income; use 30% slab as default
}


def estimate_post_tax_return(annual_return: float, category: str, holding_years: float = 1.5) -> float:
    """Return post-tax annual return estimate. Assumes >1 yr by default."""
    if category in ("equity",):
        if holding_years < 1:
            effective_tax = TAX_SLABS["equity_stcg"]
        else:
            effective_tax = TAX_SLABS["equity_ltcg"] * 0.7  # blended (exemption applies)
    else:
        effective_tax = TAX_SLABS["debt_flat"]
    return annual_return * (1 - effective_tax)
