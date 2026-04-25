"""
app.py — Efficient Frontier Portfolio Optimizer
Main Streamlit entrypoint. Manages session state and screen routing.
"""

import io
import time
import numpy as np
import pandas as pd
import streamlit as st
import os
from dotenv import load_dotenv

load_dotenv() # Load environment variables from .env

from modules.risk_profiler import (calculate_risk_profile, adjust_for_preferences,
                                    get_risk_score, estimate_post_tax_return, TAX_SLABS)
from modules.data_fetcher import build_combined_returns, get_asset_universe
from modules.portfolio_optimizer import (monte_carlo_portfolios, efficient_frontier,
                                          optimize_max_sharpe, equal_weight_portfolio,
                                          compute_asset_individual_stats, diversification_score,
                                          RISK_FREE_RATE_ANNUAL)
from modules.projections import (sip_future_value, monte_carlo_future_paths,
                                  goal_sip_required, goal_achievement_probability)
from modules.visualizations import (plot_efficient_frontier, plot_correlation_heatmap,
                                     plot_allocation_pie, plot_sector_bar, plot_asset_class_donut,
                                     plot_sip_projection, plot_risk_contribution, plot_comparison_bars)
from modules.llm_engine import (get_llm_explanation, explain_chart, explain_tax_logic, 
                                 explain_monte_carlo, get_portfolio_summary, get_final_recommendation,
                                 get_chat_response)

st.set_page_config(page_title="Portfolio Optimizer", page_icon="📈", layout="wide")

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Outfit:wght@400;600;800&display=swap');

:root {
    --primary: #6366F1;
    --primary-glow: rgba(99, 102, 241, 0.5);
    --secondary: #10B981;
    --bg-deep: #0F172A;
    --bg-card: rgba(30, 41, 59, 0.7);
    --text-main: #F8FAFC;
    --text-muted: #94A3B8;
    --glass-border: rgba(255, 255, 255, 0.1);
}

/* Global Styles */
.main {
    background-color: var(--bg-deep);
    color: var(--text-main);
    font-family: 'Inter', sans-serif;
}

h1, h2, h3, .section-title {
    font-family: 'Outfit', sans-serif;
    letter-spacing: -0.02em;
}

/* Sidebar Styling */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1E293B 0%, #0F172A 100%);
    border-right: 1px solid var(--glass-border);
}

[data-testid="stSidebar"] .stMarkdown {
    color: var(--text-main);
}

/* Glassmorphism Cards */
.metric-card {
    background: var(--bg-card);
    backdrop-filter: blur(12px);
    border: 1px solid var(--glass-border);
    border-radius: 16px;
    padding: 24px;
    margin-bottom: 16px;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
    position: relative;
    overflow: hidden;
}

.metric-card:hover {
    transform: translateY(-4px);
    border-color: var(--primary-glow);
    box-shadow: 0 12px 24px rgba(0, 0, 0, 0.3), 0 0 15px var(--primary-glow);
}

.metric-card::before {
    content: "";
    position: absolute;
    top: 0; left: 0; width: 4px; height: 100%;
    background: var(--primary);
}

.metric-card h3 {
    margin: 0 0 8px;
    font-size: 14px;
    color: var(--text-muted);
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

.metric-card p {
    margin: 0;
    font-size: 32px;
    font-weight: 800;
    color: #FFFFFF;
    font-family: 'Outfit', sans-serif;
}

.metric-card small {
    color: var(--text-muted);
    font-size: 13px;
    display: block;
    margin-top: 4px;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #6366F1 0%, #4F46E5 100%);
    color: white;
    border: none;
    border-radius: 12px;
    padding: 12px 24px;
    font-weight: 600;
    font-size: 15px;
    transition: all 0.2s;
    width: 100%;
    box-shadow: 0 4px 12px rgba(99, 102, 241, 0.3);
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 16px rgba(99, 102, 241, 0.4);
    background: linear-gradient(135deg, #818CF8 0%, #6366F1 100%);
}

/* Section Title */
.section-title {
    font-size: 32px;
    font-weight: 800;
    color: #FFFFFF;
    margin: 32px 0 16px;
    background: linear-gradient(90deg, #FFFFFF 0%, #94A3B8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

/* Animations */
@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(20px); }
    to { opacity: 1; transform: translateY(0); }
}

.stApp .block-container {
    animation: fadeInUp 0.6s ease-out;
}

/* Progress Bar */
.stProgress > div > div > div > div {
    background: linear-gradient(90deg, #6366F1 0%, #10B981 100%);
}

/* Inputs & Sliders */
.stSlider [data-baseweb="slider"] {
    background: var(--bg-card);
}

/* Custom Alert Box */
.insight-card, .warning-card {
    border-radius: 14px;
    padding: 20px;
    margin-bottom: 16px;
    font-size: 15px;
    backdrop-filter: blur(8px);
    border: 1px solid rgba(255,255,255,0.05);
}

.insight-card {
    background: rgba(16, 185, 129, 0.1);
    border-left: 4px solid #10B981;
    color: #D1FAE5;
}

.warning-card {
    background: rgba(245, 158, 11, 0.1);
    border-left: 4px solid #F59E0B;
    color: #FEF3C7;
}

/* Hide Streamlit Branding */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

SCREENS = ["Profile", "Preferences", "Optimization", "Allocation",
           "Frontier", "Diversification", "Comparison", "Projections", "Insights & Export", "AI Concierge"]

def init_state():
    defaults = dict(screen=0, profile=None, preferences=None, returns=None,
                    meta=None, mc_portfolios=None, frontier=None, optimal=None,
                    gmvp=None, ew_portfolio=None, asset_stats=None,
                    mean_returns=None, cov_matrix=None, 
                    groq_key=os.getenv("GROQ_API_KEY", ""),
                    chat_history=[])
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

def sidebar():
    with st.sidebar:
        st.markdown("## 📈 Portfolio Optimizer")
        st.markdown("---")
        pct = (st.session_state.screen) / (len(SCREENS) - 1)
        st.progress(pct)
        for i, s in enumerate(SCREENS):
            active = i == st.session_state.screen
            icon = "●" if active else "○"
            color = "#6366F1" if active else "#94A3B8"
            st.markdown(f'<div style="color:{color}; font-weight:{"600" if active else "400"}; padding: 4px 0;">{icon} {s}</div>', unsafe_allow_html=True)
        st.markdown("---")
        if st.session_state.profile:
            p = st.session_state.profile
            st.markdown(f"**Risk Profile:** {p['badge']} {p['name']}")
            st.markdown(f"**Risk Score:** {get_risk_score(p['name'])}/100")
        
        st.markdown("---")
        if st.button("🔄 Start Over", width="stretch"):
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            st.rerun()

def card(title, value, subtitle="Post-Tax & Inflation Adjusted", color="#6366F1"):
    st.markdown(f'''
    <div class="metric-card" style="border-left-color:{color}">
        <h3>{title}</h3>
        <p>{value}</p>
        <small>{subtitle}</small>
    </div>
    ''', unsafe_allow_html=True)

def fmt_inr(v):
    if v >= 10_000_000:
        return f"₹{(v/10_000_000):.2f} Cr"
    elif v >= 100_000:
        return f"₹{(v/100_000):.2f} L"
    return f"₹{v:,.0f}"

# ── Screen 1: Profile ─────────────────────────────────────────────────────────
def screen_profile():
    st.markdown('<div class="section-title">🚀 Portfolio Intel</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="background: rgba(99, 102, 241, 0.1); border-radius: 12px; padding: 20px; border-left: 4px solid #6366F1; margin-bottom: 24px;">
        <h2 style="margin:0; font-size: 20px; color: #FFFFFF;">Welcome to the next generation of asset management.</h2>
        <p style="margin:8px 0 0; color: #94A3B8; font-size: 15px;">Tell us about your financial profile to build a custom, risk-optimized portfolio.</p>
    </div>
    """, unsafe_allow_html=True)
    
    c1, c2 = st.columns(2)
    with c1:
        age = st.slider("Your Age", 18, 80, 30)
        initial = st.number_input("Initial Investment (₹)", 0, 100_000_000, 100_000, step=10_000)
    with c2:
        horizon = st.select_slider("Investment Horizon", [1, 3, 5, 7, 10, 15, 20], value=10)
        monthly_sip = st.number_input("Monthly SIP (₹)", 0, 1_000_000, 5_000, step=1_000)
    target = st.number_input("Target Corpus (₹, optional)", 0, 1_000_000_000, 0, step=100_000)

    profile = calculate_risk_profile(age, horizon)
    st.info(f"**Calculated Profile:** {profile['badge']} {profile['name']} — {profile['description']}")

    override = st.selectbox("Override risk profile (optional)", ["Auto", "Aggressive", "Moderate", "Conservative", "Very Conservative"])
    if override != "Auto":
        profile = calculate_risk_profile(age, horizon, override=override)

    if st.button("Next →"):
        st.session_state.profile = profile
        st.session_state.user = dict(age=age, initial=initial, monthly_sip=monthly_sip, horizon=horizon, target=target)
        st.session_state.screen = 1
        st.rerun()

    with st.expander("🔍 View Supported Asset Universe"):
        universe = get_asset_universe({"interested_international": True, "interested_commodities": True, "willing_bonds": True, "open_reits": True})
        
        st.markdown("### 📊 Dataset Details")
        st.markdown("""
        - **Data Source**: [yfinance](https://finance.yahoo.com/) for live assets; Synthetic models for fixed income.
        - **Time Period**: 5 Years of monthly historical data.
        - **Frequency**: Monthly Log-Returns (to model compounding correctly).
        - **FX Model**: International assets (USD) are automatically converted to **INR** using historical exchange rates to model FX risk.
        """)
        
        st.markdown("### 🛠️ Synthetic Asset Assumptions")
        st.markdown("""
        - **Fixed Deposits**: Modeled as low-volatility assets (~0.5% annual std dev) with zero equity correlation.
        - **Bonds**: Modeled with interest rate sensitivity (duration risk) and negative correlation to equities.
        - **REITs**: Modeled with hybrid characteristics (equity-like volatility but sectoral real-estate constraints).
        """)

        rows = []
        for t, info in universe["live"].items():
            rows.append({"Asset": info["label"], "Category": info["category"].upper(), "Sector": info["sector"], "Currency": info.get("currency", "INR")})
        for k, info in universe["hardcoded"].items():
            rows.append({"Asset": info["label"], "Category": info["category"].upper(), "Sector": info["sector"], "Currency": "INR (Assumed)"})
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

# ── Screen 2: Preferences ─────────────────────────────────────────────────────
def screen_preferences():
    st.markdown('<div class="section-title">🎯 Asset Constraints</div>', unsafe_allow_html=True)
    sectors = st.multiselect("Preferred sectors (equity picks)", ["Technology", "Healthcare", "Finance", "Energy", "Consumer"], default=["Technology", "Finance"])
    c1, c2 = st.columns(2)
    with c1:
        willing_bonds  = st.toggle("Include Bonds / Fixed Income?", True)
        interested_intl = st.toggle("International markets (US)?", False)
    with c2:
        interested_comm = st.toggle("Commodities (Gold)?", True)
        open_reits      = st.toggle("REITs (Real Estate)?", False)

    prefs = dict(sectors=sectors, willing_bonds=willing_bonds,
                 interested_international=interested_intl,
                 interested_commodities=interested_comm, open_reits=open_reits)

    col1, col2 = st.columns([1, 5])
    with col1:
        if st.button("← Back"):
            st.session_state.screen = 0; st.rerun()
    with col2:
        if st.button("Run Optimization →"):
            st.session_state.profile = adjust_for_preferences(st.session_state.profile, prefs)
            st.session_state.preferences = prefs
            st.session_state.screen = 2
            st.rerun()

# ── Screen 3: Optimization (auto-runs) ───────────────────────────────────────
def screen_optimization():
    st.markdown('<div class="section-title">⚡ Computing Optimal Strategy</div>', unsafe_allow_html=True)
    st.markdown('<p style="color: #94A3B8; margin-bottom: 24px;">Our engine is simulating thousands of market scenarios to find your ideal risk-reward balance.</p>', unsafe_allow_html=True)
    
    if st.session_state.mc_portfolios is None:
        bar = st.progress(0, text="Starting…")
        def cb(msg, pct): bar.progress(pct, text=msg)

        universe = get_asset_universe(st.session_state.preferences)
        returns, meta = build_combined_returns(universe, progress_cb=cb)

        if returns.empty or len(returns.columns) < 2:
            st.error("Not enough data fetched. Check your internet connection and try again.")
            return

        cb("Computing statistics…", 0.6)
        mean_ret = returns.mean()
        cov      = returns.cov()
        profile  = st.session_state.profile

        cb("Running Monte Carlo (10,000 portfolios)…", 0.7)
        mc = monte_carlo_portfolios(mean_ret, cov, profile, meta, n=10_000)

        cb("Solving efficient frontier…", 0.85)
        ef = efficient_frontier(mean_ret, cov, profile, meta, n_points=20)

        from modules.portfolio_optimizer import equal_weight_portfolio, compute_asset_individual_stats, find_gmvp
        gmvp = find_gmvp(mc) if not mc.empty else None
        ew   = equal_weight_portfolio(mean_ret, cov)
        asset_stats = compute_asset_individual_stats(mean_ret, cov)

        st.session_state.update(dict(returns=returns, meta=meta, mean_returns=mean_ret,
                                      cov_matrix=cov, mc_portfolios=mc, frontier=ef,
                                      gmvp=gmvp, ew_portfolio=ew, asset_stats=asset_stats))
        bar.progress(1.0, text="Done!")
        st.rerun()

    mc = st.session_state.mc_portfolios
    st.success("Simulations complete! We've identified 3 distinct strategies for you.")
    
    # Selection UI
    top_3 = mc[mc["strategy"] != "Random"].copy()
    
    # Sort for consistent display: Balanced -> Optimal -> Growth
    order = {"Balanced (Min Risk)": 0, "Optimal (Max Sharpe)": 1, "Growth (Higher Return)": 2}
    top_3["order"] = top_3["strategy"].map(order)
    top_3 = top_3.sort_values("order")

    cols = st.columns(3)
    for i, (idx, row) in enumerate(top_3.iterrows()):
        with cols[i]:
            st.markdown(f"### {row['strategy']}")
            st.metric("Expected Return", f"{row['return']:.1%}")
            st.metric("Volatility", f"{row['volatility']:.1%}")
            st.metric("Sharpe Ratio", f"{row['sharpe']:.2f}")
            if st.button(f"Choose {row['strategy'].split(' ')[0]}", key=f"btn_{i}"):
                # If Balanced (Min Risk), use GMVP explicitly
                if "Balanced" in row["strategy"]:
                    st.session_state.optimal = st.session_state.gmvp
                else:
                    st.session_state.optimal = row
                st.session_state.screen = 3
                st.rerun()

    # LLM insight about Monte Carlo
    if st.session_state.groq_key:
        st.markdown("---")
        if st.button("🤖 Explain why we use Monte Carlo?"):
            with st.spinner("..."):
                explanation = explain_monte_carlo(10000, mc["sharpe"].max(), st.session_state.groq_key)
                st.info(explanation)

# ── Screen 4: Allocation ──────────────────────────────────────────────────────
def screen_allocation():
    st.markdown('<div class="section-title">💎 Recommended Allocation</div>', unsafe_allow_html=True)
    opt  = st.session_state.optimal
    meta = st.session_state.meta
    user = st.session_state.user
    profile = st.session_state.profile

    tickers = [c for c in meta.index if c in opt.index]
    weights = opt[tickers]
    div_score = diversification_score(weights.values, st.session_state.cov_matrix.loc[tickers, tickers].values)

    c1, c2, c3, c4 = st.columns(4)
    with c1: card("Expected Annual Return", f"{opt['return']:.2%}", "Post-Tax & Inflation", "#6366F1")
    with c2: card("Annual Volatility", f"{opt['volatility']:.2%}", "1σ risk", "#EF4444")
    with c3: card("Sharpe Ratio", f"{opt['sharpe']:.2f}", f"RF = {RISK_FREE_RATE_ANNUAL:.1%}", "#10B981")
    with c4: card("Diversification", f"{div_score}/10", "Inv. HHI score", "#F59E0B")
    
    # Return Estimation Explanation
    from modules.portfolio_optimizer import get_return_explanation
    with st.expander("📝 How we estimate these returns? (Tax & Costs)"):
        st.markdown(get_return_explanation())

    # Post-tax estimate
    avg_cat = meta.loc[tickers].groupby("asset_class")["asset_class"].count()
    eq_wt = weights[meta.loc[tickers, "asset_class"] == "equity"].sum()
    blended_posttax = opt["return"] * (1 - 0.10 * eq_wt - 0.30 * (1 - eq_wt)) * 0.7
    st.caption(f"💡 Estimated post-tax return (LTCG blended): **{blended_posttax:.2%}** p.a.")

    col1, col2 = st.columns([1.1, 1])
    with col1:
        st.plotly_chart(plot_allocation_pie(weights, meta), use_container_width=True)
    with col2:
        # Table
        rows = []
        for t in tickers:
            if weights[t] < 0.001: continue
            amt = user["initial"] * weights[t]
            ret = st.session_state.mean_returns[t] * 12
            cat = meta.loc[t, "category"]
            post_tax = estimate_post_tax_return(ret, cat)
            rows.append({"Asset": meta.loc[t, "label"], "Allocation": f"{weights[t]:.1%}",
                         "Amount (₹)": f"₹{amt:,.0f}", "Exp. Return": f"{ret:.2%}",
                         "Post-Tax & Inf.": f"{post_tax:.2%}"})
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    if st.button("Next: Efficient Frontier →"):
        st.session_state.screen = 4; st.rerun()

# ── Screen 5: Efficient Frontier ─────────────────────────────────────────────
def screen_frontier():
    st.markdown('<div class="section-title">📈 Efficient Frontier Analysis</div>', unsafe_allow_html=True)
    fig = plot_efficient_frontier(
        st.session_state.mc_portfolios, st.session_state.frontier,
        st.session_state.optimal, st.session_state.gmvp,
        st.session_state.asset_stats, st.session_state.meta)
    st.plotly_chart(fig, use_container_width=True)
    
    if st.session_state.groq_key:
        if st.button("🤖 Explain the Efficient Frontier"):
            with st.spinner("..."):
                summary = f"Optimal Sharpe: {st.session_state.optimal['sharpe']:.2f}, Asset Count: {len(st.session_state.asset_stats)}"
                st.info(explain_chart("Efficient Frontier", summary, st.session_state.groq_key))
    st.caption("⭐ Green star = Max Sharpe (recommended) · 🔶 Diamond = Min Variance · Colored dots = 10,000 random portfolios")
    if st.button("Next: Diversification →"):
        st.session_state.screen = 5; st.rerun()

# ── Screen 6: Diversification ─────────────────────────────────────────────────
def screen_diversification():
    st.markdown('<div class="section-title">🔍 Diversification Metrics</div>', unsafe_allow_html=True)
    opt  = st.session_state.optimal
    meta = st.session_state.meta
    tickers = [c for c in meta.index if c in opt.index]
    weights = opt[tickers]

    c1, c2 = st.columns(2)
    with c1:
        corr_matrix = st.session_state.returns[tickers].corr()
        st.plotly_chart(plot_correlation_heatmap(corr_matrix, meta), use_container_width=True)
        if st.session_state.groq_key:
            if st.button("🤖 Explain Correlations"):
                with st.spinner("..."):
                    summary = f"Correlation matrix for {len(tickers)} assets."
                    st.info(explain_chart("Correlation Matrix", summary, st.session_state.groq_key))
    with c2:
        st.plotly_chart(plot_sector_bar(weights, meta), use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        st.plotly_chart(plot_asset_class_donut(weights, meta), width="stretch")
    with c4:
        st.plotly_chart(plot_risk_contribution(weights, st.session_state.cov_matrix.loc[tickers, tickers], meta), width="stretch")

    if st.button("Next: Comparison →"):
        st.session_state.screen = 6; st.rerun()

def stream_text(text: str, delay: float = 0.01):
    for word in text.split(" "):
        yield word + " "
        time.sleep(delay)


# ── Screen 10: Interactive AI Chat ────────────────────────────────────────────
def screen_chat():
    st.markdown('<div class="section-title">💬 AI Portfolio Concierge</div>', unsafe_allow_html=True)
    
    opt     = st.session_state.optimal
    profile = st.session_state.profile
    user    = st.session_state.user
    meta    = st.session_state.meta
    
    # Display mini metrics for context
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("Return (Real)", f"{opt['return']:.2%}")
    with c2: st.metric("Risk", f"{opt['volatility']:.2%}")
    with c3: st.metric("Sharpe", f"{opt['sharpe']:.2f}")
    with c4: st.metric("Horizon", f"{user['horizon']}Y")

    st.markdown("---")
    
    # Prepare Context String
    tickers = [c for c in meta.index if c in opt.index and opt[c] > 0.001]
    weights_str = ", ".join([f"{meta.loc[t, 'label']}: {opt[t]:.1%}" for t in tickers])
    context = f"""
    Risk Profile: {profile['name']}
    Investment Horizon: {user['horizon']} years
    Target Corpus: ₹{user.get('target', 0):,.0f}
    Initial Amount: ₹{user['initial']:,.0f} | Monthly SIP: ₹{user['monthly_sip']:,.0f}
    
    Optimized Portfolio Stats:
    - Expected Return: {opt['return']:.2%}
    - Volatility: {opt['volatility']:.2%}
    - Sharpe Ratio: {opt['sharpe']:.2f}
    
    Asset Allocations:
    {weights_str}
    """

    # Auto-seed initial message if empty
    if not st.session_state.chat_history and st.session_state.groq_key:
        with st.spinner("..."):
            initial_prompt = "Briefly explain the core benefits of this specific diversified portfolio for my profile."
            st.session_state.chat_history.append({"role": "user", "content": initial_prompt})
            response = get_chat_response(st.session_state.chat_history, context, st.session_state.groq_key)
            st.session_state.chat_history.append({"role": "assistant", "content": response})
        st.rerun()

    # Chat Interface
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ask me anything about your portfolio..."):
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            if not st.session_state.groq_key:
                response = "⚠️ Please set your Groq API Key in the .env file to use the AI Advisor."
            else:
                with st.spinner("..."):
                    response = get_chat_response(st.session_state.chat_history, context, st.session_state.groq_key)
                st.write_stream(stream_text(response))
            st.session_state.chat_history.append({"role": "assistant", "content": response})

    if st.button("⬅️ Back to Insights"):
        st.session_state.screen = 8; st.rerun()

# ── Screen 7: Comparison ──────────────────────────────────────────────────────
def screen_comparison():
    st.markdown('<div class="section-title">⚖️ Performance Benchmarking</div>', unsafe_allow_html=True)
    opt = st.session_state.optimal
    ew  = st.session_state.ew_portfolio
    gmvp = st.session_state.gmvp

    cmp = pd.DataFrame({
        "Recommended": {"Exp. Return (Real)": opt["return"], "Volatility": opt["volatility"], "Sharpe Ratio": opt["sharpe"]},
        "Equal Weight": {"Exp. Return (Real)": ew["return"],  "Volatility": ew["volatility"],  "Sharpe Ratio": ew["sharpe"]},
        "Min Variance":  {"Exp. Return (Real)": gmvp["return"], "Volatility": gmvp["volatility"], "Sharpe Ratio": gmvp["sharpe"]},
    })

    st.plotly_chart(plot_comparison_bars(cmp), width="stretch")

    # Rebalancing suggestions
    meta = st.session_state.meta
    opt_tickers = [t for t in meta.index if t in opt.index and opt[t] > 0.001]
    
    st.markdown("**Tax & Inflation Methodology:**")
    st.info("All returns shown are 'Real Returns' — net of simulated LTCG taxes, rebalancing costs, and an assumed inflation rate of 6.0% p.a.")
    
    st.markdown("**Detailed Tax Estimate (LTCG, >1 year holding):**")
    user = st.session_state.user
    for t in opt_tickers:
        alloc = opt[t]; amt = user["initial"] * alloc
        label = meta.loc[t, "label"]
        st.markdown(f"- **Buy** {label}: invest **₹{amt:,.0f}** ({alloc:.1%} of portfolio)")

    # Tax note
    st.markdown("**Tax Estimate (LTCG, >1 year holding):**")
    eq_wt = sum(opt[t] for t in opt_tickers if meta.loc[t, "asset_class"] == "equity")
    annual_gain = user["initial"] * opt["return"]
    eq_gain = annual_gain * eq_wt
    ltcg_tax = max(0, eq_gain - TAX_SLABS["equity_ltcg_exemption"]) * TAX_SLABS["equity_ltcg"]
    debt_tax  = annual_gain * (1 - eq_wt) * TAX_SLABS["debt_flat"]
    st.info(f"Estimated annual tax on equity gains: **₹{ltcg_tax:,.0f}** (LTCG 12.5% above ₹1.25L) | Debt/FD: **₹{debt_tax:,.0f}** (@ 30%)")

    if st.button("Next: Projections →"):
        st.session_state.screen = 7; st.rerun()

# ── Screen 8: Projections ─────────────────────────────────────────────────────
def screen_projections():
    st.markdown('<div class="section-title">🔮 Wealth Projections</div>', unsafe_allow_html=True)
    opt  = st.session_state.optimal
    user = st.session_state.user
    horizon = user["horizon"]
    initial = user["initial"]
    sip     = user["monthly_sip"]
    ann_ret = opt["return"]
    ann_vol = opt["volatility"]

    sip_data  = sip_future_value(initial, sip, ann_ret, horizon)
    paths_df, worst_case = monte_carlo_future_paths(initial, sip, ann_ret, ann_vol, horizon, n_paths=1000)

    st.plotly_chart(plot_sip_projection(paths_df, sip_data, initial), use_container_width=True)
    
    c1, c2, c3, c4 = st.columns(4)
    with c1: card("Median Final Value", fmt_inr(paths_df.iloc[-1]["p50"]), f"In {horizon} years", "#6366F1")
    with c2: card("Total Invested", fmt_inr(sip_data["total_invested"]), "Lump sum + SIP", "#94A3B8")
    with c3: card("Expected Gains", fmt_inr(sip_data["total_gains"]), "Tax-adjusted", "#10B981")
    with c4: card("Worst Case Drawdown", f"{worst_case:.1%}", "95% confidence", "#EF4444")

    if user.get("target", 0) > 0:
        target = user["target"]
        prob   = goal_achievement_probability(paths_df, target, horizon)
        req_sip = goal_sip_required(target, initial, ann_ret, horizon)
        st.success(f"🎯 **{prob:.0%} probability** of reaching ₹{target:,.0f} in {horizon} years · Required SIP if not met: ₹{req_sip:,.0f}/month")

    if st.button("Next: Insights & Export →"):
        st.session_state.screen = 8; st.rerun()

# ── Screen 9: Insights & Export ───────────────────────────────────────────────
def screen_insights():
    st.markdown('<div class="section-title">💡 Final Insights & Export</div>', unsafe_allow_html=True)
    opt  = st.session_state.optimal
    meta = st.session_state.meta
    cov  = st.session_state.cov_matrix
    user = st.session_state.user
    profile = st.session_state.profile

    tickers = [c for c in meta.index if c in opt.index and opt[c] > 0.001]
    weights = opt[tickers]

    # Auto-generated insights
    insights, warnings = [], []

    # Sector concentration check
    sector_w = meta.loc[tickers].copy(); sector_w["w"] = weights.values
    for sec, grp in sector_w.groupby("sector"):
        total = grp["w"].sum()
        if total > 0.25: warnings.append(f"⚠️ {sec} sector at {total:.0%} — above 25% cap")

    # Diversification
    div = diversification_score(weights.values, cov.loc[tickers, tickers].values)
    if div < 5:
        warnings.append("Diversification score is low. Consider adding uncorrelated assets.")
    else:
        insights.append(f"Diversification score: {div}/10 — well spread across {len(tickers)} assets.")

    # Gold / commodity hedge
    gold_tickers = [t for t in tickers if meta.loc[t, "category"] == "commodity"]
    if gold_tickers:
        insights.append("Gold is included — typically shows negative correlation with equities, acting as a portfolio hedge.")
    else:
        insights.append("Adding a small gold allocation (5–10%) can reduce portfolio volatility with minimal return sacrifice.")

    # Sharpe vs equal weight
    ew_sharpe = st.session_state.ew_portfolio["sharpe"]
    improvement = (opt["sharpe"] - ew_sharpe) / abs(ew_sharpe) * 100 if ew_sharpe != 0 else 0
    insights.append(f"Optimized Sharpe ratio ({opt['sharpe']:.2f}) is {improvement:+.1f}% vs equal-weight ({ew_sharpe:.2f}).")

    for ins in insights:
        st.markdown(f'<div class="insight-card">✅ {ins}</div>', unsafe_allow_html=True)
    for w in warnings:
        st.markdown(f'<div class="warning-card">{w}</div>', unsafe_allow_html=True)

    if st.session_state.groq_key:
        st.markdown("---")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("🤖 Generate Portfolio Summary"):
                with st.spinner("..."):
                    summary = get_portfolio_summary(weights.to_dict(), opt, st.session_state.groq_key)
                    st.info(summary)
        with c2:
            if st.button("🤖 Get Final Recommendation"):
                with st.spinner("..."):
                    rec = get_final_recommendation(profile['name'], user['horizon'], st.session_state.groq_key)
                    st.info(rec)

    st.markdown("---")

    # ── Exports ───────────────────────────────────────────────────────────────
    st.markdown("#### Download Reports")
    rows = []
    for t in tickers:
        amt = user["initial"] * opt[t]
        ret = st.session_state.mean_returns[t] * 12
        rows.append({"Asset": meta.loc[t, "label"], "Ticker": t,
                     "Allocation_%": round(opt[t] * 100, 2), "Amount_INR": round(amt, 2),
                     "Exp_Return_%": round(ret * 100, 2), "Sector": meta.loc[t, "sector"]})
    alloc_df = pd.DataFrame(rows)

    c1, c2, c3 = st.columns(3)
    with c1:
        csv = alloc_df.to_csv(index=False)
        st.download_button("⬇️ Download CSV", csv, "portfolio_allocation.csv", "text/csv")

    with c2:
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as writer:
            alloc_df.to_excel(writer, sheet_name="Allocation", index=False)
            if st.session_state.frontier is not None and not st.session_state.frontier.empty:
                st.session_state.frontier[["return", "volatility", "sharpe"]].to_excel(writer, sheet_name="Frontier", index=False)
        st.download_button("⬇️ Download Excel", buf.getvalue(), "portfolio_report.xlsx",
                           "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    with c3:
        try:
            from fpdf import FPDF
            from fpdf.enums import XPos, YPos
            pdf = FPDF(); pdf.add_page(); pdf.set_font("Helvetica", "B", 16)
            pdf.cell(0, 10, "Portfolio Optimization Report", new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")
            pdf.set_font("Helvetica", size=11)
            pdf.cell(0, 8, f"Risk Profile: {profile['name']}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.cell(0, 8, f"Expected Return: {opt['return']:.2%} | Volatility: {opt['volatility']:.2%} | Sharpe: {opt['sharpe']:.2f}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.ln(4); pdf.set_font("Helvetica", "B", 12); pdf.cell(0, 8, "Asset Allocation", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.set_font("Helvetica", size=10)
            for _, r in alloc_df.iterrows():
                # Replace Unicode characters that Helvetica doesn't support
                asset_label = r['Asset'].replace('—', '-').replace('₹', 'Rs.')
                pdf.cell(0, 7, f"  {asset_label}: {r['Allocation_%']:.2f}% - Rs.{r['Amount_INR']:,.0f}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            st.download_button("⬇️ Download PDF", bytes(pdf.output()), "portfolio_report.pdf", "application/pdf")
        except Exception as e:
            st.caption(f"PDF unavailable: {e}")

    st.markdown("---")
    if st.button("Final Step: Chat with AI Advisor →"):
        st.session_state.screen = 9; st.rerun()


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    init_state()
    sidebar()

    s = st.session_state.screen
    if   s == 0: screen_profile()
    elif s == 1: screen_preferences()
    elif s == 2: screen_optimization()
    elif s == 3: screen_allocation()
    elif s == 4: screen_frontier()
    elif s == 5: screen_diversification()
    elif s == 6: screen_comparison()
    elif s == 7: screen_projections()
    elif s == 8: screen_insights()
    elif s == 9: screen_chat()

if __name__ == "__main__":
    main()
