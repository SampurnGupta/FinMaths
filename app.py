"""
app.py — Efficient Frontier Portfolio Optimizer
Main Streamlit entrypoint. Manages session state and screen routing.
"""

import io
import numpy as np
import pandas as pd
import streamlit as st

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

st.set_page_config(page_title="Portfolio Optimizer", page_icon="📈", layout="wide")

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.block-container { padding-top: 1.5rem; }
section[data-testid="stSidebar"] { background: #1E2139; }
section[data-testid="stSidebar"] * { color: #E5E7EB !important; }
section[data-testid="stSidebar"] .stProgress > div > div { background: #4F46E5; }
.metric-card {
  background: #fff; border-radius: 12px; padding: 18px 22px;
  box-shadow: 0 1px 6px rgba(0,0,0,0.07); border-left: 4px solid #4F46E5;
  margin-bottom: 12px;
}
.metric-card h3 { margin: 0 0 4px; font-size: 13px; color: #6B7280; font-weight: 500; }
.metric-card p  { margin: 0; font-size: 26px; font-weight: 700; color: #111827; }
.metric-card small { color: #6B7280; font-size: 12px; }
.section-title { font-size: 20px; font-weight: 700; color: #111827; margin: 24px 0 12px; }
.insight-card {
  background: #F0FDF4; border-radius: 10px; padding: 14px 18px;
  border-left: 4px solid #10B981; margin-bottom: 10px; font-size: 14px; color: #065F46;
}
.warning-card {
  background: #FEF9C3; border-radius: 10px; padding: 14px 18px;
  border-left: 4px solid #F59E0B; margin-bottom: 10px; font-size: 14px; color: #92400E;
}
.stButton > button {
  background: #4F46E5; color: white; border: none; border-radius: 8px;
  padding: 10px 28px; font-weight: 600; font-size: 14px; transition: all 0.2s;
}
.stButton > button:hover { background: #4338CA; transform: translateY(-1px); }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

SCREENS = ["Profile", "Preferences", "Optimization", "Allocation",
           "Frontier", "Diversification", "Comparison", "Projections", "Insights & Export"]

def init_state():
    defaults = dict(screen=0, profile=None, preferences=None, returns=None,
                    meta=None, mc_portfolios=None, frontier=None, optimal=None,
                    gmvp=None, ew_portfolio=None, asset_stats=None,
                    mean_returns=None, cov_matrix=None)
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
            icon = "✅" if i < st.session_state.screen else ("▶" if i == st.session_state.screen else "○")
            st.markdown(f"{'**' if i == st.session_state.screen else ''}{icon} {s}{'**' if i == st.session_state.screen else ''}")
        st.markdown("---")
        if st.session_state.profile:
            p = st.session_state.profile
            st.markdown(f"**Risk Profile:** {p['badge']} {p['name']}")
            st.markdown(f"**Risk Score:** {get_risk_score(p['name'])}/100")
        if st.button("🔄 Start Over", use_container_width=True):
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            st.rerun()

def card(title, value, subtitle="", color="#4F46E5"):
    st.markdown(f'<div class="metric-card" style="border-left-color:{color}"><h3>{title}</h3><p>{value}</p><small>{subtitle}</small></div>', unsafe_allow_html=True)

def fmt_inr(v): return f"₹{v:,.0f}"

# ── Screen 1: Profile ─────────────────────────────────────────────────────────
def screen_profile():
    st.markdown('<div class="section-title">👤 Your Investment Profile</div>', unsafe_allow_html=True)
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

# ── Screen 2: Preferences ─────────────────────────────────────────────────────
def screen_preferences():
    st.markdown('<div class="section-title">🎯 Investment Preferences</div>', unsafe_allow_html=True)
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
    st.markdown('<div class="section-title">⚙️ Running Optimization…</div>', unsafe_allow_html=True)
    if st.session_state.mc_portfolios is not None:
        st.session_state.screen = 3; st.rerun(); return

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

    cb("Finding optimal portfolio…", 0.92)
    optimal = optimize_max_sharpe(mean_ret, cov, profile, meta)
    if optimal is None and not mc.empty:
        from modules.portfolio_optimizer import find_max_sharpe
        optimal = find_max_sharpe(mc)

    from modules.portfolio_optimizer import find_gmvp
    gmvp = find_gmvp(mc) if not mc.empty else None
    ew   = equal_weight_portfolio(mean_ret, cov)
    asset_stats = compute_asset_individual_stats(mean_ret, cov)

    st.session_state.update(dict(returns=returns, meta=meta, mean_returns=mean_ret,
                                  cov_matrix=cov, mc_portfolios=mc, frontier=ef,
                                  optimal=optimal, gmvp=gmvp, ew_portfolio=ew,
                                  asset_stats=asset_stats))
    bar.progress(1.0, text="Done!")
    st.session_state.screen = 3
    st.rerun()

# ── Screen 4: Allocation ──────────────────────────────────────────────────────
def screen_allocation():
    st.markdown('<div class="section-title">📊 Recommended Portfolio</div>', unsafe_allow_html=True)
    opt  = st.session_state.optimal
    meta = st.session_state.meta
    user = st.session_state.user
    profile = st.session_state.profile

    tickers = [c for c in meta.index if c in opt.index]
    weights = opt[tickers]
    div_score = diversification_score(weights.values, st.session_state.cov_matrix.loc[tickers, tickers].values)

    c1, c2, c3, c4 = st.columns(4)
    with c1: card("Expected Annual Return", f"{opt['return']:.2%}", "Pre-tax", "#4F46E5")
    with c2: card("Annual Volatility", f"{opt['volatility']:.2%}", "1σ risk", "#EF4444")
    with c3: card("Sharpe Ratio", f"{opt['sharpe']:.2f}", f"RF = {RISK_FREE_RATE_ANNUAL:.1%}", "#10B981")
    with c4: card("Diversification", f"{div_score}/10", "Inv. HHI score", "#F59E0B")

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
                         "Post-Tax": f"{post_tax:.2%}"})
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    if st.button("Next: Efficient Frontier →"):
        st.session_state.screen = 4; st.rerun()

# ── Screen 5: Efficient Frontier ─────────────────────────────────────────────
def screen_frontier():
    st.markdown('<div class="section-title">📉 Efficient Frontier</div>', unsafe_allow_html=True)
    fig = plot_efficient_frontier(
        st.session_state.mc_portfolios, st.session_state.frontier,
        st.session_state.optimal, st.session_state.gmvp,
        st.session_state.asset_stats, st.session_state.meta)
    st.plotly_chart(fig, use_container_width=True)
    st.caption("⭐ Green star = Max Sharpe (recommended) · 🔶 Diamond = Min Variance · Colored dots = 10,000 random portfolios")
    if st.button("Next: Diversification →"):
        st.session_state.screen = 5; st.rerun()

# ── Screen 6: Diversification ─────────────────────────────────────────────────
def screen_diversification():
    st.markdown('<div class="section-title">🔍 Diversification Analysis</div>', unsafe_allow_html=True)
    opt  = st.session_state.optimal
    meta = st.session_state.meta
    tickers = [c for c in meta.index if c in opt.index]
    weights = opt[tickers]

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(plot_correlation_heatmap(
            st.session_state.cov_matrix.loc[tickers, tickers] /
            np.outer(st.session_state.cov_matrix.loc[tickers, tickers].values.diagonal()**0.5,
                     st.session_state.cov_matrix.loc[tickers, tickers].values.diagonal()**0.5),
            meta), use_container_width=True)
    with c2:
        st.plotly_chart(plot_sector_bar(weights, meta), use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        st.plotly_chart(plot_asset_class_donut(weights, meta), use_container_width=True)
    with c4:
        st.plotly_chart(plot_risk_contribution(weights, st.session_state.cov_matrix.loc[tickers, tickers], meta), use_container_width=True)

    if st.button("Next: Comparison →"):
        st.session_state.screen = 6; st.rerun()

# ── Screen 7: Comparison ──────────────────────────────────────────────────────
def screen_comparison():
    st.markdown('<div class="section-title">⚖️ Portfolio Comparison</div>', unsafe_allow_html=True)
    opt = st.session_state.optimal
    ew  = st.session_state.ew_portfolio
    gmvp = st.session_state.gmvp

    cmp = pd.DataFrame({
        "Recommended": {"Expected Return": opt["return"], "Volatility": opt["volatility"], "Sharpe Ratio": opt["sharpe"]},
        "Equal Weight": {"Expected Return": ew["return"],  "Volatility": ew["volatility"],  "Sharpe Ratio": ew["sharpe"]},
        "Min Variance":  {"Expected Return": gmvp["return"], "Volatility": gmvp["volatility"], "Sharpe Ratio": gmvp["sharpe"]},
    })

    st.plotly_chart(plot_comparison_bars(cmp), use_container_width=True)

    # Rebalancing suggestions
    meta = st.session_state.meta
    opt_tickers = [c for c in meta.index if c in opt.index and opt[c] > 0.001]
    st.markdown("#### Rebalancing Suggestions")
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
    st.markdown('<div class="section-title">🔮 Future Projections</div>', unsafe_allow_html=True)
    opt  = st.session_state.optimal
    user = st.session_state.user
    horizon = user["horizon"]
    initial = user["initial"]
    sip     = user["monthly_sip"]
    ann_ret = opt["return"]
    ann_vol = opt["volatility"]

    sip_data  = sip_future_value(initial, sip, ann_ret, horizon)
    paths_df  = monte_carlo_future_paths(initial, sip, ann_ret, ann_vol, horizon, n_paths=1000)

    st.plotly_chart(plot_sip_projection(paths_df, sip_data, initial), use_container_width=True)

    c1, c2, c3 = st.columns(3)
    with c1: card("Median Final Value", fmt_inr(paths_df.iloc[-1]["p50"]), f"In {horizon} years", "#4F46E5")
    with c2: card("Total Invested", fmt_inr(sip_data["total_invested"]), "Lump sum + SIP", "#6B7280")
    with c3: card("Expected Gains", fmt_inr(sip_data["total_gains"]), "Pre-tax", "#10B981")

    if user.get("target", 0) > 0:
        target = user["target"]
        prob   = goal_achievement_probability(paths_df, target, horizon)
        req_sip = goal_sip_required(target, initial, ann_ret, horizon)
        st.success(f"🎯 **{prob:.0%} probability** of reaching ₹{target:,.0f} in {horizon} years · Required SIP if not met: ₹{req_sip:,.0f}/month")

    if st.button("Next: Insights & Export →"):
        st.session_state.screen = 8; st.rerun()

# ── Screen 9: Insights & Export ───────────────────────────────────────────────
def screen_insights():
    st.markdown('<div class="section-title">💡 Insights & Export</div>', unsafe_allow_html=True)
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

    st.markdown("#### Checklist")
    st.checkbox("Open / fund brokerage account")
    st.checkbox("Place buy orders per recommended allocation")
    st.checkbox("Set up monthly SIP / auto-invest")
    st.checkbox("Set quarterly rebalancing reminder")

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
            pdf = FPDF(); pdf.add_page(); pdf.set_font("Helvetica", "B", 16)
            pdf.cell(0, 10, "Portfolio Optimization Report", ln=True, align="C")
            pdf.set_font("Helvetica", size=11)
            pdf.cell(0, 8, f"Risk Profile: {profile['name']}", ln=True)
            pdf.cell(0, 8, f"Expected Return: {opt['return']:.2%} | Volatility: {opt['volatility']:.2%} | Sharpe: {opt['sharpe']:.2f}", ln=True)
            pdf.ln(4); pdf.set_font("Helvetica", "B", 12); pdf.cell(0, 8, "Asset Allocation", ln=True)
            pdf.set_font("Helvetica", size=10)
            for _, r in alloc_df.iterrows():
                pdf.cell(0, 7, f"  {r['Asset']}: {r['Allocation_%']:.2f}% — ₹{r['Amount_INR']:,.0f}", ln=True)
            st.download_button("⬇️ Download PDF", bytes(pdf.output()), "portfolio_report.pdf", "application/pdf")
        except Exception as e:
            st.caption(f"PDF unavailable: {e}")


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

if __name__ == "__main__":
    main()
