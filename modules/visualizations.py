"""
visualizations.py
All Plotly chart generation functions. Uses a consistent, minimal theme.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ── Theme ─────────────────────────────────────────────────────────────────────
COLORS = {
    "primary":   "#4F46E5",   # indigo
    "secondary": "#06B6D4",   # cyan
    "success":   "#10B981",   # emerald
    "warning":   "#F59E0B",   # amber
    "danger":    "#EF4444",   # red
    "neutral":   "#6B7280",   # gray
    "bg":        "#FFFFFF",
    "grid":      "#F3F4F6",
    "text":      "#111827",
    "subtext":   "#6B7280",
    "mc_scatter":"rgba(99,102,241,0.18)",
}

LAYOUT_BASE = dict(
    paper_bgcolor=COLORS["bg"],
    plot_bgcolor=COLORS["bg"],
    font=dict(family="Inter, sans-serif", color=COLORS["text"], size=12),
    margin=dict(l=40, r=20, t=50, b=40),
    xaxis=dict(gridcolor=COLORS["grid"], zerolinecolor=COLORS["grid"]),
    yaxis=dict(gridcolor=COLORS["grid"], zerolinecolor=COLORS["grid"]),
    legend=dict(bgcolor="rgba(0,0,0,0)", borderwidth=0),
    hoverlabel=dict(bgcolor="white", font_size=12, bordercolor="#E5E7EB"),
)


def _pct(val: float) -> str:
    return f"{val * 100:.2f}%"


# ── 1. Efficient Frontier ─────────────────────────────────────────────────────
def plot_efficient_frontier(
    mc_portfolios: pd.DataFrame,
    frontier: pd.DataFrame,
    optimal: pd.Series,
    gmvp: pd.Series,
    asset_stats: pd.DataFrame,
    meta: pd.DataFrame,
) -> go.Figure:
    fig = go.Figure()

    # Monte Carlo scatter
    fig.add_trace(go.Scatter(
        x=mc_portfolios["volatility"],
        y=mc_portfolios["return"],
        mode="markers",
        marker=dict(
            color=mc_portfolios["sharpe"],
            colorscale="Viridis",
            size=3,
            opacity=0.5,
            colorbar=dict(title="Sharpe", thickness=12, len=0.6),
        ),
        name="MC Portfolios",
        hovertemplate="Vol: %{x:.2%}<br>Return: %{y:.2%}<br>Sharpe: %{marker.color:.2f}<extra></extra>",
    ))

    # Efficient frontier line
    if not frontier.empty:
        fig.add_trace(go.Scatter(
            x=frontier["volatility"],
            y=frontier["return"],
            mode="lines",
            line=dict(color=COLORS["primary"], width=2.5),
            name="Efficient Frontier",
            hovertemplate="Vol: %{x:.2%}<br>Return: %{y:.2%}<extra></extra>",
        ))

    # Capital Allocation Line
    rf = 0.065
    if optimal is not None and "volatility" in optimal:
        max_vol = max(mc_portfolios["volatility"].max(), optimal["volatility"] * 1.5)
        cal_x = [0, max_vol]
        slope = (optimal["return"] - rf) / optimal["volatility"]
        cal_y = [rf, rf + slope * max_vol]
        fig.add_trace(go.Scatter(
            x=cal_x, y=cal_y, mode="lines",
            line=dict(color=COLORS["warning"], width=1.5, dash="dash"),
            name="CAL (Risk-Free → Optimal)",
        ))

    # Individual assets
    for t, row in asset_stats.iterrows():
        label = meta.loc[t, "label"] if t in meta.index else t
        fig.add_trace(go.Scatter(
            x=[row["volatility"]], y=[row["return"]],
            mode="markers+text",
            marker=dict(symbol="circle", size=8, color=COLORS["neutral"], line=dict(color="white", width=1)),
            text=[label[:10]], textposition="top center",
            textfont=dict(size=9, color=COLORS["subtext"]),
            name=label, showlegend=False,
            hovertemplate=f"<b>{label}</b><br>Vol: {_pct(row['volatility'])}<br>Return: {_pct(row['return'])}<extra></extra>",
        ))

    # GMVP
    if gmvp is not None:
        fig.add_trace(go.Scatter(
            x=[gmvp["volatility"]], y=[gmvp["return"]],
            mode="markers", marker=dict(symbol="diamond", size=13, color=COLORS["warning"], line=dict(color="white", width=1.5)),
            name="Min Variance", hovertemplate=f"<b>GMVP</b><br>Vol: {_pct(gmvp['volatility'])}<br>Return: {_pct(gmvp['return'])}<br>Sharpe: {gmvp['sharpe']:.2f}<extra></extra>",
        ))

    # Optimal (Max Sharpe)
    if optimal is not None:
        fig.add_trace(go.Scatter(
            x=[optimal["volatility"]], y=[optimal["return"]],
            mode="markers", marker=dict(symbol="star", size=18, color=COLORS["success"], line=dict(color="white", width=1.5)),
            name="Max Sharpe (Recommended)", hovertemplate=f"<b>Recommended</b><br>Vol: {_pct(optimal['volatility'])}<br>Return: {_pct(optimal['return'])}<br>Sharpe: {optimal['sharpe']:.2f}<extra></extra>",
        ))

    fig.update_layout(
        **LAYOUT_BASE,
        title=dict(text="Efficient Frontier", font=dict(size=16, color=COLORS["text"])),
        xaxis_title="Annual Volatility (Risk)",
        yaxis_title="Annual Expected Return",
        xaxis_tickformat=".1%",
        yaxis_tickformat=".1%",
        height=520,
    )
    return fig


# ── 2. Correlation Heatmap ────────────────────────────────────────────────────
def plot_correlation_heatmap(corr_matrix: pd.DataFrame, meta: pd.DataFrame) -> go.Figure:
    labels = [meta.loc[t, "label"] if t in meta.index else t for t in corr_matrix.index]
    z = corr_matrix.values

    fig = go.Figure(go.Heatmap(
        z=z, x=labels, y=labels,
        colorscale=[
            [0.0, "#EF4444"], [0.5, "#F9FAFB"], [1.0, "#4F46E5"]
        ],
        zmin=-1, zmax=1,
        text=np.round(z, 2),
        texttemplate="%{text}",
        textfont=dict(size=9),
        hoverongaps=False,
        colorbar=dict(title="ρ", thickness=12),
    ))

    fig.update_layout(
        **LAYOUT_BASE,
        title=dict(text="Asset Correlation Matrix", font=dict(size=16)),
        height=450,
        xaxis=dict(tickfont=dict(size=9), tickangle=-30),
        yaxis=dict(tickfont=dict(size=9), autorange="reversed"),
    )
    return fig


# ── 3. Allocation Pie Chart ───────────────────────────────────────────────────
def plot_allocation_pie(weights: pd.Series, meta: pd.DataFrame) -> go.Figure:
    w = weights[weights > 0.001].sort_values(ascending=False)
    labels = [meta.loc[t, "label"] if t in meta.index else t for t in w.index]
    colors = px.colors.qualitative.Set2[:len(w)]

    fig = go.Figure(go.Pie(
        labels=labels,
        values=w.values,
        hole=0.42,
        textinfo="label+percent",
        textfont=dict(size=11),
        marker=dict(colors=colors, line=dict(color="white", width=2)),
        hovertemplate="<b>%{label}</b><br>Allocation: %{percent}<extra></extra>",
    ))

    fig.update_layout(
        **LAYOUT_BASE,
        title=dict(text="Recommended Portfolio Allocation", font=dict(size=16)),
        showlegend=False,
        height=420,
        margin=dict(l=10, r=10, t=50, b=10),
    )
    return fig


# ── 4. Sector Exposure Bar ────────────────────────────────────────────────────
def plot_sector_bar(weights: pd.Series, meta: pd.DataFrame) -> go.Figure:
    df = meta.copy()
    df["weight"] = weights.reindex(df.index).fillna(0)
    sector_weights = df.groupby("sector")["weight"].sum().sort_values(ascending=True)

    colors = [COLORS["danger"] if v > 0.25 else COLORS["primary"] for v in sector_weights.values]

    fig = go.Figure(go.Bar(
        x=sector_weights.values,
        y=sector_weights.index,
        orientation="h",
        marker=dict(color=colors, line=dict(color="white", width=0.5)),
        text=[f"{v:.1%}" for v in sector_weights.values],
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>Allocation: %{x:.1%}<extra></extra>",
    ))

    fig.add_vline(x=0.25, line_dash="dash", line_color=COLORS["warning"],
                  annotation_text="25% cap", annotation_position="top")

    fig.update_layout(
        **LAYOUT_BASE,
        title=dict(text="Sector Exposure", font=dict(size=16)),
        xaxis_title="Allocation",
        xaxis_tickformat=".0%",
        height=400,
    )
    return fig


# ── 5. Asset Class Donut ──────────────────────────────────────────────────────
def plot_asset_class_donut(weights: pd.Series, meta: pd.DataFrame) -> go.Figure:
    df = meta.copy()
    df["weight"] = weights.reindex(df.index).fillna(0)
    class_weights = df.groupby("asset_class")["weight"].sum()

    color_map = {"equity": COLORS["primary"], "debt": COLORS["success"], "alt": COLORS["warning"]}
    label_map = {"equity": "Equity", "debt": "Debt / Fixed Income", "alt": "Alternatives"}

    labels = [label_map.get(k, k) for k in class_weights.index]
    colors = [color_map.get(k, COLORS["neutral"]) for k in class_weights.index]

    fig = go.Figure(go.Pie(
        labels=labels, values=class_weights.values, hole=0.55,
        textinfo="label+percent",
        marker=dict(colors=colors, line=dict(color="white", width=3)),
        hovertemplate="<b>%{label}</b><br>%{percent}<extra></extra>",
    ))
    fig.update_layout(
        **LAYOUT_BASE,
        title=dict(text="Asset Class Breakdown", font=dict(size=16)),
        showlegend=False, height=380,
        margin=dict(l=10, r=10, t=50, b=10),
    )
    return fig


# ── 6. SIP Projection Chart ───────────────────────────────────────────────────
def plot_sip_projection(paths_df: pd.DataFrame, sip_data: dict, initial: float) -> go.Figure:
    fig = go.Figure()

    # Percentile bands
    fig.add_trace(go.Scatter(
        x=pd.concat([paths_df["year"], paths_df["year"][::-1]]),
        y=pd.concat([paths_df["p95"], paths_df["p5"][::-1]]),
        fill="toself", fillcolor="rgba(79,70,229,0.08)",
        line=dict(color="rgba(0,0,0,0)"),
        name="5th–95th Percentile", showlegend=True,
    ))
    fig.add_trace(go.Scatter(
        x=pd.concat([paths_df["year"], paths_df["year"][::-1]]),
        y=pd.concat([paths_df["p75"], paths_df["p25"][::-1]]),
        fill="toself", fillcolor="rgba(79,70,229,0.15)",
        line=dict(color="rgba(0,0,0,0)"),
        name="25th–75th Percentile", showlegend=True,
    ))

    # Median line
    fig.add_trace(go.Scatter(
        x=paths_df["year"], y=paths_df["p50"],
        mode="lines", line=dict(color=COLORS["primary"], width=2.5),
        name="Median Outcome",
        hovertemplate="Year %{x}<br>Median: ₹%{y:,.0f}<extra></extra>",
    ))

    # Deterministic SIP line (constant return)
    if sip_data and "year_by_year" in sip_data:
        yby = sip_data["year_by_year"]
        fig.add_trace(go.Scatter(
            x=yby["year"], y=yby["total_invested"],
            mode="lines", line=dict(color=COLORS["neutral"], width=1.5, dash="dot"),
            name="Total Invested",
            hovertemplate="Year %{x}<br>Invested: ₹%{y:,.0f}<extra></extra>",
        ))

    # Initial investment line
    fig.add_hline(y=initial, line_dash="dot", line_color=COLORS["neutral"],
                  annotation_text="Initial", annotation_position="right")

    fig.update_layout(
        **LAYOUT_BASE,
        title=dict(text="Portfolio Value Projection", font=dict(size=16)),
        xaxis_title="Year",
        yaxis_title="Portfolio Value (₹)",
        yaxis_tickprefix="₹",
        yaxis_tickformat=",.0f",
        height=460,
    )
    return fig


# ── 7. Risk Contribution Pie ──────────────────────────────────────────────────
def plot_risk_contribution(weights: pd.Series, cov_matrix: pd.DataFrame, meta: pd.DataFrame) -> go.Figure:
    w = weights.values
    sigma = cov_matrix.values
    portfolio_var = w @ sigma @ w
    marginal = sigma @ w
    contrib = w * marginal / portfolio_var if portfolio_var > 0 else w

    df = pd.DataFrame({
        "ticker": weights.index,
        "contrib": contrib,
    })
    df["label"] = df["ticker"].map(lambda t: meta.loc[t, "label"] if t in meta.index else t)
    df = df[df["contrib"] > 0.001]

    fig = go.Figure(go.Pie(
        labels=df["label"], values=df["contrib"], hole=0.45,
        textinfo="label+percent", textfont=dict(size=10),
        marker=dict(colors=px.colors.qualitative.Pastel, line=dict(color="white", width=2)),
        hovertemplate="<b>%{label}</b><br>Risk Contribution: %{percent}<extra></extra>",
    ))
    fig.update_layout(
        **LAYOUT_BASE,
        title=dict(text="Risk Contribution by Asset", font=dict(size=16)),
        showlegend=False, height=380,
        margin=dict(l=10, r=10, t=50, b=10),
    )
    return fig


# ── 8. Comparison Bar Chart ───────────────────────────────────────────────────
def plot_comparison_bars(comparison_df: pd.DataFrame) -> go.Figure:
    metrics = ["Expected Return", "Volatility", "Sharpe Ratio"]
    portfolios = comparison_df.columns.tolist()
    color_list = [COLORS["primary"], COLORS["success"], COLORS["warning"], COLORS["neutral"]]

    fig = make_subplots(rows=1, cols=3, subplot_titles=metrics)

    for col_i, metric in enumerate(metrics):
        for p_i, portfolio in enumerate(portfolios):
            val = comparison_df.loc[metric, portfolio]
            fig.add_trace(go.Bar(
                x=[portfolio], y=[val],
                name=portfolio if col_i == 0 else None,
                showlegend=(col_i == 0),
                marker_color=color_list[p_i % len(color_list)],
                text=[f"{val:.2f}" if metric == "Sharpe Ratio" else f"{val:.1%}"],
                textposition="outside",
                hovertemplate=f"<b>{portfolio}</b><br>{metric}: {val:.3f}<extra></extra>",
            ), row=1, col=col_i + 1)

        if metric != "Sharpe Ratio":
            fig.update_yaxes(tickformat=".1%", row=1, col=col_i + 1)

    fig.update_layout(
        **LAYOUT_BASE,
        title=dict(text="Portfolio Comparison", font=dict(size=16)),
        height=420,
        barmode="group",
    )
    return fig
