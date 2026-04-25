# Efficient Frontier Portfolio Optimizer

A Streamlit app for multi-asset portfolio optimization using the Efficient Frontier methodology.

## Setup

```bash
# Create and activate virtual environment (optional but recommended)
python -m venv venv
venv\Scripts\activate    # Windows

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

## Features

- **Risk Profiling** — Auto-calculated from age + horizon with manual override
- **Asset Universe** — Indian equities, US markets, Gold, Bonds (Indian + International), FDs, REITs
- **Portfolio Optimization** — Max Sharpe via scipy SLSQP + 10,000 Monte Carlo portfolios
- **Efficient Frontier** — 20-point minimum variance curve with Capital Allocation Line
- **Diversification Analysis** — Correlation heatmap, sector exposure, risk contribution
- **SIP Projections** — 1,000 Monte Carlo future paths (p5/p50/p95 bands)
- **Tax Estimates** — Simplified LTCG/STCG for equity; flat rate for debt
- **Export** — CSV, Excel (multi-sheet), PDF report

## Constraints

- Max 15% in any single asset
- Max 25% per sector
- Asset class bounds mapped to risk profile (equity/debt/alternatives)

## Tech Stack

Python 3.11 · Streamlit · yfinance · scipy · Plotly · fpdf2 · openpyxl
