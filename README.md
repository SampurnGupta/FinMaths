# 📈 FinMaths: Institutional-Grade Portfolio Optimizer

A premium, glassmorphism-inspired financial dashboard built with Streamlit and powered by Groq AI. This tool performs advanced Modern Portfolio Theory (MPT) optimization using real-time market data, adjusted for multi-currency risk, taxes, and inflation.

## 🚀 Key Features

### 1. Advanced Financial Engineering
- **Log Returns**: Modeled for accurate multi-period compounding and statistical robustness.
- **FX Modeling**: Automatic USD to INR conversion using live exchange rates to account for currency risk on international assets.
- **Real Returns**: All projections are adjusted for **Tax (LTCG/Debt)** and **Inflation (6% p.a.)**.
- **Synthetic Assets**: Stochastic noise and duration modeling for FDs, Bonds, and REITs to maintain data consistency.

### 2. Premium Interactive UI
- **Glassmorphism Design**: Midnight navy theme with blur effects, neon accents, and modern typography (Outfit & Inter).
- **Dynamic Visualizations**: Neon-styled Efficient Frontier, Correlation Heatmaps, and Monte Carlo projections.
- **Animated Transitions**: Smooth fade-in effects and interactive components.

### 3. AI Portfolio Concierge (Groq)
- **Interactive Chat**: Ask follow-up questions about your specific portfolio results in a natural chat interface.
- **Context-Aware Insights**: AI Advisor is grounded in your generated metrics, risk profile, and asset allocations.
- **Automated Briefing**: Get an immediate expert summary of your diversification benefits upon opening the chat.

## 🛠️ Installation

1.  **Clone the Repository**:
    ```bash
    git clone <repo-url>
    cd FinMaths
    ```

2.  **Set Up Environment**:
    ```bash
    python -m venv venv
    .\venv\Scripts\activate
    pip install -r requirements.txt
    ```

3.  **Configure API Key**:
    Create a `.env` file in the root directory:
    ```env
    GROQ_API_KEY=your_key_here
    ```

4.  **Run the App**:
    ```bash
    streamlit run app.py
    ```

## 📊 Methodology
- **Optimization**: Uses Scipy's `SLSQP` optimizer to maximize the Sharpe Ratio.
- **Simulation**: 10,000-point Monte Carlo for Efficient Frontier discovery.
- **Constraints**: Sector caps (25%), asset caps (15%), and risk-profile specific class bounds.
- **Projections**: 1,000-path Monte Carlo wealth simulations with 95% confidence intervals and worst-case drawdown analysis.
