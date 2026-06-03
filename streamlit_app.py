import streamlit as st

st.set_page_config(page_title="Crypto Prediction Dashboard", layout="wide", page_icon="🪙")

from crypto_predictor import (
    COINS, SCENARIO_PARAMS,
    fetch_fear_and_greed, run_coin_scenario
)

COIN_LABELS     = {"bitcoin": "₿ Bitcoin (BTC)", "ethereum": "Ξ Ethereum (ETH)"}
SCENARIO_LABELS = {"base": "⚪ Base", "bullish": "🟢 Bullish", "bearish": "🔴 Bearish"}
HORIZON_LABELS  = {"7d": "7 Days", "30d": "30 Days", "90d": "90 Days"}


@st.cache_data(show_spinner="Fetching market data & running predictions... (~30–60s)")
def load_all_data():
    coins     = list(COINS.keys())
    scenarios = list(SCENARIO_PARAMS.keys())
    fng_df    = fetch_fear_and_greed(days=365)
    cached_df = {}
    panels    = {}
    for coin_id in coins:
        panels[coin_id] = {}
        for scenario in scenarios:
            try:
                panels[coin_id][scenario] = run_coin_scenario(
                    coin_id, 365, scenario, fng_df, cached_df
                )
            except Exception as e:
                st.warning(f"Skipped {coin_id}/{scenario}: {e}")
    return panels


panels = load_all_data()

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("## 🪙 Crypto Prediction Dashboard")
st.caption("BTC & ETH · 7d / 30d / 90d forecasts · Data: CoinGecko, Fear & Greed, News, Reddit")

with st.expander("📖 How to Use This Dashboard"):
    st.markdown("""
**What is Cryptocurrency?**
Digital money that exists only online — no bank or government controls it. **Bitcoin (BTC)** is the most well-known (like digital gold). **Ethereum (ETH)** is the second largest, powering many apps and digital contracts. Prices change constantly based on supply, demand, and public sentiment — just like stocks.

**What Does This Dashboard Do?**
It tracks the current price of Bitcoin and Ethereum and uses a mathematical model (Prophet) to predict where the price might go over the next 7, 30, or 90 days. Data sources:
- **CoinGecko** — real-time and historical price data
- **Fear & Greed Index** — measures how nervous or excited the market is (0 = extreme fear, 100 = extreme greed)
- **News headlines** — scans crypto news sites for positive or negative coverage
- **Reddit** — reads community sentiment from crypto forums

**How to Read the Chart**
- The **solid orange/blue line** is the real historical price.
- The **dashed line** after "TODAY" is the model's best guess for the future price.
- The **shaded band** is the uncertainty range — the price will likely land somewhere inside it. Wider = more uncertainty.
- The **dotted white line** is the recent price average, showing the general trend.

**The Three Selectors**
- **Crypto** — switch between Bitcoin and Ethereum.
- **Scenario** — ⚪ Base (honest best guess) · 🟢 Bullish (optimistic) · 🔴 Bearish (cautious).
- **Horizon** — how far ahead to predict. 7 Days is most reliable; 90 Days is more speculative.

**RSI** measures buying/selling speed. Below 30 = oversold (may bounce). Above 70 = overbought (may pull back).

**MACD** shows momentum. Positive = buying momentum growing. Negative = selling momentum growing.
    """)

st.divider()

# ── Controls ─────────────────────────────────────────────────────────────────
col1, col2, col3 = st.columns(3)
with col1:
    coin_options = {COIN_LABELS[c]: c for c in panels}
    coin_label   = st.selectbox("Crypto", list(coin_options.keys()))
    coin_id      = coin_options[coin_label]

with col2:
    sc_options = {SCENARIO_LABELS[s]: s for s in SCENARIO_PARAMS}
    sc_label   = st.selectbox("Scenario", list(sc_options.keys()))
    scenario   = sc_options[sc_label]

with col3:
    hz_label  = st.selectbox("Horizon", list(HORIZON_LABELS.values()))
    horizon   = {v: k for k, v in HORIZON_LABELS.items()}[hz_label]

st.divider()

# ── Chart ─────────────────────────────────────────────────────────────────────
if coin_id in panels and scenario in panels[coin_id]:
    data = panels[coin_id][scenario]

    price = data["price"]
    st.metric(label=COIN_LABELS[coin_id], value=f"${price:,.2f}")

    fig = data["charts_fig"][horizon]
    st.plotly_chart(fig, use_container_width=True)

    # Fear & Greed bar chart
    fng_df = data.get("fng_df")
    if fng_df is not None and not fng_df.empty:
        import plotly.graph_objects as go
        import pandas as pd
        fng = fng_df.copy().sort_values("date").tail(90)
        fng["x"] = pd.to_datetime(fng["date"]).dt.strftime("%Y-%m-%d")
        colors = fng["fng"].apply(
            lambda v: "#ef4444" if v < 40 else ("#22c55e" if v > 60 else "#f59e0b")
        )
        mood_fig = go.Figure(go.Bar(
            x=fng["x"], y=fng["fng"],
            marker_color=colors,
            hovertemplate="%{x}: %{y}<extra></extra>"
        ))
        mood_fig.update_layout(
            title="📊 Market Mood (Fear & Greed) — Last 90 Days",
            template="plotly_dark",
            paper_bgcolor="#0a0f1e",
            plot_bgcolor="#0d1526",
            height=220,
            margin=dict(l=40, r=20, t=40, b=40),
            yaxis=dict(range=[0, 100], gridcolor="rgba(255,255,255,0.05)"),
            xaxis=dict(showgrid=False),
        )
        st.plotly_chart(mood_fig, use_container_width=True)

    # Explanation
    explanation = data.get("explanation_html", "")
    if explanation:
        st.markdown("---")
        st.markdown(explanation, unsafe_allow_html=True)
else:
    st.error("No data available for this selection.")
