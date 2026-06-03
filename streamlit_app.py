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
    d = data.get("explanation_data")
    if d:
        st.markdown("---")

        st.markdown("### 📈 Trend & Price Structure")
        st.markdown(
            f"{d['name']} is **${d['current_price']:,.2f}**, "
            f"{d['vs_ma30_label']} its 30-day MA, "
            f"{d['vs_ma90_label']} its 90-day MA, and "
            f"{d['vs_ma200_label']} its 200-day MA. "
            f"Short-term **{d['trend_short']}**, {d['trend_long']}. "
            f"{d['ma_cross'].capitalize()} structure. {d['bb_text'].capitalize()}."
        )

        st.markdown("### ⚡ Momentum Indicators")
        def pill(label, color):
            bg  = {"green": "#16a34a22", "red": "#dc262622", "gray": "#47556922"}[color]
            bdr = {"green": "#16a34a66", "red": "#dc262666", "gray": "#47556966"}[color]
            clr = {"green": "#4ade80",   "red": "#f87171",   "gray": "#94a3b8"}[color]
            return f'<span style="background:{bg};border:1px solid {bdr};color:{clr};border-radius:999px;padding:2px 10px;font-size:12px;font-weight:600;white-space:nowrap">{label}</span>'

        def score_color(score):
            if score > 0.05:  return "green"
            if score < -0.05: return "red"
            return "gray"

        def fng_color(val):
            if val >= 60: return "green"
            if val <= 40: return "red"
            return "gray"

        m1, m2, m3 = st.columns(3)
        rsi = d['rsi']
        rsi_color = "green" if rsi < 30 else "red" if rsi > 70 else "green" if rsi > 55 else "red" if rsi < 45 else "gray"
        rsi_label = "Oversold — potential bounce" if rsi < 30 else "Overbought — potential pullback" if rsi > 70 else "Bullish neutral" if rsi > 55 else "Bearish neutral" if rsi < 45 else "Neutral"
        m1.metric("RSI", f"{rsi:.1f}")
        m1.markdown(pill(rsi_label, rsi_color), unsafe_allow_html=True)

        macd_color = "green" if d['macd_bull'] else "red"
        macd_label = "▲ Bullish" if d['macd_bull'] else "▼ Bearish"
        m2.metric("MACD Histogram", f"{d['macd_hist']:+.2f}")
        m2.markdown(pill(macd_label, macd_color), unsafe_allow_html=True)

        vol = d['volatility_pct']
        vol_color = "red" if vol > 80 else "gray" if vol > 40 else "green"
        vol_label = "High volatility" if vol > 80 else "Moderate" if vol > 40 else "Low volatility"
        m3.metric("14d Volatility", f"{vol:.1f}%")
        m3.markdown(pill(vol_label, vol_color), unsafe_allow_html=True)

        st.markdown("### 🌐 Sentiment")
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Fear & Greed", f"{d['fng_val']:.0f}")
        s1.markdown(pill(d['fng_cls'], fng_color(d['fng_val'])), unsafe_allow_html=True)
        s2.metric("News", f"{d['news_score']:+.2f}")
        s2.markdown(pill("Positive" if d['news_score'] > 0.05 else "Negative" if d['news_score'] < -0.05 else "Neutral", score_color(d['news_score'])), unsafe_allow_html=True)
        s3.metric("Reddit", f"{d['x_score']:+.2f}")
        s3.markdown(pill("Positive" if d['x_score'] > 0.05 else "Negative" if d['x_score'] < -0.05 else "Neutral", score_color(d['x_score'])), unsafe_allow_html=True)
        s4.metric("Composite", f"{d['today_sentiment']:+.3f}")
        s4.markdown(pill("Bullish bias" if d['today_sentiment'] > 0.05 else "Bearish bias" if d['today_sentiment'] < -0.05 else "Neutral", score_color(d['today_sentiment'])), unsafe_allow_html=True)

        st.markdown("### 🔮 Forecast Summary")
        import pandas as pd
        rows = []
        for f in d["forecasts"]:
            rows.append({
                "Horizon": f["horizon"],
                "Target Price": f"${f['target']:,.2f}",
                "Change": f"{f['change_pct']:+.1f}%",
                "80% Range": f"${f['lower']:,.0f} – ${f['upper']:,.0f}",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        st.caption(f"The model projects a **{d['magnitude']} {d['direction']}** trajectory over 90 days under the **{d['scenario_label']}** scenario.")

        st.markdown("### 💡 Why This Scenario?")
        scenario_text = {
            "bullish": "The **Bullish scenario** amplifies positive sentiment signals and increases Prophet's changepoint flexibility to follow upward momentum. Positive sentiment scores are weighted 1.5×.",
            "bearish": "The **Bearish scenario** amplifies negative signals and anchors the model against upside momentum. Negative sentiment scores are weighted 1.5×.",
            "base": "The **Base scenario** is the neutral benchmark — no directional bias, sentiment at face value (1× weight), default Prophet settings. The most statistically honest forecast.",
        }.get(d["scenario"], "")
        st.info(scenario_text)
        st.warning("⚠️ Crypto markets are highly unpredictable. These forecasts are research tools, not financial advice.")
else:
    st.error("No data available for this selection.")
