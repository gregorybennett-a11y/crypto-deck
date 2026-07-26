import streamlit as st

st.set_page_config(page_title="Crypto Prediction Dashboard", layout="wide", page_icon="🪙")

from crypto_predictor import (
    COINS, SCENARIO_PARAMS,
    fetch_fear_and_greed, run_coin_scenario
)

COIN_LABELS     = {"bitcoin": "₿ Bitcoin (BTC)", "ethereum": "Ξ Ethereum (ETH)"}
COIN_SHORT      = {"bitcoin": "BTC", "ethereum": "ETH"}
COIN_SYMBOL     = {"bitcoin": "₿", "ethereum": "Ξ"}
COIN_GRADIENT   = {"bitcoin": "linear-gradient(135deg,#F7931A,#fbbf24)",
                   "ethereum": "linear-gradient(135deg,#627EEA,#22d3ee)"}
SCENARIO_LABELS = {"base": "Base", "bullish": "Bullish", "bearish": "Bearish"}
HORIZON_LABELS  = {"7d": "7 Days", "30d": "30 Days", "90d": "90 Days"}

# ── Global design system ──────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500;600;700&display=swap');

html, body, [class*="st-"], .stApp { font-family: 'Inter', -apple-system, 'Segoe UI', sans-serif; }
/* Restore Streamlit's Material icon font (expander arrows, sidebar collapse, etc.) */
[data-testid="stIconMaterial"], [class*="material-symbols"], .material-symbols-rounded {
    font-family: 'Material Symbols Rounded' !important;
    font-weight: normal; font-style: normal; letter-spacing: normal;
    text-transform: none; white-space: nowrap; word-wrap: normal; direction: ltr;
}

.stApp {
  background:
    radial-gradient(1100px 520px at 12% -8%, rgba(99,102,241,.18), transparent 60%),
    radial-gradient(900px 480px at 88% -4%, rgba(34,211,238,.12), transparent 55%),
    radial-gradient(1000px 620px at 50% 112%, rgba(167,139,250,.09), transparent 60%),
    #05070d;
}
[data-testid="stHeader"] { background: rgba(5,7,13,.72); backdrop-filter: blur(10px); }
.block-container { padding-top: 1.6rem; max-width: 1180px; }

/* ── Hero ── */
.cx-hero {
  position: relative; overflow: hidden;
  background: rgba(15,23,42,.55);
  border: 1px solid rgba(148,163,184,.14);
  border-radius: 20px; padding: 1.6rem 1.9rem 1.5rem;
  backdrop-filter: blur(14px);
  box-shadow: 0 12px 40px rgba(2,6,23,.5);
  margin-bottom: 1.1rem;
}
.cx-hero::before {
  content:''; position:absolute; inset:0 0 auto 0; height:3px;
  background: linear-gradient(90deg,#6366f1,#22d3ee,#a78bfa);
}
.cx-hero .cx-badge {
  display:inline-flex; align-items:center; gap:.4rem;
  font-size:.66rem; font-weight:700; letter-spacing:.14em; text-transform:uppercase;
  color:#22d3ee; background:rgba(34,211,238,.08);
  border:1px solid rgba(34,211,238,.25); border-radius:999px;
  padding:.25rem .7rem; margin-bottom:.7rem;
}
.cx-hero .cx-badge .dot {
  width:7px; height:7px; border-radius:50%; background:#22d3ee;
  box-shadow:0 0 8px #22d3ee; animation: cx-pulse 2s infinite;
}
@keyframes cx-pulse { 50% { opacity:.35; } }
.cx-hero h1 {
  margin:0; font-size:2.05rem; font-weight:800; letter-spacing:-.02em; line-height:1.15;
  background: linear-gradient(135deg,#e2e8f0 25%,#818cf8 65%,#22d3ee);
  -webkit-background-clip:text; background-clip:text; -webkit-text-fill-color:transparent;
}
.cx-hero p { margin:.45rem 0 0; color:#94a3b8; font-size:.9rem; }

/* ── Price banner ── */
.cx-price {
  display:flex; align-items:center; gap:1.1rem;
  background: rgba(15,23,42,.55); border:1px solid rgba(148,163,184,.14);
  border-radius:18px; padding:1.05rem 1.4rem; backdrop-filter: blur(12px);
  box-shadow: 0 8px 28px rgba(2,6,23,.45); margin: .2rem 0 .9rem;
}
.cx-price .coin-ic {
  width:52px; height:52px; border-radius:14px; flex:0 0 52px;
  display:flex; align-items:center; justify-content:center;
  font-size:1.6rem; font-weight:800; color:#0b1120;
  box-shadow: 0 6px 18px rgba(2,6,23,.5);
}
.cx-price .p-name { font-size:.72rem; font-weight:700; letter-spacing:.12em;
  text-transform:uppercase; color:#94a3b8; }
.cx-price .p-val { font-family:'JetBrains Mono',monospace; font-size:1.85rem;
  font-weight:700; color:#f1f5f9; line-height:1.2; }
.cx-chip {
  display:inline-block; font-size:.7rem; font-weight:600; color:#cbd5e1;
  background:rgba(99,102,241,.10); border:1px solid rgba(99,102,241,.3);
  border-radius:999px; padding:.22rem .65rem; margin-left:.35rem;
}

/* ── Section headers ── */
.cx-sec {
  display:flex; align-items:center; gap:.6rem;
  font-size:1.02rem; font-weight:700; color:#e2e8f0;
  margin: 1.4rem 0 .7rem; letter-spacing:-.01em;
}
.cx-sec::before {
  content:''; width:4px; height:1.05rem; border-radius:4px;
  background: linear-gradient(180deg,#6366f1,#22d3ee);
}

/* ── Glass cards / narrative ── */
.cx-card {
  background: rgba(15,23,42,.5); border:1px solid rgba(148,163,184,.13);
  border-radius:14px; padding:1rem 1.2rem; backdrop-filter: blur(10px);
  color:#cbd5e1; font-size:.92rem; line-height:1.65;
}
.cx-card b, .cx-card strong { color:#f1f5f9; }

/* ── Forecast cards ── */
.fc-grid { display:flex; gap:.8rem; }
.fc-card {
  flex:1; background: rgba(15,23,42,.5); border:1px solid rgba(148,163,184,.13);
  border-radius:14px; padding: .95rem 1.05rem; backdrop-filter: blur(10px);
  transition: transform .15s ease, border-color .15s ease;
}
.fc-card:hover { transform: translateY(-2px); }
.fc-card.active { border-color: rgba(99,102,241,.55);
  box-shadow: 0 0 0 1px rgba(99,102,241,.35), 0 10px 30px rgba(99,102,241,.12); }
.fc-card .fc-h { font-size:.68rem; font-weight:700; letter-spacing:.12em;
  text-transform:uppercase; color:#94a3b8; margin-bottom:.35rem; }
.fc-card .fc-t { font-family:'JetBrains Mono',monospace; font-size:1.3rem;
  font-weight:700; color:#f1f5f9; }
.fc-card .fc-c { font-size:.85rem; font-weight:700; margin:.15rem 0 .4rem; }
.fc-card .fc-r { font-size:.72rem; color:#7c8aa0; font-family:'JetBrains Mono',monospace; }
.fc-up   { color:#34d399; } .fc-down { color:#fb7185; }

/* ── Streamlit widget polish ── */
div[data-testid="stMetric"] {
  background: rgba(15,23,42,.5); border:1px solid rgba(148,163,184,.13);
  border-radius:14px; padding:.85rem 1rem .7rem; backdrop-filter: blur(10px);
}
div[data-testid="stMetric"] label { color:#94a3b8 !important; }
div[data-testid="stMetricValue"] { font-family:'JetBrains Mono',monospace; font-weight:700; }

div[data-baseweb="select"] > div {
  background: rgba(15,23,42,.7) !important;
  border-color: rgba(99,102,241,.28) !important;
  border-radius: 10px !important;
}
div[data-testid="stExpander"] {
  background: rgba(15,23,42,.45); border:1px solid rgba(148,163,184,.13);
  border-radius: 14px; overflow:hidden;
}
div[data-testid="stExpander"] summary { font-weight:600; }
[data-testid="stDataFrame"] { border:1px solid rgba(148,163,184,.13); border-radius:12px; }
hr { border-color: rgba(148,163,184,.12) !important; }

div[data-testid="stAlert"] { border-radius: 12px; backdrop-filter: blur(8px); }
</style>
""", unsafe_allow_html=True)


@st.cache_data(ttl=14400, show_spinner="Fetching market data & running predictions... (~30–60s)")
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


def polish(fig, plot_bg="rgba(13,20,38,0.45)"):
    """Apply the shared dark-fintech theme to any Plotly figure."""
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=plot_bg,
        font=dict(family="Inter, -apple-system, sans-serif", size=12, color="#cbd5e1"),
        hoverlabel=dict(bgcolor="#111a30", bordercolor="rgba(99,102,241,.45)",
                        font=dict(family="Inter, sans-serif", size=12, color="#e2e8f0")),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    fig.update_layout(title_font=dict(family="Inter, sans-serif", size=17, color="#f1f5f9"))
    fig.update_xaxes(gridcolor="rgba(148,163,184,.08)", zerolinecolor="rgba(148,163,184,.08)")
    fig.update_yaxes(gridcolor="rgba(148,163,184,.08)", zerolinecolor="rgba(148,163,184,.08)")
    return fig


panels = load_all_data()

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="cx-hero">
  <div class="cx-badge"><span class="dot"></span>Live ML Forecasts</div>
  <h1>Crypto Prediction Dashboard</h1>
  <p>BTC &amp; ETH · 7d / 30d / 90d Prophet forecasts · CoinGecko · Fear &amp; Greed · News · Reddit</p>
</div>
""", unsafe_allow_html=True)

with st.expander("How to Use This Dashboard"):
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
- **Scenario** — Base (honest best guess) · Bullish (optimistic) · Bearish (cautious).
- **Horizon** — how far ahead to predict. 7 Days is most reliable; 90 Days is more speculative.

**RSI** measures buying/selling speed. Below 30 = oversold (may bounce). Above 70 = overbought (may pull back).

**MACD** shows momentum. Positive = buying momentum growing. Negative = selling momentum growing.
    """)

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

# ── Chart ─────────────────────────────────────────────────────────────────────
if coin_id in panels and scenario in panels[coin_id]:
    data = panels[coin_id][scenario]

    price = data["price"]
    st.markdown(f"""
<div class="cx-price">
  <div class="coin-ic" style="background:{COIN_GRADIENT[coin_id]}">{COIN_SYMBOL[coin_id]}</div>
  <div>
    <div class="p-name">{coin_id.capitalize()} · {COIN_SHORT[coin_id]}</div>
    <div class="p-val">${price:,.2f}</div>
  </div>
  <div style="margin-left:auto;text-align:right">
    <span class="cx-chip">{sc_label}</span>
    <span class="cx-chip">{hz_label}</span>
  </div>
</div>
""", unsafe_allow_html=True)

    fig = data["charts_fig"][horizon]
    fig.update_layout(title_text=(
        f"<b>{coin_id.capitalize()}</b>  ·  {scenario.capitalize()} Scenario  —  {hz_label} Forecast"))
    st.plotly_chart(polish(fig), use_container_width=True)

    # Fear & Greed bar chart
    fng_df = data.get("fng_df")
    if fng_df is not None and not fng_df.empty:
        import plotly.graph_objects as go
        import pandas as pd
        fng = fng_df.copy().sort_values("date").tail(90)
        fng["x"] = pd.to_datetime(fng["date"]).dt.strftime("%Y-%m-%d")
        colors = fng["fng"].apply(
            lambda v: "#fb7185" if v < 40 else ("#34d399" if v > 60 else "#fbbf24")
        )
        mood_fig = go.Figure(go.Bar(
            x=fng["x"], y=fng["fng"],
            marker_color=colors,
            marker_line_width=0,
            hovertemplate="%{x}: %{y}<extra></extra>"
        ))
        mood_fig.update_layout(
            title="Market Mood (Fear & Greed) — Last 90 Days",
            template="plotly_dark",
            height=230,
            bargap=0.25,
            margin=dict(l=40, r=20, t=44, b=40),
            yaxis=dict(range=[0, 100]),
            xaxis=dict(showgrid=False),
        )
        st.plotly_chart(polish(mood_fig), use_container_width=True)

    # Explanation
    d = data.get("explanation_data")
    if d:
        st.markdown('<div class="cx-sec">Trend &amp; Price Structure</div>', unsafe_allow_html=True)
        st.markdown(
            f"""<div class="cx-card">{d['name']} is <b>${d['current_price']:,.2f}</b>,
            {d['vs_ma30_label']} its 30-day MA,
            {d['vs_ma90_label']} its 90-day MA, and
            {d['vs_ma200_label']} its 200-day MA.
            Short-term <b>{d['trend_short']}</b>, {d['trend_long']}.
            {d['ma_cross'].capitalize()} structure. {d['bb_text'].capitalize()}.</div>""",
            unsafe_allow_html=True,
        )

        st.markdown('<div class="cx-sec">Momentum Indicators</div>', unsafe_allow_html=True)
        def pill(label, color):
            bg  = {"green": "#34d39918", "red": "#fb718518", "gray": "#64748b1e"}[color]
            bdr = {"green": "#34d39955", "red": "#fb718555", "gray": "#64748b55"}[color]
            clr = {"green": "#34d399",   "red": "#fb7185",   "gray": "#94a3b8"}[color]
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

        st.markdown('<div class="cx-sec">Sentiment</div>', unsafe_allow_html=True)
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Fear & Greed", f"{d['fng_val']:.0f}")
        s1.markdown(pill(d['fng_cls'], fng_color(d['fng_val'])), unsafe_allow_html=True)
        s2.metric("News", f"{d['news_score']:+.2f}")
        s2.markdown(pill("Positive" if d['news_score'] > 0.05 else "Negative" if d['news_score'] < -0.05 else "Neutral", score_color(d['news_score'])), unsafe_allow_html=True)
        s3.metric("Reddit", f"{d['x_score']:+.2f}")
        s3.markdown(pill("Positive" if d['x_score'] > 0.05 else "Negative" if d['x_score'] < -0.05 else "Neutral", score_color(d['x_score'])), unsafe_allow_html=True)
        s4.metric("Composite", f"{d['today_sentiment']:+.3f}")
        s4.markdown(pill("Bullish bias" if d['today_sentiment'] > 0.05 else "Bearish bias" if d['today_sentiment'] < -0.05 else "Neutral", score_color(d['today_sentiment'])), unsafe_allow_html=True)

        st.markdown('<div class="cx-sec">Forecast Summary</div>', unsafe_allow_html=True)
        hz_names = {"7d": "7-Day Target", "30d": "30-Day Target", "90d": "90-Day Target"}
        cards = ""
        for f in d["forecasts"]:
            up = f["change_pct"] >= 0
            cls = "fc-up" if up else "fc-down"
            arrow = "▲" if up else "▼"
            active = " active" if f["horizon"] == horizon else ""
            cards += f"""
  <div class="fc-card{active}">
    <div class="fc-h">{hz_names.get(f["horizon"], f["horizon"])}</div>
    <div class="fc-t">${f['target']:,.2f}</div>
    <div class="fc-c {cls}">{arrow} {f['change_pct']:+.1f}%</div>
    <div class="fc-r">80% range&nbsp; ${f['lower']:,.0f} – ${f['upper']:,.0f}</div>
  </div>"""
        st.markdown(f'<div class="fc-grid">{cards}\n</div>', unsafe_allow_html=True)
        st.caption(f"The model projects a **{d['magnitude']} {d['direction']}** trajectory over 90 days under the **{scenario.capitalize()}** scenario.")

        st.markdown('<div class="cx-sec">Why This Scenario?</div>', unsafe_allow_html=True)
        scenario_text = {
            "bullish": "The **Bullish scenario** amplifies positive sentiment signals and increases Prophet's changepoint flexibility to follow upward momentum. Positive sentiment scores are weighted 1.5×.",
            "bearish": "The **Bearish scenario** amplifies negative signals and anchors the model against upside momentum. Negative sentiment scores are weighted 1.5×.",
            "base": "The **Base scenario** is the neutral benchmark — no directional bias, sentiment at face value (1× weight), default Prophet settings. The most statistically honest forecast.",
        }.get(d["scenario"], "")
        st.info(scenario_text)
        st.warning("Crypto markets are highly unpredictable. These forecasts are research tools, not financial advice.")
else:
    st.error("No data available for this selection.")
