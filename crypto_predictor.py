"""
crypto_predictor.py
===================
BTC & ETH prediction model with sentiment analysis.

Sources:
  • Price data    — Yahoo Finance (free, no key)
  • Sentiment     — Fear & Greed Index, Reddit (r/bitcoin, r/ethereum etc.),
                    CoinDesk & Cointelegraph RSS headlines

Forecast horizons: 7 days | 30 days | 90 days (rolling from today)

Scenario modes:
  --scenario bullish  Weight positive sentiment up, aggressive changepoints
  --scenario bearish  Weight negative sentiment up, conservative changepoints
  --scenario base     Neutral (default)

Requirements:
    pip install requests pandas numpy scikit-learn prophet plotly \
                feedparser vaderSentiment beautifulsoup4 lxml

Usage:
    python crypto_predictor.py
    python crypto_predictor.py --coin bitcoin --scenario bullish
    python crypto_predictor.py --days 365 --scenario bearish --output ./charts
"""

import argparse
import sys
import time
import warnings
import logging
warnings.filterwarnings("ignore")
logging.getLogger("prophet").setLevel(logging.WARNING)
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)

import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta, timezone

# ── Optional imports (graceful fallback if missing) ─────────────────────────
try:
    import feedparser
    HAS_FEEDPARSER = True
except ImportError:
    HAS_FEEDPARSER = False
    print("⚠ feedparser not installed — news RSS disabled. pip install feedparser")

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER = SentimentIntensityAnalyzer()
    HAS_VADER = True
except ImportError:
    HAS_VADER = False
    print("⚠ vaderSentiment not installed — text sentiment disabled. pip install vaderSentiment")

try:
    from bs4 import BeautifulSoup
    HAS_BS4 = True
except ImportError:
    HAS_BS4 = False

from prophet import Prophet


# ═══════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════

FNG_API = "https://api.alternative.me/fng/"

YAHOO_TICKER = {
    "bitcoin":  "BTC-USD",
    "ethereum": "ETH-USD",
}

SCENARIO_PARAMS = {
    "base": {
        "changepoint_prior_scale": 0.05,
        "seasonality_prior_scale": 10,
        "sentiment_weight": 1.0,
        "label": "⚪ Base",
        "color": "#94a3b8",
    },
    "bullish": {
        "changepoint_prior_scale": 0.15,   # more flexible upward moves
        "seasonality_prior_scale": 15,
        "sentiment_weight": 1.5,            # amplify positive sentiment
        "label": "🟢 Bullish",
        "color": "#22c55e",
    },
    "bearish": {
        "changepoint_prior_scale": 0.02,   # more rigid / mean-reverting
        "seasonality_prior_scale": 5,
        "sentiment_weight": 1.5,            # amplify negative sentiment
        "label": "🔴 Bearish",
        "color": "#ef4444",
    },
}

HORIZONS = {"7d": 7, "30d": 30, "90d": 90}

COLORS = {
    "bitcoin":  {"primary": "#F7931A"},
    "ethereum": {"primary": "#627EEA"},
}

RSS_FEEDS = [
    ("CoinDesk",       "https://feeds.feedburner.com/CoinDesk"),
    ("Cointelegraph",  "https://cointelegraph.com/rss"),
    ("Bitcoin.com",    "https://news.bitcoin.com/feed/"),
    ("Decrypt",        "https://decrypt.co/feed"),
]

REDDIT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json",
}

# Subreddits to pull from per coin
REDDIT_SUBS = {
    "bitcoin":  ["bitcoin", "CryptoCurrency", "BitcoinMarkets"],
    "ethereum": ["ethereum", "CryptoCurrency", "ethtrader"],
}


# ═══════════════════════════════════════════════════════════════════════════
# 1. DATA FETCHING
# ═══════════════════════════════════════════════════════════════════════════

KRAKEN_PAIR = {"bitcoin": "XBTUSD", "ethereum": "ETHUSD"}


def _fetch_ohlcv_kraken(coin_id: str, days: int) -> pd.DataFrame:
    pair  = KRAKEN_PAIR.get(coin_id, f"{coin_id.upper()}USD")
    since = int((pd.Timestamp.now() - pd.Timedelta(days=days)).timestamp())
    r = requests.get("https://api.kraken.com/0/public/OHLC",
                     params={"pair": pair, "interval": 1440, "since": since}, timeout=30)
    r.raise_for_status()
    result = r.json()
    if result.get("error"):
        raise RuntimeError(result["error"])
    data = next(iter(result["result"].values()))
    df = pd.DataFrame(data, columns=["time","open","high","low","close","vwap","volume","count"])
    df["date"]   = pd.to_datetime(df["time"].astype(int), unit="s").dt.normalize()
    df["close"]  = pd.to_numeric(df["close"],  errors="coerce").astype(float)
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce").astype(float)
    df["market_cap"] = 0.0
    return df[["date","close","volume","market_cap"]].dropna(subset=["close"]).sort_values("date").reset_index(drop=True)


def _fetch_ohlcv_yfinance(coin_id: str, days: int) -> pd.DataFrame:
    import yfinance as yf
    ticker = YAHOO_TICKER.get(coin_id, f"{coin_id.upper()}-USD")
    period = "2y" if days > 365 else "1y"
    raw = yf.download(ticker, period=period, interval="1d", progress=False, auto_adjust=True)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    df = pd.DataFrame({
        "date":       pd.to_datetime(raw.index).normalize(),
        "close":      pd.to_numeric(raw["Close"], errors="coerce").astype(float),
        "volume":     pd.to_numeric(raw["Volume"] if "Volume" in raw.columns else 0, errors="coerce").astype(float),
        "market_cap": 0.0,
    })
    return df.dropna(subset=["close"]).sort_values("date").reset_index(drop=True)


def fetch_ohlcv(coin_id: str, days: int = 365) -> pd.DataFrame:
    """Fetch daily OHLCV — Kraken primary, yfinance fallback."""
    try:
        print(f"  Fetching {days}d {coin_id} from Kraken...")
        df = _fetch_ohlcv_kraken(coin_id, days)
        if df.empty:
            raise ValueError("Empty response")
    except Exception as e:
        print(f"  Kraken failed ({e}), falling back to yfinance...")
        df = _fetch_ohlcv_yfinance(coin_id, days)
    print(f"  ✓ {len(df)} records  |  ${df['close'].min():,.0f} – ${df['close'].max():,.0f}")
    return df


def fetch_current_price(coin_id: str) -> float:
    """Get latest price — Kraken primary, yfinance fallback."""
    try:
        pair = KRAKEN_PAIR.get(coin_id, f"{coin_id.upper()}USD")
        r = requests.get("https://api.kraken.com/0/public/Ticker",
                         params={"pair": pair}, timeout=10)
        r.raise_for_status()
        return float(next(iter(r.json()["result"].values()))["c"][0])
    except Exception:
        import yfinance as yf
        ticker = YAHOO_TICKER.get(coin_id, f"{coin_id.upper()}-USD")
        return float(yf.Ticker(ticker).fast_info["last_price"])


# ═══════════════════════════════════════════════════════════════════════════
# 2. SENTIMENT SOURCES
# ═══════════════════════════════════════════════════════════════════════════

def fetch_fear_and_greed(days: int = 365) -> pd.DataFrame:
    """
    Returns daily Fear & Greed scores (0=Extreme Fear, 100=Extreme Greed).
    Normalised to [-1, +1] for use as a Prophet regressor.
    """
    print("  Fetching Fear & Greed Index...")
    try:
        r = requests.get(FNG_API, params={"limit": days, "format": "json"}, timeout=10)
        r.raise_for_status()
        records = r.json().get("data", [])
        df = pd.DataFrame(records)
        df["date"]  = pd.to_datetime(df["timestamp"].astype(int), unit="s").dt.normalize()
        df["fng"]   = df["value"].astype(float)
        df["fng_norm"] = (df["fng"] - 50) / 50   # scale to [-1, +1]
        df = df[["date", "fng", "fng_norm", "value_classification"]].sort_values("date")
        print(f"  ✓ Fear & Greed: {len(df)} days  |  Today: {df['fng'].iloc[-1]:.0f} ({df['value_classification'].iloc[-1]})")
        return df
    except Exception as e:
        print(f"  ✗ Fear & Greed failed: {e}")
        return pd.DataFrame(columns=["date", "fng", "fng_norm", "value_classification"])


def score_text(text: str) -> float:
    """VADER compound score in [-1, +1]. Returns 0 if VADER unavailable."""
    if not HAS_VADER or not text:
        return 0.0
    return VADER.polarity_scores(text)["compound"]


def fetch_news_sentiment(coin_keywords: list[str]) -> float:
    """
    Pull recent headlines from RSS feeds, score each one, return mean sentiment.
    Returns float in [-1, +1].
    """
    if not HAS_FEEDPARSER:
        return 0.0

    print("  Fetching news headlines...")
    scores = []
    keywords_lower = [k.lower() for k in coin_keywords]

    for source, url in RSS_FEEDS:
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries[:30]:
                title = getattr(entry, "title", "")
                summary = getattr(entry, "summary", "")
                text = f"{title} {summary}".lower()
                if any(kw in text for kw in keywords_lower):
                    scores.append(score_text(f"{title} {summary}"))
        except Exception:
            continue

    if scores:
        mean = float(np.mean(scores))
        print(f"  ✓ News sentiment: {mean:+.3f}  ({len(scores)} relevant headlines)")
        return mean
    print("  ⚠ No relevant news headlines found")
    return 0.0


def fetch_reddit_sentiment(coin_id: str) -> float:
    """
    Pull posts from crypto subreddits via Reddit's public RSS feeds.
    No account, API key, or authentication needed.
    Scores post titles with VADER, returns mean sentiment in [-1, +1].
    """
    if not (HAS_VADER and HAS_FEEDPARSER):
        return 0.0

    subs = REDDIT_SUBS.get(coin_id, ["CryptoCurrency"])
    print(f"  Fetching Reddit sentiment for {coin_id} ({', '.join(f'r/{s}' for s in subs)})...")

    scores = []
    for sub in subs:
        for sort in ["hot", "top"]:
            try:
                url  = f"https://www.reddit.com/r/{sub}/{sort}.rss?limit=25&t=day"
                feed = feedparser.parse(url)
                entries = feed.get("entries", [])
                if not entries:
                    continue
                for entry in entries[:25]:
                    title   = entry.get("title", "")
                    summary = entry.get("summary", "")
                    text    = f"{title} {summary}".strip()
                    if text:
                        scores.append(score_text(text))
                print(f"    [Reddit] r/{sub}/{sort} → {len(entries)} posts")
            except Exception as e:
                print(f"    [Reddit] r/{sub}/{sort} → error: {e}")
                continue

    if not scores:
        print("  ⚠ Reddit RSS not accessible — skipping")
        return 0.0

    mean = float(np.mean(scores))
    print(f"  ✓ Reddit sentiment: {mean:+.3f}  ({len(scores)} posts across {len(subs)} subreddits)")
    return mean


def build_sentiment_series(
    df: pd.DataFrame,
    coin_id: str,
    scenario: str,
    fng_df: pd.DataFrame,
    news_score: float = None,
    x_score: float = None,
) -> pd.DataFrame:
    """
    Merge Fear & Greed into the price dataframe.
    Compute today's composite sentiment from all sources.
    Returns df with new columns: fng_norm, sentiment_composite.
    """
    coin_keywords = {
        "bitcoin":  ["bitcoin", "btc", "$btc"],
        "ethereum": ["ethereum", "eth", "$eth"],
    }.get(coin_id, [coin_id])

    # Scores passed in from run_coin to avoid double-scraping;
    # if called standalone, fetch here.
    if news_score is None:
        news_score = fetch_news_sentiment(coin_keywords)
    if x_score is None:
        x_score = fetch_reddit_sentiment(coin_id)

    # Composite today (weighted average)
    w = SCENARIO_PARAMS[scenario]["sentiment_weight"]
    if scenario == "bullish":
        # clip negative scores, amplify positive
        news_score = max(news_score, 0) * w
        x_score    = max(x_score, 0) * w
    elif scenario == "bearish":
        # clip positive scores, amplify negative
        news_score = min(news_score, 0) * w
        x_score    = min(x_score, 0) * w

    today_composite = float(np.mean([s for s in [news_score, x_score] if s != 0.0] or [0.0]))
    print(f"  Today composite sentiment: {today_composite:+.3f}  ({SCENARIO_PARAMS[scenario]['label']})")

    # Merge historical F&G into price df
    if not fng_df.empty:
        df = df.merge(fng_df[["date", "fng_norm"]], on="date", how="left")
        df["fng_norm"] = df["fng_norm"].ffill().fillna(0.0)
    else:
        df["fng_norm"] = 0.0

    # sentiment_composite: historical uses F&G; today patch with live composite
    df["sentiment_composite"] = df["fng_norm"].copy()
    # Blend today's live score with last historical F&G
    if len(df) > 0:
        df.loc[df.index[-1], "sentiment_composite"] = today_composite

    return df, today_composite


# ═══════════════════════════════════════════════════════════════════════════
# 3. TECHNICAL INDICATORS
# ═══════════════════════════════════════════════════════════════════════════

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    c = df["close"]

    for w in [7, 14, 30, 90, 200]:
        df[f"ma_{w}"] = c.rolling(w).mean()
    for w in [12, 26]:
        df[f"ema_{w}"] = c.ewm(span=w, adjust=False).mean()

    df["macd"]        = df["ema_12"] - df["ema_26"]
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"]   = df["macd"] - df["macd_signal"]

    delta = c.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / loss.replace(0, np.nan)
    df["rsi"] = 100 - (100 / (1 + rs))

    df["bb_mid"]   = c.rolling(20).mean()
    std20          = c.rolling(20).std()
    df["bb_upper"] = df["bb_mid"] + 2 * std20
    df["bb_lower"] = df["bb_mid"] - 2 * std20
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_mid"]

    df["log_ret"]    = np.log(c / c.shift(1))
    df["volatility"] = df["log_ret"].rolling(14).std() * np.sqrt(365)
    df["volume_change"] = df["volume"].pct_change()

    return df


# ═══════════════════════════════════════════════════════════════════════════
# 4. PROPHET MODEL
# ═══════════════════════════════════════════════════════════════════════════

def run_prophet(
    df: pd.DataFrame,
    horizon_days: int,
    scenario: str,
    today_sentiment: float,
) -> pd.DataFrame:
    """
    Fit Prophet on log-price with volume, volatility, and sentiment regressors.
    Returns forecast df with original-scale yhat/yhat_lower/yhat_upper.
    """
    params = SCENARIO_PARAMS[scenario]

    train = df[["date", "close", "volume", "volatility", "sentiment_composite"]].dropna().copy()
    train = train.rename(columns={"date": "ds", "close": "y"})
    train["y"] = np.log(train["y"])

    # Normalise continuous regressors
    reg_stats = {}
    for col in ["volume", "volatility"]:
        mu, sigma = train[col].mean(), train[col].std()
        reg_stats[col] = (mu, sigma if sigma else 1)
        train[col] = (train[col] - mu) / reg_stats[col][1]

    # sentiment_composite is already in [-1,+1]
    if scenario == "bullish":
        train["sentiment_composite"] = train["sentiment_composite"].clip(lower=0)
    elif scenario == "bearish":
        train["sentiment_composite"] = train["sentiment_composite"].clip(upper=0)

    m = Prophet(
        changepoint_prior_scale=params["changepoint_prior_scale"],
        seasonality_prior_scale=params["seasonality_prior_scale"],
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        interval_width=0.80,
    )
    m.add_regressor("volume")
    m.add_regressor("volatility")
    m.add_regressor("sentiment_composite")
    m.fit(train, iter=500)

    future = m.make_future_dataframe(periods=horizon_days)

    # Forward-fill regressors
    last_vol  = (train["volume"].iloc[-1] * reg_stats["volume"][1]) + reg_stats["volume"][0]
    last_vola = (train["volatility"].iloc[-1] * reg_stats["volatility"][1]) + reg_stats["volatility"][0]

    hist_idx = train.set_index("ds")
    hist_idx = hist_idx[~hist_idx.index.duplicated(keep="last")]
    for col in ["volume", "volatility", "sentiment_composite"]:
        future[col] = hist_idx[col].reindex(future["ds"]).values

    # Fill future dates
    future["volume"]     = future["volume"].fillna(
        (train["volume"].iloc[-1])
    )
    future["volatility"] = future["volatility"].fillna(
        (train["volatility"].iloc[-1])
    )
    # Future sentiment: use today's live composite
    sent_future = today_sentiment
    if scenario == "bullish":
        sent_future = max(sent_future, 0)
    elif scenario == "bearish":
        sent_future = min(sent_future, 0)
    future["sentiment_composite"] = future["sentiment_composite"].fillna(sent_future)

    forecast = m.predict(future)
    for col in ["yhat", "yhat_lower", "yhat_upper"]:
        forecast[col] = np.exp(forecast[col])

    return forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].rename(columns={"ds": "date"})


def build_forecasts(df: pd.DataFrame, scenario: str, today_sentiment: float) -> dict:
    forecasts = {}
    for label, days in HORIZONS.items():
        print(f"    Training Prophet ({label}, {SCENARIO_PARAMS[scenario]['label']})...")
        forecasts[label] = run_prophet(df, days, scenario, today_sentiment)
    return forecasts


# ═══════════════════════════════════════════════════════════════════════════
# 5. TERMINAL SUMMARY
# ═══════════════════════════════════════════════════════════════════════════

def print_summary(df, forecasts, coin_id, current_price, scenario, today_sentiment, fng_df):
    sp = SCENARIO_PARAMS[scenario]
    print(f"\n{'═'*58}")
    print(f"  {coin_id.upper()}  —  ${current_price:,.2f}  |  Scenario: {sp['label']}")
    print(f"{'═'*58}")

    last = df.iloc[-1]
    rsi_flag = "⚠ Overbought" if last["rsi"] > 70 else ("⚠ Oversold" if last["rsi"] < 30 else "✓ Neutral")
    macd_flag = "▲ Bullish" if last["macd_hist"] > 0 else "▼ Bearish"

    print(f"  RSI (14):     {last['rsi']:.1f}  {rsi_flag}")
    print(f"  MACD hist:    {last['macd_hist']:.2f}  {macd_flag}")
    print(f"  Volatility:   {last['volatility']*100:.1f}% (14d annualised)")
    print(f"  MA 30/90/200: ${last['ma_30']:,.0f} / ${last['ma_90']:,.0f} / ${last['ma_200']:,.0f}")

    if not fng_df.empty:
        latest_fng = fng_df.iloc[-1]
        print(f"  Fear & Greed: {latest_fng['fng']:.0f} — {latest_fng['value_classification']}")
    print(f"  Live sentiment (composite): {today_sentiment:+.3f}")

    print()
    for label, fc in forecasts.items():
        future_fc = fc[fc["date"] > df["date"].max()]
        if future_fc.empty:
            continue
        t = future_fc.iloc[-1]
        chg = (t["yhat"] - current_price) / current_price * 100
        print(f"  {label} forecast:  ${t['yhat']:>11,.2f}  ({chg:+.1f}%)")
        print(f"              range:  ${t['yhat_lower']:,.0f} – ${t['yhat_upper']:,.0f}")
    print()


# ═══════════════════════════════════════════════════════════════════════════
# 6. PLOTLY DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════

FC_COLORS = {"7d": "#22c55e", "30d": "#eab308", "90d": "#ef4444"}


def hex_to_rgba(hex_color: str, alpha: float = 0.16) -> str:
    """Convert #rrggbb to rgba(r,g,b,alpha) for Plotly compatibility."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"

def _to_date_str(series) -> list:
    """Convert any date series to plain YYYY-MM-DD strings for Plotly."""
    return pd.to_datetime(series).dt.strftime("%Y-%m-%d").tolist()


def build_chart(
    df: pd.DataFrame,
    forecasts: dict,
    coin_id: str,
    current_price: float,
    scenario: str,
    fng_df: pd.DataFrame,
    horizon: str = "90d",   # "7d", "30d", or "90d"
) -> go.Figure:
    """Returns a single Plotly figure for the given horizon."""
    primary = COLORS.get(coin_id, {"primary": "#888"})["primary"]
    name    = coin_id.capitalize()
    sp      = SCENARIO_PARAMS[scenario]
    today_str = pd.Timestamp.now().strftime("%Y-%m-%d")

    # How many days of history to show per horizon
    history_days = {"7d": 14, "30d": 60, "90d": 90}
    horizon_days = {"7d": 7,  "30d": 30, "90d": 90}
    fc_label     = {"7d": "1-Week Forecast", "30d": "1-Month Forecast", "90d": "3-Month Forecast"}
    fc_color     = {"7d": "#22c55e", "30d": "#f59e0b", "90d": "#a78bfa"}
    hist_window  = history_days[horizon]

    # ── History slice ─────────────────────────────────────────────────────
    df2 = df.copy()
    df2["close"] = pd.to_numeric(df2["close"], errors="coerce")
    df2["x"]    = pd.to_datetime(df2["date"]).dt.strftime("%Y-%m-%d")
    df2 = df2.dropna(subset=["close"]).sort_values("x").reset_index(drop=True)
    hist = df2.tail(hist_window).copy()

    last_x     = hist["x"].iloc[-1]
    last_price = float(hist["close"].iloc[-1])

    fig = go.Figure()

    # Actual price line
    fig.add_trace(go.Scatter(
        x=hist["x"].tolist(),
        y=hist["close"].astype(float).tolist(),
        name="Actual Price",
        mode="lines",
        line=dict(color=primary, width=3),
        hovertemplate="<b>%{x}</b><br>$%{y:,.2f}<extra></extra>",
    ))

    # Moving average (use shorter MA for shorter horizons)
    ma_col = "ma_7" if horizon == "7d" else "ma_30"
    if ma_col in hist.columns:
        ma_hist = hist.dropna(subset=[ma_col])
        if not ma_hist.empty:
            ma_label = "7-Day Average" if horizon == "7d" else "30-Day Average"
            fig.add_trace(go.Scatter(
                x=ma_hist["x"].tolist(),
                y=ma_hist[ma_col].astype(float).tolist(),
                name=ma_label,
                mode="lines",
                line=dict(color="rgba(255,255,255,0.35)", width=1.5, dash="dot"),
                hovertemplate=f"{ma_label}: $%{{y:,.2f}}<extra></extra>",
            ))

    # Only the relevant forecast
    fc = forecasts.get(horizon)
    if fc is not None:
        fc2 = fc.copy()
        fc2["x"] = pd.to_datetime(fc2["date"]).dt.strftime("%Y-%m-%d")
        future = fc2[fc2["x"] > last_x].copy()
        if not future.empty:
            c   = fc_color[horizon]
            lbl = fc_label[horizon]
            yhat    = future["yhat"].astype(float).tolist()
            yhat_up = future["yhat_upper"].astype(float).tolist()
            yhat_lo = future["yhat_lower"].astype(float).tolist()
            fut_xs  = future["x"].tolist()

            # Confidence band
            fig.add_trace(go.Scatter(
                x=fut_xs + fut_xs[::-1],
                y=yhat_up + yhat_lo[::-1],
                fill="toself", fillcolor=hex_to_rgba(c, 0.15),
                line=dict(color="rgba(0,0,0,0)"),
                showlegend=False, hoverinfo="skip",
            ))
            # Upper / lower bound lines (subtle)
            fig.add_trace(go.Scatter(
                x=fut_xs, y=yhat_up,
                name="Upper bound", mode="lines",
                line=dict(color=c, width=1, dash="dot"), opacity=0.5,
            ))
            fig.add_trace(go.Scatter(
                x=fut_xs, y=yhat_lo,
                name="Lower bound", mode="lines",
                line=dict(color=c, width=1, dash="dot"), opacity=0.5,
            ))
            # Forecast line
            end_price = float(future["yhat"].iloc[-1])
            chg = (end_price - current_price) / current_price * 100
            fig.add_trace(go.Scatter(
                x=[last_x] + fut_xs,
                y=[last_price] + yhat,
                name=f"{lbl}  ({chg:+.1f}%)",
                mode="lines",
                line=dict(color=c, width=3, dash="dash"),
                hovertemplate=f"<b>%{{x}}</b><br>{lbl}: $%{{y:,.2f}}<extra></extra>",
            ))

    # TODAY line
    fig.add_vline(x=today_str, line_dash="dash",
                  line_color="rgba(255,255,255,0.3)", line_width=2)

    # Y-axis: fit history + forecast yhat only (no extreme lower bounds in range)
    hist_prices = hist["close"].astype(float).dropna().tolist()
    fc_yhat = []
    if fc is not None:
        fc2 = fc.copy()
        fc2["x"] = pd.to_datetime(fc2["date"]).dt.strftime("%Y-%m-%d")
        future = fc2[fc2["x"] > last_x]
        fc_yhat = future["yhat"].astype(float).tolist()
    combined = hist_prices + fc_yhat
    if combined:
        p_min, p_max = min(combined), max(combined)
        pad = (p_max - p_min) * 0.12
        y_range = [p_min - pad, p_max + pad]
    else:
        y_range = None

    horizon_label = {"7d": "7-Day", "30d": "30-Day", "90d": "90-Day"}[horizon]
    scenario_badge = {"base": "Balanced", "bullish": "Optimistic", "bearish": "Cautious"}[scenario]
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0a0f1e",
        plot_bgcolor="#0d1526",
        title=dict(
            text=(
                f"<b>{name}</b>  ·  {sp['label']} ({scenario_badge})"
                f"  —  {horizon_label} Forecast"
            ),
            font=dict(size=16, color="#f1f5f9"), x=0.5,
        ),
        height=520,
        autosize=True,
        hovermode="x unified",
        legend=dict(orientation="h", y=-0.28, x=0, font=dict(size=12), bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=60, r=20, t=60, b=150),
        yaxis=dict(title="Price (USD)", tickprefix="$", tickformat=",.0f",
                   gridcolor="rgba(255,255,255,0.05)", range=y_range),
        xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)", tickangle=-30),
    )
    return fig


def build_mood_html(fng_df: pd.DataFrame) -> str:
    """Standalone HTML bar chart for Fear & Greed — shared across all horizons."""
    if fng_df.empty:
        return ""
    fng = fng_df.copy()
    fng["x"] = pd.to_datetime(fng["date"]).dt.strftime("%Y-%m-%d")
    fng = fng.sort_values("x").tail(90)
    bars = ""
    for _, row in fng.iterrows():
        v   = float(row["fng"])
        clr = "#22c55e" if v >= 60 else ("#ef4444" if v <= 40 else "#f59e0b")
        bars += (
            f'<div title="{row["x"]}: {v:.0f}/100" '
            f'style="flex:1;height:{v}%;background:{clr};opacity:0.85;'
            f'min-width:2px;border-radius:1px 1px 0 0"></div>'
        )
    return f"""
<div style="background:#0d1526;border-radius:8px;padding:16px 20px;margin-top:12px;
            font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif">
  <div style="font-size:13px;color:#94a3b8;margin-bottom:10px;font-weight:600">
    📊 MARKET MOOD (FEAR &amp; GREED)  —  Last 90 Days
    <span style="float:right;font-size:12px">
      <span style="color:#ef4444">■ Fear (&lt;40)</span>&nbsp;&nbsp;
      <span style="color:#f59e0b">■ Neutral</span>&nbsp;&nbsp;
      <span style="color:#22c55e">■ Greed (&gt;60)</span>
    </span>
  </div>
  <div style="display:flex;align-items:flex-end;height:80px;gap:1px;
              border-bottom:1px solid #1e293b;padding-bottom:2px">
    {bars}
  </div>
  <div style="display:flex;justify-content:space-between;
              font-size:11px;color:#475569;margin-top:4px">
    <span>{fng['x'].iloc[0]}</span>
    <span style="color:#94a3b8">Today: {float(fng['fng'].iloc[-1]):.0f}/100 — {fng['value_classification'].iloc[-1]}</span>
    <span>{fng['x'].iloc[-1]}</span>
  </div>
</div>"""


# ═══════════════════════════════════════════════════════════════════════════
# 7. EXPLANATION GENERATOR
# ═══════════════════════════════════════════════════════════════════════════

def _tag(label: str, value: str, color: str) -> str:
    return (
        f'<span style="background:{color}22;border:1px solid {color}66;'
        f'color:{color};border-radius:4px;padding:2px 8px;font-size:12px;'
        f'font-weight:600;margin-right:6px">{label}: {value}</span>'
    )


def _pill(text: str, color: str) -> str:
    return (
        f'<span style="background:{color};color:#0f172a;border-radius:12px;'
        f'padding:3px 10px;font-size:12px;font-weight:700;margin:2px">{text}</span>'
    )


def _section(title: str, body: str) -> str:
    return f"""
    <div style="margin-bottom:44px">
      <h3 style="color:#94a3b8;font-size:16px;text-transform:uppercase;
                 letter-spacing:1.2px;margin:0 0 18px 0;border-bottom:1px solid #1e293b;
                 padding-bottom:12px">{title}</h3>
      <div style="color:#cbd5e1;line-height:1.85;font-size:15.5px">{body}</div>
    </div>"""


def generate_explanation_data(
    df: pd.DataFrame,
    forecasts: dict,
    coin_id: str,
    current_price: float,
    scenario: str,
    today_sentiment: float,
    fng_df: pd.DataFrame,
    news_score: float = 0.0,
    x_score: float = 0.0,
) -> dict:
    """Return structured explanation data for native Streamlit rendering."""
    sp   = SCENARIO_PARAMS[scenario]
    name = coin_id.capitalize()
    last = df.dropna(subset=["rsi", "macd_hist", "ma_30"]).iloc[-1]

    price_vs_ma30  = (current_price - last["ma_30"])  / last["ma_30"]  * 100
    price_vs_ma90  = (current_price - last["ma_90"])  / last["ma_90"]  * 100
    price_vs_ma200 = (current_price - last["ma_200"]) / last["ma_200"] * 100

    def vs_label(pct):
        if pct > 5:   return f"well above ({pct:+.1f}%)"
        if pct > 0:   return f"slightly above ({pct:+.1f}%)"
        if pct > -5:  return f"slightly below ({pct:+.1f}%)"
        return f"well below ({pct:+.1f}%)"

    trend_short = "uptrend" if price_vs_ma30 > 0 else "downtrend"
    trend_long  = "above its long-term average" if price_vs_ma200 > 0 else "below its long-term average"
    ma_cross = "bullish golden-cross" if last["ma_30"] > last["ma_90"] else "bearish death-cross"
    bb_pos = (current_price - last["bb_lower"]) / (last["bb_upper"] - last["bb_lower"])
    bb_text = ("near the upper Bollinger Band (potential resistance)" if bb_pos > 0.85
               else "near the lower Bollinger Band (potential support/oversold)" if bb_pos < 0.15
               else f"mid-Bollinger Bands ({bb_pos*100:.0f}% of range, no extreme)")

    rsi = last["rsi"]
    rsi_zone = ("🔴 Overbought (>70)" if rsi > 70 else "🟢 Oversold (<30)" if rsi < 30
                else "🟢 Bullish neutral" if rsi > 55 else "🟠 Bearish neutral" if rsi < 45 else "⚪ Neutral")
    macd_bull = last["macd_hist"] > 0
    vol_pct = last["volatility"] * 100

    fng_val = float(fng_df["fng"].iloc[-1]) if not fng_df.empty else 50.0
    fng_cls = fng_df["value_classification"].iloc[-1] if not fng_df.empty else "Unknown"

    forecasts_list = []
    chg_90 = 0.0
    for label, fc in forecasts.items():
        future = fc[fc["date"] > df["date"].max()]
        if future.empty:
            continue
        t = future.iloc[-1]
        chg = (t["yhat"] - current_price) / current_price * 100
        if label == "90d":
            chg_90 = chg
        forecasts_list.append({
            "horizon": label,
            "target": t["yhat"],
            "change_pct": chg,
            "lower": t["yhat_lower"],
            "upper": t["yhat_upper"],
        })

    direction = "upward" if chg_90 > 0 else "downward"
    magnitude = "strongly" if abs(chg_90) > 25 else "moderately" if abs(chg_90) > 10 else "mildly"

    return {
        "name": name,
        "scenario_label": sp["label"],
        "current_price": current_price,
        "trend_short": trend_short,
        "trend_long": trend_long,
        "ma_cross": ma_cross,
        "bb_text": bb_text,
        "price_vs_ma30": price_vs_ma30,
        "price_vs_ma90": price_vs_ma90,
        "price_vs_ma200": price_vs_ma200,
        "vs_ma30_label": vs_label(price_vs_ma30),
        "vs_ma90_label": vs_label(price_vs_ma90),
        "vs_ma200_label": vs_label(price_vs_ma200),
        "rsi": rsi,
        "rsi_zone": rsi_zone,
        "macd_hist": last["macd_hist"],
        "macd_bull": macd_bull,
        "volatility_pct": vol_pct,
        "fng_val": fng_val,
        "fng_cls": fng_cls,
        "news_score": news_score,
        "x_score": x_score,
        "today_sentiment": today_sentiment,
        "forecasts": forecasts_list,
        "direction": direction,
        "magnitude": magnitude,
        "scenario": scenario,
    }


def generate_explanation(
    df: pd.DataFrame,
    forecasts: dict,
    coin_id: str,
    current_price: float,
    scenario: str,
    today_sentiment: float,
    fng_df: pd.DataFrame,
    news_score: float = 0.0,
    x_score: float = 0.0,
) -> str:
    """
    Build a self-contained HTML explanation block that describes WHY
    the model predicts what it predicts, drawing on every data source.
    """
    sp   = SCENARIO_PARAMS[scenario]
    name = coin_id.capitalize()
    last = df.dropna(subset=["rsi", "macd_hist", "ma_30"]).iloc[-1]
    now  = datetime.now().strftime("%B %d, %Y at %H:%M UTC")

    sc_color = sp["color"]

    # ── Trend analysis ────────────────────────────────────────────────────
    price_vs_ma30  = (current_price - last["ma_30"])  / last["ma_30"]  * 100
    price_vs_ma90  = (current_price - last["ma_90"])  / last["ma_90"]  * 100
    price_vs_ma200 = (current_price - last["ma_200"]) / last["ma_200"] * 100

    def vs_label(pct):
        if pct > 5:   return f"well above ({pct:+.1f}%)"
        if pct > 0:   return f"slightly above ({pct:+.1f}%)"
        if pct > -5:  return f"slightly below ({pct:+.1f}%)"
        return f"well below ({pct:+.1f}%)"

    trend_short = "uptrend" if price_vs_ma30 > 0 else "downtrend"
    trend_long  = "above its long-term average" if price_vs_ma200 > 0 else "below its long-term average"

    trend_body = (
        f"{name} is currently priced at <strong>${current_price:,.2f}</strong>, "
        f"sitting {vs_label(price_vs_ma30)} its 30-day MA, "
        f"{vs_label(price_vs_ma90)} its 90-day MA, and "
        f"{vs_label(price_vs_ma200)} its 200-day MA. "
        f"This places {name} in a short-term <strong>{trend_short}</strong> and "
        f"{trend_long} structurally."
    )

    # MA crossover signal
    if last["ma_30"] > last["ma_90"]:
        trend_body += (
            " The 30-day MA is <strong>above</strong> the 90-day MA — a bullish golden-cross structure."
        )
    else:
        trend_body += (
            " The 30-day MA is <strong>below</strong> the 90-day MA — a bearish death-cross structure."
        )

    # Bollinger band context
    bb_pos = (current_price - last["bb_lower"]) / (last["bb_upper"] - last["bb_lower"])
    if bb_pos > 0.85:
        trend_body += " Price is near the <strong>upper Bollinger Band</strong>, suggesting potential short-term resistance or overbought conditions."
    elif bb_pos < 0.15:
        trend_body += " Price is near the <strong>lower Bollinger Band</strong>, suggesting potential support or oversold bounce territory."
    else:
        trend_body += f" Price sits in the <strong>middle of the Bollinger Bands</strong> ({bb_pos*100:.0f}% of the range), indicating no extreme deviation."

    # ── Momentum ──────────────────────────────────────────────────────────
    rsi = last["rsi"]
    if rsi > 70:
        rsi_text = f"RSI of <strong>{rsi:.1f}</strong> is in overbought territory (&gt;70). Momentum is strong but a pullback or consolidation is historically more likely from this level."
        rsi_tag  = _tag("RSI", f"{rsi:.1f}", "#ef4444")
    elif rsi < 30:
        rsi_text = f"RSI of <strong>{rsi:.1f}</strong> is in oversold territory (&lt;30). Selling pressure has been extreme; a relief rally is historically more probable from this zone."
        rsi_tag  = _tag("RSI", f"{rsi:.1f}", "#22c55e")
    elif rsi > 55:
        rsi_text = f"RSI of <strong>{rsi:.1f}</strong> is in bullish neutral territory. Momentum favors buyers without being stretched."
        rsi_tag  = _tag("RSI", f"{rsi:.1f}", "#22c55e")
    elif rsi < 45:
        rsi_text = f"RSI of <strong>{rsi:.1f}</strong> is in bearish neutral territory. Momentum leans toward sellers but is not yet extreme."
        rsi_tag  = _tag("RSI", f"{rsi:.1f}", "#f97316")
    else:
        rsi_text = f"RSI of <strong>{rsi:.1f}</strong> is neutral (45–55). No dominant momentum signal — the market is in balance."
        rsi_tag  = _tag("RSI", f"{rsi:.1f}", "#94a3b8")

    macd_bull = last["macd_hist"] > 0
    macd_text = (
        f"MACD histogram is <strong>{'positive' if macd_bull else 'negative'} ({last['macd_hist']:+.2f})</strong>, "
        f"confirming {'bullish' if macd_bull else 'bearish'} momentum. "
        f"The MACD line ({'above' if last['macd'] > last['macd_signal'] else 'below'} signal) "
        f"{'supports continued upward' if macd_bull else 'warns of continued downward'} pressure."
    )
    macd_tag = _tag("MACD", f"{'▲' if macd_bull else '▼'} {last['macd_hist']:+.2f}", "#22c55e" if macd_bull else "#ef4444")

    vol_pct = last["volatility"] * 100
    vol_text = (
        f"14-day annualised volatility is <strong>{vol_pct:.1f}%</strong>. "
        + ("This is elevated, widening forecast confidence intervals." if vol_pct > 80
           else "This is moderate, giving the model reasonable confidence in its range." if vol_pct > 40
           else "This is relatively low, suggesting a tighter forecast range.")
    )

    momentum_body = f"{rsi_tag}{macd_tag}<br><br>{rsi_text}<br><br>{macd_text}<br><br>{vol_text}"

    # ── Sentiment ─────────────────────────────────────────────────────────
    fng_val  = float(fng_df["fng"].iloc[-1])  if not fng_df.empty else 50.0
    fng_cls  = fng_df["value_classification"].iloc[-1] if not fng_df.empty else "Unknown"

    def fng_color(v):
        if v >= 60: return "#22c55e"
        if v <= 40: return "#ef4444"
        return "#eab308"

    fng_tag   = _tag("Fear & Greed", f"{fng_val:.0f} — {fng_cls}", fng_color(fng_val))
    news_tag  = _tag("News", f"{news_score:+.2f}", "#22c55e" if news_score > 0.05 else "#ef4444" if news_score < -0.05 else "#94a3b8")
    x_tag     = _tag("Reddit", f"{x_score:+.2f}", "#22c55e" if x_score > 0.05 else "#ef4444" if x_score < -0.05 else "#94a3b8")
    comp_tag  = _tag("Composite", f"{today_sentiment:+.2f}", "#22c55e" if today_sentiment > 0.05 else "#ef4444" if today_sentiment < -0.05 else "#94a3b8")

    def sentiment_prose(score, source):
        if score > 0.2:  return f"{source} is <strong>strongly positive</strong> ({score:+.2f})"
        if score > 0.05: return f"{source} leans <strong>mildly positive</strong> ({score:+.2f})"
        if score < -0.2: return f"{source} is <strong>strongly negative</strong> ({score:+.2f})"
        if score < -0.05:return f"{source} leans <strong>mildly negative</strong> ({score:+.2f})"
        return f"{source} is <strong>neutral</strong> ({score:+.2f})"

    sentiment_body = (
        f"{fng_tag}{news_tag}{x_tag}{comp_tag}<br><br>"
        f"The <strong>Fear & Greed Index</strong> is at <strong>{fng_val:.0f} ({fng_cls})</strong>. "
        + (
            "Historically, extreme fear has coincided with market bottoms and attractive buying opportunities." if fng_val <= 25 else
            "Extreme greed has historically preceded corrections; late-cycle caution is warranted." if fng_val >= 75 else
            "Greed is moderately elevated — the market is optimistic but not at historically dangerous levels." if fng_val >= 55 else
            "Fear is moderately elevated — uncertainty is present but not extreme." if fng_val <= 45 else
            "The market is in a neutral sentiment zone, with no strong crowd bias detected."
        )
        + f"<br><br>{sentiment_prose(news_score, 'Crypto news coverage')}. "
        + f"{sentiment_prose(x_score, 'Reddit community sentiment')}. "
        + f"The composite sentiment score is <strong>{today_sentiment:+.3f}</strong>, "
        + ("which <strong>supports the bullish scenario</strong> weighting." if today_sentiment > 0.1 else
           "which <strong>supports the bearish scenario</strong> weighting." if today_sentiment < -0.1 else
           "which is broadly neutral and does not strongly favour either direction.")
    )

    # ── Forecasts ─────────────────────────────────────────────────────────
    forecast_rows = ""
    for label, fc in forecasts.items():
        future = fc[fc["date"] > df["date"].max()]
        if future.empty:
            continue
        t    = future.iloc[-1]
        chg  = (t["yhat"] - current_price) / current_price * 100
        clr  = "#22c55e" if chg > 0 else "#ef4444"
        forecast_rows += f"""
        <tr>
          <td style="padding:8px 12px;color:#94a3b8;font-weight:600">{label}</td>
          <td style="padding:8px 12px">${t['yhat']:,.2f}</td>
          <td style="padding:8px 12px;color:{clr};font-weight:700">{chg:+.1f}%</td>
          <td style="padding:8px 12px;color:#64748b">${t['yhat_lower']:,.0f} – ${t['yhat_upper']:,.0f}</td>
          <td style="padding:8px 12px;color:#64748b">
            {'Mild move; inside normal volatility range.' if abs(chg) < 10 else
             'Significant move; would confirm current trend.' if abs(chg) < 25 else
             'Large move; requires sustained catalyst to materialise.'}
          </td>
        </tr>"""

    # Overall forecast narrative
    chg_90 = 0.0
    if "90d" in forecasts:
        fut = forecasts["90d"][forecasts["90d"]["date"] > df["date"].max()]
        if not fut.empty:
            chg_90 = (fut.iloc[-1]["yhat"] - current_price) / current_price * 100

    direction = "upward" if chg_90 > 0 else "downward"
    magnitude = "strongly" if abs(chg_90) > 25 else "moderately" if abs(chg_90) > 10 else "mildly"

    forecast_body = f"""
    <table style="width:100%;border-collapse:collapse;font-size:13px">
      <thead>
        <tr style="border-bottom:1px solid #1e293b;color:#64748b">
          <th style="padding:8px 12px;text-align:left">Horizon</th>
          <th style="padding:8px 12px;text-align:left">Target Price</th>
          <th style="padding:8px 12px;text-align:left">Change</th>
          <th style="padding:8px 12px;text-align:left">80% Range</th>
          <th style="padding:8px 12px;text-align:left">Context</th>
        </tr>
      </thead>
      <tbody>{forecast_rows}</tbody>
    </table>
    <br>
    <p>The model projects a <strong>{magnitude} {direction}</strong> trajectory over the next 90 days under the <strong>{sp['label']}</strong> scenario.
    {"The bullish scenario amplifies positive sentiment signals and allows the model more flexibility to follow upward price trends." if scenario == "bullish" else
     "The bearish scenario amplifies negative sentiment signals and constrains the model toward mean reversion and downside risks." if scenario == "bearish" else
     "The base scenario applies no directional bias — the forecast reflects what the data and historical patterns suggest without adjustment."}
    </p>
    """

    # ── Scenario rationale ────────────────────────────────────────────────
    scenario_body = f"""
    <div style="border-left:3px solid {sc_color};padding-left:14px">
    {"<p>The <strong>Bullish scenario</strong> is chosen when the weight of evidence tilts positive: upward price trend, RSI not yet overbought, MACD rising, Fear &amp; Greed moving toward greed, and positive social sentiment. Under this mode, Prophet's changepoint flexibility is increased so the model can follow upward momentum rather than snap back to a mean. Positive sentiment scores from X and news are amplified 1.5× as regressors, pulling forecasts higher.</p>" if scenario == "bullish" else
     "<p>The <strong>Bearish scenario</strong> is chosen when evidence tilts negative: price below key MAs, RSI declining or overbought with divergence, MACD histogram negative, Fear &amp; Greed in fear territory, and negative social sentiment. Under this mode, the model is anchored more tightly, resisting upside momentum and letting negative sentiment pull forecasts lower. Negative sentiment scores are amplified 1.5×.</p>" if scenario == "bearish" else
     "<p>The <strong>Base scenario</strong> is the neutral benchmark. No directional bias is applied. Sentiment is used at face value (1× weight). Changepoint flexibility is set to Prophet's recommended default. This scenario is the most statistically honest — it describes what the model expects if the market continues following its historical seasonal and trend patterns without outsized external events.</p>"}
    <p style="color:#64748b;font-size:13px">
      <strong>Important:</strong> Crypto markets are highly unpredictable. These forecasts are generated from historical patterns and current sentiment — they are research tools, not financial advice. Always use multiple signals and your own judgment before making any decisions.
    </p>
    </div>"""

    # ── Assemble full HTML block ───────────────────────────────────────────
    html = f"""
<div style="font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
            background:#0f172a;color:#e2e8f0;padding:28px 32px;
            border-radius:12px;margin-top:24px;border:1px solid #1e293b">

  <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:24px">
    <div>
      <h2 style="margin:0;font-size:20px;color:#f1f5f9">
        {name} — {sp['label']} Scenario Analysis
      </h2>
      <p style="margin:4px 0 0 0;color:#64748b;font-size:13px">
        Generated {now} · Data: Yahoo Finance, Fear &amp; Greed Index, News RSS, Reddit
      </p>
    </div>
    <div style="text-align:right">
      {_pill(f"${current_price:,.2f}", sc_color)}
      {_pill(f"{chg_90:+.1f}% (90d)", sc_color)}
    </div>
  </div>

  {_section("📈 Trend & Price Structure", trend_body)}
  {_section("⚡ Momentum Indicators", momentum_body)}
  {_section("🌐 Sentiment (Fear & Greed · News · Reddit)", sentiment_body)}
  {_section("🔮 Forecast Summary", forecast_body)}
  {_section("💡 Why This Scenario?", scenario_body)}

</div>
"""
    return html


def build_combined_dashboard(panels: dict, out_path: str):
    """
    Assemble all coin × scenario panels into one HTML file.
    Each panel contains 3 horizon chart divs (7d/30d/90d) toggled by a selector.
    """
    now = datetime.now().strftime("%B %d, %Y at %H:%M UTC")
    coins     = [c for c in panels if panels[c]]
    scenarios = list(panels[coins[0]].keys()) if coins else []
    horizons  = ["7d", "30d", "90d"]

    if not coins or not scenarios:
        print("  ✗ No panels to render — dashboard not saved.")
        return

    coin_labels     = {"bitcoin": "₿ Bitcoin (BTC)", "ethereum": "Ξ Ethereum (ETH)"}
    scenario_labels = {"base": "⚪ Base", "bullish": "🟢 Bullish", "bearish": "🔴 Bearish"}
    scenario_colors = {"base": "#94a3b8", "bullish": "#22c55e", "bearish": "#ef4444"}
    horizon_labels  = {"7d": "7 Days", "30d": "30 Days", "90d": "90 Days"}

    # ── Panel divs ────────────────────────────────────────────────────────
    panel_divs = ""
    first_panel = True
    for coin in coins:
        for sc in scenarios:
            if sc not in panels[coin]:
                continue
            panel_id  = f"panel_{coin}_{sc}"
            p_visible = "block" if first_panel else "none"
            first_panel = False
            data = panels[coin][sc]

            # Build the 3 horizon chart divs inside this panel
            horizon_divs = ""
            for i, hz in enumerate(horizons):
                ch_html  = data["charts_html"].get(hz, "")
                hz_visible = "block" if i == 0 else "none"
                horizon_divs += f"""
          <div id="hz_{coin}_{sc}_{hz}" class="hz-chart" style="display:{hz_visible}">
            {ch_html}
          </div>"""

            panel_divs += f"""
      <div id="{panel_id}" class="panel" style="display:{p_visible}">
        {horizon_divs}
        {data['mood_html']}
        {data['explanation_html']}
      </div>"""

    # ── Selector buttons ──────────────────────────────────────────────────
    coin_tabs = ""
    for i, coin in enumerate(coins):
        active   = "coin-tab-active" if i == 0 else ""
        first_sc = next(iter(panels[coin]), None)
        price    = panels[coin][first_sc]["price"] if first_sc else 0.0
        coin_tabs += f"""
        <button class="coin-tab {active}" onclick="selectCoin('{coin}')" data-coin="{coin}">
          {coin_labels.get(coin, coin.capitalize())}
          <span class="price-badge" id="price_{coin}">${price:,.2f}</span>
        </button>"""

    sc_buttons = ""
    for i, sc in enumerate(scenarios):
        active = "sc-btn-active" if i == 0 else ""
        clr    = scenario_colors[sc]
        sc_buttons += f"""
        <button class="sc-btn {active}" onclick="selectScenario('{sc}')"
                data-scenario="{sc}" style="--sc-color:{clr}">
          {scenario_labels.get(sc, sc)}
        </button>"""

    hz_buttons = ""
    for i, hz in enumerate(horizons):
        active = "hz-btn-active" if i == 0 else ""
        hz_buttons += f"""
        <button class="hz-btn {active}" onclick="selectHorizon('{hz}')" data-horizon="{hz}">
          {horizon_labels[hz]}
        </button>"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Crypto Prediction Dashboard</title>
  <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
  <style>
    *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      background: #070d1a;
      color: #e2e8f0;
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
      min-height: 100vh;
    }}
    .header {{
      background: #0f172a;
      border-bottom: 1px solid #1e293b;
      padding: 16px 28px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      position: sticky;
      top: 0;
      z-index: 100;
    }}
    .header-title {{ font-size: 18px; font-weight: 700; color: #f1f5f9; letter-spacing: -0.3px; }}
    .header-sub   {{ font-size: 12px; color: #475569; margin-top: 2px; }}
    .selectors {{
      background: #0c1526;
      border-bottom: 1px solid #1e293b;
      padding: 12px 28px;
      display: flex;
      gap: 28px;
      align-items: center;
      flex-wrap: wrap;
    }}
    .selector-group {{ display: flex; align-items: center; gap: 8px; }}
    .selector-label {{
      font-size: 11px; font-weight: 600; text-transform: uppercase;
      letter-spacing: 1px; color: #475569; white-space: nowrap;
    }}
    .divider {{ width: 1px; height: 32px; background: #1e293b; }}

    /* Coin tabs */
    .coin-tab {{
      background: transparent; border: 1px solid #1e293b; color: #94a3b8;
      border-radius: 8px; padding: 7px 14px; font-size: 14px; font-weight: 600;
      cursor: pointer; display: flex; align-items: center; gap: 8px; transition: all 0.15s;
    }}
    .coin-tab:hover {{ border-color: #334155; color: #cbd5e1; }}
    .coin-tab-active {{ background: #1e3a5f; border-color: #3b82f6; color: #93c5fd; }}
    .price-badge {{
      background: #0f172a; border: 1px solid #1e293b; border-radius: 4px;
      padding: 2px 7px; font-size: 12px; color: #64748b; font-weight: 500;
    }}
    .coin-tab-active .price-badge {{ border-color: #3b82f6; color: #93c5fd; }}

    /* Scenario buttons */
    .sc-btn {{
      background: transparent; border: 1px solid #1e293b; color: #94a3b8;
      border-radius: 8px; padding: 7px 14px; font-size: 13px; font-weight: 600;
      cursor: pointer; transition: all 0.15s;
    }}
    .sc-btn:hover {{ border-color: var(--sc-color); color: var(--sc-color); }}
    .sc-btn-active {{
      border-color: var(--sc-color); color: var(--sc-color);
      background: color-mix(in srgb, var(--sc-color) 12%, transparent);
    }}

    /* Horizon buttons */
    .hz-btn {{
      background: transparent; border: 1px solid #1e293b; color: #94a3b8;
      border-radius: 8px; padding: 7px 14px; font-size: 13px; font-weight: 600;
      cursor: pointer; transition: all 0.15s;
    }}
    .hz-btn:hover {{ border-color: #60a5fa; color: #60a5fa; }}
    .hz-btn-active {{ background: #1e3a5f; border-color: #60a5fa; color: #60a5fa; }}

    .content {{ padding: 20px 24px 40px; max-width: 1600px; margin: 0 auto; }}
    .panel {{ display: none; }}
    .hz-chart {{ display: none; }}

    /* Help button */
    .help-btn {{
      background: transparent; border: 1px solid #1e293b; color: #64748b;
      border-radius: 8px; padding: 7px 14px; font-size: 13px; font-weight: 600;
      cursor: pointer; transition: all 0.15s; display: flex; align-items: center; gap: 6px;
    }}
    .help-btn:hover {{ border-color: #60a5fa; color: #60a5fa; }}

    /* Modal overlay */
    .modal-overlay {{
      display: none; position: fixed; inset: 0;
      background: rgba(0,0,0,0.75); z-index: 1000;
      align-items: center; justify-content: center;
    }}
    .modal-overlay.open {{ display: flex; }}
    .modal {{
      background: #0f172a; border: 1px solid #1e293b; border-radius: 16px;
      padding: 36px 40px; max-width: 720px; width: 90%; max-height: 85vh;
      overflow-y: auto; position: relative;
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }}
    .modal h2 {{
      color: #f1f5f9; font-size: 22px; margin: 0 0 6px 0;
    }}
    .modal .modal-sub {{
      color: #475569; font-size: 13px; margin-bottom: 28px;
    }}
    .modal h3 {{
      color: #60a5fa; font-size: 14px; text-transform: uppercase;
      letter-spacing: 1px; margin: 24px 0 10px 0;
      border-bottom: 1px solid #1e293b; padding-bottom: 8px;
    }}
    .modal p, .modal li {{
      color: #cbd5e1; font-size: 14px; line-height: 1.75; margin-bottom: 8px;
    }}
    .modal ul {{ padding-left: 20px; margin-bottom: 12px; }}
    .modal .close-btn {{
      position: absolute; top: 20px; right: 24px;
      background: transparent; border: none; color: #475569;
      font-size: 22px; cursor: pointer; line-height: 1;
    }}
    .modal .close-btn:hover {{ color: #f1f5f9; }}
    .modal .credit {{
      margin-top: 32px; padding-top: 20px; border-top: 1px solid #1e293b;
      color: #475569; font-size: 13px; text-align: center;
    }}
    .modal .credit strong {{ color: #94a3b8; }}

    /* Footer */
    .footer {{
      text-align: center; padding: 24px; color: #334155; font-size: 12px;
      border-top: 1px solid #0f172a;
    }}
    .footer strong {{ color: #475569; }}

    /* Make Plotly charts fill their container */
    .hz-chart > div, .js-plotly-plot, .plotly-graph-div {{
      width: 100% !important;
      max-width: 100% !important;
    }}

    /* ── Mobile responsive ── */
    @media (max-width: 640px) {{
      .header {{
        padding: 12px 16px;
        flex-direction: column;
        align-items: flex-start;
        gap: 8px;
      }}
      .header-title {{ font-size: 15px; }}
      .selectors {{
        padding: 10px 12px;
        gap: 10px;
        flex-direction: column;
        align-items: flex-start;
      }}
      .selector-group {{ flex-wrap: wrap; gap: 6px; }}
      .divider {{ display: none; }}
      .coin-tab {{ font-size: 13px; padding: 6px 10px; }}
      .sc-btn, .hz-btn {{ font-size: 12px; padding: 6px 10px; }}
      .price-badge {{ font-size: 11px; }}
      .content {{ padding: 12px 10px 32px; }}
      .modal {{ padding: 24px 18px; }}
      .modal h2 {{ font-size: 18px; }}
      .modal p, .modal li {{ font-size: 13px; }}
    }}
  </style>
</head>
<body>

  <!-- Help Modal -->
  <div class="modal-overlay" id="helpModal" onclick="if(event.target===this) closeHelp()">
    <div class="modal">
      <button class="close-btn" onclick="closeHelp()">✕</button>
      <h2>📖 How to Use This Dashboard</h2>
      <p class="modal-sub">A plain-English guide — no crypto knowledge required.</p>

      <h3>What is Cryptocurrency?</h3>
      <p>Cryptocurrency is digital money that exists only on the internet — no bank or government controls it. <strong>Bitcoin (BTC)</strong> is the most well-known, like digital gold. <strong>Ethereum (ETH)</strong> is the second largest, and powers many apps and digital contracts online. Their prices change constantly based on supply, demand, and public sentiment — just like stocks.</p>

      <h3>What Does This Dashboard Do?</h3>
      <p>It tracks the current price of Bitcoin and Ethereum and uses a mathematical model to predict where the price might go over the next 7, 30, or 90 days. It pulls data from multiple sources every morning to keep predictions fresh:</p>
      <ul>
        <li><strong>Yahoo Finance</strong> — real-time and historical price data</li>
        <li><strong>Fear &amp; Greed Index</strong> — measures how nervous or excited the overall crypto market is (0 = everyone is panicking, 100 = everyone is overconfident)</li>
        <li><strong>News headlines</strong> — scans crypto news sites for positive or negative coverage</li>
        <li><strong>Reddit</strong> — reads community sentiment from crypto forums</li>
      </ul>

      <h3>How to Read the Chart</h3>
      <ul>
        <li>The <strong>solid orange (or blue) line</strong> is the real historical price — what actually happened.</li>
        <li>The <strong>dashed line</strong> after "TODAY" is the model's best guess for the future price.</li>
        <li>The <strong>shaded band</strong> around the dashed line is the uncertainty range — the price will likely land somewhere inside it. A wider band means more uncertainty.</li>
        <li>The <strong>dotted white line</strong> is the recent price average — useful for seeing the general direction of the trend.</li>
      </ul>

      <h3>The Three Selectors</h3>
      <ul>
        <li><strong>Crypto</strong> — switch between Bitcoin (BTC) and Ethereum (ETH).</li>
        <li><strong>Scenario</strong> — choose how the model interprets sentiment:
          <ul>
            <li>⚪ <strong>Base</strong> — neutral, no bias. The model's honest best guess.</li>
            <li>🟢 <strong>Bullish</strong> — assumes positive sentiment is more meaningful. Shows the optimistic case.</li>
            <li>🔴 <strong>Bearish</strong> — assumes negative sentiment is more meaningful. Shows the cautious case.</li>
          </ul>
        </li>
        <li><strong>Horizon</strong> — how far ahead to predict. <strong>7 Days</strong> is the most reliable. <strong>90 Days</strong> is more speculative — the band will be much wider.</li>
      </ul>

      <h3>The Market Mood Bar</h3>
      <p>The bar chart below the price chart shows the <strong>Fear &amp; Greed Index</strong> over the last 90 days. <span style="color:#ef4444">Red bars</span> mean people were fearful (prices often bounce from here). <span style="color:#22c55e">Green bars</span> mean people were greedy (sometimes a warning sign). <span style="color:#f59e0b">Yellow</span> is neutral.</p>

      <h3>What is RSI?</h3>
      <p><strong>RSI (Relative Strength Index)</strong> is a number between 0 and 100 that measures how fast the price has been moving up or down recently. Think of it like a speedometer for buying and selling pressure:</p>
      <ul>
        <li><strong>Below 30 — Oversold:</strong> The price has dropped very fast and a lot of people have been selling. Historically, this is often a sign the price may bounce back up — sellers may be running out of steam.</li>
        <li><strong>Above 70 — Overbought:</strong> The price has risen very fast and a lot of people have been buying. This can be a warning sign that the price may pull back — buyers may be getting overexcited.</li>
        <li><strong>Between 30–70 — Neutral:</strong> Normal territory, no extreme signal in either direction.</li>
      </ul>

      <h3>What is MACD?</h3>
      <p><strong>MACD (Moving Average Convergence Divergence)</strong> is a tool that shows whether the recent price momentum is speeding up or slowing down. Don't worry about the name — here's all you need to know:</p>
      <ul>
        <li>If the <strong>MACD histogram is positive (above zero)</strong>, it means buying momentum is growing — a bullish signal.</li>
        <li>If the <strong>MACD histogram is negative (below zero)</strong>, it means selling momentum is growing — a bearish signal.</li>
        <li>The bigger the number (positive or negative), the stronger the momentum in that direction.</li>
      </ul>
      <p>Both RSI and MACD are used together with other signals — no single indicator tells the whole story.</p>

      <h3>The Analysis Section</h3>
      <p>Below the charts is a written breakdown explaining <em>why</em> the model predicts what it does — covering price trends, momentum (RSI &amp; MACD), sentiment from news and Reddit, and a forecast summary table.</p>

      <h3>Important Disclaimer</h3>
      <p style="color:#f59e0b">⚠ This dashboard is a <strong>research tool</strong>, not financial advice. Cryptocurrency prices are highly unpredictable. Never invest money you cannot afford to lose, and always do your own research before making any financial decisions.</p>

      <div class="credit">
        Created by <strong>G. Bennett Robble</strong><br>
        Powered by Yahoo Finance · Fear &amp; Greed Index · News RSS · Reddit · Prophet AI Model
      </div>
    </div>
  </div>

  <div class="header">
    <div>
      <div class="header-title">🔮 Crypto Prediction Dashboard</div>
      <div class="header-sub">BTC &amp; ETH · 7d / 30d / 90d forecasts · Updated {now}</div>
    </div>
    <div style="display:flex;align-items:center;gap:16px">
      <span style="font-size:12px;color:#334155">Data: Yahoo Finance · Fear &amp; Greed · News RSS · Reddit</span>
      <button class="help-btn" onclick="openHelp()">❓ How to Use</button>
    </div>
  </div>

  <div class="selectors">
    <div class="selector-group">
      <span class="selector-label">Crypto</span>
      {coin_tabs}
    </div>
    <div class="divider"></div>
    <div class="selector-group">
      <span class="selector-label">Scenario</span>
      {sc_buttons}
    </div>
    <div class="divider"></div>
    <div class="selector-group">
      <span class="selector-label">Horizon</span>
      {hz_buttons}
    </div>
  </div>

  <div class="content">
    {panel_divs}
  </div>

  <div class="footer">
    Created by <strong>G. Bennett Robble</strong> &nbsp;·&nbsp;
    Crypto Prediction Dashboard &nbsp;·&nbsp;
    Updated daily &nbsp;·&nbsp;
    <span style="color:#1e293b">Not financial advice</span>
  </div>

  <script>
    let activeCoin     = '{coins[0]}';
    let activeScenario = '{scenarios[0]}';
    let activeHorizon  = '7d';

    function showPanel() {{
      document.querySelectorAll('.panel').forEach(p => p.style.display = 'none');
      const panel = document.getElementById('panel_' + activeCoin + '_' + activeScenario);
      if (!panel) return;
      panel.style.display = 'block';

      // Show the right horizon chart, hide others in this panel
      panel.querySelectorAll('.hz-chart').forEach(d => d.style.display = 'none');
      const hzDiv = document.getElementById('hz_' + activeCoin + '_' + activeScenario + '_' + activeHorizon);
      if (hzDiv) {{
        hzDiv.style.display = 'block';
        hzDiv.querySelectorAll('.js-plotly-plot').forEach(p => {{
          if (window.Plotly) Plotly.Plots.resize(p);
        }});
      }}
    }}

    function selectCoin(coin) {{
      activeCoin = coin;
      document.querySelectorAll('.coin-tab').forEach(b =>
        b.classList.toggle('coin-tab-active', b.dataset.coin === coin));
      showPanel();
    }}

    function selectScenario(sc) {{
      activeScenario = sc;
      document.querySelectorAll('.sc-btn').forEach(b =>
        b.classList.toggle('sc-btn-active', b.dataset.scenario === sc));
      showPanel();
    }}

    function selectHorizon(hz) {{
      activeHorizon = hz;
      document.querySelectorAll('.hz-btn').forEach(b =>
        b.classList.toggle('hz-btn-active', b.dataset.horizon === hz));
      showPanel();
    }}

    // Show first panel on load
    showPanel();

    function openHelp() {{
      document.getElementById('helpModal').classList.add('open');
      document.body.style.overflow = 'hidden';
    }}
    function closeHelp() {{
      document.getElementById('helpModal').classList.remove('open');
      document.body.style.overflow = '';
    }}
    document.addEventListener('keydown', e => {{ if (e.key === 'Escape') closeHelp(); }});
  </script>

</body>
</html>"""

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"\n  ✓ Combined dashboard saved → {out_path}")


# ═══════════════════════════════════════════════════════════════════════════
# 8. MAIN
# ═══════════════════════════════════════════════════════════════════════════

COINS = {"bitcoin": "BTC", "ethereum": "ETH"}


def run_coin_scenario(
    coin_id: str,
    days: int,
    scenario: str,
    fng_df: pd.DataFrame,
    cached_df: dict,          # shared per-coin cache {coin_id: (df, price, news_score, x_score)}
) -> dict:
    """
    Run one coin × scenario. Returns panel dict:
      {"chart_html": str, "explanation_html": str, "price": float}
    Caches heavy fetches (price data, sentiment) per coin so they're not
    repeated across the 3 scenarios for the same coin.
    """
    sp = SCENARIO_PARAMS[scenario]
    print(f"\n  [{coin_id.upper()} / {sp['label']}]")

    # ── Fetch once per coin ────────────────────────────────────────────────
    if coin_id not in cached_df:
        df = fetch_ohlcv(coin_id, days=days)
        current_price = fetch_current_price(coin_id)
        print(f"  Current price: ${current_price:,.2f}")
        df = add_indicators(df)

        coin_keywords = {
            "bitcoin":  ["bitcoin", "btc", "$btc"],
            "ethereum": ["ethereum", "eth", "$eth"],
        }.get(coin_id, [coin_id])
        news_score = fetch_news_sentiment(coin_keywords)
        x_score    = fetch_reddit_sentiment(coin_id)
        cached_df[coin_id] = (df, current_price, news_score, x_score)
    else:
        df, current_price, news_score, x_score = cached_df[coin_id]

    # ── Scenario-specific work ─────────────────────────────────────────────
    df_sc, today_sentiment = build_sentiment_series(
        df.copy(), coin_id, scenario, fng_df, news_score, x_score
    )

    print(f"    Building forecasts...")
    forecasts = build_forecasts(df_sc, scenario, today_sentiment)
    print_summary(df_sc, forecasts, coin_id, current_price, scenario, today_sentiment, fng_df)

    # Build one chart per horizon
    charts_html = {}
    charts_fig = {}
    for hz in ["7d", "30d", "90d"]:
        fig = build_chart(df_sc, forecasts, coin_id, current_price, scenario, fng_df, horizon=hz)
        charts_fig[hz] = fig
        charts_html[hz] = fig.to_html(
            include_plotlyjs=False, full_html=False,
            div_id=f"chart_{coin_id}_{scenario}_{hz}"
        )

    mood_html = build_mood_html(fng_df)

    explanation_html = generate_explanation(
        df_sc, forecasts, coin_id, current_price, scenario,
        today_sentiment, fng_df, news_score, x_score
    )

    explanation_data = generate_explanation_data(
        df_sc, forecasts, coin_id, current_price, scenario,
        today_sentiment, fng_df, news_score, x_score
    )

    return {
        "charts_html": charts_html,
        "charts_fig":  charts_fig,
        "mood_html":   mood_html,
        "explanation_html": explanation_html,
        "explanation_data": explanation_data,
        "price": current_price,
        "fng_df": fng_df,
        "news_score": news_score,
        "x_score": x_score,
        "today_sentiment": today_sentiment,
    }


def main():
    parser = argparse.ArgumentParser(description="BTC & ETH crypto prediction model")
    parser.add_argument("--coin", choices=[*COINS.keys(), "all"], default="all")
    parser.add_argument("--scenario", choices=["base", "bullish", "bearish", "all"], default="all",
                        help="Scenario mode: base | bullish | bearish | all (default: all)")
    parser.add_argument("--days", type=int, default=730,
                        help="Days of historical data (default: 730)")
    parser.add_argument("--output", type=str, default=".",
                        help="Output directory for the dashboard HTML")
    args = parser.parse_args()

    coins     = list(COINS.keys()) if args.coin == "all" else [args.coin]
    scenarios = list(SCENARIO_PARAMS.keys()) if args.scenario == "all" else [args.scenario]

    print(f"\n{'═'*58}")
    print("  CRYPTO PREDICTION DASHBOARD  —  BTC & ETH")
    print(f"  Horizons: 7d | 30d | 90d  (rolling from today)")
    print(f"  Sentiment sources: Fear & Greed · News RSS · Reddit")
    print(f"  Coins: {', '.join(c.upper() for c in coins)}")
    print(f"  Scenarios: {', '.join(SCENARIO_PARAMS[s]['label'] for s in scenarios)}")
    print(f"{'═'*58}")

    fng_df     = fetch_fear_and_greed(days=args.days)
    cached_df  = {}   # per-coin cache to avoid repeat fetches
    panels     = {}   # {coin_id: {scenario: panel_dict}}

    for coin_id in coins:
        panels[coin_id] = {}
        for scenario in scenarios:
            try:
                panels[coin_id][scenario] = run_coin_scenario(
                    coin_id, args.days, scenario, fng_df, cached_df
                )
            except requests.exceptions.HTTPError as e:
                print(f"  ✗ API error ({coin_id}/{scenario}): {e}")
            except Exception as e:
                print(f"  ✗ Error ({coin_id}/{scenario}): {e}")
                raise

    # Build single combined dashboard
    out_path = f"{args.output}/crypto_dashboard.html"
    build_combined_dashboard(panels, out_path)

    print(f"\n{'═'*58}")
    print("  Done! Open crypto_dashboard.html in your browser.")
    print(f"  Use the Crypto and Scenario selectors at the top")
    print(f"  to switch between views.")
    print(f"{'═'*58}\n")


if __name__ == "__main__":
    main()
