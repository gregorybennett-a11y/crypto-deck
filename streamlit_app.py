import streamlit as st
import sys
import os
import tempfile

st.set_page_config(page_title="Crypto Prediction Dashboard", layout="wide")

st.title("🪙 Crypto Prediction Dashboard")
st.caption("Generating predictions for BTC & ETH — this takes about 30–60 seconds...")

@st.cache_data(show_spinner="Running prediction model... please wait")
def generate_dashboard():
    # Import from the existing script
    from crypto_predictor import (
        COINS, SCENARIO_PARAMS,
        fetch_fear_and_greed, run_coin_scenario, build_combined_dashboard
    )
    import requests

    coins = list(COINS.keys())
    scenarios = list(SCENARIO_PARAMS.keys())
    days = 730

    fng_df = fetch_fear_and_greed(days=days)
    cached_df = {}
    panels = {}

    for coin_id in coins:
        panels[coin_id] = {}
        for scenario in scenarios:
            try:
                panels[coin_id][scenario] = run_coin_scenario(
                    coin_id, days, scenario, fng_df, cached_df
                )
            except Exception as e:
                st.warning(f"Skipped {coin_id}/{scenario}: {e}")

    # Write HTML to a temp file, read it back
    with tempfile.NamedTemporaryFile(suffix=".html", delete=False, mode="w") as f:
        tmp_path = f.name

    build_combined_dashboard(panels, tmp_path)

    with open(tmp_path, "r", encoding="utf-8") as f:
        html = f.read()

    os.unlink(tmp_path)
    return html


html_content = generate_dashboard()

if html_content:
    st.components.v1.html(html_content, height=1400, scrolling=True)
else:
    st.error("Dashboard could not be generated. Check the logs.")
