import streamlit as st
import pandas as pd
import joblib
import requests
from bs4 import BeautifulSoup
import re

st.set_page_config(page_title="NTB AI Predictor", layout="centered")
st.title("🇳🇬 NTB Stop Rate Predictor v3.0")
st.markdown("**Live CBN Scraper + March 2026 Lags**")

# Load models
@st.cache_resource
def load_models():
    return {t: joblib.load(f"gti_model_tenor_{t}D.pkl") for t in [91,182,364]}

models = load_models()

# Scraper
def scrape_cbn_data():
    data = {"policy_rate": 27.25, "inflation_rate": 15.06}
    try:
        r = requests.get("https://www.cbn.gov.ng/MonetaryPolicy/decisions.html", timeout=10)
        match = re.search(r"to\s+(\d{1,2}\.?\d{0,2})\s+per cent", r.text, re.IGNORECASE)
        if match: data["policy_rate"] = float(match.group(1))
    except: pass
    
    try:
        r = requests.get("https://www.cbn.gov.ng/rates/inflrates.html", timeout=10)
        soup = BeautifulSoup(r.text, "html.parser")
        table = soup.find("table")
        if table:
            cells = table.find_all("tr")[-1].find_all("td")
            infl_str = re.sub(r"[^0-9.]", "", cells[1].text.strip()) if len(cells)>1 else ""
            if infl_str: data["inflation_rate"] = float(infl_str)
    except: pass
    return data

# Inputs
st.sidebar.header("Market Inputs")
bid_cover = st.sidebar.number_input("Bid Cover Ratio", value=2.85, step=0.01)
liquidity = st.sidebar.number_input("Liquidity (NGN bn)", value=2780.0, step=10.0)
omo_days = st.sidebar.number_input("OMO Maturity (days)", value=45, step=1)

lag1 = st.sidebar.number_input("lag1_rate (most recent)", value=15.95, step=0.01)
lag2 = st.sidebar.number_input("lag2_rate", value=16.65, step=0.01)
lag3 = st.sidebar.number_input("lag3_rate", value=16.72, step=0.01)

col1, col2 = st.sidebar.columns(2)
scrape_btn = col1.button("🔄 Scrape MPR + Inflation")
fill_btn = col2.button("📅 Fill Latest Lags (Mar 2026)")

if fill_btn:
    lag1 = 15.95
    lag2 = 16.65
    lag3 = 16.72
    st.sidebar.success("Latest auction lags loaded!")

if scrape_btn:
    auto = scrape_cbn_data()
    st.sidebar.success(f"MPR: {auto['policy_rate']}% | Inflation: {auto['inflation_rate']}%")

# Predict
if st.button("🚀 Predict NTB Stop Rates", type="primary"):
    auto = scrape_cbn_data()
    features = {
        "bid_cover": bid_cover,
        "liquidity": liquidity,
        "omo_maturity": omo_days,
        "MPR": auto["policy_rate"],
        "inflation": auto["inflation_rate"],
        "lag1": lag1,
        "lag2": lag2,
        "lag3": lag3
    }
    
    st.subheader("📊 Features Used")
    st.dataframe(pd.DataFrame([features]), use_container_width=True)
    
    st.subheader("🎯 PREDICTED STOP RATES")
    cols = st.columns(3)
    for i, (tenor, pipe) in enumerate(models.items()):
        ty = tenor / 365.0
        feat = {
            "lag1_rate": features["lag1"], "lag2_rate": features["lag2"], "lag3_rate": features["lag3"],
            "ma3_rate": (features["lag1"] + features["lag2"] + features["lag3"]) / 3,
            "bid_cover": features["bid_cover"],
            "MPR": features["MPR"],
            "inflation": features["inflation"],
            "liquidity": features["liquidity"],
            "omo_maturity": features["omo_maturity"],
            "tenor_years": ty
        }
        pred = round(pipe.predict(pd.DataFrame([feat]))[0], 4)
        cols[i].metric(f"{tenor}D Tenor", f"{pred}%")
