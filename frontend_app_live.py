import os
import requests
import pandas as pd
import streamlit as st

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
TENORS = [91, 182, 364]

st.set_page_config(page_title="NG NTB Live Frontend", page_icon="🇳🇬", layout="wide")
st.title("NG NTB Live Frontend")
st.caption("Frontend for the FastAPI backend, live snapshots, and prediction history.")


def api_get(path: str):
    return requests.get(f"{API_BASE_URL}{path}", timeout=20)


def api_post(path: str, payload: dict):
    return requests.post(f"{API_BASE_URL}{path}", json=payload, timeout=20)


with st.sidebar:
    st.subheader("Backend")
    st.write(f"API base URL: `{API_BASE_URL}`")
    if st.button("Check health", use_container_width=True):
        try:
            res = api_get("/health")
            st.json(res.json())
        except Exception as exc:
            st.error(str(exc))

left, right = st.columns([1, 1.2], gap="large")

with left:
    st.subheader("Manual Snapshot Entry")
    tenor = st.selectbox("Tenor", TENORS)
    auction_date = st.date_input("Auction date")

    c1, c2 = st.columns(2)
    with c1:
        lag1_stop = st.number_input("Lag 1 stop", value=16.10, step=0.01)
        lag2_stop = st.number_input("Lag 2 stop", value=15.95, step=0.01)
        lag3_stop = st.number_input("Lag 3 stop", value=15.80, step=0.01)
        offer_amt = st.number_input("Offer amount (NGN bn)", value=80.0, step=10.0)
        prev_offer = st.number_input("Previous offer (NGN bn)", value=80.0, step=10.0)
        prev_bid_cover = st.number_input("Previous bid cover", value=2.20, step=0.05)
    with c2:
        sec_rate = st.number_input("Secondary rate", value=16.0943, step=0.0001, format="%.4f")
        sec_rate_5d_ago = st.number_input("Secondary rate 5D ago", value=16.05, step=0.0001, format="%.4f")
        system_liquidity = st.number_input("System liquidity (NGN bn)", value=2780.0, step=10.0)
        mpr = st.number_input("MPR", value=26.50, step=0.25)
        inflation = st.number_input("Inflation", value=15.06, step=0.01)
        source = st.text_input("Source", value="manual")

    if st.button("Save snapshot", type="primary", use_container_width=True):
        payload = {
            "auction_date": str(auction_date),
            "tenor_days": tenor,
            "lag1_stop": lag1_stop,
            "lag2_stop": lag2_stop,
            "lag3_stop": lag3_stop,
            "offer_amt": offer_amt,
            "prev_offer": prev_offer,
            "prev_bid_cover": prev_bid_cover,
            "sec_rate": sec_rate,
            "sec_rate_5d_ago": sec_rate_5d_ago,
            "system_liquidity": system_liquidity,
            "mpr": mpr,
            "inflation": inflation,
            "source": source,
        }
        try:
            res = api_post("/snapshots", payload)
            st.success("Snapshot saved")
            st.json(res.json())
        except Exception as exc:
            st.error(str(exc))

with right:
    st.subheader("Latest Snapshots")
    if st.button("Refresh latest snapshots", use_container_width=True):
        try:
            res = api_get("/snapshots/latest")
            data = res.json()
            rows = []
            for tenor_key, row in data.items():
                if row:
                    rows.append(row)
            if rows:
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            else:
                st.info("No snapshots found.")
        except Exception as exc:
            st.error(str(exc))

    st.subheader("Run Latest Prediction")
    if st.button("Predict from latest snapshots", use_container_width=True):
        try:
            res = api_post("/predict/latest", {})
            st.dataframe(pd.DataFrame(res.json()), use_container_width=True, hide_index=True)
        except Exception as exc:
            st.error(str(exc))

    st.subheader("Prediction History")
    if st.button("Load prediction history", use_container_width=True):
        try:
            res = api_get("/predictions/history?limit=20")
            st.dataframe(pd.DataFrame(res.json()), use_container_width=True, hide_index=True)
        except Exception as exc:
            st.error(str(exc))
