import os
import requests
import pandas as pd
import streamlit as st

# Streamlit Cloud stores secrets in st.secrets; env var is fallback for local use.
try:
    API_BASE_URL = st.secrets["API_BASE_URL"]
except (KeyError, FileNotFoundError):
    API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
TENORS = [91, 182, 364]

st.set_page_config(page_title="NG NTB Live Frontend", page_icon="🇳🇬", layout="wide")
st.title("NG NTB Live Frontend")
st.caption(
    "Operator console for the FastAPI backend: live snapshots, predictions, "
    "actual auction results, accuracy tracking, and model retraining."
)


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


# =========================================================
# LIVE TRAINING LOOP
# =========================================================
st.divider()
st.header("Live Training Loop")
st.caption(
    "Record realised auction outcomes, measure how predictions performed, "
    "and retrain the models on the accumulated history stored in the database."
)

tab_results, tab_accuracy, tab_train, tab_history, tab_import = st.tabs(
    ["Record Result", "Accuracy", "Retrain Models", "Training History", "Bulk Import"]
)

# --- Record actual auction result ---------------------------------------
with tab_results:
    st.subheader("Record Actual Stop Rate")
    st.write(
        "After an auction settles, record the realised stop rate. This is the "
        "training target the models learn from."
    )

    rc1, rc2, rc3 = st.columns(3)
    with rc1:
        result_tenor = st.selectbox("Tenor", TENORS, key="result_tenor")
    with rc2:
        result_date = st.date_input("Auction date", key="result_date")
    with rc3:
        actual_stop_rate = st.number_input(
            "Actual stop rate (%)", value=16.00, step=0.01, format="%.4f", key="actual_rate"
        )
    result_source = st.text_input("Source", value="manual", key="result_source")

    if st.button("Save auction result", type="primary", use_container_width=True):
        payload = {
            "auction_date": str(result_date),
            "tenor_days": result_tenor,
            "actual_stop_rate": actual_stop_rate,
            "source": result_source,
        }
        try:
            res = api_post("/auctions/results", payload)
            if res.status_code == 200:
                st.success(f"Saved {result_tenor}D result for {result_date}.")
                st.json(res.json())
            else:
                st.error(f"{res.status_code}: {res.text}")
        except Exception as exc:
            st.error(str(exc))

    st.markdown("**Recorded Results**")
    if st.button("Load recorded results", use_container_width=True):
        try:
            res = api_get("/auctions/results?limit=50")
            data = res.json()
            if data:
                st.dataframe(pd.DataFrame(data), use_container_width=True, hide_index=True)
            else:
                st.info("No auction results recorded yet.")
        except Exception as exc:
            st.error(str(exc))

# --- Accuracy -----------------------------------------------------------
with tab_accuracy:
    st.subheader("Prediction Accuracy")
    st.write(
        "Compares logged predictions against recorded actuals, matched on "
        "auction date and tenor. Positive bias means the model over-predicts."
    )

    if st.button("Compute accuracy", type="primary", use_container_width=True):
        try:
            res = api_get("/accuracy")
            data = res.json()
            df = pd.DataFrame(data)

            cols = st.columns(len(TENORS))
            for col, item in zip(cols, data):
                tenor = item["tenor_days"]
                n = item["n_compared"]
                mae = item["mae"]
                if n and mae is not None:
                    col.metric(f"{tenor}D MAE", f"{mae:.4f}%", help=f"{n} predictions compared")
                    col.caption(f"RMSE {item['rmse']:.4f} · bias {item['bias']:+.4f}")
                else:
                    col.metric(f"{tenor}D MAE", "—")
                    col.caption("No matched predictions yet")

            st.dataframe(df, use_container_width=True, hide_index=True)
        except Exception as exc:
            st.error(str(exc))

# --- Retrain ------------------------------------------------------------
with tab_train:
    st.subheader("Retrain Models")
    st.write(
        "Retrains every tenor that has at least 12 matched snapshot + result "
        "pairs. New models are saved and loaded immediately — the next "
        "prediction uses them with no restart."
    )

    if st.button("Retrain now", type="primary", use_container_width=True):
        try:
            with st.spinner("Training models from stored history…"):
                res = api_post("/train", {})
            results = res.json()
            for item in results:
                tenor = item["tenor_days"]
                status = item["status"]
                if status == "ok":
                    st.success(
                        f"**{tenor}D** — trained on {item['n_samples']} samples "
                        f"(version {item['model_version']})"
                    )
                    mc1, mc2, mc3 = st.columns(3)
                    mc1.metric("MAE", f"{item['mae']:.4f}%")
                    mc2.metric("RMSE", f"{item['rmse']:.4f}%")
                    mc3.metric("R²", f"{item['r2']:.4f}")
                elif status == "skip":
                    st.warning(f"**{tenor}D** — {item['message']}")
                else:
                    st.error(f"**{tenor}D** — {item['message']}")
        except Exception as exc:
            st.error(str(exc))

# --- Bulk historical import ---------------------------------------------
with tab_import:
    st.subheader("Bulk Historical Import")
    st.write(
        "Upload a CSV of past auction data to seed the database in one step. "
        "Each row must contain both the pre-auction snapshot features **and** "
        "the realised `actual_stop_rate`. Once imported you can retrain "
        "immediately — no need to collect 12 live points first."
    )

    IMPORT_COLUMNS = [
        "auction_date", "tenor_days",
        "lag1_stop", "lag2_stop", "lag3_stop",
        "offer_amt", "prev_offer", "prev_bid_cover",
        "sec_rate", "sec_rate_5d_ago",
        "system_liquidity", "mpr", "inflation",
        "actual_stop_rate", "source",
    ]

    with st.expander("CSV column reference"):
        st.code(", ".join(IMPORT_COLUMNS))
        st.caption(
            "`auction_date` — YYYY-MM-DD  |  `tenor_days` — 91 / 182 / 364  |  "
            "`source` — optional label (e.g. `historical`)"
        )

    uploaded = st.file_uploader(
        "Choose historical CSV", type=["csv"], key="bulk_import_csv"
    )

    if uploaded is not None:
        try:
            df_preview = pd.read_csv(uploaded)
            uploaded.seek(0)
            st.write(f"**{len(df_preview)} rows** detected — preview:")
            st.dataframe(df_preview.head(5), use_container_width=True, hide_index=True)

            missing_cols = [c for c in IMPORT_COLUMNS if c != "source" and c not in df_preview.columns]
            if missing_cols:
                st.error(f"Missing required columns: {', '.join(missing_cols)}")
            else:
                if st.button("Import into database", type="primary", use_container_width=True):
                    df_import = pd.read_csv(uploaded)
                    if "source" not in df_import.columns:
                        df_import["source"] = "historical_import"
                    df_import["source"] = df_import["source"].fillna("historical_import")

                    payload = df_import[IMPORT_COLUMNS].to_dict(orient="records")
                    for row in payload:
                        row["auction_date"] = str(row["auction_date"])
                        row["tenor_days"] = int(row["tenor_days"])

                    with st.spinner("Importing…"):
                        res = api_post("/import/historical", payload)

                    if res.status_code == 200:
                        result = res.json()
                        st.success(
                            f"Imported **{result['imported_snapshots']} snapshots** "
                            f"and **{result['imported_results']} results**."
                            + (f"  Skipped: {result['skipped']}" if result["skipped"] else "")
                        )
                        ready = result["ready_to_train"]
                        rcols = st.columns(3)
                        for col, (tenor_key, is_ready) in zip(rcols, ready.items()):
                            col.metric(
                                tenor_key,
                                "Ready to train ✓" if is_ready else "Need more data",
                            )
                        if result["errors"]:
                            with st.expander("Row errors"):
                                for err in result["errors"]:
                                    st.warning(err)
                    else:
                        st.error(f"{res.status_code}: {res.text}")
        except Exception as exc:
            st.error(str(exc))

# --- Training history ---------------------------------------------------
with tab_history:
    st.subheader("Training History")
    if st.button("Load training history", use_container_width=True):
        try:
            res = api_get("/training/history?limit=20")
            data = res.json()
            if data:
                st.dataframe(pd.DataFrame(data), use_container_width=True, hide_index=True)
            else:
                st.info("No training runs logged yet.")
        except Exception as exc:
            st.error(str(exc))
