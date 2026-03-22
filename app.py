import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="NG NTB Stop Rate Predictor v5.0",
    page_icon="🇳🇬",
    layout="wide"
)

# =========================================================
# STYLING
# =========================================================
st.markdown("""
<style>
    .block-container {
        padding-top: 1rem;
        padding-bottom: 2rem;
        max-width: 1450px;
    }

    .hero {
        padding: 1.4rem 1.5rem;
        border-radius: 22px;
        background: linear-gradient(135deg, #0f172a 0%, #111827 60%, #1e293b 100%);
        border: 1px solid rgba(148,163,184,0.18);
        box-shadow: 0 12px 28px rgba(0,0,0,0.18);
        margin-bottom: 1rem;
    }

    .hero h1 {
        margin: 0;
        color: #f8fafc;
        font-size: 2.4rem;
        line-height: 1.05;
    }

    .hero p {
        margin: 0.45rem 0 0 0;
        color: #cbd5e1;
        font-size: 0.98rem;
    }

    .subtle {
        color: #94a3b8;
        font-size: 0.9rem;
    }

    div[data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: 18px !important;
        border: 1px solid rgba(148,163,184,0.16) !important;
        background: rgba(15,23,42,0.52) !important;
        box-shadow: 0 8px 22px rgba(0,0,0,0.10);
    }

    div[data-testid="stMetric"] {
        background: rgba(15,23,42,0.75);
        border: 1px solid rgba(148,163,184,0.16);
        padding: 0.85rem 1rem;
        border-radius: 16px;
    }

    .stButton > button {
        border-radius: 12px !important;
        font-weight: 600 !important;
        border: 1px solid rgba(148,163,184,0.22) !important;
    }
</style>
""", unsafe_allow_html=True)

# =========================================================
# CONSTANTS
# =========================================================
TENORS = [91, 182, 364]

MODEL_FILES = {
    91: "gti_ntb_v5_91D.pkl",
    182: "gti_ntb_v5_182D.pkl",
    364: "gti_ntb_v5_364D.pkl",
}

FEATURE_ORDER = [
    "lag1_stop",
    "lag2_stop",
    "lag3_stop",
    "ma3_stop",
    "delta_stop_1",
    "offer_amt",
    "offer_change",
    "prev_bid_cover",
    "sec_rate",
    "sec_rate_change_5d",
    "sec_minus_lag1",
    "system_liquidity",
    "mpr",
    "inflation",
]

DEFAULTS = {
    "auction_date": pd.Timestamp.today().date(),
    "system_liquidity": 2780.0,
    "mpr": 26.50,
    "inflation": 15.06,

    "offer_91": 80.0,
    "offer_182": 120.0,
    "offer_364": 500.0,

    "prev_offer_91": 80.0,
    "prev_offer_182": 120.0,
    "prev_offer_364": 500.0,

    "prev_bid_cover_91": 2.20,
    "prev_bid_cover_182": 1.90,
    "prev_bid_cover_364": 3.10,

    "lag1_stop_91": 16.10,
    "lag2_stop_91": 15.95,
    "lag3_stop_91": 15.80,

    "lag1_stop_182": 16.35,
    "lag2_stop_182": 16.20,
    "lag3_stop_182": 16.05,

    "lag1_stop_364": 16.85,
    "lag2_stop_364": 16.70,
    "lag3_stop_364": 16.55,

    "sec_rate_91": 16.25,
    "sec_rate_182": 16.50,
    "sec_rate_364": 16.95,

    "sec_rate_5d_ago_91": 16.05,
    "sec_rate_5d_ago_182": 16.30,
    "sec_rate_5d_ago_364": 16.70,
}

# =========================================================
# HELPERS
# =========================================================
def init_state():
    for k, v in DEFAULTS.items():
        if k not in st.session_state:
            st.session_state[k] = v

def load_demo_values():
    for k, v in DEFAULTS.items():
        st.session_state[k] = v

@st.cache_resource
def load_models():
    models = {}
    missing = []

    for tenor, fname in MODEL_FILES.items():
        if Path(fname).exists():
            models[tenor] = joblib.load(fname)
        else:
            missing.append(fname)

    return models, missing

def derive_tenor_features(tenor: int) -> dict:
    lag1 = float(st.session_state[f"lag1_stop_{tenor}"])
    lag2 = float(st.session_state[f"lag2_stop_{tenor}"])
    lag3 = float(st.session_state[f"lag3_stop_{tenor}"])

    offer_amt = float(st.session_state[f"offer_{tenor}"])
    prev_offer = float(st.session_state[f"prev_offer_{tenor}"])

    sec_rate = float(st.session_state[f"sec_rate_{tenor}"])
    sec_rate_5d_ago = float(st.session_state[f"sec_rate_5d_ago_{tenor}"])

    return {
        "lag1_stop": lag1,
        "lag2_stop": lag2,
        "lag3_stop": lag3,
        "ma3_stop": round((lag1 + lag2 + lag3) / 3.0, 6),
        "delta_stop_1": round(lag1 - lag2, 6),
        "offer_amt": offer_amt,
        "offer_change": round(offer_amt - prev_offer, 6),
        "prev_bid_cover": float(st.session_state[f"prev_bid_cover_{tenor}"]),
        "sec_rate": sec_rate,
        "sec_rate_change_5d": round(sec_rate - sec_rate_5d_ago, 6),
        "sec_minus_lag1": round(sec_rate - lag1, 6),
        "system_liquidity": float(st.session_state["system_liquidity"]),
        "mpr": float(st.session_state["mpr"]),
        "inflation": float(st.session_state["inflation"]),
    }

def build_feature_table() -> pd.DataFrame:
    rows = []
    for tenor in TENORS:
        feat = derive_tenor_features(tenor)
        rows.append({"tenor": f"{tenor}D", **feat})
    return pd.DataFrame(rows)

def interpret_result(pred: float, sec_rate: float, lag1_stop: float) -> str:
    spread_to_sec = pred - sec_rate
    spread_to_lag1 = pred - lag1_stop

    if spread_to_sec <= -0.10 and spread_to_lag1 <= -0.05:
        return "Softer than current market tone."
    if spread_to_sec >= 0.10 and spread_to_lag1 >= 0.05:
        return "Higher / more pressured than recent market tone."
    return "Broadly in line with recent market tone."

def predict_all(models: dict):
    preds = {}
    feature_rows = []

    for tenor in TENORS:
        feat = derive_tenor_features(tenor)
        feature_rows.append({"tenor": f"{tenor}D", **feat})

        if tenor not in models:
            preds[tenor] = "Model file not found"
            continue

        X = pd.DataFrame([feat])[FEATURE_ORDER]

        try:
            yhat = float(models[tenor].predict(X)[0])
            preds[tenor] = round(yhat, 4)
        except Exception as e:
            preds[tenor] = f"{type(e).__name__}: {e}"

    return preds, pd.DataFrame(feature_rows)

# =========================================================
# INIT
# =========================================================
init_state()
models, missing_models = load_models()

# =========================================================
# HEADER
# =========================================================
st.markdown("""
<div class="hero">
    <h1>🇳🇬 NG NTB Stop Rate Predictor v5.0</h1>
    <p>Pre-auction predictor using tenor-specific stop-rate history, tenor-specific offer, prior demand, current secondary market tone, and shared liquidity.</p>
    <p class="subtle">This version uses the same schema as the v5 retrained models.</p>
</div>
""", unsafe_allow_html=True)

if missing_models:
    st.warning(
        "Missing model file(s): " + ", ".join(missing_models) +
        ". Upload the new v5 model files before predicting."
    )

left, right = st.columns([1.05, 1.45], gap="large")

# =========================================================
# LEFT PANEL
# =========================================================
with left:
    with st.container(border=True):
        st.subheader("1) Shared Market Inputs")

        a, b = st.columns(2)
        with a:
            st.date_input(
                "Auction date",
                key="auction_date",
                help="Auction date for the scenario being assessed."
            )
        with b:
            if st.button(
                "Load demo values",
                use_container_width=True,
                help="Load a starter scenario for testing."
            ):
                load_demo_values()
                st.success("Demo values loaded.")

        c1, c2 = st.columns(2)
        with c1:
            st.number_input(
                "System liquidity (NGN bn)",
                key="system_liquidity",
                step=10.0,
                help="Estimated net system liquidity before the auction."
            )
        with c2:
            st.number_input(
                "MPR (%)",
                key="mpr",
                step=0.25,
                help="Current Monetary Policy Rate."
            )

        c3, c4 = st.columns(2)
        with c3:
            st.number_input(
                "Inflation (%)",
                key="inflation",
                step=0.01,
                help="Latest inflation reading."
            )
        with c4:
            total_offer = (
                float(st.session_state["offer_91"])
                + float(st.session_state["offer_182"])
                + float(st.session_state["offer_364"])
            )
            st.metric("Total current offer", f"{total_offer:,.2f} bn")

    with st.container(border=True):
        st.subheader("2) Inputs by Tenor")

        tabs = st.tabs(["91D", "182D", "364D"])

        for tenor, tab in zip(TENORS, tabs):
            with tab:
                st.markdown(f"**{tenor}D scenario inputs**")

                r1, r2 = st.columns(2)
                with r1:
                    st.number_input(
                        f"{tenor}D offer amount (NGN bn)",
                        key=f"offer_{tenor}",
                        step=10.0,
                        min_value=0.0,
                        help=f"Current amount offered for the {tenor}D tenor."
                    )
                with r2:
                    st.number_input(
                        f"{tenor}D previous offer (NGN bn)",
                        key=f"prev_offer_{tenor}",
                        step=10.0,
                        min_value=0.0,
                        help=f"Offer amount for the previous {tenor}D auction."
                    )

                r3, r4 = st.columns(2)
                with r3:
                    st.number_input(
                        f"{tenor}D previous bid cover",
                        key=f"prev_bid_cover_{tenor}",
                        step=0.05,
                        min_value=0.0,
                        help=f"Previous auction bid cover for the {tenor}D tenor."
                    )
                with r4:
                    offer_change = float(st.session_state[f"offer_{tenor}"]) - float(st.session_state[f"prev_offer_{tenor}"])
                    st.metric(f"{tenor}D offer change", f"{offer_change:,.2f} bn")

                l1, l2, l3 = st.columns(3)
                with l1:
                    st.number_input(
                        f"{tenor}D lag 1 stop",
                        key=f"lag1_stop_{tenor}",
                        step=0.01,
                        help=f"Most recent stop rate for the {tenor}D tenor."
                    )
                with l2:
                    st.number_input(
                        f"{tenor}D lag 2 stop",
                        key=f"lag2_stop_{tenor}",
                        step=0.01,
                        help=f"Second most recent stop rate for the {tenor}D tenor."
                    )
                with l3:
                    st.number_input(
                        f"{tenor}D lag 3 stop",
                        key=f"lag3_stop_{tenor}",
                        step=0.01,
                        help=f"Third most recent stop rate for the {tenor}D tenor."
                    )

                s1, s2 = st.columns(2)
                with s1:
                    st.number_input(
                        f"{tenor}D secondary rate",
                        key=f"sec_rate_{tenor}",
                        step=0.01,
                        help=f"Current secondary market yield closest to the {tenor}D tenor."
                    )
                with s2:
                    st.number_input(
                        f"{tenor}D secondary rate 5D ago",
                        key=f"sec_rate_5d_ago_{tenor}",
                        step=0.01,
                        help=f"Secondary market yield for the same tenor proxy five trading days ago."
                    )

                feat_preview = derive_tenor_features(tenor)
                p1, p2, p3 = st.columns(3)
                with p1:
                    st.metric(f"{tenor}D MA3 stop", f"{feat_preview['ma3_stop']:.2f}%")
                with p2:
                    st.metric(f"{tenor}D stop momentum", f"{feat_preview['delta_stop_1']:+.2f}")
                with p3:
                    st.metric(f"{tenor}D secondary vs lag1", f"{feat_preview['sec_minus_lag1']:+.2f}")

# =========================================================
# RIGHT PANEL
# =========================================================
with right:
    with st.container(border=True):
        st.subheader("3) Feature Preview")
        preview_df = build_feature_table()
        st.dataframe(preview_df, use_container_width=True, hide_index=True)

        if st.button(
            "🚀 Predict Stop Rates",
            type="primary",
            use_container_width=True,
            help="Run all three tenor models using the current pre-auction scenario."
        ):
            preds, pred_features = predict_all(models)
            st.session_state["predictions"] = preds
            st.session_state["pred_features"] = pred_features

    if "predictions" in st.session_state:
        with st.container(border=True):
            st.subheader("4) Predicted Stop Rates")

            c1, c2, c3 = st.columns(3)

            for idx, tenor in enumerate(TENORS):
                val = st.session_state["predictions"][tenor]
                col = [c1, c2, c3][idx]

                if isinstance(val, (float, int)):
                    sec_rate = float(st.session_state[f"sec_rate_{tenor}"])
                    lag1 = float(st.session_state[f"lag1_stop_{tenor}"])
                    interpretation = interpret_result(val, sec_rate, lag1)

                    col.metric(f"{tenor}D Estimated Stop Rate", f"{val:.2f}%")
                    col.caption(interpretation)
                else:
                    col.error(f"{tenor}D prediction failed")
                    col.caption(str(val))

        with st.container(border=True):
            st.subheader("5) Model Inputs Used")
            st.dataframe(st.session_state["pred_features"], use_container_width=True, hide_index=True)

    with st.container(border=True):
        st.subheader("Definitions")
        st.markdown("""
- **Lag 1 / Lag 2 / Lag 3 stop**: the last three stop rates for that tenor.  
- **MA3 stop**: average of the last three stop rates.  
- **Stop momentum**: latest stop-rate change, calculated as **lag1 − lag2**.  
- **Offer amount**: current supply for that tenor.  
- **Offer change**: current offer minus previous offer.  
- **Previous bid cover**: prior auction demand proxy for that tenor.  
- **Secondary rate**: current market yield closest to that tenor.  
- **Secondary rate 5D ago**: same tenor proxy five trading days earlier.  
- **Secondary vs lag1**: current secondary rate minus most recent stop rate.  
- **System liquidity**: estimated net market liquidity before auction.
        """)
        st.info(
            "This is a pre-auction tool. It should use previous or observable market variables only, not current-auction realised results."
        )
        st.caption("For internal decision support only. Not an official auction result.")
