import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="NG NTB Stop Rate Predictor",
    page_icon="🇳🇬",
    layout="wide"
)

# =========================================================
# STYLING
# =========================================================
st.markdown("""
<style>
    .block-container {
        padding-top: 0.9rem;
        padding-bottom: 1.8rem;
        max-width: 1450px;
    }

    .hero {
        padding: 1.25rem 1.4rem;
        border-radius: 22px;
        background: linear-gradient(135deg, #0b1220 0%, #111827 55%, #182235 100%);
        border: 1px solid rgba(148,163,184,0.15);
        box-shadow: 0 14px 30px rgba(0,0,0,0.16);
        margin-bottom: 1rem;
    }

    .hero-top {
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 1rem;
        flex-wrap: wrap;
    }

    .hero h1 {
        margin: 0;
        color: #f8fafc;
        font-size: 2.2rem;
        line-height: 1.05;
        font-weight: 700;
        letter-spacing: -0.02em;
    }

    .hero-badges {
        display: flex;
        gap: 0.5rem;
        flex-wrap: wrap;
    }

    .badge {
        display: inline-block;
        padding: 0.4rem 0.75rem;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 600;
        color: #dbeafe;
        background: rgba(30, 41, 59, 0.85);
        border: 1px solid rgba(148,163,184,0.18);
    }

    .section-label {
        font-size: 0.8rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #94a3b8;
        margin-bottom: 0.25rem;
    }

    div[data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: 18px !important;
        border: 1px solid rgba(148,163,184,0.14) !important;
        background: rgba(15,23,42,0.50) !important;
        box-shadow: 0 8px 20px rgba(0,0,0,0.08);
    }

    div[data-testid="stMetric"] {
        background: rgba(15,23,42,0.78);
        border: 1px solid rgba(148,163,184,0.14);
        padding: 0.85rem 1rem;
        border-radius: 16px;
    }

    .stButton > button {
        border-radius: 12px !important;
        font-weight: 600 !important;
        border: 1px solid rgba(148,163,184,0.20) !important;
    }

    .small-note {
        color: #94a3b8;
        font-size: 0.9rem;
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

    "sec_rate_91": 16.0943,
    "sec_rate_182": 17.5339,
    "sec_rate_364": 19.6057,

    "sec_rate_5d_ago_91": 16.05,
    "sec_rate_5d_ago_182": 17.30,
    "sec_rate_5d_ago_364": 19.20,
}

# =========================================================
# HELPERS
# =========================================================
def init_state():
    for k, v in DEFAULTS.items():
        if k not in st.session_state:
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
        return "Higher than recent market tone."
    return "Broadly in line with market tone."

def build_tenor_interpretation(tenor: int, pred: float) -> str:
    sec_rate = float(st.session_state[f"sec_rate_{tenor}"])
    lag1 = float(st.session_state[f"lag1_stop_{tenor}"])
    offer_amt = float(st.session_state[f"offer_{tenor}"])
    prev_offer = float(st.session_state[f"prev_offer_{tenor}"])
    prev_bid_cover = float(st.session_state[f"prev_bid_cover_{tenor}"])

    spread_to_sec = round(pred - sec_rate, 2)
    spread_to_lag1 = round(pred - lag1, 2)
    offer_change = round(offer_amt - prev_offer, 2)

    if spread_to_sec >= 0.15:
        market_tone = "pricing above current secondary market tone"
    elif spread_to_sec <= -0.15:
        market_tone = "pricing below current secondary market tone"
    else:
        market_tone = "pricing broadly in line with current secondary market tone"

    if spread_to_lag1 >= 0.10:
        stop_comparison = "above the most recent stop rate"
    elif spread_to_lag1 <= -0.10:
        stop_comparison = "below the most recent stop rate"
    else:
        stop_comparison = "close to the most recent stop rate"

    if offer_change > 0:
        supply_view = f"supply is higher by {offer_change:,.2f}bn versus the previous offer"
    elif offer_change < 0:
        supply_view = f"supply is lower by {abs(offer_change):,.2f}bn versus the previous offer"
    else:
        supply_view = "supply is unchanged versus the previous offer"

    if prev_bid_cover >= 3.0:
        demand_view = "prior demand was strong"
    elif prev_bid_cover >= 2.0:
        demand_view = "prior demand was moderate"
    else:
        demand_view = "prior demand was relatively soft"

    return (
        f"{tenor}D is indicating {market_tone}, {stop_comparison}. "
        f"Model context suggests {supply_view}, while {demand_view} "
        f"(previous bid cover: {prev_bid_cover:.2f}x)."
    )

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
    <div class="hero-top">
        <h1>NG NTB Stop Rate Predictor</h1>
        <div class="hero-badges">
            <span class="badge">Treasury Desk</span>
            <span class="badge">Internal Use</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

if missing_models:
    st.error(
        "Missing model file(s): " + ", ".join(missing_models) +
        ". Upload the new v5 model files before predicting."
    )
    st.stop()

left, right = st.columns([1.02, 1.48], gap="large")

# =========================================================
# LEFT PANEL
# =========================================================
with left:
    with st.container(border=True):
        st.markdown('<div class="section-label">Market Inputs</div>', unsafe_allow_html=True)
        st.subheader("Shared Inputs")

        st.date_input(
            "Auction date",
            key="auction_date",
            help="Auction date for the scenario being assessed."
        )

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
            st.metric("Calculated total offer", f"{total_offer:,.2f} bn")

    with st.container(border=True):
        st.markdown('<div class="section-label">Auction Inputs</div>', unsafe_allow_html=True)
        st.subheader("By Tenor")

        tabs = st.tabs(["91D", "182D", "364D"])

        for tenor, tab in zip(TENORS, tabs):
            with tab:
                st.markdown(f"**{tenor}D Inputs**")

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
                        step=0.0001,
                        format="%.4f",
                        help=f"Current secondary market yield closest to the {tenor}D tenor."
                    )
                with s2:
                    st.number_input(
                        f"{tenor}D secondary rate 5D ago",
                        key=f"sec_rate_5d_ago_{tenor}",
                        step=0.0001,
                        format="%.4f",
                        help=f"Secondary market yield for the same tenor proxy five trading days ago."
                    )

                feat_preview = derive_tenor_features(tenor)
                p1, p2, p3 = st.columns(3)
                with p1:
                    st.metric("MA3 stop", f"{feat_preview['ma3_stop']:.2f}%")
                with p2:
                    st.metric("Momentum", f"{feat_preview['delta_stop_1']:+.2f}")
                with p3:
                    st.metric("Sec vs lag1", f"{feat_preview['sec_minus_lag1']:+.2f}")

# =========================================================
# RIGHT PANEL
# =========================================================
with right:
    with st.container(border=True):
        st.markdown('<div class="section-label">Model View</div>', unsafe_allow_html=True)
        st.subheader("Feature Preview")
        preview_df = build_feature_table()
        st.dataframe(preview_df, use_container_width=True, hide_index=True)

        if st.button(
            "Predict Stop Rates",
            type="primary",
            use_container_width=True,
            help="Run all three tenor models using the current pre-auction scenario."
        ):
            preds, pred_features = predict_all(models)
            st.session_state["predictions"] = preds
            st.session_state["pred_features"] = pred_features

    if "predictions" in st.session_state:
        with st.container(border=True):
            st.markdown('<div class="section-label">Output</div>', unsafe_allow_html=True)
            st.subheader("Predicted Stop Rates")

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
            st.markdown('<div class="section-label">Interpretation</div>', unsafe_allow_html=True)
            st.subheader("Auction Interpretation")

            for tenor in TENORS:
                val = st.session_state["predictions"][tenor]

                if isinstance(val, (float, int)):
                    st.markdown(f"**{tenor}D:** {build_tenor_interpretation(tenor, float(val))}")
                else:
                    st.markdown(f"**{tenor}D:** Interpretation unavailable because prediction failed.")

        with st.container(border=True):
            st.markdown('<div class="section-label">Audit Trail</div>', unsafe_allow_html=True)
            st.subheader("Model Inputs Used")
            st.dataframe(st.session_state["pred_features"], use_container_width=True, hide_index=True)

    with st.container(border=True):
        st.markdown('<div class="section-label">Note</div>', unsafe_allow_html=True)
        st.subheader("Model Use")
        st.caption("For internal treasury decision support only.")
