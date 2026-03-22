import streamlit as st
import pandas as pd
import joblib
import requests
from bs4 import BeautifulSoup
import re

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="NG NTB Stop Rate Predictor v4.0",
    page_icon="🇳🇬",
    layout="wide"
)

# -----------------------------
# STYLING
# -----------------------------
st.markdown("""
<style>
    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 2rem;
        max-width: 1400px;
    }

    .hero {
        padding: 1.35rem 1.5rem;
        border-radius: 24px;
        background: linear-gradient(135deg, #0f172a 0%, #111827 55%, #1f2937 100%);
        border: 1px solid rgba(148, 163, 184, 0.20);
        box-shadow: 0 12px 30px rgba(0,0,0,0.22);
        margin-bottom: 1rem;
    }

    .hero h1 {
        margin: 0;
        font-size: 2.6rem;
        line-height: 1.05;
        color: #f8fafc;
    }

    .hero p {
        margin: 0.5rem 0 0 0;
        color: #cbd5e1;
        font-size: 1rem;
    }

    .small-muted {
        color: #94a3b8;
        font-size: 0.92rem;
    }

    div[data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: 20px !important;
        border: 1px solid rgba(148, 163, 184, 0.18) !important;
        background: rgba(15, 23, 42, 0.55) !important;
        box-shadow: 0 8px 24px rgba(0,0,0,0.14);
    }

    div[data-testid="stMetric"] {
        background: rgba(15, 23, 42, 0.75);
        border: 1px solid rgba(148, 163, 184, 0.18);
        padding: 0.85rem 1rem;
        border-radius: 18px;
    }

    .stButton > button {
        border-radius: 14px !important;
        font-weight: 600 !important;
        border: 1px solid rgba(148, 163, 184, 0.25) !important;
    }

    div[data-testid="stDataFrame"] {
        border-radius: 16px;
        overflow: hidden;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------
# DEFAULTS / STARTER SCENARIO
# -----------------------------
DEFAULTS = {
    "MPR": 26.50,
    "inflation": 15.06,
    "liquidity": 2780.0,
    "omo_maturity": 45,
    "predictions": None,
    "prediction_features": None,
    # liquidity estimator defaults
    "liq_opening": 1800.0,
    "liq_omo_maturities": 350.0,
    "liq_ntb_maturities": 250.0,
    "liq_bond_coupons": 120.0,
    "liq_faac_other_inflows": 500.0,
    "liq_crr_debits": 100.0,
    "liq_omo_sales": 50.0,
    "liq_ntb_auction_debit": 60.0,
    "liq_tax_govt_outflows": 20.0,
    "liq_other_drains": 10.0,
}

# Starter scenario only. Replace with actual tenor-specific latest values if you have them.
STARTER_SCENARIO = {
    "MPR": 26.50,
    "inflation": 15.06,
    "liquidity": 2780.0,
    "omo_maturity": 45,
    "tenors": {
        91:  {"offered": 100.0, "bids": 285.0, "lag1": 15.95, "lag2": 16.65, "lag3": 16.72},
        182: {"offered": 100.0, "bids": 285.0, "lag1": 15.95, "lag2": 16.65, "lag3": 16.72},
        364: {"offered": 100.0, "bids": 285.0, "lag1": 15.95, "lag2": 16.65, "lag3": 16.72},
    }
}

TENORS = [91, 182, 364]

# -----------------------------
# HELPERS
# -----------------------------
def init_state():
    for k, v in DEFAULTS.items():
        if k not in st.session_state:
            st.session_state[k] = v

    for tenor in TENORS:
        if f"offered_{tenor}" not in st.session_state:
            st.session_state[f"offered_{tenor}"] = STARTER_SCENARIO["tenors"][tenor]["offered"]
        if f"bids_{tenor}" not in st.session_state:
            st.session_state[f"bids_{tenor}"] = STARTER_SCENARIO["tenors"][tenor]["bids"]
        if f"lag1_{tenor}" not in st.session_state:
            st.session_state[f"lag1_{tenor}"] = STARTER_SCENARIO["tenors"][tenor]["lag1"]
        if f"lag2_{tenor}" not in st.session_state:
            st.session_state[f"lag2_{tenor}"] = STARTER_SCENARIO["tenors"][tenor]["lag2"]
        if f"lag3_{tenor}" not in st.session_state:
            st.session_state[f"lag3_{tenor}"] = STARTER_SCENARIO["tenors"][tenor]["lag3"]

def load_starter_scenario():
    st.session_state["MPR"] = STARTER_SCENARIO["MPR"]
    st.session_state["inflation"] = STARTER_SCENARIO["inflation"]
    st.session_state["liquidity"] = STARTER_SCENARIO["liquidity"]
    st.session_state["omo_maturity"] = STARTER_SCENARIO["omo_maturity"]

    for tenor in TENORS:
        st.session_state[f"offered_{tenor}"] = STARTER_SCENARIO["tenors"][tenor]["offered"]
        st.session_state[f"bids_{tenor}"] = STARTER_SCENARIO["tenors"][tenor]["bids"]
        st.session_state[f"lag1_{tenor}"] = STARTER_SCENARIO["tenors"][tenor]["lag1"]
        st.session_state[f"lag2_{tenor}"] = STARTER_SCENARIO["tenors"][tenor]["lag2"]
        st.session_state[f"lag3_{tenor}"] = STARTER_SCENARIO["tenors"][tenor]["lag3"]

def derive_bid_cover(tenor: int) -> float:
    offered = st.session_state.get(f"offered_{tenor}", 0.0)
    bids = st.session_state.get(f"bids_{tenor}", 0.0)
    if offered and offered > 0:
        return round(bids / offered, 4)
    return 0.0

def estimate_liquidity() -> float:
    return round(
        st.session_state["liq_opening"]
        + st.session_state["liq_omo_maturities"]
        + st.session_state["liq_ntb_maturities"]
        + st.session_state["liq_bond_coupons"]
        + st.session_state["liq_faac_other_inflows"]
        - st.session_state["liq_crr_debits"]
        - st.session_state["liq_omo_sales"]
        - st.session_state["liq_ntb_auction_debit"]
        - st.session_state["liq_tax_govt_outflows"]
        - st.session_state["liq_other_drains"],
        2
    )

def scrape_cbn_data():
    data = {
        "policy_rate": st.session_state.get("MPR", 26.50),
        "inflation_rate": st.session_state.get("inflation", 15.06)
    }

    try:
        r = requests.get("https://www.cbn.gov.ng/MonetaryPolicy/decisions.html", timeout=12)
        match = re.search(r"to\s+(\d{1,2}\.?\d{0,2})\s+per cent", r.text, re.IGNORECASE)
        if match:
            data["policy_rate"] = float(match.group(1))
    except Exception:
        pass

    try:
        r = requests.get("https://www.cbn.gov.ng/rates/inflrates.html", timeout=12)
        soup = BeautifulSoup(r.text, "html.parser")
        table = soup.find("table")
        if table:
            cells = table.find_all("tr")[-1].find_all("td")
            infl_str = re.sub(r"[^0-9.]", "", cells[1].text.strip()) if len(cells) > 1 else ""
            if infl_str:
                data["inflation_rate"] = float(infl_str)
    except Exception:
        pass

    return data

def build_model_features(tenor: int) -> dict:
    lag1 = st.session_state[f"lag1_{tenor}"]
    lag2 = st.session_state[f"lag2_{tenor}"]
    lag3 = st.session_state[f"lag3_{tenor}"]

    return {
        "lag1_rate": lag1,
        "lag2_rate": lag2,
        "lag3_rate": lag3,
        "ma3_rate": (lag1 + lag2 + lag3) / 3.0,
        "bid_cover": derive_bid_cover(tenor),
        "MPR": st.session_state["MPR"],
        "inflation": st.session_state["inflation"],
        "liquidity": st.session_state["liquidity"],
        "omo_maturity": st.session_state["omo_maturity"],
        "tenor_years": tenor / 365.0
    }

@st.cache_resource
def load_models():
    return {t: joblib.load(f"gti_model_tenor_{t}D.pkl") for t in TENORS}

# -----------------------------
# INIT
# -----------------------------
init_state()
models = load_models()

# -----------------------------
# HEADER
# -----------------------------
st.markdown("""
<div class="hero">
    <h1>🇳🇬 NG NTB Stop Rate Predictor v4.0</h1>
    <p>Tenor-specific lags, tenor-specific bid cover, shared market inputs, and a cleaner executive layout.</p>
    <p class="small-muted">
        Use shared macro conditions across the auction, then enter each tenor’s own demand and last three stop rates.
    </p>
</div>
""", unsafe_allow_html=True)

left, right = st.columns([1.05, 1.45], gap="large")

# -----------------------------
# LEFT PANEL
# -----------------------------
with left:
    with st.container(border=True):
        st.subheader("1) Shared Market Inputs")

        btn1, btn2 = st.columns(2)
        with btn1:
            if st.button(
                "🔄 Scrape latest MPR & inflation",
                use_container_width=True,
                help="Fetch the latest available MPR and inflation values from the live CBN pages and update the two fields."
            ):
                auto = scrape_cbn_data()
                st.session_state["MPR"] = auto["policy_rate"]
                st.session_state["inflation"] = auto["inflation_rate"]
                st.success(f"Updated: MPR {auto['policy_rate']}% | Inflation {auto['inflation_rate']}%")

        with btn2:
            if st.button(
                "🧩 Load starter scenario",
                use_container_width=True,
                help="Load a starter set of values so you can test the app quickly. Replace them with your actual auction values."
            ):
                load_starter_scenario()
                st.success("Starter scenario loaded.")

        c1, c2 = st.columns(2)
        with c1:
            st.number_input(
                "MPR (%)",
                min_value=0.0,
                step=0.25,
                key="MPR",
                help="Current Monetary Policy Rate used as a shared macro input across all three tenor models."
            )
        with c2:
            st.number_input(
                "Inflation (%)",
                min_value=0.0,
                step=0.01,
                key="inflation",
                help="Latest inflation reading used as a shared macroeconomic input across all three tenor models."
            )

        c3, c4 = st.columns(2)
        with c3:
            st.number_input(
                "Liquidity (NGN bn)",
                step=10.0,
                key="liquidity",
                help="Shared market liquidity condition used by all three tenor models. This is the estimated net system liquidity around the auction."
            )
        with c4:
            st.number_input(
                "OMO Maturity (days)",
                min_value=0,
                step=1,
                key="omo_maturity",
                help="Days to maturity for the relevant OMO condition being used as a market input."
            )

        st.caption(
            "Liquidity is shared across the auction. Bid cover and lags are now tenor-specific."
        )

        with st.expander("Optional: liquidity estimator", expanded=False):
            st.markdown("Estimate liquidity from major inflows and drains, then push it into the model input.")

            a1, a2 = st.columns(2)
            with a1:
                st.number_input("Opening liquidity", step=10.0, key="liq_opening")
                st.number_input("OMO maturities", step=10.0, key="liq_omo_maturities")
                st.number_input("NTB maturities", step=10.0, key="liq_ntb_maturities")
                st.number_input("Bond coupons/redemptions", step=10.0, key="liq_bond_coupons")
                st.number_input("FAAC / other inflows", step=10.0, key="liq_faac_other_inflows")

            with a2:
                st.number_input("CRR debits", step=10.0, key="liq_crr_debits")
                st.number_input("OMO sales", step=10.0, key="liq_omo_sales")
                st.number_input("NTB auction debit", step=10.0, key="liq_ntb_auction_debit")
                st.number_input("Tax / govt outflows", step=10.0, key="liq_tax_govt_outflows")
                st.number_input("Other drains", step=10.0, key="liq_other_drains")

            est_liq = estimate_liquidity()
            st.metric("Estimated liquidity", f"{est_liq:,.2f} bn")

            if st.button(
                "Apply estimated liquidity",
                use_container_width=True,
                help="Copy the estimated liquidity into the main Liquidity input used by the model."
            ):
                st.session_state["liquidity"] = est_liq
                st.success("Estimated liquidity applied to shared market inputs.")

            st.caption(
                "Formula: opening liquidity + OMO maturities + NTB maturities + bond coupons/redemptions + FAAC/other inflows "
                "− CRR debits − OMO sales − NTB auction debit − tax/govt outflows − other drains."
            )

    with st.container(border=True):
        st.subheader("2) Auction Inputs by Tenor")

        tabs = st.tabs(["91D", "182D", "364D"])

        for tenor, tab in zip(TENORS, tabs):
            with tab:
                st.markdown(f"**{tenor}D inputs**")
                x1, x2 = st.columns(2)

                with x1:
                    st.number_input(
                        "Amount offered (NGN bn)",
                        min_value=0.0,
                        step=10.0,
                        key=f"offered_{tenor}",
                        help=f"Amount offered for the {tenor}D tenor. Used with total bids to calculate tenor-specific bid cover."
                    )

                with x2:
                    st.number_input(
                        "Total bids received (NGN bn)",
                        min_value=0.0,
                        step=10.0,
                        key=f"bids_{tenor}",
                        help=f"Total bids received for the {tenor}D tenor."
                    )

                bid_cover_val = derive_bid_cover(tenor)
                st.metric(f"{tenor}D calculated bid cover", f"{bid_cover_val:.2f}x")
                st.caption("Bid cover = total bids received ÷ amount offered")

                l1, l2, l3 = st.columns(3)
                with l1:
                    st.number_input(
                        "Lag 1",
                        step=0.01,
                        key=f"lag1_{tenor}",
                        help=f"Most recent stop rate for the {tenor}D tenor."
                    )
                with l2:
                    st.number_input(
                        "Lag 2",
                        step=0.01,
                        key=f"lag2_{tenor}",
                        help=f"Second most recent stop rate for the {tenor}D tenor."
                    )
                with l3:
                    st.number_input(
                        "Lag 3",
                        step=0.01,
                        key=f"lag3_{tenor}",
                        help=f"Third most recent stop rate for the {tenor}D tenor."
                    )

                ma3 = round(
                    (
                        st.session_state[f"lag1_{tenor}"]
                        + st.session_state[f"lag2_{tenor}"]
                        + st.session_state[f"lag3_{tenor}"]
                    ) / 3.0,
                    4
                )
                st.caption(f"{tenor}D rolling 3-auction average: **{ma3:.4f}%**")

# -----------------------------
# RIGHT PANEL
# -----------------------------
with right:
    with st.container(border=True):
        st.subheader("Scenario Summary")

        summary_rows = []
        for tenor in TENORS:
            lag1 = st.session_state[f"lag1_{tenor}"]
            lag2 = st.session_state[f"lag2_{tenor}"]
            lag3 = st.session_state[f"lag3_{tenor}"]

            summary_rows.append({
                "tenor": f"{tenor}D",
                "amount_offered": st.session_state[f"offered_{tenor}"],
                "total_bids": st.session_state[f"bids_{tenor}"],
                "bid_cover": derive_bid_cover(tenor),
                "lag1": lag1,
                "lag2": lag2,
                "lag3": lag3,
                "ma3_rate": round((lag1 + lag2 + lag3) / 3.0, 4),
                "MPR": st.session_state["MPR"],
                "inflation": st.session_state["inflation"],
                "liquidity": st.session_state["liquidity"],
                "omo_maturity": st.session_state["omo_maturity"]
            })

        st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

        if st.button(
            "🚀 Predict NTB Stop Rates",
            type="primary",
            use_container_width=True,
            help="Run the models using the current shared market inputs and each tenor’s own lags and bid cover."
        ):
            preds = {}
            feature_rows = []

            for tenor, pipe in models.items():
                feat = build_model_features(tenor)
                feature_rows.append({"tenor": f"{tenor}D", **feat})

                try:
                    pred = float(pipe.predict(pd.DataFrame([feat]))[0])
                    preds[tenor] = round(pred, 4)
                except Exception as e:
                    preds[tenor] = f"{type(e).__name__}: {e}"

            st.session_state["predictions"] = preds
            st.session_state["prediction_features"] = feature_rows

    if st.session_state["predictions"] is not None:
        with st.container(border=True):
            st.subheader("Predicted Stop Rates")
            r1, r2, r3 = st.columns(3)

            for idx, tenor in enumerate(TENORS):
                val = st.session_state["predictions"][tenor]
                col = [r1, r2, r3][idx]

                if isinstance(val, (float, int)):
                    col.metric(f"{tenor}D Estimated Stop Rate", f"{val:.2f}%")
                else:
                    col.error(f"{tenor}D failed")
                    col.caption(str(val))

        with st.container(border=True):
            st.subheader("Model Inputs Sent to Each Tenor Model")
            st.dataframe(
                pd.DataFrame(st.session_state["prediction_features"]),
                use_container_width=True,
                hide_index=True
            )

    with st.container(border=True):
        st.subheader("Definitions")
        st.markdown("""
- **Bid cover ratio** = **total bids received ÷ amount offered** for that tenor.  
- **Tenor-specific lags** = the last three stop rates for that tenor only.  
- **Liquidity** = a shared auction-market liquidity condition. A practical treasury estimate is:  
  **opening liquidity + OMO maturities + NTB maturities + bond coupons/redemptions + FAAC/other inflows − CRR debits − OMO sales − NTB auction debit − tax/govt outflows − other drains**.
        """)

        st.info(
            "Current models are still the existing saved models, but each tenor now receives its own lags and bid cover at prediction time. "
            "The next best upgrade is retraining the models end-to-end on tenor-specific histories."
        )

        st.caption(
            "For internal decision support only. Model outputs are estimates, not official auction results."
        )
